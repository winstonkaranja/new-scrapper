"""
SheriaHub Kenya Case Law Scraper for RAG Pipeline
===================================================
Scrapes ~280K Kenya court cases + downloads PDFs.
Outputs structured JSONL for RAG ingestion.

Usage:
    uv run sheriahub_kenya_scraper.py --phase 1
    uv run sheriahub_kenya_scraper.py --phase 2 --email your@email.com
    uv run sheriahub_kenya_scraper.py --phase 2 --cookie "PHPSESSID=abc123..."
    uv run sheriahub_kenya_scraper.py --stats
"""

import json
import logging
import os
import re
import sys
import time
import random
import getpass
import argparse
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────

BASE_URL = "https://sheriahub.com"
LOGIN_URL = f"{BASE_URL}/login"
SITEMAP_INDEX_URL = f"{BASE_URL}/cases/sitemap-index.xml"

OUTPUT_DIR = Path("sheriahub_kenya_data")
URLS_FILE = OUTPUT_DIR / "kenya_case_urls.txt"
JSONL_FILE = OUTPUT_DIR / "kenya_cases.jsonl"
FAILED_FILE = OUTPUT_DIR / "failed_urls.txt"
PROGRESS_FILE = OUTPUT_DIR / "scrape_progress.json"
LOG_FILE = OUTPUT_DIR / "scraper.log"
SESSION_FILE = OUTPUT_DIR / ".session_cookies.json"

# S3 config (matches existing scrapper bucket)
S3_BUCKET = "denning-kenya-law"
S3_PREFIX = "sheriahub"  # s3://denning-kenya-law/sheriahub/{year}/{month}/{slug}.pdf

# Rate limiting
MIN_DELAY = 1.5
MAX_DELAY = 3.0
MAX_RETRIES = 3
RETRY_BACKOFF = 5
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ─── SETUP ──────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

session = requests.Session()
session.headers.update(HEADERS)


# ─── AUTH ───────────────────────────────────────────────────────────────

def login(email: str, password: str = None) -> bool:
    """Login to SheriaHub via POST /login (simple PHP form, no CSRF)."""
    if password is None:
        password = getpass.getpass(f"Enter password for {email}: ")

    logger.info(f"Logging in as {email}...")

    payload = {
        "email": email,
        "password": password,
        "url": f"{BASE_URL}/dashboard",
    }

    resp = session.post(LOGIN_URL, data=payload, allow_redirects=True, timeout=REQUEST_TIMEOUT)

    if "dashboard" in resp.url or "PHPSESSID" in str(session.cookies):
        logger.info("Login successful!")
        save_session_cookies()
        return True

    logger.error("Login failed. Check your credentials.")
    if "Invalid" in resp.text or "incorrect" in resp.text.lower():
        logger.error("  Server says: Invalid email or password.")
    return False


def load_cookie_string(cookie_str: str):
    """Load cookies from a raw cookie string (from browser dev tools)."""
    for pair in cookie_str.split(";"):
        pair = pair.strip()
        if "=" in pair:
            name, value = pair.split("=", 1)
            session.cookies.set(name.strip(), value.strip(), domain="sheriahub.com")
    logger.info(f"Loaded {len(session.cookies)} cookies from string")
    save_session_cookies()


def save_session_cookies():
    """Persist session cookies to disk for resume across runs."""
    cookies = {}
    for c in session.cookies:
        cookies[c.name] = {"value": c.value, "domain": c.domain, "path": c.path}
    with open(SESSION_FILE, "w") as f:
        json.dump(cookies, f)
    logger.info(f"  Session cookies saved to {SESSION_FILE}")


def load_session_cookies() -> bool:
    """Load previously saved session cookies."""
    if not SESSION_FILE.exists():
        return False
    try:
        with open(SESSION_FILE, "r") as f:
            cookies = json.load(f)
        for name, data in cookies.items():
            session.cookies.set(name, data["value"], domain=data.get("domain", "sheriahub.com"))
        logger.info(f"Loaded {len(cookies)} saved session cookies")
        return True
    except Exception as e:
        logger.warning(f"Could not load saved cookies: {e}")
        return False


def ensure_auth(email=None, password=None, cookie=None) -> bool:
    """
    Ensure we have a valid authenticated session.
    Tries: (1) saved cookies, (2) --cookie flag, (3) --email login.
    """
    if load_session_cookies() and verify_auth():
        return True
    logger.warning("Saved session expired or missing, need fresh login")

    if cookie:
        load_cookie_string(cookie)
        if verify_auth():
            return True
        logger.warning("Provided cookies are invalid")

    if email:
        return login(email, password)

    logger.error("No authentication method provided. Use --email or --cookie")
    return False


def verify_auth() -> bool:
    """Verify the current session is authenticated by checking a PDF download."""
    test_url = f"{BASE_URL}/cases/ke/caselaw/republic-v-sanya-2026-kehc-2277-klr.pdf"
    try:
        resp = session.get(test_url, timeout=REQUEST_TIMEOUT, stream=True)
        content_type = resp.headers.get("content-type", "")
        resp.close()
        is_pdf = "pdf" in content_type.lower()
        if is_pdf:
            logger.info("  Session is authenticated (PDF access confirmed)")
        else:
            logger.info(f"  Not authenticated (got {content_type})")
        return is_pdf
    except Exception as e:
        logger.warning(f"Auth check failed: {e}")
        return False


# ─── UTILITIES ──────────────────────────────────────────────────────────

def polite_sleep():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def fetch(url, retries=MAX_RETRIES, stream=False):
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                logger.error(f"  All {retries} attempts failed for {url}")
                return None


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"phase1_done": False, "phase2_last_index": 0, "phase3_last_index": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def generate_case_id(url):
    slug = url.rstrip("/").split("/")[-1]
    return f"ke-{slug}"


# ─── PHASE 1: COLLECT URLS FROM SITEMAPS ────────────────────────────────

def phase1_collect_urls():
    """Parse Kenya case sitemaps to get all case URLs. No auth needed."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Collecting Kenya case URLs from sitemaps")
    logger.info("=" * 60)

    resp = fetch(SITEMAP_INDEX_URL)
    if not resp:
        logger.error("Failed to fetch sitemap index.")
        return

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ElementTree.fromstring(resp.content)

    all_sitemaps = [loc.text for loc in root.findall(".//sm:loc", ns)]
    ke_sitemaps = sorted([u for u in all_sitemaps if "ke-cases-sitemap" in u])

    logger.info(f"Found {len(ke_sitemaps)} Kenya case sitemap files")

    all_urls = []
    for i, sitemap_url in enumerate(ke_sitemaps, 1):
        logger.info(f"  [{i}/{len(ke_sitemaps)}] {sitemap_url}")
        resp = fetch(sitemap_url)
        if not resp:
            continue
        root = ElementTree.fromstring(resp.content)
        urls = [loc.text for loc in root.findall(".//sm:loc", ns)]
        all_urls.extend(urls)
        logger.info(f"    -> {len(urls)} URLs (total: {len(all_urls)})")
        polite_sleep()

    all_urls = list(dict.fromkeys(all_urls))  # deduplicate

    with open(URLS_FILE, "w") as f:
        f.write("\n".join(all_urls))

    logger.info(f"Phase 1 complete: {len(all_urls)} URLs -> {URLS_FILE}")

    progress = load_progress()
    progress["phase1_done"] = True
    progress["total_urls"] = len(all_urls)
    save_progress(progress)


# ─── PHASE 2: SCRAPE CASES + DOWNLOAD PDFS ─────────────────────────────

def parse_date(date_str):
    """Parse various date formats into YYYY-MM-DD."""
    # Remove ordinal suffixes: "19th" / "19 th" -> "19"
    cleaned = re.sub(r"(\d+)\s*(st|nd|rd|th)\b", r"\1", date_str.strip())
    # Ensure space between day and month: "19Feb" -> "19 Feb"
    cleaned = re.sub(r"(\d)([A-Z])", r"\1 \2", cleaned)
    # Collapse extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for fmt in [
        "%Y-%m-%d", "%d/%m/%Y",
        "%d %b %Y", "%d %B %Y", "%B %d, %Y",
    ]:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def extract_json_ld(soup):
    """Extract structured data from JSON-LD script tags."""
    data = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(script.string)
            graph = ld.get("@graph", [ld]) if isinstance(ld, dict) else ld
            for item in graph:
                if isinstance(item, dict) and item.get("@type") == "Article":
                    data["date_published"] = item.get("datePublished", "")
                    data["date_modified"] = item.get("dateModified", "")
                    author = item.get("author", {})
                    if isinstance(author, dict):
                        data["author_name"] = author.get("name", "")
        except (json.JSONDecodeError, TypeError):
            continue
    return data


def parse_case_page(html, url):
    """Extract all structured data from a case page."""
    soup = BeautifulSoup(html, "lxml")

    case = {
        "id": generate_case_id(url),
        "jurisdiction": "Kenya",
        "jurisdiction_code": "ke",
        "source_url": url,
        "pdf_url": url.rstrip("/") + ".pdf",
    }

    # ── Data attributes from #meta-table div ──
    meta_div = soup.find("div", id="meta-table")
    if meta_div:
        case["slug"] = meta_div.get("data-case-slug", "")
        case["unique_id"] = meta_div.get("data-case-uniqueid", "")

    # ── Citation from #citation ──
    citation_el = soup.find(id="citation")
    if citation_el:
        h1 = citation_el.find("h1")
        if h1:
            case["citation"] = h1.get_text(strip=True)

    # ── Metadata table ──
    table = soup.find("table")
    if table:
        first_row = table.find("tr")
        if first_row:
            title_cell = first_row.find("td", colspan=True) or first_row.find("th", colspan=True)
            if title_cell:
                case["title_full"] = title_cell.get_text(strip=True)

        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).upper()
                val = cells[1].get_text(separator=" ", strip=True).rstrip(",").strip()

                if key == "COURT":
                    case["court"] = val
                elif key == "JUDGES":
                    case["judges"] = [j.strip().rstrip(",") for j in val.split(",") if j.strip()]
                elif key == "PARTIES":
                    case["parties_raw"] = val
                elif key == "ADVOCATES":
                    case["advocates"] = [a.strip().rstrip(",") for a in val.split(",") if a.strip()]
                elif key == "CASE CLASS":
                    case["case_class"] = val
                elif key == "CASE ACTION":
                    case["case_action"] = val
                elif key == "CASE OUTCOME":
                    case["case_outcome"] = val
                elif key == "DATE DELIVERED":
                    case["date_delivered_raw"] = val
                    case["date_delivered"] = parse_date(val)
                elif key == "COUNTY":
                    case["county"] = val
                elif key == "COURT DIVISION":
                    case["court_division"] = val

    # ── Title fallback ──
    if "title_full" not in case:
        h_tag = soup.find("h1") or soup.find("h2")
        if h_tag:
            case["title_full"] = h_tag.get_text(strip=True)

    case["title"] = case.get("title_full", "").split(" - ")[0].strip()

    # ── Judgment text (div.judgement#case) ──
    judgment_div = soup.find("div", id="case", class_="judgement")
    if not judgment_div:
        # Fallback: any div with class containing "judgement"
        judgment_div = soup.find("div", class_=lambda c: c and "judgement" in c)

    if judgment_div:
        for tag in judgment_div.find_all("h2"):
            tag.decompose()
        for tag in judgment_div.find_all("a", class_=lambda c: c and "share" in str(c).lower()):
            tag.decompose()

        text = judgment_div.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"Tweet\s*Share\s*LinkedIn", "", text).strip()
        case["judgment_text"] = text
    else:
        case["judgment_text"] = ""

    # ── JSON-LD structured data ──
    json_ld = extract_json_ld(soup)
    if json_ld.get("date_published"):
        case["date_published"] = json_ld["date_published"]
    if json_ld.get("date_modified"):
        case["date_modified"] = json_ld["date_modified"]

    # ── Similar Cases ──
    similar_cases = []
    for p in soup.find_all("p"):
        if p.get_text(strip=True) == "Similar Cases":
            parent = p.find_parent("div")
            if parent:
                for a in parent.find_all("a", href=True):
                    if "/caselaw/" in a["href"]:
                        href = a["href"]
                        full_url = href if href.startswith("http") else BASE_URL + href
                        similar_cases.append({
                            "url": full_url,
                            "title": a.get_text(strip=True),
                        })
            break
    case["similar_cases"] = similar_cases
    case["similar_cases_count"] = len(similar_cases)

    # ── Parse plaintiff/defendant ──
    parties_raw = case.get("parties_raw", "")
    if " v " in parties_raw or " V " in parties_raw:
        parts = re.split(r"\s+[vV]\s+", parties_raw, maxsplit=1)
        case["plaintiff"] = parts[0].strip() if len(parts) > 0 else ""
        case["defendant"] = parts[1].strip() if len(parts) > 1 else ""

    # ── RAG metadata ──
    case["text_length"] = len(case.get("judgment_text", ""))
    case["scraped_at"] = datetime.now(timezone.utc).isoformat()

    return case


def init_s3():
    """Initialize S3 client and verify credentials."""
    try:
        s3 = boto3.client("s3")
        s3.list_buckets()
        logger.info(f"S3 connected (bucket: {S3_BUCKET})")
        return s3
    except NoCredentialsError:
        logger.error("AWS credentials not found. Configure via env vars or ~/.aws/credentials")
        return None
    except Exception as e:
        logger.error(f"S3 init failed: {e}")
        return None


def s3_key_exists(s3, key):
    """Check if an object already exists in S3."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def build_s3_key(slug, date_delivered=None):
    """Build S3 key: sheriahub/{year}/{month}/{slug}.pdf"""
    if date_delivered and len(date_delivered) >= 7:
        year = date_delivered[:4]
        month = date_delivered[5:7]
    else:
        now = datetime.now(timezone.utc)
        year = str(now.year)
        month = f"{now.month:02d}"
    return f"{S3_PREFIX}/{year}/{month}/{slug}.pdf"


def upload_pdf_to_s3(s3, pdf_url, s3_key):
    """Download PDF and stream directly to S3. Returns s3:// URI on success."""
    try:
        resp = fetch(pdf_url, stream=True)
        if not resp:
            return None

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            logger.warning(f"  Not a PDF ({content_type}): {pdf_url}")
            resp.close()
            return None

        pdf_bytes = resp.content
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=pdf_bytes)
        return f"s3://{S3_BUCKET}/{s3_key}"
    except Exception as e:
        logger.error(f"  S3 upload error: {e}")
        return None


def phase2_scrape_and_download(email=None, password=None, cookie=None):
    """Scrape all case pages AND upload PDFs to S3 in one pass."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Scraping cases + uploading PDFs to S3")
    logger.info("=" * 60)

    if not URLS_FILE.exists():
        logger.error(f"URL file not found: {URLS_FILE}. Run --phase 1 first.")
        return

    # ── Initialize S3 ──
    s3 = init_s3()
    if not s3:
        logger.error("S3 not available. Will scrape metadata only (no PDFs).")

    # ── Authenticate for PDF access ──
    if not ensure_auth(email=email, password=password, cookie=cookie):
        logger.error("Cannot authenticate. PDFs will be skipped.")
        logger.error("Use: --email your@email.com  OR  --cookie 'PHPSESSID=...'")
        auth_ok = False
    else:
        auth_ok = True

    can_upload = s3 is not None and auth_ok

    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    progress = load_progress()
    start_index = progress.get("phase2_last_index", 0)

    logger.info(f"Total: {len(urls)} | Resuming from: {start_index}")

    scraped = 0
    pdfs_uploaded = 0
    errors = 0

    with open(JSONL_FILE, "a", encoding="utf-8") as out:
        pbar = tqdm(
            range(start_index, len(urls)),
            initial=start_index,
            total=len(urls),
            desc="Scraping+S3",
        )
        for i in pbar:
            url = urls[i]
            slug = url.rstrip("/").split("/")[-1]

            try:
                # ── 1. Scrape case page ──
                resp = fetch(url)
                if not resp:
                    errors += 1
                    continue

                case_data = parse_case_page(resp.text, url)

                # ── 2. Upload PDF to S3 ──
                if can_upload:
                    s3_key = build_s3_key(slug, case_data.get("date_delivered"))

                    if s3_key_exists(s3, s3_key):
                        case_data["pdf_uploaded"] = True
                        case_data["pdf_s3_uri"] = f"s3://{S3_BUCKET}/{s3_key}"
                    else:
                        pdf_url = case_data["pdf_url"]
                        s3_uri = upload_pdf_to_s3(s3, pdf_url, s3_key)
                        if s3_uri:
                            case_data["pdf_uploaded"] = True
                            case_data["pdf_s3_uri"] = s3_uri
                            pdfs_uploaded += 1
                        else:
                            case_data["pdf_uploaded"] = False
                            # Session might have expired — try re-auth once
                            if not verify_auth():
                                logger.warning("Session expired. Re-authenticating...")
                                if ensure_auth(email=email, password=password, cookie=cookie):
                                    s3_uri = upload_pdf_to_s3(s3, pdf_url, s3_key)
                                    if s3_uri:
                                        case_data["pdf_uploaded"] = True
                                        case_data["pdf_s3_uri"] = s3_uri
                                        pdfs_uploaded += 1
                                else:
                                    can_upload = False
                                    logger.error("Re-auth failed. Continuing without PDFs.")
                else:
                    case_data["pdf_uploaded"] = False

                # ── 3. Write JSONL ──
                out.write(json.dumps(case_data, ensure_ascii=False) + "\n")
                scraped += 1

                pbar.set_postfix(scraped=scraped, s3=pdfs_uploaded, err=errors)

                if i % 10 == 0:
                    progress["phase2_last_index"] = i
                    save_progress(progress)
                    out.flush()

                polite_sleep()

            except KeyboardInterrupt:
                logger.info(f"\nInterrupted at index {i}. Progress saved. Resume anytime.")
                progress["phase2_last_index"] = i
                save_progress(progress)
                return

            except Exception as e:
                logger.error(f"Error [{i}] {url}: {e}")
                errors += 1
                continue

    progress["phase2_last_index"] = len(urls)
    save_progress(progress)

    logger.info(f"\nPhase 2 complete!")
    logger.info(f"  Cases scraped:    {scraped:,}")
    logger.info(f"  PDFs -> S3:       {pdfs_uploaded:,}")
    logger.info(f"  Errors:           {errors:,}")
    logger.info(f"  Output:           {JSONL_FILE}")
    logger.info(f"  S3 bucket:        s3://{S3_BUCKET}/{S3_PREFIX}/")


def log_failed_url(url, reason=""):
    """Append a failed URL to the failed_urls.txt file."""
    with open(FAILED_FILE, "a") as f:
        f.write(f"{url}\t{reason}\n")


def retry_failed(email=None, password=None, cookie=None):
    """Retry all URLs in failed_urls.txt."""
    if not FAILED_FILE.exists():
        # Try to extract from log
        logger.info("No failed_urls.txt found. Extracting from scraper.log...")
        _extract_failed_from_log()

    if not FAILED_FILE.exists() or FAILED_FILE.stat().st_size == 0:
        logger.info("No failed URLs to retry.")
        return

    with open(FAILED_FILE, "r") as f:
        urls = list(set(line.split("\t")[0].strip() for line in f if line.strip()))

    # Separate page URLs from PDF URLs
    page_urls = [u for u in urls if not u.endswith(".pdf")]
    pdf_urls = [u for u in urls if u.endswith(".pdf")]

    logger.info(f"Retrying {len(urls)} failed URLs ({len(page_urls)} pages, {len(pdf_urls)} PDFs)")

    if not ensure_auth(email=email, password=password, cookie=cookie):
        logger.error("Cannot authenticate for retry.")
        return

    s3 = init_s3()
    succeeded = []

    # Retry failed page scrapes
    for url in tqdm(page_urls, desc="Retrying pages"):
        resp = fetch(url)
        if not resp:
            continue
        case_data = parse_case_page(resp.text, url)
        slug = url.rstrip("/").split("/")[-1]

        if s3:
            s3_key = build_s3_key(slug, case_data.get("date_delivered"))
            pdf_url = case_data["pdf_url"]
            s3_uri = upload_pdf_to_s3(s3, pdf_url, s3_key)
            if s3_uri:
                case_data["pdf_uploaded"] = True
                case_data["pdf_s3_uri"] = s3_uri
            else:
                case_data["pdf_uploaded"] = False

        with open(JSONL_FILE, "a", encoding="utf-8") as out:
            out.write(json.dumps(case_data, ensure_ascii=False) + "\n")
        succeeded.append(url)
        polite_sleep()

    # Retry failed PDF uploads
    for pdf_url in tqdm(pdf_urls, desc="Retrying PDFs"):
        slug = pdf_url.rstrip("/").split("/")[-1].replace(".pdf", "")
        if s3:
            s3_key = build_s3_key(slug)
            if s3_key_exists(s3, s3_key):
                succeeded.append(pdf_url)
                continue
            s3_uri = upload_pdf_to_s3(s3, pdf_url, s3_key)
            if s3_uri:
                succeeded.append(pdf_url)
                # Update JSONL record
                _update_jsonl_pdf_status(slug, s3_uri)
        polite_sleep()

    # Rewrite failed_urls.txt with only still-failing URLs
    remaining = [u for u in urls if u not in succeeded]
    with open(FAILED_FILE, "w") as f:
        for u in remaining:
            f.write(f"{u}\n")

    logger.info(f"Retry complete: {len(succeeded)} succeeded, {len(remaining)} still failing")


def _extract_failed_from_log():
    """Extract failed URLs from scraper.log into failed_urls.txt."""
    if not LOG_FILE.exists():
        return
    failed = set()
    with open(LOG_FILE, "r") as f:
        for line in f:
            if "All 3 attempts failed" in line:
                url = line.strip().split()[-1]
                if url.startswith("http"):
                    failed.add(url)
    if failed:
        with open(FAILED_FILE, "w") as f:
            for u in sorted(failed):
                f.write(f"{u}\n")
        logger.info(f"Extracted {len(failed)} failed URLs from log")


def _update_jsonl_pdf_status(slug, s3_uri):
    """Update a JSONL record's PDF status after successful retry."""
    lines = []
    updated = False
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("id") == f"ke-{slug}" and not data.get("pdf_uploaded"):
                data["pdf_uploaded"] = True
                data["pdf_s3_uri"] = s3_uri
                updated = True
            lines.append(json.dumps(data, ensure_ascii=False) + "\n")
    if updated:
        with open(JSONL_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)


# ─── STATS ──────────────────────────────────────────────────────────────

def show_stats():
    if not JSONL_FILE.exists():
        print("No data yet. Run --phase 2 first.")
        return

    courts, classes, years = {}, {}, {}
    total = with_text = with_similar = with_pdf = 0

    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            total += 1
            courts[case.get("court", "?")] = courts.get(case.get("court", "?"), 0) + 1
            cc = case.get("case_class", "?")
            classes[cc] = classes.get(cc, 0) + 1
            d = case.get("date_delivered", "")
            if d and len(d) >= 4:
                y = d[:4]
                years[y] = years.get(y, 0) + 1
            if case.get("text_length", 0) > 100:
                with_text += 1
            if case.get("similar_cases_count", 0) > 0:
                with_similar += 1
            if case.get("pdf_uploaded"):
                with_pdf += 1

    print(f"\n{'='*50}")
    print(f"  KENYA CASES SCRAPE STATISTICS")
    print(f"{'='*50}")
    print(f"  Total cases scraped:   {total:,}")
    print(f"  With judgment text:    {with_text:,}")
    print(f"  With similar cases:    {with_similar:,}")
    print(f"  PDFs on S3:            {with_pdf:,}")

    print(f"\n  Top 10 Courts:")
    for c, n in sorted(courts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {c}: {n:,}")

    print(f"\n  Case Classes:")
    for c, n in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n:,}")

    print(f"\n  Top 10 Years:")
    for y, n in sorted(years.items(), key=lambda x: -x[1])[:10]:
        print(f"    {y}: {n:,}")


# ─── MAIN ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SheriaHub Kenya Scraper")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2],
        help="1 = collect URLs from sitemaps, 2 = scrape pages + download PDFs",
    )
    parser.add_argument("--email", type=str, help="SheriaHub login email")
    parser.add_argument("--password", type=str, help="SheriaHub password (or set SHERIAHUB_PASSWORD env var)")
    parser.add_argument(
        "--cookie", type=str,
        help='Raw cookie string, e.g. "PHPSESSID=abc123; remember_xxx=yyy"',
    )
    parser.add_argument("--stats", action="store_true", help="Show scraping stats")
    parser.add_argument("--retry", action="store_true", help="Retry all failed URLs")

    args = parser.parse_args()

    # Resolve password from flag or env var
    password = args.password or os.environ.get("SHERIAHUB_PASSWORD")

    if args.stats:
        show_stats()
    elif args.retry:
        retry_failed(email=args.email, password=password, cookie=args.cookie)
    elif args.phase == 1:
        phase1_collect_urls()
    elif args.phase == 2:
        phase2_scrape_and_download(email=args.email, password=password, cookie=args.cookie)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  uv run sheriahub_kenya_scraper.py --phase 1")
        print("  uv run sheriahub_kenya_scraper.py --phase 2 --email you@gmail.com")
        print('  uv run sheriahub_kenya_scraper.py --phase 2 --cookie "PHPSESSID=abc..."')
        print("  uv run sheriahub_kenya_scraper.py --retry --email you@gmail.com")
        print("  uv run sheriahub_kenya_scraper.py --stats")


if __name__ == "__main__":
    main()

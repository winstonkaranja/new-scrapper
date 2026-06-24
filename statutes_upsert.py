"""
Pinecone Bulk Upsert for Kenya Statute PDFs
=============================================
Reads statute PDFs from Statutes/ directory (A-Z subfolders), extracts text
with PyMuPDF, chunks, embeds via Google Gemini, and bulk upserts to Pinecone.

Usage:
    uv run python statutes_upsert.py              # run full upsert
    uv run python statutes_upsert.py --test 10    # test with first 10 PDFs
    uv run python statutes_upsert.py --stats      # show index stats
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional

import fitz
from google import genai
from google.genai import types
from pinecone import Pinecone
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} environment variable is required")
    return value


PINECONE_API_KEY = _require_env("PINECONE_API_KEY")
GOOGLE_API_KEY = _require_env("GOOGLE_API_KEY")

INDEX_NAME = "denningv2"
NAMESPACE = "statutes"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMS = 1024

STATUTES_DIR = Path("Statutes")
DATA_DIR = Path("sheriahub_kenya_data")
PROGRESS_PATH = DATA_DIR / "statutes_upsert_progress.json"
FAILURES_PATH = DATA_DIR / "statutes_upsert_failures.log"

S3_BUCKET = "denning-kenya-law"

# Chunking parameters (~6000 tokens target)
CHUNK_SIZE_CHARS = 24_000
OVERLAP_CHARS = 2_000

# Batching and rate limiting
EMBED_BATCH_SIZE = 10
UPSERT_BATCH_SIZE = 100
EMBED_DELAY = 1.0
PROGRESS_SAVE_INTERVAL = 10

MAX_RETRIES = 5
INITIAL_BACKOFF = 10.0

# ─── LOGGING ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─── PDF DISCOVERY ──────────────────────────────────────────────────────

def discover_pdfs() -> list[Path]:
    """
    Find all statute PDFs under Statutes/<letter>/*.pdf.
    Returns paths sorted alphabetically for deterministic resume ordering.
    """
    pdfs = sorted(STATUTES_DIR.rglob("*.pdf"))
    return pdfs


def slugify(filename: str) -> str:
    """Convert a PDF filename to a URL-safe slug. 'Refugees Act.pdf' -> 'refugees-act'."""
    name = Path(filename).stem
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


# ─── PDF TEXT EXTRACTION ────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_citation(text: str) -> str:
    """
    Try to extract the Cap number from statute text.
    Looks for patterns like 'Cap. 173', 'CAP. 6', 'Cap 108A'.
    """
    match = re.search(r"[Cc][Aa][Pp]\.?\s*(\d+[A-Za-z]?)", text[:3000])
    if match:
        return f"Cap. {match.group(1)}"
    return ""


# ─── CHUNKING ───────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE_CHARS characters.

    Strategy: split on paragraph boundaries first (double newline),
    fall back to sentence boundaries (period + space), then hard split.
    """
    if not text or len(text) <= CHUNK_SIZE_CHARS:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE_CHARS

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try paragraph boundary within the last 20% of the chunk
        search_start = start + int(CHUNK_SIZE_CHARS * 0.8)
        boundary = text.rfind("\n\n", search_start, end)

        if boundary == -1:
            # Fall back to sentence boundary
            match = None
            for m in re.finditer(r"\.\s", text[search_start:end]):
                match = m
            if match:
                boundary = search_start + match.start() + 1
            else:
                boundary = end

        chunk = text[start:boundary]
        if not chunk:
            break
        chunks.append(chunk)

        next_start = boundary - OVERLAP_CHARS
        start = max(next_start, start + 1)

    return chunks


# ─── METADATA ───────────────────────────────────────────────────────────

def _classify_statute_authority_rank(slug: str, title: str, citation: str) -> str:
    """Lazy-import the WI-7a classifier so classification logic stays
    single-sourced in audit_authority_ranks.py."""
    from audit_authority_ranks import classify_statute
    rank, _ = classify_statute({
        "slug": slug,
        "title": title,
        "case_id": f"statute-{slug}",
        "case_action": "Statute",
    })
    return rank


def build_metadata(
    slug: str,
    title: str,
    citation: str,
    letter: str,
    full_text_length: int,
    chunk_text_str: str,
    chunk_idx: int,
    total_chunks: int,
) -> dict:
    """Build the Pinecone metadata dict for a single statute vector."""
    return {
        "case_id": f"statute-{slug}",
        "doc_type": "statute",
        "title": title,
        "citation": citation,
        "jurisdiction": "Kenya",
        "court": "",
        "judges": "",
        "case_action": "Statute",
        "date_delivered": "",
        "county": "",
        "plaintiff": "",
        "defendant": "",
        "source_url": "",
        "pdf_s3_uri": f"s3://{S3_BUCKET}/statutes/{letter}/{slug}.pdf",
        "slug": slug,
        "text_length": full_text_length,
        "chunk_index": chunk_idx,
        "total_chunks": total_chunks,
        "chunk_text": chunk_text_str[:CHUNK_SIZE_CHARS],
        # WI-7: authority_rank tagged at ingest time — see pinecone_upsert.py
        "authority_rank": _classify_statute_authority_rank(slug, title, citation),
    }


# ─── EMBEDDING ──────────────────────────────────────────────────────────

def create_gemini_client() -> genai.Client:
    return genai.Client(api_key=GOOGLE_API_KEY)


def embed_texts_with_retry(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Gemini with exponential backoff."""
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMS),
            )
            return [emb.values for emb in result.embeddings]

        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(
                keyword in error_str
                for keyword in ("rate", "limit", "quota", "429", "500", "503", "timeout", "unavailable")
            )

            if is_transient and attempt < MAX_RETRIES:
                logger.warning(
                    f"Embedding attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {backoff:.0f}s..."
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


# ─── PINECONE ───────────────────────────────────────────────────────────

def init_pinecone():
    """Initialize Pinecone client and return the index handle."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    logger.info(f"Connected to Pinecone index '{INDEX_NAME}'.")
    return pc, index


def upsert_with_retry(index, vectors: list[tuple], namespace: str):
    """Upsert a batch of vectors with exponential backoff."""
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            return
        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(
                keyword in error_str
                for keyword in ("rate", "limit", "timeout", "unavailable", "503", "500")
            )

            if is_transient and attempt < MAX_RETRIES:
                logger.warning(
                    f"Upsert attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {backoff:.0f}s..."
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


# ─── PROGRESS ───────────────────────────────────────────────────────────

def load_progress() -> int:
    """Load the last successfully processed file index."""
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r") as f:
                data = json.load(f)
            return data.get("last_file_index", 0)
        except (json.JSONDecodeError, KeyError):
            return 0
    return 0


def save_progress(last_file_index: int):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_file_index": last_file_index}, f)


def log_failure(pdf_path: str, error: str):
    with open(FAILURES_PATH, "a") as f:
        f.write(f"{pdf_path}\t{error}\n")


# ─── CORE PIPELINE ──────────────────────────────────────────────────────

def process_single_pdf(pdf_path: Path) -> list[tuple[str, str, dict]]:
    """
    Extract text from a PDF, chunk it, and return (vector_id, chunk_text, metadata)
    triples ready for embedding. Returns empty list if no text is extractable.
    """
    title = pdf_path.stem
    slug = slugify(pdf_path.name)
    letter = pdf_path.parent.name

    text = extract_pdf_text(pdf_path)
    text = text.strip()

    if not text:
        return []

    citation = extract_citation(text)
    chunks = chunk_text(text)
    total_chunks = len(chunks)

    triples = []
    for chunk_idx, chunk in enumerate(chunks):
        vector_id = f"statute-{slug}_chunk_{chunk_idx}"
        metadata = build_metadata(
            slug=slug,
            title=title,
            citation=citation,
            letter=letter,
            full_text_length=len(text),
            chunk_text_str=chunk,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
        )
        triples.append((vector_id, chunk, metadata))

    return triples


def run_upsert(test_limit: Optional[int] = None):
    """Main pipeline: discover PDFs, extract, chunk, embed, upsert."""

    DATA_DIR.mkdir(exist_ok=True)

    # ── Discover PDFs ──
    all_pdfs = discover_pdfs()
    total_pdfs = len(all_pdfs)
    logger.info(f"Found {total_pdfs} statute PDFs in {STATUTES_DIR}/")

    if total_pdfs == 0:
        logger.error(f"No PDFs found in {STATUTES_DIR}/. Check the directory structure.")
        return

    # ── Apply test limit ──
    if test_limit:
        all_pdfs = all_pdfs[:test_limit]
        logger.info(f"TEST MODE: processing first {test_limit} PDFs")

    # ── Resume ──
    start_index = load_progress()
    if start_index > 0:
        logger.info(f"Resuming from file index {start_index}")

    pdfs_to_process = all_pdfs[start_index:]
    if not pdfs_to_process:
        logger.info("All PDFs already processed. Nothing to do.")
        return

    # ── Initialize clients ──
    gemini_client = create_gemini_client()
    _, index = init_pinecone()

    # ── Pipeline state ──
    vectors_upserted = 0
    pdfs_processed = 0
    pdfs_failed = 0
    pdfs_skipped = 0

    # Accumulate triples across multiple PDFs for efficient batching
    pending_triples: list[tuple[str, str, dict]] = []
    pending_upsert_future: Optional[Future] = None

    executor = ThreadPoolExecutor(max_workers=2)

    pbar = tqdm(
        total=len(pdfs_to_process),
        desc="Statutes",
        unit="pdf",
        dynamic_ncols=True,
    )

    def flush_pipeline(triples: list[tuple[str, str, dict]]) -> int:
        """Embed all pending triples and upsert. Returns vector count."""
        nonlocal pending_upsert_future
        if not triples:
            return 0

        texts = [t[1] for t in triples]
        ids_and_meta = [(t[0], t[2]) for t in triples]

        # Embed in sub-batches with rate limit delay
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]
            embeddings = embed_texts_with_retry(gemini_client, batch_texts)
            all_embeddings.extend(embeddings)
            if i + EMBED_BATCH_SIZE < len(texts):
                time.sleep(EMBED_DELAY)

        # Build vector tuples
        vectors = [
            (vid, emb, meta)
            for (vid, meta), emb in zip(ids_and_meta, all_embeddings)
        ]

        # Wait for any previous upsert to finish
        if pending_upsert_future is not None:
            pending_upsert_future.result()

        # Upsert in sub-batches, last one async for overlap with next embed
        upsert_batches = [
            vectors[i : i + UPSERT_BATCH_SIZE]
            for i in range(0, len(vectors), UPSERT_BATCH_SIZE)
        ]

        for batch in upsert_batches[:-1]:
            upsert_with_retry(index, batch, NAMESPACE)

        if upsert_batches:
            pending_upsert_future = executor.submit(
                upsert_with_retry, index, upsert_batches[-1], NAMESPACE
            )

        return len(vectors)

    current_file_index = start_index

    try:
        for i, pdf_path in enumerate(pdfs_to_process):
            current_file_index = start_index + i

            try:
                triples = process_single_pdf(pdf_path)

                if not triples:
                    pdfs_skipped += 1
                    logger.debug(f"Skipped (no text): {pdf_path.name}")
                    pbar.update(1)
                    continue

                pending_triples.extend(triples)

                # Flush when we have enough chunks for at least one embed batch
                if len(pending_triples) >= EMBED_BATCH_SIZE:
                    count = flush_pipeline(pending_triples)
                    vectors_upserted += count
                    pending_triples = []

                pdfs_processed += 1
                pbar.update(1)
                pbar.set_postfix(
                    vecs=vectors_upserted,
                    skip=pdfs_skipped,
                    fail=pdfs_failed,
                )

            except Exception as e:
                log_failure(str(pdf_path), str(e))
                pdfs_failed += 1
                logger.error(f"Failed: {pdf_path.name}: {e}")
                pbar.update(1)
                continue

            # Save progress every N files
            if (i + 1) % PROGRESS_SAVE_INTERVAL == 0:
                if pending_upsert_future is not None:
                    pending_upsert_future.result()
                    pending_upsert_future = None
                save_progress(current_file_index + 1)

        # Flush remaining triples
        if pending_triples:
            try:
                count = flush_pipeline(pending_triples)
                vectors_upserted += count
            except Exception as e:
                logger.error(f"Final flush failed: {e}")
                pdfs_failed += 1

        # Wait for any in-flight upsert
        if pending_upsert_future is not None:
            pending_upsert_future.result()

        save_progress(current_file_index + 1)
        pbar.close()

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Saving progress...")
        if pending_upsert_future is not None:
            try:
                pending_upsert_future.result(timeout=10)
            except Exception:
                pass
        save_progress(current_file_index)
        pbar.close()
        logger.info(f"Progress saved at file index {current_file_index}. Resume anytime.")
        return

    finally:
        executor.shutdown(wait=False)

    # ── Summary ──
    logger.info(f"\nUpsert complete!")
    logger.info(f"  PDFs processed:    {pdfs_processed:,}")
    logger.info(f"  PDFs skipped:      {pdfs_skipped:,}")
    logger.info(f"  PDFs failed:       {pdfs_failed:,}")
    logger.info(f"  Vectors upserted:  {vectors_upserted:,}")
    logger.info(f"  Progress saved to: {PROGRESS_PATH}")
    if pdfs_failed > 0:
        logger.info(f"  Failures logged:   {FAILURES_PATH}")


# ─── STATS ──────────────────────────────────────────────────────────────

def show_stats():
    """Display Pinecone index statistics and local PDF counts."""
    _, index = init_pinecone()
    stats = index.describe_index_stats()

    print(f"\n{'='*50}")
    print(f"  PINECONE INDEX STATS: {INDEX_NAME}")
    print(f"{'='*50}")
    print(f"  Dimension:         {stats.dimension}")
    print(f"  Total vectors:     {stats.total_vector_count:,}")

    if stats.namespaces:
        print(f"\n  Namespaces:")
        for ns_name, ns_stats in stats.namespaces.items():
            print(f"    {ns_name}: {ns_stats.vector_count:,} vectors")

    # Local progress
    all_pdfs = discover_pdfs()
    last_index = load_progress()
    print(f"\n  Statute PDFs found:   {len(all_pdfs)}")
    print(f"  PDFs processed:       {last_index}")
    print(f"  PDFs remaining:       {max(0, len(all_pdfs) - last_index)}")
    print()


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk upsert Kenya statute PDFs to Pinecone",
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Process only the first N PDFs (for testing)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show Pinecone index stats and exit",
    )

    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.stats:
        show_stats()
    else:
        if not STATUTES_DIR.exists():
            logger.error(f"Statutes directory not found: {STATUTES_DIR}")
            logger.error("Expected structure: Statutes/A/*.pdf, Statutes/B/*.pdf, ...")
            return
        run_upsert(test_limit=args.test)


if __name__ == "__main__":
    main()

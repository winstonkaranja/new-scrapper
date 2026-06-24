"""
Backfill chunk_text metadata for existing Pinecone vectors.
=============================================================
Fixes the 1000-char truncation bug in legacy vectors by re-reading
the source (JSONL for sheriahub, PDFs for statutes), re-chunking
with the same algorithm as the upsert scripts, and calling
index.update() with the FULL chunk text — no re-embedding.

Usage:
    # Dry-run (default — prints what it would update, makes NO writes)
    PINECONE_API_KEY=... uv run python backfill_chunk_text_metadata.py --namespace sheriahub

    # Limit to first 100 records for a pilot:
    PINECONE_API_KEY=... uv run python backfill_chunk_text_metadata.py --namespace sheriahub --limit 100 --dry-run

    # Really write (only after you've verified a dry-run pilot):
    PINECONE_API_KEY=... uv run python backfill_chunk_text_metadata.py --namespace sheriahub --yes-really-write

    # Resume after interruption (uses progress JSON):
    PINECONE_API_KEY=... uv run python backfill_chunk_text_metadata.py --namespace sheriahub --yes-really-write --resume

Safety:
    - Default mode is --dry-run. You must pass --yes-really-write to actually update.
    - Progress is saved every PROGRESS_SAVE_INTERVAL records.
    - If a vector does not exist in Pinecone, it's logged as "missing" and skipped
      (no phantom upserts).
    - If update fails, it's logged to a failures file and skipped.
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from pinecone import Pinecone

# Reuse chunking from the upsert scripts so byte-for-byte identical output.
# Import here is local to each module (different source per namespace).
from pinecone_upsert import chunk_text as chunk_case_text
from statutes_upsert import (
    chunk_text as chunk_statute_text,
    discover_pdfs,
    extract_pdf_text,
    extract_citation,
    slugify,
)


# ─── CONFIG ─────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} environment variable is required")
    return value


INDEX_NAME = "denningv2"
CHUNK_SIZE_CHARS = 24_000

DATA_DIR = Path("sheriahub_kenya_data")
JSONL_PATH = DATA_DIR / "kenya_cases.jsonl"
PROGRESS_PATH = DATA_DIR / "backfill_progress.json"
FAILURES_PATH = DATA_DIR / "backfill_failures.log"

# Throttling: Pinecone serverless can handle many QPS but be polite.
MAX_WORKERS = 10
UPDATE_RETRY_MAX = 3
INITIAL_BACKOFF = 5.0
PROGRESS_SAVE_INTERVAL = 100

# ─── LOGGING ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "backfill.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── PROGRESS ───────────────────────────────────────────────────────────


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"sheriahub_last_line": 0, "statutes_completed_slugs": []}


def save_progress(state: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(state, indent=2))


def log_failure(vec_id: str, error: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(FAILURES_PATH, "a") as f:
        f.write(f"{vec_id}\t{error}\n")


# ─── UPDATE PRIMITIVES ──────────────────────────────────────────────────


def update_one_with_retry(index, vec_id: str, new_chunk_text: str, namespace: str, dry_run: bool) -> tuple[str, str]:
    """
    Return (vec_id, status) where status is one of:
      "updated"     — write succeeded (or would succeed in dry-run)
      "skipped"     — vector not in Pinecone (phantom ID)
      "failed"      — retries exhausted
    """
    if dry_run:
        return vec_id, "updated"

    backoff = INITIAL_BACKOFF
    for attempt in range(1, UPDATE_RETRY_MAX + 1):
        try:
            index.update(
                id=vec_id,
                set_metadata={"chunk_text": new_chunk_text[:CHUNK_SIZE_CHARS]},
                namespace=namespace,
            )
            return vec_id, "updated"
        except Exception as e:
            err_str = str(e).lower()
            # 404 / missing vector => not a retry, it's a gap
            if "404" in err_str or "not found" in err_str:
                return vec_id, "skipped"
            if attempt < UPDATE_RETRY_MAX:
                logger.warning(
                    "update %s attempt %d/%d failed: %s — retrying in %.0fs",
                    vec_id, attempt, UPDATE_RETRY_MAX, e, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                log_failure(vec_id, str(e))
                return vec_id, "failed"
    return vec_id, "failed"


def update_batch_concurrent(index, updates: list[tuple[str, str]], namespace: str, dry_run: bool) -> dict:
    """
    updates: list of (vector_id, full_chunk_text).
    Returns status counts: {"updated": N, "skipped": N, "failed": N}.
    """
    counts = {"updated": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(update_one_with_retry, index, vec_id, text, namespace, dry_run)
            for vec_id, text in updates
        ]
        for fut in as_completed(futures):
            _, status = fut.result()
            counts[status] += 1
    return counts


# ─── SHERIAHUB BACKFILL (from JSONL) ────────────────────────────────────


def backfill_sheriahub(index, limit: int | None, dry_run: bool, resume: bool) -> dict:
    state = load_progress() if resume else {"sheriahub_last_line": 0, "statutes_completed_slugs": []}
    start_line = state.get("sheriahub_last_line", 0)

    logger.info("sheriahub backfill: start_line=%d limit=%s dry_run=%s", start_line, limit, dry_run)
    totals = {"records_processed": 0, "chunks_generated": 0, "updated": 0, "skipped": 0, "failed": 0}

    batch: list[tuple[str, str]] = []
    BATCH_FLUSH = 200  # flush every N chunks

    with open(JSONL_PATH, "r") as f:
        for line_num, line in enumerate(f):
            if line_num < start_line:
                continue
            if limit and totals["records_processed"] >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            judgment = (rec.get("judgment_text") or "").strip()
            if not judgment:
                continue
            record_id = rec.get("id")
            if not record_id:
                continue

            chunks = chunk_case_text(judgment)
            for chunk_idx, chunk in enumerate(chunks):
                vec_id = f"{record_id}_chunk_{chunk_idx}"
                batch.append((vec_id, chunk))
            totals["chunks_generated"] += len(chunks)
            totals["records_processed"] += 1

            if len(batch) >= BATCH_FLUSH:
                counts = update_batch_concurrent(index, batch, namespace="sheriahub", dry_run=dry_run)
                totals["updated"] += counts["updated"]
                totals["skipped"] += counts["skipped"]
                totals["failed"] += counts["failed"]
                batch = []

            if totals["records_processed"] % PROGRESS_SAVE_INTERVAL == 0:
                state["sheriahub_last_line"] = line_num + 1
                save_progress(state)
                logger.info(
                    "sheriahub: records=%d chunks=%d updated=%d skipped=%d failed=%d",
                    totals["records_processed"], totals["chunks_generated"],
                    totals["updated"], totals["skipped"], totals["failed"],
                )

    if batch:
        counts = update_batch_concurrent(index, batch, namespace="sheriahub", dry_run=dry_run)
        totals["updated"] += counts["updated"]
        totals["skipped"] += counts["skipped"]
        totals["failed"] += counts["failed"]

    state["sheriahub_last_line"] = line_num + 1 if totals["records_processed"] else start_line
    save_progress(state)
    return totals


# ─── STATUTES BACKFILL (from PDFs) ──────────────────────────────────────


def backfill_statutes(index, limit: int | None, dry_run: bool, resume: bool) -> dict:
    state = load_progress() if resume else {"sheriahub_last_line": 0, "statutes_completed_slugs": []}
    completed = set(state.get("statutes_completed_slugs", []))

    pdfs = discover_pdfs()
    logger.info("statutes backfill: %d PDFs discovered, %d already completed, dry_run=%s",
                len(pdfs), len(completed), dry_run)

    totals = {"records_processed": 0, "chunks_generated": 0, "updated": 0, "skipped": 0, "failed": 0}
    batch: list[tuple[str, str]] = []
    BATCH_FLUSH = 200

    for pdf_path in pdfs:
        if limit and totals["records_processed"] >= limit:
            break
        slug = slugify(pdf_path.name)
        if slug in completed:
            continue
        try:
            text = extract_pdf_text(pdf_path)
        except Exception as e:
            logger.warning("pdf extract failed %s: %s", pdf_path, e)
            continue

        chunks = chunk_statute_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            vec_id = f"statute-{slug}_chunk_{chunk_idx}"
            batch.append((vec_id, chunk))
        totals["chunks_generated"] += len(chunks)
        totals["records_processed"] += 1
        completed.add(slug)

        if len(batch) >= BATCH_FLUSH:
            counts = update_batch_concurrent(index, batch, namespace="statutes", dry_run=dry_run)
            totals["updated"] += counts["updated"]
            totals["skipped"] += counts["skipped"]
            totals["failed"] += counts["failed"]
            batch = []

        if totals["records_processed"] % PROGRESS_SAVE_INTERVAL == 0:
            state["statutes_completed_slugs"] = sorted(completed)
            save_progress(state)
            logger.info(
                "statutes: pdfs=%d chunks=%d updated=%d skipped=%d failed=%d",
                totals["records_processed"], totals["chunks_generated"],
                totals["updated"], totals["skipped"], totals["failed"],
            )

    if batch:
        counts = update_batch_concurrent(index, batch, namespace="statutes", dry_run=dry_run)
        totals["updated"] += counts["updated"]
        totals["skipped"] += counts["skipped"]
        totals["failed"] += counts["failed"]

    state["statutes_completed_slugs"] = sorted(completed)
    save_progress(state)
    return totals


# ─── MAIN ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backfill chunk_text metadata on denningv2")
    parser.add_argument("--namespace", choices=["sheriahub", "statutes", "all"], default="sheriahub")
    parser.add_argument("--limit", type=int, help="Stop after N source records (for pilot runs)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Print actions, make no writes (default)")
    parser.add_argument("--yes-really-write", action="store_true", help="Opt-in to real writes (disables --dry-run)")
    parser.add_argument("--resume", action="store_true", help="Resume from backfill_progress.json")
    args = parser.parse_args()

    dry_run = not args.yes_really_write

    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    logger.info("Index '%s': %d total vectors", INDEX_NAME, stats.total_vector_count)

    mode = "DRY-RUN (no writes)" if dry_run else "LIVE WRITES"
    logger.info("=== Backfill mode: %s ===", mode)

    if args.namespace in ("sheriahub", "all"):
        totals = backfill_sheriahub(index, args.limit, dry_run, args.resume)
        logger.info("sheriahub totals: %s", totals)

    if args.namespace in ("statutes", "all"):
        totals = backfill_statutes(index, args.limit, dry_run, args.resume)
        logger.info("statutes totals: %s", totals)

    logger.info("=== Backfill done (%s) ===", mode)


if __name__ == "__main__":
    main()

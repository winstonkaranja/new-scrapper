"""
Pinecone Bulk Upsert for Kenya Court Cases
===========================================
Reads JSONL records, chunks judgment text, embeds via Google Gemini,
and bulk upserts to Pinecone with resume capability.

Usage:
    uv run python pinecone_upsert.py              # run full upsert
    uv run python pinecone_upsert.py --test 100   # test with first 100 records
    uv run python pinecone_upsert.py --stats      # show index stats
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

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
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
NAMESPACE = "sheriahub"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMS = 1024

DATA_DIR = Path("sheriahub_kenya_data")
JSONL_PATH = DATA_DIR / "kenya_cases.jsonl"
PROGRESS_PATH = DATA_DIR / "upsert_progress.json"
FAILURES_PATH = DATA_DIR / "upsert_failures.log"

# Chunking parameters (~6000 tokens target)
CHUNK_SIZE_CHARS = 24_000
OVERLAP_CHARS = 2_000

# Batching parameters
EMBED_BATCH_SIZE = 10        # texts per Gemini API call (smaller to avoid rate limits)
UPSERT_BATCH_SIZE = 100     # vectors per Pinecone upsert call
PROGRESS_SAVE_INTERVAL = 50 # save progress every N records
EMBED_DELAY = 1.0           # seconds between embedding batches (rate limit guard)

MAX_RETRIES = 5
INITIAL_BACKOFF = 10.0      # seconds, doubles each retry (10, 20, 40, 80, 160)

# ─── LOGGING ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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

        # Try paragraph boundary (double newline) within the last 20% of the chunk
        search_start = start + int(CHUNK_SIZE_CHARS * 0.8)
        boundary = text.rfind("\n\n", search_start, end)

        if boundary == -1:
            # Fall back to sentence boundary (. followed by space or newline)
            match = None
            for m in re.finditer(r'\.\s', text[search_start:end]):
                match = m
            if match:
                boundary = search_start + match.start() + 1  # include the period
            else:
                # Hard split at chunk boundary
                boundary = end

        chunk = text[start:boundary]
        if not chunk:
            break
        chunks.append(chunk)

        next_start = boundary - OVERLAP_CHARS
        # Prevent infinite loop: always advance past the previous start
        start = max(next_start, start + 1)

    return chunks


# ─── METADATA ───────────────────────────────────────────────────────────


def _classify_case_authority_rank(record: dict) -> str:
    """Lazy-import the WI-7a classifier to avoid circular imports and
    keep classification logic single-sourced in audit_authority_ranks.py.
    """
    from audit_authority_ranks import classify_case
    # audit_authority_ranks.classify_case expects a metadata-shaped dict
    # with court/citation/case_action keys — our record dict has the same
    # fields under the same names (the upsert populates them directly).
    rank, _ = classify_case({
        "court": record.get("court", ""),
        "citation": record.get("citation", ""),
        "case_action": record.get("case_action", ""),
    })
    return rank


def build_metadata(record: dict, chunk_text_str: str, chunk_idx: int, total_chunks: int) -> dict:
    """Build the Pinecone metadata dict for a single vector."""
    return {
        "case_id": record["id"],
        "citation": record.get("citation", ""),
        "court": record.get("court", ""),
        "judges": ", ".join(record.get("judges", [])),
        "case_action": record.get("case_action", ""),
        "date_delivered": record.get("date_delivered", ""),
        "county": record.get("county", ""),
        "plaintiff": record.get("plaintiff", ""),
        "defendant": record.get("defendant", ""),
        "title": record.get("title", ""),
        "source_url": record.get("pdf_s3_uri", record.get("source_url", "")),
        "pdf_s3_uri": record.get("pdf_s3_uri", ""),
        "slug": record.get("slug", ""),
        "jurisdiction": record.get("jurisdiction", "Kenya"),
        "text_length": record.get("text_length", 0),
        "chunk_index": chunk_idx,
        "total_chunks": total_chunks,
        "chunk_text": chunk_text_str[:CHUNK_SIZE_CHARS],
        # WI-7: authority_rank tagged at ingest time so fresh scrapes don't
        # require a follow-up backfill pass. Classification is deterministic
        # per audit_authority_ranks.classify_case.
        "authority_rank": _classify_case_authority_rank(record),
    }


def build_vector_id(record_id: str, chunk_idx: int) -> str:
    """Build deterministic vector ID: {record_id}_chunk_{chunk_idx}."""
    return f"{record_id}_chunk_{chunk_idx}"


# ─── EMBEDDING ──────────────────────────────────────────────────────────

def create_gemini_client() -> genai.Client:
    """Initialize Google Gemini client."""
    return genai.Client(api_key=GOOGLE_API_KEY)


def embed_texts_with_retry(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts via Gemini with exponential backoff.

    Returns a list of embedding vectors (list of floats) in the same order.
    Raises on exhausted retries so the caller can handle it.
    """
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
                logger.warning(f"Embedding attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


# ─── PINECONE ───────────────────────────────────────────────────────────

def init_pinecone() -> tuple:
    """Initialize Pinecone client and ensure the index exists. Returns (pc, index)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        logger.info(f"Creating Pinecone index '{INDEX_NAME}' (serverless, cosine, {EMBEDDING_DIMS}d)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status.get("ready"):
            logger.info("  Waiting for index to be ready...")
            time.sleep(3)
        logger.info(f"  Index '{INDEX_NAME}' is ready.")
    else:
        logger.info(f"Using existing Pinecone index '{INDEX_NAME}'.")

    index = pc.Index(INDEX_NAME)
    return pc, index


def upsert_with_retry(index, vectors: list[tuple], namespace: str):
    """Upsert a batch of vectors with exponential backoff on transient errors."""
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
                logger.warning(f"Upsert attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


# ─── PROGRESS ───────────────────────────────────────────────────────────

def load_progress() -> int:
    """Load the last successfully processed line number."""
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r") as f:
                data = json.load(f)
            return data.get("last_line", 0)
        except (json.JSONDecodeError, KeyError):
            return 0
    return 0


def save_progress(last_line: int):
    """Persist the last successfully processed line number."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"last_line": last_line}, f)


def log_failure(record_id: str, error: str):
    """Append a failed record to the failures log."""
    with open(FAILURES_PATH, "a") as f:
        f.write(f"{record_id}\t{error}\n")


# ─── CORE PIPELINE ──────────────────────────────────────────────────────

def process_records(records: list[dict]) -> list[tuple[str, str, dict]]:
    """
    Process a batch of records into (vector_id, chunk_text, metadata) triples.

    Returns a flat list of all chunks across all records, ready for embedding.
    Skips records with empty judgment_text.
    """
    triples = []

    for record in records:
        judgment = record.get("judgment_text", "").strip()
        if not judgment:
            continue

        chunks = chunk_text(judgment)
        total_chunks = len(chunks)

        for chunk_idx, chunk in enumerate(chunks):
            vector_id = build_vector_id(record["id"], chunk_idx)
            metadata = build_metadata(record, chunk, chunk_idx, total_chunks)
            triples.append((vector_id, chunk, metadata))

    return triples


def run_upsert(test_limit: Optional[int] = None):
    """Main upsert pipeline: read JSONL, chunk, embed, upsert."""

    # ── Initialize clients ──
    gemini_client = create_gemini_client()
    _, index = init_pinecone()

    # ── Load resume state ──
    start_line = load_progress()
    if start_line > 0:
        logger.info(f"Resuming from line {start_line}")

    # ── Count total lines for progress bar ──
    if test_limit:
        total_lines = test_limit
        logger.info(f"TEST MODE: processing first {test_limit} records")
    else:
        with open(JSONL_PATH, "r") as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"Total records in JSONL: {total_lines:,}")

    effective_total = min(total_lines, test_limit) if test_limit else total_lines
    records_to_process = max(0, effective_total - start_line)

    if records_to_process == 0:
        logger.info("All records already processed. Nothing to do.")
        return

    # ── Pipeline ──
    vectors_upserted = 0
    records_processed = 0
    records_failed = 0

    # Accumulator for vectors waiting to be embedded and upserted
    pending_triples: list[tuple[str, str, dict]] = []
    pending_upsert_future: Optional[Future] = None

    executor = ThreadPoolExecutor(max_workers=2)

    pbar = tqdm(
        total=records_to_process,
        desc="Upserting",
        unit="rec",
        dynamic_ncols=True,
    )

    def flush_pipeline(triples: list[tuple[str, str, dict]]):
        """Embed all pending triples and upsert in batches. Returns vector count."""
        nonlocal pending_upsert_future
        if not triples:
            return 0

        texts = [t[1] for t in triples]
        ids_and_meta = [(t[0], t[2]) for t in triples]

        # Embed in sub-batches of EMBED_BATCH_SIZE with rate limit delay
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]
            embeddings = embed_texts_with_retry(gemini_client, batch_texts)
            all_embeddings.extend(embeddings)
            if i + EMBED_BATCH_SIZE < len(texts):
                time.sleep(EMBED_DELAY)

        # Build vector tuples: (id, values, metadata)
        vectors = [
            (vid, emb, meta)
            for (vid, meta), emb in zip(ids_and_meta, all_embeddings)
        ]

        # Wait for any previous upsert to finish before starting new ones
        if pending_upsert_future is not None:
            pending_upsert_future.result()

        # Upsert in sub-batches of UPSERT_BATCH_SIZE
        # Submit the last sub-batch asynchronously; do the rest synchronously
        upsert_batches = [
            vectors[i : i + UPSERT_BATCH_SIZE]
            for i in range(0, len(vectors), UPSERT_BATCH_SIZE)
        ]

        for batch in upsert_batches[:-1]:
            upsert_with_retry(index, batch, NAMESPACE)

        if upsert_batches:
            # Submit last batch to overlap with next embedding call
            pending_upsert_future = executor.submit(
                upsert_with_retry, index, upsert_batches[-1], NAMESPACE
            )

        return len(vectors)

    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            # Skip already-processed lines
            for _ in range(start_line):
                next(f, None)

            line_num = start_line
            batch_records: list[dict] = []

            for line in f:
                if test_limit and line_num >= test_limit:
                    break

                line_num += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    log_failure(f"line_{line_num}", f"JSON decode error: {e}")
                    records_failed += 1
                    pbar.update(1)
                    continue

                judgment = record.get("judgment_text", "").strip()
                if not judgment:
                    pbar.update(1)
                    records_processed += 1
                    continue

                batch_records.append(record)

                # Process in batches to amortize embedding API calls.
                # We accumulate records until we have enough chunks to fill
                # at least one embedding batch, then flush.
                if len(batch_records) >= EMBED_BATCH_SIZE:
                    try:
                        triples = process_records(batch_records)
                        count = flush_pipeline(triples)
                        vectors_upserted += count
                    except Exception as e:
                        for rec in batch_records:
                            log_failure(rec["id"], str(e))
                            records_failed += 1
                        logger.error(f"Batch failed at line {line_num}: {e}")

                    records_processed += len(batch_records)
                    pbar.update(len(batch_records))
                    pbar.set_postfix(
                        vectors=vectors_upserted,
                        failed=records_failed,
                    )
                    batch_records = []

                # Save progress periodically
                if line_num % PROGRESS_SAVE_INTERVAL == 0:
                    # Ensure any pending upsert completes before recording progress
                    if pending_upsert_future is not None:
                        pending_upsert_future.result()
                        pending_upsert_future = None
                    save_progress(line_num)

            # Flush remaining records
            if batch_records:
                try:
                    triples = process_records(batch_records)
                    count = flush_pipeline(triples)
                    vectors_upserted += count
                except Exception as e:
                    for rec in batch_records:
                        log_failure(rec["id"], str(e))
                        records_failed += 1
                    logger.error(f"Final batch failed: {e}")

                records_processed += len(batch_records)
                pbar.update(len(batch_records))

        # Wait for any in-flight upsert
        if pending_upsert_future is not None:
            pending_upsert_future.result()

        save_progress(line_num)
        pbar.close()

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Saving progress...")
        if pending_upsert_future is not None:
            try:
                pending_upsert_future.result(timeout=10)
            except Exception:
                pass
        save_progress(line_num)
        pbar.close()
        logger.info(f"Progress saved at line {line_num}. Resume anytime.")
        return

    finally:
        executor.shutdown(wait=False)

    # ── Summary ──
    logger.info(f"\nUpsert complete!")
    logger.info(f"  Records processed: {records_processed:,}")
    logger.info(f"  Records failed:    {records_failed:,}")
    logger.info(f"  Vectors upserted:  {vectors_upserted:,}")
    logger.info(f"  Progress saved to: {PROGRESS_PATH}")
    if records_failed > 0:
        logger.info(f"  Failures logged:   {FAILURES_PATH}")


# ─── STATS ──────────────────────────────────────────────────────────────

def show_stats():
    """Display Pinecone index statistics."""
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

    # Show local progress
    last_line = load_progress()
    if last_line > 0:
        print(f"\n  Local progress:    {last_line:,} lines processed")
    print()


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk upsert Kenya court cases to Pinecone",
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Process only the first N records (for testing)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show Pinecone index stats and exit",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        if not JSONL_PATH.exists():
            logger.error(f"JSONL file not found: {JSONL_PATH}")
            logger.error("Run the scraper first to generate case data.")
            return
        run_upsert(test_limit=args.test)


if __name__ == "__main__":
    main()

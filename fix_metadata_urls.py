"""
Fix Pinecone Metadata: Replace source_url with S3 Key
======================================================
Reads JSONL to get case IDs + S3 URIs, then builds vector IDs
and updates Pinecone metadata directly — no slow listing needed.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pinecone import Pinecone
from tqdm import tqdm

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} environment variable is required")
    return value


PINECONE_API_KEY = _require_env("PINECONE_API_KEY")
INDEX_NAME = "denningv2"
NAMESPACE = "sheriahub"
S3_BUCKET = "denning-kenya-law"
S3_PREFIX = "sheriahub"
JSONL_PATH = Path("sheriahub_kenya_data/kenya_cases.jsonl")
PROGRESS_PATH = Path("sheriahub_kenya_data/fix_urls_progress.json")
CHUNK_SIZE_CHARS = 24_000
MAX_WORKERS = 8


def build_s3_uri(slug, date_delivered):
    year = date_delivered[:4] if date_delivered and len(date_delivered) >= 4 else "unknown"
    month = date_delivered[5:7] if date_delivered and len(date_delivered) >= 7 else "00"
    return f"s3://{S3_BUCKET}/{S3_PREFIX}/{year}/{month}/{slug}.pdf"


def estimate_chunks(text_length):
    """Estimate number of chunks for a record based on text length."""
    if text_length <= CHUNK_SIZE_CHARS:
        return 1
    # Account for overlap
    return max(1, (text_length - CHUNK_SIZE_CHARS) // (CHUNK_SIZE_CHARS - 2000) + 2)


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"last_line": 0, "updated": 0, "skipped": 0}


def save_progress(state):
    PROGRESS_PATH.write_text(json.dumps(state))


def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    state = load_progress()
    start_line = state["last_line"]
    updated = state["updated"]
    skipped = state["skipped"]

    # Count total lines
    total = sum(1 for _ in open(JSONL_PATH))
    print(f"Total JSONL records: {total}, resuming from line {start_line}")

    # Build updates from JSONL — we know the vector IDs from case_id + chunk count
    updates = []  # (vector_id, s3_uri)

    with open(JSONL_PATH) as f:
        for i, line in enumerate(tqdm(f, total=total, desc="Reading JSONL")):
            if i < start_line:
                continue
            rec = json.loads(line)
            case_id = rec["id"]
            s3_uri = rec.get("pdf_s3_uri", "")
            if not s3_uri:
                slug = rec.get("slug", "")
                date = rec.get("date_delivered", "")
                if slug:
                    s3_uri = build_s3_uri(slug, date)
                else:
                    continue

            text_len = rec.get("text_length", 0)
            num_chunks = estimate_chunks(text_len)

            for c in range(num_chunks):
                vid = f"{case_id}_chunk_{c}"
                updates.append((vid, s3_uri))

    print(f"Built {len(updates)} vector updates")

    # Apply updates in batches
    batch_size = 100
    pbar = tqdm(total=len(updates), desc="Updating Pinecone")

    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for vid, s3_uri in batch:
                fut = executor.submit(
                    index.update,
                    id=vid,
                    set_metadata={"source_url": s3_uri},
                    namespace=NAMESPACE,
                )
                futures.append(fut)

            for fut in as_completed(futures):
                try:
                    fut.result()
                    updated += 1
                except Exception:
                    skipped += 1
                pbar.update(1)

        # Save progress every 50 batches
        if (i // batch_size) % 50 == 0:
            line_progress = min(start_line + (i * total // len(updates)), total)
            save_progress({"last_line": line_progress, "updated": updated, "skipped": skipped})

    pbar.close()
    save_progress({"last_line": total, "updated": updated, "skipped": skipped})
    print(f"\nDone! Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()

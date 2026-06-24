"""
Backfill `authority_rank` metadata for existing Pinecone vectors — WI-7b.

Re-classifies every existing vector in both denningv2 namespaces and
writes the `authority_rank` field via `index.update(id=..., set_metadata=...)`.
No re-embedding, no chunk rewrite — metadata-only update.

Shares the classifier implementation with `audit_authority_ranks.py`
(WI-7a) so the audit and migration agree on rank assignment by
construction — if one says a document is `supreme_court`, the other
writes `supreme_court` on it.

Usage:
    # Dry-run (default — print what would change, no writes)
    PINECONE_API_KEY=... uv run python backfill_authority_rank.py --namespace statutes

    # Pilot on one namespace with a small limit
    PINECONE_API_KEY=... uv run python backfill_authority_rank.py \\
        --namespace statutes --limit 50 --yes-really-write

    # Full write (after dry-run + advocate review signoff per WI-7e)
    PINECONE_API_KEY=... uv run python backfill_authority_rank.py \\
        --namespace all --yes-really-write --resume

Safety:
    - Dry-run is the default. `--yes-really-write` required for writes.
    - Resumable via progress JSON checkpoint every 100 docs.
    - Idempotent: re-running after completion is a no-op (documents that
      already carry the correct authority_rank are skipped).
    - Failed updates logged to backfill_authority_rank_failures.log.

Plan ref: /Users/winston/.claude/plans/calm-fluttering-flute.md §WI-7b.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pinecone import Pinecone

# Reuse the classifier from WI-7a audit — single source of truth for
# rank assignment logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_authority_ranks import (  # noqa: E402
    classify_case,
    classify_statute,
    iter_ids,
    FETCH_BATCH,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} environment variable is required")
    return value


INDEX_NAME = "denningv2"
DATA_DIR = Path("sheriahub_kenya_data")
PROGRESS_PATH = DATA_DIR / "authority_rank_backfill_progress.json"
FAILURES_PATH = DATA_DIR / "backfill_authority_rank_failures.log"

MAX_WORKERS = 10              # parallel update threads
UPDATE_RETRY_MAX = 3
INITIAL_BACKOFF = 5.0
PROGRESS_SAVE_INTERVAL = 100  # save progress every N documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "authority_rank_backfill.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── Progress / failure tracking ──────────────────────────────────────────


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"sheriahub_completed_doc_keys": [], "statutes_completed_doc_keys": []}


def save_progress(state: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(state, indent=2))


def log_failure(vec_id: str, error: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(FAILURES_PATH, "a") as f:
        f.write(f"{vec_id}\t{error}\n")


# ─── Update primitives ────────────────────────────────────────────────────


def update_one(index, vec_id: str, authority_rank: str, namespace: str, dry_run: bool) -> tuple[str, str]:
    """Return (vec_id, status). status ∈ {'updated', 'skipped', 'failed'}."""
    if dry_run:
        return vec_id, "updated"

    backoff = INITIAL_BACKOFF
    for attempt in range(1, UPDATE_RETRY_MAX + 1):
        try:
            index.update(
                id=vec_id,
                set_metadata={"authority_rank": authority_rank},
                namespace=namespace,
            )
            return vec_id, "updated"
        except Exception as e:
            err_str = str(e).lower()
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


def update_batch_concurrent(
    index, updates: list[tuple[str, str]], namespace: str, dry_run: bool
) -> dict:
    """Updates: list of (vec_id, authority_rank). Returns status counts."""
    counts = {"updated": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(update_one, index, vec_id, rank, namespace, dry_run)
            for vec_id, rank in updates
        ]
        for fut in as_completed(futures):
            _, status = fut.result()
            counts[status] += 1
    return counts


# ─── Namespace backfill ───────────────────────────────────────────────────


def backfill_namespace(
    index, namespace: str, limit: int | None, dry_run: bool, resume: bool
) -> dict:
    """Classify every vector in `namespace` and write authority_rank.

    De-duplicates by document key so each unique document gets classified
    once, but the metadata update fires for every chunk (so chunks of the
    same document all carry the same rank).
    """
    state = load_progress() if resume else {
        "sheriahub_completed_doc_keys": [],
        "statutes_completed_doc_keys": [],
    }
    completed_key = f"{namespace}_completed_doc_keys"
    completed: set[str] = set(state.get(completed_key, []))

    classifier = classify_case if namespace == "sheriahub" else classify_statute
    logger.info(
        "backfill %s: dry_run=%s limit=%s already_completed=%d",
        namespace, dry_run, limit, len(completed),
    )

    # Walk: for each document, classify once, then queue update for every
    # chunk of that document. Collect chunk IDs for each classified doc_key.
    doc_rank: dict[str, str] = {}        # doc_key → authority_rank
    pending_updates: list[tuple[str, str]] = []  # (vec_id, rank)
    BATCH_FLUSH = 200

    totals = {"documents_classified": 0, "chunks_queued": 0,
              "updated": 0, "skipped": 0, "failed": 0}
    start = time.time()

    # Fetch metadata for each batch of IDs → classify → queue updates
    id_batch: list[str] = []

    def process_batch(id_batch: list[str]):
        nonlocal pending_updates
        if not id_batch:
            return
        resp = index.fetch(ids=id_batch, namespace=namespace)
        vectors = resp.vectors if hasattr(resp, "vectors") else {}
        for vec_id, v in vectors.items():
            md = v.metadata if hasattr(v, "metadata") else {}
            doc_key = md.get("case_id") or md.get("slug") or vec_id.split("_chunk_")[0]
            if doc_key in completed:
                continue  # already migrated in a prior run

            # Classify the document once; cache the result
            if doc_key not in doc_rank:
                rank, _ = classifier(md)
                doc_rank[doc_key] = rank
                totals["documents_classified"] += 1

            pending_updates.append((vec_id, doc_rank[doc_key]))
            totals["chunks_queued"] += 1

    for vec_id in iter_ids(index, namespace):
        id_batch.append(vec_id)
        if len(id_batch) >= FETCH_BATCH:
            process_batch(id_batch)
            id_batch = []
            if len(pending_updates) >= BATCH_FLUSH:
                counts = update_batch_concurrent(index, pending_updates, namespace, dry_run)
                totals["updated"] += counts["updated"]
                totals["skipped"] += counts["skipped"]
                totals["failed"] += counts["failed"]
                # Mark docs complete
                for doc_key in doc_rank:
                    completed.add(doc_key)
                doc_rank.clear()
                pending_updates = []

                if totals["documents_classified"] % PROGRESS_SAVE_INTERVAL < BATCH_FLUSH:
                    state[completed_key] = sorted(completed)
                    save_progress(state)
                    elapsed = time.time() - start
                    logger.info(
                        "%s: docs=%d chunks=%d updated=%d skipped=%d failed=%d (%.0fs)",
                        namespace, totals["documents_classified"], totals["chunks_queued"],
                        totals["updated"], totals["skipped"], totals["failed"], elapsed,
                    )
                if limit and totals["documents_classified"] >= limit:
                    break
    process_batch(id_batch)

    # Final flush
    if pending_updates:
        counts = update_batch_concurrent(index, pending_updates, namespace, dry_run)
        totals["updated"] += counts["updated"]
        totals["skipped"] += counts["skipped"]
        totals["failed"] += counts["failed"]
    for doc_key in doc_rank:
        completed.add(doc_key)
    state[completed_key] = sorted(completed)
    save_progress(state)

    return totals


# ─── Entry ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="WI-7b authority_rank metadata backfill")
    parser.add_argument(
        "--namespace", choices=["sheriahub", "statutes", "all"], default="sheriahub",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Stop after N unique documents per namespace (for pilot runs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Preview only, make no writes (default)",
    )
    parser.add_argument(
        "--yes-really-write", action="store_true",
        help="Opt-in to real writes (disables --dry-run)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from authority_rank_backfill_progress.json",
    )
    args = parser.parse_args()

    dry_run = not args.yes_really_write

    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    logger.info("Index '%s': %d total vectors", INDEX_NAME, stats.total_vector_count)

    mode = "DRY-RUN (no writes)" if dry_run else "LIVE WRITES"
    logger.info("=== authority_rank backfill mode: %s ===", mode)

    namespaces = ["statutes", "sheriahub"] if args.namespace == "all" else [args.namespace]
    for ns in namespaces:
        totals = backfill_namespace(index, ns, args.limit, dry_run, args.resume)
        logger.info("%s totals: %s", ns, totals)

    logger.info("=== authority_rank backfill done (%s) ===", mode)


if __name__ == "__main__":
    main()

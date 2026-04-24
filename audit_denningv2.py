"""
Pinecone denningv2 quality audit
=================================
Walks both namespaces (sheriahub + statutes) and reports:
  - chunk_text length distribution (detects truncation survivors)
  - documents missing case_name / title metadata
  - documents with very low chunk counts (suspected incomplete ingest)
  - coverage of a known-good landmark list

Usage:
    PINECONE_API_KEY=... uv run python audit_denningv2.py
    PINECONE_API_KEY=... uv run python audit_denningv2.py --namespace statutes
    PINECONE_API_KEY=... uv run python audit_denningv2.py --out audit.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from pinecone import Pinecone


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} environment variable is required")
    return value


INDEX_NAME = "denningv2"
TRUNCATION_THRESHOLD = 5_000  # chunk_text shorter than this is suspicious
FETCH_BATCH = 100             # Pinecone fetch() limit per call

# Landmarks we expect to find. Extend as needed.
LANDMARK_TOKENS = {
    "Dina Management [2023] KESC 8": ["kesc-8", "[2023] kesc 8"],
    "Dina Management [2022] KESC 24": ["kesc-24", "[2022] kesc 24"],
    "JOO v MBO [2023] KESC 4": ["kesc-4", "[2023] kesc 4"],
    "Echaria v Echaria [2007]": ["echaria"],
    "Data Protection Act Cap. 411C": ["data-protection", "data protection act"],
}


def iter_ids(index, namespace: str):
    """Paginate all vector IDs in a namespace via index.list()."""
    for page in index.list(namespace=namespace):
        for vector_id in page:
            yield vector_id


def audit_namespace(index, namespace: str) -> dict:
    """Walk the namespace and return a metric dict."""
    print(f"\n=== auditing namespace '{namespace}' ===", flush=True)

    # Group chunks by document (case_id or slug)
    doc_chunks: dict[str, list[dict]] = defaultdict(list)
    len_histogram = Counter()
    total_vectors = 0
    truncated_chunks = 0
    landmark_hits: dict[str, int] = {name: 0 for name in LANDMARK_TOKENS}

    batch: list[str] = []
    start = time.time()

    def flush(batch: list[str]):
        nonlocal total_vectors, truncated_chunks
        if not batch:
            return
        resp = index.fetch(ids=batch, namespace=namespace)
        vectors = resp.vectors if hasattr(resp, "vectors") else resp.get("vectors", {})
        for vec_id, vec in vectors.items():
            md = vec.metadata if hasattr(vec, "metadata") else vec.get("metadata", {}) or {}
            chunk_text = md.get("chunk_text", "") or ""
            chunk_len = len(chunk_text)
            len_histogram[chunk_len // 1000 * 1000] += 1
            if chunk_len < TRUNCATION_THRESHOLD:
                truncated_chunks += 1

            doc_key = md.get("case_id") or md.get("slug") or vec_id.split("_chunk_")[0]
            doc_chunks[doc_key].append({
                "id": vec_id,
                "chunk_index": md.get("chunk_index"),
                "total_chunks": md.get("total_chunks"),
                "chunk_len": chunk_len,
                "case_name": md.get("title") or md.get("case_name") or "",
                "citation": md.get("citation", ""),
                "slug": md.get("slug", ""),
            })
            total_vectors += 1

            haystack = " ".join(str(md.get(k, "")) for k in ("title", "citation", "slug", "case_id")).lower()
            for name, tokens in LANDMARK_TOKENS.items():
                if any(tok.lower() in haystack for tok in tokens):
                    landmark_hits[name] += 1

    for vec_id in iter_ids(index, namespace):
        batch.append(vec_id)
        if len(batch) >= FETCH_BATCH:
            flush(batch)
            batch = []
            if total_vectors % 1000 == 0:
                elapsed = time.time() - start
                print(f"  scanned {total_vectors} vectors ({elapsed:.1f}s)", flush=True)
    flush(batch)

    empty_name_docs = [k for k, chunks in doc_chunks.items()
                       if not any(c["case_name"] for c in chunks)]
    low_chunk_docs = [(k, len(v)) for k, v in doc_chunks.items() if len(v) < 3]

    return {
        "namespace": namespace,
        "total_vectors": total_vectors,
        "total_documents": len(doc_chunks),
        "truncated_chunks": truncated_chunks,
        "truncated_chunk_fraction": round(truncated_chunks / total_vectors, 4) if total_vectors else 0,
        "chunk_len_histogram": dict(sorted(len_histogram.items())),
        "documents_missing_case_name": len(empty_name_docs),
        "documents_missing_case_name_sample": empty_name_docs[:20],
        "documents_with_fewer_than_3_chunks": len(low_chunk_docs),
        "low_chunk_documents_sample": sorted(low_chunk_docs, key=lambda x: x[1])[:20],
        "landmark_hits": landmark_hits,
        "duration_seconds": round(time.time() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit denningv2 index quality")
    parser.add_argument("--namespace", choices=["sheriahub", "statutes", "all"], default="all")
    parser.add_argument("--out", type=Path, help="Write JSON report to this path")
    args = parser.parse_args()

    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"Index '{INDEX_NAME}' total vectors: {stats.total_vector_count}")
    for ns, data in (stats.namespaces or {}).items():
        print(f"  {ns}: {data.vector_count} vectors")

    namespaces = ["sheriahub", "statutes"] if args.namespace == "all" else [args.namespace]
    reports = [audit_namespace(index, ns) for ns in namespaces]

    summary = {"index": INDEX_NAME, "reports": reports}
    print("\n=== SUMMARY ===")
    for r in reports:
        frac = r["truncated_chunk_fraction"] * 100
        print(f"\n[{r['namespace']}]  {r['total_vectors']} vectors  |  {r['total_documents']} documents")
        print(f"  chunks with chunk_text < {TRUNCATION_THRESHOLD} chars: {r['truncated_chunks']} ({frac:.1f}%)")
        print(f"  documents missing case_name/title: {r['documents_missing_case_name']}")
        print(f"  documents with <3 chunks: {r['documents_with_fewer_than_3_chunks']}")
        print(f"  landmark coverage:")
        for name, hits in r["landmark_hits"].items():
            mark = "OK" if hits > 0 else "MISS"
            print(f"    [{mark}] {name}: {hits} chunk(s)")

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nJSON report written to {args.out}")


if __name__ == "__main__":
    main()

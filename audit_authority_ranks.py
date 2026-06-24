"""
Authority-rank classifier audit — WI-7a.

Walks both denningv2 namespaces and applies a deterministic rank
classifier to every vector based on existing metadata (`court`,
`citation`, `case_action` for case law; `title`, `slug`, `case_id`
for statutes).

Outputs:
  1. Console summary — rank distribution per namespace, auto-assignment %
  2. JSON report (--out) — full structured output
  3. Review-pole JSON (--review-pole-out) — every unmapped document
     with its metadata, ready for senior-advocate review per WI-7e

Read-only. No writes to the index. Run alongside live query traffic
without conflict. Sibling of audit_denningv2.py.

Usage:
    PINECONE_API_KEY=... uv run python audit_authority_ranks.py
    PINECONE_API_KEY=... uv run python audit_authority_ranks.py --namespace statutes
    PINECONE_API_KEY=... uv run python audit_authority_ranks.py \\
        --out rank_audit.json --review-pole-out review_pole.json

Plan ref: /Users/winston/.claude/plans/calm-fluttering-flute.md §WI-7a.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
# Pinecone fetch() URL cap — see audit_denningv2.py for context.
FETCH_BATCH = 40


# ─── Rank enum — the WI-7 authority_rank metadata field domain ─────────────

CASE_RANKS = (
    "supreme_court",
    "court_of_appeal",
    "high_court",
    "elrc",
    "elc",
    "magistrate",
    "tribunal",
    "kadhi_court",
    "small_claims",
    "unknown_case",
)

STATUTE_RANKS = (
    "constitution",
    "act",
    "regulations",
    "rules",
    "practice_direction",
    "gazette_notice",
    "unknown_statute",
)


# ─── Case-law classifier ───────────────────────────────────────────────────

# Pattern: (regex, rank). Order matters — first match wins. Specialised
# courts come before "high_court" so KEELRC/KEELC don't fall through to
# the generic high-court match. AKN-code patterns (KESC, KECA, etc.) are
# the modern identifiers post-2018 Kenya Law AKN migration; full-name
# patterns catch pre-AKN records.
_CASE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Supreme Court
    (re.compile(r"supreme\s*court", re.I), "supreme_court"),
    (re.compile(r"\bKESC\b", re.I), "supreme_court"),
    # Court of Appeal
    (re.compile(r"court\s*of\s*appeal", re.I), "court_of_appeal"),
    (re.compile(r"\bKECA\b", re.I), "court_of_appeal"),
    # ELRC — check before "employment" or "high_court" generic match
    (re.compile(r"employment\s*and\s*labour\s*relations", re.I), "elrc"),
    (re.compile(r"\bKEELRC\b", re.I), "elrc"),
    # ELC
    (re.compile(r"environment\s*and\s*land", re.I), "elc"),
    (re.compile(r"\bKEELC\b", re.I), "elc"),
    # High Court — must come AFTER all specialised-court patterns
    (re.compile(r"high\s*court", re.I), "high_court"),
    (re.compile(r"\bKEHC\b", re.I), "high_court"),
    # Magistrate
    (re.compile(r"magistrate", re.I), "magistrate"),
    (re.compile(r"\bKEMC\b", re.I), "magistrate"),
    (re.compile(r"\bCMC\b", re.I), "magistrate"),
    (re.compile(r"\bSRM\b", re.I), "magistrate"),
    # Tribunals (includes capital-markets and other specialised bodies)
    (re.compile(r"tribunal", re.I), "tribunal"),
    (re.compile(r"\bKECMT\b", re.I), "tribunal"),
    # Kadhi Court — modern AKN code + full-name fallback
    (re.compile(r"\bKEKC\b", re.I), "kadhi_court"),
    (re.compile(r"kadhi", re.I), "kadhi_court"),
    # Small Claims
    (re.compile(r"small\s*claims", re.I), "small_claims"),
]


def classify_case(metadata: dict) -> tuple[str, str]:
    """Return (rank, reason). reason = classifier source, for debugging."""
    court = metadata.get("court") or ""
    citation = metadata.get("citation") or ""
    case_action = metadata.get("case_action") or ""

    for field_val, field_name in ((court, "court"), (citation, "citation"), (case_action, "case_action")):
        for pattern, rank in _CASE_PATTERNS:
            if pattern.search(field_val):
                return rank, f"{field_name}~{pattern.pattern!r}"

    return "unknown_case", "no match"


# ─── Statute classifier ────────────────────────────────────────────────────

_STATUTE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Constitution first — highest authority
    (re.compile(r"constitution\s*of\s*kenya", re.I), "constitution"),
    (re.compile(r"constitution-of-kenya", re.I), "constitution"),
    # Practice Directions
    (re.compile(r"practice\s*direction", re.I), "practice_direction"),
    (re.compile(r"practice-direction", re.I), "practice_direction"),
    # Rules (Civil Procedure Rules, High Court Rules, etc.)
    (re.compile(r"\bRules\b", re.I), "rules"),
    (re.compile(r"-rules$", re.I), "rules"),
    # Regulations (subsidiary legislation)
    (re.compile(r"\bRegulations\b", re.I), "regulations"),
    (re.compile(r"-regulations$", re.I), "regulations"),
    # Gazette notices
    (re.compile(r"gazette\s*notice", re.I), "gazette_notice"),
    # Acts — explicit "Act" in title
    (re.compile(r"\bAct\b", re.I), "act"),
    # Cap. N convention — catches Code statutes like Penal Code (Cap 63)
    # that don't have "Act" in the title
    (re.compile(r"\bCap\.?\s*\d+", re.I), "act"),
    # "Statute" fallback in case_action
    (re.compile(r"statute", re.I), "act"),
]


def classify_statute(metadata: dict) -> tuple[str, str]:
    title = metadata.get("title") or ""
    slug = metadata.get("slug") or ""
    case_id = metadata.get("case_id") or ""
    case_action = metadata.get("case_action") or ""

    for field_val, field_name in (
        (slug, "slug"),
        (title, "title"),
        (case_id, "case_id"),
        (case_action, "case_action"),
    ):
        for pattern, rank in _STATUTE_PATTERNS:
            if pattern.search(field_val):
                return rank, f"{field_name}~{pattern.pattern!r}"

    return "unknown_statute", "no match"


# ─── Namespace walker ──────────────────────────────────────────────────────


def iter_ids(index, namespace: str, page_limit: int = 99):
    """Paginate vector IDs via list_paginated() — avoids the index.list()
    generator's 414 URI-Too-Large trap on deep pagination."""
    pagination_token = None
    while True:
        kwargs = {"namespace": namespace, "limit": page_limit}
        if pagination_token:
            kwargs["pagination_token"] = pagination_token
        resp = index.list_paginated(**kwargs)
        for vec in resp.vectors or []:
            vec_id = vec.id if hasattr(vec, "id") else vec.get("id")
            if vec_id:
                yield vec_id
        pagination = resp.pagination if hasattr(resp, "pagination") else None
        pagination_token = getattr(pagination, "next", None) if pagination else None
        if not pagination_token:
            return


def audit_namespace(index, namespace: str, classifier, limit: int | None = None) -> dict:
    """Walk the namespace (or a `limit` sample), classify, return aggregates.

    De-duplicates by document key (case_id / slug / document_id) so multi-
    chunk documents count once — the authority rank is a per-document
    property, not a per-chunk one.
    """
    print(f"\n=== auditing namespace '{namespace}' "
          f"(limit={limit if limit else 'all'}) ===", flush=True)

    seen_docs: set[str] = set()
    counts: Counter = Counter()
    reasons: Counter = Counter()
    # review_pole: every unmapped doc's full metadata for WI-7e
    review_pole: list[dict] = []
    examples: dict[str, list] = defaultdict(list)
    total_docs = 0
    total_vectors = 0
    batch: list[str] = []
    start = time.time()

    def flush(batch: list[str]):
        nonlocal total_docs, total_vectors
        if not batch:
            return
        resp = index.fetch(ids=batch, namespace=namespace)
        vectors = resp.vectors if hasattr(resp, "vectors") else {}
        for vec_id, v in vectors.items():
            total_vectors += 1
            md = v.metadata if hasattr(v, "metadata") else {}
            doc_key = md.get("case_id") or md.get("slug") or vec_id.split("_chunk_")[0]
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)

            rank, reason = classifier(md)
            counts[rank] += 1
            reasons[reason[:80]] += 1
            total_docs += 1

            if rank.startswith("unknown"):
                review_pole.append({
                    "doc_key": doc_key,
                    "sample_vector_id": vec_id,
                    "court": md.get("court", ""),
                    "citation": md.get("citation", ""),
                    "title": md.get("title", ""),
                    "slug": md.get("slug", ""),
                    "case_id": md.get("case_id", ""),
                    "case_action": md.get("case_action", ""),
                    "date_delivered": md.get("date_delivered", ""),
                    "jurisdiction": md.get("jurisdiction", ""),
                })

            if len(examples[rank]) < 5:
                examples[rank].append({
                    "doc_key": doc_key[:100],
                    "court": (md.get("court") or "")[:80],
                    "citation": (md.get("citation") or "")[:100],
                    "title": (md.get("title") or "")[:80],
                })

    for vec_id in iter_ids(index, namespace):
        batch.append(vec_id)
        if len(batch) >= FETCH_BATCH:
            flush(batch)
            batch = []
            if total_vectors and total_vectors % 2000 == 0:
                elapsed = time.time() - start
                print(f"  scanned {total_vectors} vectors, {total_docs} unique docs ({elapsed:.1f}s)", flush=True)
            if limit and total_docs >= limit:
                break
    flush(batch)

    return {
        "namespace": namespace,
        "total_vectors": total_vectors,
        "total_documents": total_docs,
        "rank_distribution": dict(counts),
        "top_reasons": dict(reasons.most_common(15)),
        "review_pole_size": len(review_pole),
        "review_pole": review_pole,
        "examples_by_rank": dict(examples),
        "duration_seconds": round(time.time() - start, 1),
    }


def print_report(report: dict) -> None:
    ns = report["namespace"]
    total = report["total_documents"]
    dist = report["rank_distribution"]
    print(f"\n[{ns}] {total} unique documents ({report['total_vectors']} vectors)")
    print(f"  duration: {report['duration_seconds']}s")
    print(f"  rank distribution:")
    for rank, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        tag = " ← REVIEW" if rank.startswith("unknown") else ""
        print(f"    {rank:<22} {count:>6} ({pct:5.1f}%){tag}")

    pole_size = report["review_pole_size"]
    clean_pct = 100 * (total - pole_size) / total if total else 0
    print(f"  CLEAN AUTO-ASSIGNMENT: {clean_pct:.2f}%")
    print(f"  REVIEW POLE: {pole_size} documents (WI-7e senior-advocate scope)")


def main():
    parser = argparse.ArgumentParser(description="WI-7a authority-rank classifier audit")
    parser.add_argument(
        "--namespace", choices=["sheriahub", "statutes", "all"], default="all",
        help="Which namespace to audit. 'all' runs both.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N unique documents per namespace (default: full walk)",
    )
    parser.add_argument(
        "--out", type=Path,
        help="Write full JSON report to this path",
    )
    parser.add_argument(
        "--review-pole-out", type=Path,
        help="Write unmapped-documents list to this path (for senior-advocate review per WI-7e)",
    )
    args = parser.parse_args()

    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"Index '{INDEX_NAME}': {stats.total_vector_count} total vectors")
    for ns, data in (stats.namespaces or {}).items():
        print(f"  {ns}: {data.vector_count} vectors")

    namespaces = ["statutes", "sheriahub"] if args.namespace == "all" else [args.namespace]
    classifiers = {"sheriahub": classify_case, "statutes": classify_statute}

    reports = []
    all_review_pole: list[dict] = []
    for ns in namespaces:
        report = audit_namespace(index, ns, classifiers[ns], limit=args.limit)
        print_report(report)
        reports.append(report)
        for entry in report["review_pole"]:
            all_review_pole.append({**entry, "namespace": ns})

    print("\n=== SUMMARY ===")
    for r in reports:
        pole = r["review_pole_size"]
        total = r["total_documents"]
        clean = 100 * (total - pole) / total if total else 0
        print(f"  [{r['namespace']}]  {total} docs  |  {clean:.2f}% clean  |  {pole} in review pole")

    if args.out:
        # Strip bulky review_pole from the main report; it goes to its own file
        trimmed = [{k: v for k, v in r.items() if k != "review_pole"} for r in reports]
        args.out.write_text(json.dumps({"index": INDEX_NAME, "reports": trimmed}, indent=2, default=str))
        print(f"\nReport written: {args.out}")

    if args.review_pole_out:
        args.review_pole_out.write_text(json.dumps({
            "index": INDEX_NAME,
            "total_review_pole_size": len(all_review_pole),
            "entries": all_review_pole,
        }, indent=2, default=str))
        print(f"Review pole written: {args.review_pole_out}  ({len(all_review_pole)} entries)")


if __name__ == "__main__":
    main()

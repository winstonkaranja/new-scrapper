"""
Ingest a single case-law PDF into the sheriahub Pinecone namespace.
===================================================================
Purpose: manually onboard a landmark judgment that was missed by the
bulk scraper (e.g., pre-AKN cases, cases scraped under a different
slug, or cases fetched from alternate sources).

The script extracts text with PyMuPDF, chunks it with the same
algorithm as pinecone_upsert.chunk_text, embeds via Google Gemini
(gemini-embedding-2-preview, 1024 dims), and upserts vectors with
the full 24k chunk_text metadata. No truncation.

Usage:
    PINECONE_API_KEY=... GOOGLE_API_KEY=... uv run python ingest_single_doc.py \\
        --pdf sheriahub_kenya_data/pdfs/dina-management-2023-kesc-30.pdf \\
        --id ke-dina-management-limited-v-county-government-mombasa-2023-kesc-30-klr-21-april-2023-judgment \\
        --title "Dina Management Limited v County Government of Mombasa & 5 others (Petition 8 (E010) of 2021) [2023] KESC 30 (KLR) (21 April 2023) (Judgment)" \\
        --citation "[2023] KESC 30 (KLR)" \\
        --court "Supreme Court" \\
        --date-delivered 2023-04-21 \\
        --dry-run

Remove --dry-run to actually embed and upsert.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import fitz

# Reuse upsert primitives
from pinecone_upsert import (
    CHUNK_SIZE_CHARS,
    INDEX_NAME,
    NAMESPACE,
    build_vector_id,
    chunk_text,
    create_gemini_client,
    embed_texts_with_retry,
    init_pinecone,
    upsert_with_retry,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} environment variable is required")
    return value


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def build_metadata(
    record_id: str,
    title: str,
    citation: str,
    court: str,
    date_delivered: str,
    slug: str,
    text_length: int,
    chunk_text_str: str,
    chunk_idx: int,
    total_chunks: int,
) -> dict:
    """Match the metadata shape used by pinecone_upsert.build_metadata."""
    return {
        "case_id": record_id,
        "citation": citation,
        "court": court,
        "judges": "",
        "case_action": "Judgment",
        "date_delivered": date_delivered,
        "county": "",
        "plaintiff": "",
        "defendant": "",
        "title": title,
        "source_url": "",
        "pdf_s3_uri": "",
        "slug": slug,
        "jurisdiction": "Kenya",
        "text_length": text_length,
        "chunk_index": chunk_idx,
        "total_chunks": total_chunks,
        "chunk_text": chunk_text_str[:CHUNK_SIZE_CHARS],
    }


def probe_existing(index, record_id: str) -> int:
    """Return how many chunks for this record_id already exist in Pinecone."""
    test_ids = [build_vector_id(record_id, i) for i in range(20)]
    resp = index.fetch(ids=test_ids, namespace=NAMESPACE)
    vectors = resp.vectors if hasattr(resp, "vectors") else {}
    return len(vectors)


def main():
    parser = argparse.ArgumentParser(description="Ingest a single case-law PDF to sheriahub")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the PDF file")
    parser.add_argument("--id", required=True, help="Deterministic record id, e.g. ke-dina-...-2023-kesc-30-klr")
    parser.add_argument("--title", required=True)
    parser.add_argument("--citation", required=True)
    parser.add_argument("--court", default="", help="e.g. Supreme Court, Court of Appeal")
    parser.add_argument("--date-delivered", default="", help="ISO date: YYYY-MM-DD")
    parser.add_argument("--slug", default="", help="URL slug (defaults to stripped id)")
    parser.add_argument("--dry-run", action="store_true", help="Extract + chunk, skip embed + upsert")
    parser.add_argument("--force", action="store_true", help="Upsert even if vectors already exist")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"error: PDF not found at {args.pdf}")
    if not args.id.startswith("ke-"):
        print(f"WARNING: id '{args.id}' does not start with 'ke-' — non-standard", file=sys.stderr)

    _require_env("PINECONE_API_KEY")

    print(f"Extracting text from {args.pdf} ...")
    text = extract_pdf_text(args.pdf)
    text_len = len(text)
    print(f"  extracted: {text_len:,} chars")

    chunks = chunk_text(text)
    print(f"  chunked: {len(chunks)} chunks (sizes: {[len(c) for c in chunks]})")

    slug = args.slug or args.id.removeprefix("ke-")

    if args.dry_run:
        print("\n=== DRY RUN — no writes ===")
        print(f"Expected vector IDs: {[build_vector_id(args.id, i) for i in range(len(chunks))]}")
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n--- chunk {i} (first 300 chars) ---\n{chunk[:300]}...")
        return

    _require_env("GOOGLE_API_KEY")

    # Init clients
    gclient = create_gemini_client()
    _, index = init_pinecone()

    existing = probe_existing(index, args.id)
    if existing and not args.force:
        print(f"\nERROR: {existing} vectors already exist for id={args.id}")
        print("Use --force to overwrite, or pick a different --id.")
        raise SystemExit(2)
    if existing and args.force:
        print(f"\nWARNING: {existing} existing vectors will be overwritten (--force)")

    print(f"\nEmbedding {len(chunks)} chunks...")
    embeddings = embed_texts_with_retry(gclient, chunks)
    print(f"  got {len(embeddings)} vectors, dim={len(embeddings[0])}")

    print(f"\nUpserting to Pinecone namespace={NAMESPACE} index={INDEX_NAME} ...")
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vec_id = build_vector_id(args.id, i)
        metadata = build_metadata(
            record_id=args.id,
            title=args.title,
            citation=args.citation,
            court=args.court,
            date_delivered=args.date_delivered,
            slug=slug,
            text_length=text_len,
            chunk_text_str=chunk,
            chunk_idx=i,
            total_chunks=len(chunks),
        )
        vectors.append((vec_id, embedding, metadata))
    upsert_with_retry(index, vectors, namespace=NAMESPACE)
    print(f"  upserted {len(vectors)} vectors")

    # Verify
    print("\nVerifying ...")
    resp = index.fetch(ids=[v[0] for v in vectors], namespace=NAMESPACE)
    found = resp.vectors if hasattr(resp, "vectors") else {}
    print(f"  {len(found)}/{len(vectors)} vectors present")
    for vid, v in sorted(found.items()):
        md = v.metadata if hasattr(v, "metadata") else {}
        clen = len(md.get("chunk_text", "") or "")
        print(f"  {vid}: chunk_text={clen} chars, title={md.get('title', '')[:60]!r}")


if __name__ == "__main__":
    main()

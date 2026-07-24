"""
scripts/export_seed_data.py — Export ingested patent data as a SQL seed file.

Produces a data-only SQL dump of patent_documents + patent_chunks (and,
optionally, patent_images) so a fresh Supabase project can load real data
after running sql_queries/001-004, instead of re-running the ingestion
pipeline (slow, OCR-dependent, and the reason this script exists).

fts_tokens is intentionally omitted from the dump — the BEFORE INSERT trigger
defined in 001_schema.sql regenerates it automatically on import, so shipping
it would just be dead weight.

Reads via the existing supabase-py client (same SUPABASE_URL / SUPABASE_ANON_KEY
as the app) — no pg_dump, no direct Postgres/DB password needed.

Usage:
  python scripts/export_seed_data.py --gzip   # writes seed_data.sql + seed_data.sql.gz; commit only the .gz
  python scripts/export_seed_data.py --out sql_queries/seed_data.sql
  python scripts/export_seed_data.py --include-images   # + patent_images (BYTEA, untested — verify before relying on it)
"""
import argparse
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase import create_client

from app.config import settings

PAGE_SIZE = 1000


def _fetch_all(supabase, table: str, columns: str):
    """Page past PostgREST's 1000-row cap, same pattern as ingest_patents.py."""
    rows = []
    offset = 0
    while True:
        resp = (
            supabase.table(table)
            .select(columns)
            .range(offset, offset + PAGE_SIZE - 1)
            .order("created_at")
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _sql_literal(value) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _insert_stmt(table: str, columns: list, row: dict) -> str:
    cols = ", ".join(columns)
    values = ", ".join(_sql_literal(row.get(c)) for c in columns)
    return f"INSERT INTO {table} ({cols}) VALUES ({values}) ON CONFLICT (id) DO NOTHING;"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="sql_queries/seed_data.sql")
    p.add_argument(
        "--include-images", action="store_true",
        help="Also export patent_images (BYTEA blobs). Not needed to demo Phases 2-4; "
             "verify the output loads correctly before relying on it.",
    )
    p.add_argument(
        "--gzip", action="store_true",
        help="Also write a .gz alongside the .sql (embeddings compress ~3x — commit the .gz, not the raw .sql).",
    )
    args = p.parse_args()

    if not settings.supabase_url or not settings.supabase_anon_key:
        print("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env", file=sys.stderr)
        sys.exit(1)

    supabase = create_client(settings.supabase_url, settings.supabase_anon_key)

    doc_cols = ["id", "patent_number", "title", "assignee", "jurisdiction", "publication_date", "created_at"]
    chunk_cols = ["id", "patent_id", "section_type", "content", "embedding", "created_at"]

    docs = _fetch_all(supabase, "patent_documents", ",".join(doc_cols))
    chunks = _fetch_all(supabase, "patent_chunks", ",".join(chunk_cols))
    print(f"Fetched {len(docs)} patent_documents, {len(chunks)} patent_chunks", file=sys.stderr)

    lines = [
        "-- Seed data exported by scripts/export_seed_data.py",
        "-- Run AFTER sql_queries/001_schema.sql (002-004 optional, only needed for their features).",
        "-- fts_tokens is intentionally omitted here -- the trigger in 001_schema.sql regenerates it on insert.",
        "",
    ]
    lines += [_insert_stmt("patent_documents", doc_cols, row) for row in docs]
    lines.append("")
    lines += [_insert_stmt("patent_chunks", chunk_cols, row) for row in chunks]

    if args.include_images:
        img_cols = ["id", "patent_id", "page_number", "width", "height", "image_data", "created_at"]
        images = _fetch_all(supabase, "patent_images", ",".join(img_cols))
        print(f"Fetched {len(images)} patent_images", file=sys.stderr)
        lines.append("")
        for row in images:
            data = row.get("image_data")
            # Postgres/PostgREST serialises bytea as its own hex literal ("\x89504e47...");
            # that text is valid directly inside a quoted ::bytea literal, no decode() needed.
            image_sql = f"'{data}'::bytea" if data else "NULL"
            other_cols = [c for c in img_cols if c != "image_data"]
            values = ", ".join(_sql_literal(row.get(c)) for c in other_cols) + ", " + image_sql
            cols_sql = ", ".join(other_cols + ["image_data"])
            lines.append(f"INSERT INTO patent_images ({cols_sql}) VALUES ({values}) ON CONFLICT (id) DO NOTHING;")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    out_path.write_text(content)

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)", file=sys.stderr)

    if args.gzip:
        gz_path = out_path.with_suffix(out_path.suffix + ".gz")
        with gzip.open(gz_path, "wt") as f:
            f.write(content)
        gz_size_kb = gz_path.stat().st_size / 1024
        print(f"Wrote {gz_path} ({gz_size_kb:.1f} KB) — commit this, not the raw .sql", file=sys.stderr)


if __name__ == "__main__":
    main()

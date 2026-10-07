#!/usr/bin/env python3
"""
MIMIC-III → DuckDB Setup Script
--------------------------------
Scans /Users/Shared/MIMIC_III/mimic-iii-clinical-database-1.4 for all
.csv and .csv.gz files and loads each one as a table in a DuckDB database
saved to /Users/Shared/MIMIC_III/MIMIC_III_DB/mimic3.duckdb

This is the loader that built the database used by every script in this folder
(step 0 of REPRODUCE.md). Copied here 2026-10-06; the only change is that both
paths can be overridden:
    MIMIC3_CSV=/path/to/mimic-iii-clinical-database-1.4   (source CSV folder)
    MIMIC3_DB=/path/to/mimic3.duckdb                      (output database)

Usage:
    pip install duckdb pandas
    python setup_mimic_duckdb.py

Optional flags:
    --threads 8           use more CPU cores (default: 4)
    --memory-limit 8GB    raise DuckDB's memory cap (default: 4GB)
    --skip-existing       skip tables that already exist in the DB
    --chunk-size 100000   rows per batch for chunked tables (default: 50000)
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import duckdb
except ImportError:
    sys.exit("duckdb is not installed. Run:  pip install duckdb pandas  then retry.")

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is not installed. Run:  pip install duckdb pandas  then retry.")

# ── paths ─────────────────────────────────────────────────────────────────────
SOURCE_DIR = Path(os.environ.get("MIMIC3_CSV", "/Users/Shared/MIMIC_III/mimic-iii-clinical-database-1.4"))
DB_PATH    = Path(os.environ.get("MIMIC3_DB", "/Users/Shared/MIMIC_III/MIMIC_III_DB/mimic3.duckdb"))
OUTPUT_DIR = DB_PATH.parent

# ── per-table overrides ───────────────────────────────────────────────────────
# "types"   : force specific column types for read_csv_auto
# "parallel": set False when quoted newlines are present in fields
# "chunked" : use the pandas chunk-insert path (for very large tables)
TABLE_CONFIG: dict[str, dict] = {
    # CPT_CD starts as all-digits then goes alphanumeric (e.g. "G0272")
    "cptevents": {
        "types": {"CPT_CD": "VARCHAR"},
    },
    # Free-text clinical notes: quoted newlines inside fields break the parallel
    # scanner, and the full table is too large to buffer in one pass → chunk it.
    "noteevents": {
        "chunked": True,
    },
    # VALUE is mostly numeric but also contains "Yes" / "No" strings
    "chartevents": {
        "types": {"VALUE": "VARCHAR"},
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def table_name_from_path(p: Path) -> str:
    name = p.name
    for ext in (".csv.gz", ".csv"):
        if name.lower().endswith(ext):
            name = name[: len(name) - len(ext)]
            break
    return name.lower()


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_val(v) -> str:
    """Format a Python value as a DuckDB SQL literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        inner = ", ".join(f"'{ck}': '{cv}'" for ck, cv in v.items())
        return "{" + inner + "}"
    if isinstance(v, str) and not v.startswith("'"):
        return f"'{v}'"
    return str(v)


def build_read_sql(csv_path: Path, extra: dict) -> str:
    opts = {
        "header":       True,
        "compression":  "'auto'",
        "null_padding": True,
    }
    opts.update(extra)
    kv = ",\n                    ".join(f"{k} = {fmt_val(v)}" for k, v in opts.items())
    return f"read_csv_auto(\n                    '{csv_path}',\n                    {kv}\n                )"


# ── standard loader (single-pass via read_csv_auto) ───────────────────────────

def load_table_standard(con, table: str, csv_path: Path, extra: dict) -> tuple[int, str]:
    """
    Try to load with given overrides, then escalate:
      1. overrides only
      2. + sample_size=-1  (full type-scan)
      3. + all_varchar=true (load everything as TEXT)
    """
    attempts = [
        ("overrides",     extra),
        ("sample_size=-1", {**extra, "sample_size": -1}),
        ("all_varchar",    {**extra, "all_varchar": True, "sample_size": -1}),
    ]
    last_exc = None
    for label, opts in attempts:
        try:
            con.execute(f"""
                CREATE OR REPLACE TABLE {table} AS
                SELECT * FROM {build_read_sql(csv_path, opts)}
            """)
            rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return rows, label
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(k in msg for k in ("Conversion Error", "parallel scanner",
                                       "could not convert", "Out of Memory")):
                continue
            raise
    raise last_exc


# ── chunked loader (pandas → DuckDB insert, fixed memory) ─────────────────────

def load_table_chunked(con, table: str, csv_path: Path,
                       chunk_size: int = 50_000) -> tuple[int, str]:
    """
    Stream the CSV through pandas in fixed-size chunks and INSERT into DuckDB.
    All columns are explicitly created as VARCHAR so that columns which happen
    to be all-NULL in the first chunk (e.g. CHARTTIME in NOTEEVENTS) don't get
    mis-typed as INT32/DOUBLE and then fail on real values later.
    """
    import csv as _csv
    import gzip as _gzip

    compression = "gzip" if str(csv_path).endswith(".gz") else None

    # ── Step 1: read just the header to get column names ─────────────────────
    opener = _gzip.open if compression == "gzip" else open
    with opener(csv_path, "rt", encoding="utf-8", errors="replace") as fh:
        header = next(_csv.reader(fh))

    # ── Step 2: create the table with all-VARCHAR schema ─────────────────────
    con.execute(f"DROP TABLE IF EXISTS {table}")
    cols_ddl = ", ".join(f'"{c}" VARCHAR' for c in header)
    con.execute(f"CREATE TABLE {table} ({cols_ddl})")

    # ── Step 3: stream data in chunks ─────────────────────────────────────────
    reader = pd.read_csv(
        csv_path,
        compression=compression,
        dtype=str,               # keep everything as text → no type-cast OOM
        keep_default_na=True,
        chunksize=chunk_size,
        on_bad_lines="skip",
    )

    total = 0
    for i, chunk in enumerate(reader):
        # Replace pandas NA with None so DuckDB stores proper NULLs
        chunk = chunk.where(pd.notna(chunk), None)

        con.register("_chunk", chunk)
        con.execute(f"INSERT INTO {table} SELECT * FROM _chunk")
        con.unregister("_chunk")

        total += len(chunk)
        if (i + 1) % 10 == 0:
            print(f"    … {total:,} rows inserted", end="\r", flush=True)

    return total, "chunked"


# ── parse args ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Load MIMIC-III CSVs into DuckDB")
    p.add_argument("--threads",      type=int, default=4)
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--chunk-size",   type=int, default=50_000)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not SOURCE_DIR.exists():
        sys.exit(f"ERROR: source directory not found:\n  {SOURCE_DIR}")

    csv_files = sorted(
        [f for f in SOURCE_DIR.iterdir()
         if f.is_file() and (f.name.lower().endswith(".csv.gz")
                             or f.name.lower().endswith(".csv"))],
        key=lambda f: f.stat().st_size,
    )
    if not csv_files:
        sys.exit(f"ERROR: no CSV files found in:\n  {SOURCE_DIR}")

    print(f"Found {len(csv_files)} CSV file(s) to load.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output database : {DB_PATH}\n")

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")

    existing = set()
    if args.skip_existing:
        existing = {r[0] for r in
                    con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}

    total_rows = 0
    errors     = []

    for idx, csv_path in enumerate(csv_files, start=1):
        table = table_name_from_path(csv_path)
        size  = human_bytes(csv_path.stat().st_size)
        cfg   = TABLE_CONFIG.get(table, {})

        if table in existing:
            print(f"[{idx:>2}/{len(csv_files)}] {table:<30s}  (skipped – already loaded)")
            continue

        print(f"[{idx:>2}/{len(csv_files)}] {table:<30s}  ({size})", end="", flush=True)
        t0 = time.time()

        try:
            if cfg.get("chunked"):
                row_count, strategy = load_table_chunked(
                    con, table, csv_path, chunk_size=args.chunk_size
                )
                # Clear the progress line
                print(" " * 40, end="\r", flush=True)
            else:
                extra = {k: v for k, v in cfg.items() if k != "chunked"}
                row_count, strategy = load_table_standard(con, table, csv_path, extra)

            elapsed = time.time() - t0
            note    = f"  [{strategy}]" if strategy != "overrides" else ""
            print(f"[{idx:>2}/{len(csv_files)}] {table:<30s}  →  {row_count:>10,} rows  ({elapsed:.1f}s){note}")
            total_rows += row_count

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n[{idx:>2}/{len(csv_files)}] {table:<30s}  ✗  FAILED ({elapsed:.1f}s)")
            errors.append((table, str(exc)))

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    tables_ok = len(csv_files) - len(errors)
    print(f"Loaded : {tables_ok}/{len(csv_files)} tables  |  {total_rows:,} total rows")
    print(f"DB size: {human_bytes(DB_PATH.stat().st_size)}")
    print(f"DB path: {DB_PATH}")

    if errors:
        print(f"\n⚠  {len(errors)} table(s) failed to load:")
        for tbl, msg in errors:
            print(f"   {tbl}: {msg}")

    print("\n── Table row counts ──────────────────────────────────────────")
    for tbl, cnt in con.execute("""
        SELECT table_name, estimated_size
        FROM duckdb_tables()
        ORDER BY table_name
    """).fetchall():
        print(f"  {tbl:<35s} {cnt:>12,}")

    con.close()
    print("\nDone. ✓")


if __name__ == "__main__":
    main()

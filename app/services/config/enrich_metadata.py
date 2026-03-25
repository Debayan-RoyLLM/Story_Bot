"""
Metadata Enrichment Script (run once / periodically)

Queries the DB to auto-extract:
  - Column data types
  - Min/Max values for numeric columns
  - Sample distinct values (top 5)
  - Sample rows (2 per table)

Outputs enriched_metadata.json — used by the prompt to give the LLM
real data context so it understands what values mean.

Usage:
    python -m app.services.config.enrich_metadata
"""

import json
import os
from pathlib import Path
from urllib.parse import quote
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path(__file__).parent
OUTPUT_FILE = CONFIG_DIR / "enriched_metadata.json"
METADATA_FILE = CONFIG_DIR / "metadata.json"


def get_engine():
    password = quote(os.environ.get("SQL_SERVER_PASSWORD", ""))
    return create_engine(
        f"mssql+pyodbc://SA:{password}@127.0.0.1:1433/sportmonk"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&Encrypt=no&TrustServerCertificate=yes"
    )


def get_column_stats(engine, schema, table):
    """Get data type, min, max, distinct count, sample values for each column."""
    full_table = f"{schema}.{table}"
    stats = {}

    # Get column names and types
    with engine.connect() as conn:
        cols = conn.execute(text("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
        """), {"schema": schema, "table": table}).fetchall()

        for col_name, data_type in cols:
            col_stats = {"type": data_type}

            try:
                # Numeric columns: get min, max
                if data_type in ("int", "bigint", "float", "decimal", "numeric", "smallint", "tinyint", "real"):
                    row = conn.execute(text(
                        f"SELECT MIN([{col_name}]), MAX([{col_name}]) FROM {full_table}"
                    )).fetchone()
                    if row:
                        col_stats["min"] = row[0]
                        col_stats["max"] = row[1]

                # Get top 5 distinct sample values
                sample_rows = conn.execute(text(
                    f"SELECT DISTINCT TOP 5 [{col_name}] FROM {full_table} WHERE [{col_name}] IS NOT NULL"
                )).fetchall()
                col_stats["sample_values"] = [str(r[0]) for r in sample_rows]

            except Exception as e:
                col_stats["error"] = str(e)[:100]

            stats[col_name] = col_stats

    return stats


def get_sample_rows(engine, schema, table, limit=2):
    """Get sample rows from a table."""
    full_table = f"{schema}.{table}"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT TOP {limit} * FROM {full_table}"))
            keys = list(result.keys())
            rows = []
            for row in result:
                rows.append({k: str(v) for k, v in zip(keys, row)})
            return rows
    except Exception:
        return []


def get_value_ranges(engine, schema, table):
    """Get realistic ranges for key numeric columns — used for sanity checks."""
    full_table = f"{schema}.{table}"
    ranges = {}
    try:
        with engine.connect() as conn:
            cols = conn.execute(text("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                AND DATA_TYPE IN ('int', 'bigint', 'float', 'decimal', 'numeric', 'real')
            """), {"schema": schema, "table": table}).fetchall()

            for col_name, _ in cols:
                row = conn.execute(text(
                    f"SELECT MIN([{col_name}]), MAX([{col_name}]), AVG(CAST([{col_name}] AS FLOAT)) FROM {full_table}"
                )).fetchone()
                if row and row[0] is not None:
                    ranges[col_name] = {
                        "min": float(row[0]) if row[0] is not None else None,
                        "max": float(row[1]) if row[1] is not None else None,
                        "avg": round(float(row[2]), 2) if row[2] is not None else None,
                    }
    except Exception:
        pass
    return ranges


def main():
    engine = get_engine()

    # Load existing metadata for descriptions
    with open(METADATA_FILE, "r") as f:
        base_metadata = json.load(f)

    enriched = {}
    tables = [
        ("history2", "fixtures__balls"),
        ("history2", "fixtures__batting"),
        ("history2", "fixtures__bowling"),
        ("history2", "fixtures__runs"),
    ]

    for schema, table in tables:
        full_name = f"{schema}.{table}"
        print(f"\nProcessing: {full_name}")

        # Column stats
        col_stats = get_column_stats(engine, schema, table)
        print(f"  Columns: {len(col_stats)}")

        # Sample rows
        samples = get_sample_rows(engine, schema, table)
        print(f"  Sample rows: {len(samples)}")

        # Value ranges
        ranges = get_value_ranges(engine, schema, table)
        print(f"  Numeric ranges: {len(ranges)}")

        # Merge with existing descriptions
        base_cols = base_metadata.get(full_name, {})
        columns = {}
        for col_name, stats in col_stats.items():
            entry = stats.copy()
            # Add description from base metadata if exists
            if col_name in base_cols:
                entry["description"] = base_cols[col_name]
            columns[col_name] = entry

        # Carry over ❌ NO entries
        for key, value in base_cols.items():
            if key.startswith("❌"):
                columns[key] = value

        enriched[full_name] = {
            "columns": columns,
            "sample_rows": samples,
            "value_ranges": ranges,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(enriched, f, indent=2, default=str)

    print(f"\nEnriched metadata saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

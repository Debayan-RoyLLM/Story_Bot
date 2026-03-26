"""
Metadata Generator — Single script to build one rich metadata.json

Auto-generates everything from the database:
  1. Column names, types, min/max, sample values     (from DB)
  2. Sample rows per table                            (from DB)
  3. Table summaries + column descriptions            (from LLM)
  4. ❌ NO forbidden entries                           (auto-derived by comparing columns across tables)

Outputs ONE metadata.json used by validators, prompt, and sanity checks.

Usage:
    python -m app.services.config.enrich_metadata

Re-run whenever:
    - New tables/columns are added to the DB
    - Schema changes
    - You want to refresh descriptions
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import quote
from openai import OpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path(__file__).parent
OUTPUT_FILE = CONFIG_DIR / "metadata.json"


# ─────────────────────────────────────────────
# DB Connection
# ─────────────────────────────────────────────

def get_engine():
    password = quote(os.environ.get("SQL_SERVER_PASSWORD", ""))
    return create_engine(
        f"mssql+pyodbc://SA:{password}@127.0.0.1:1433/sportmonk"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&Encrypt=no&TrustServerCertificate=yes"
    )


# ─────────────────────────────────────────────
# LLM Helpers
# ─────────────────────────────────────────────

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return str(obj)


def extract_json(text_output):
    text_output = re.sub(r"```json|```", "", text_output).strip()
    match = re.search(r"\{.*\}", text_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def ask_llm(prompt, client, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        usage = response.usage
        print(f"    Tokens — in: {usage.prompt_tokens}, out: {usage.completion_tokens}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    LLM error: {e}")
        return ""


# ─────────────────────────────────────────────
# DB Introspection
# ─────────────────────────────────────────────

def get_columns(engine, schema, table):
    """Get column names and data types."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
        """), {"schema": schema, "table": table}).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_column_stats(engine, schema, table, col_name, data_type):
    """Get 2 sample values for a column."""
    full_table = f"{schema}.{table}"
    stats = {}

    try:
        with engine.connect() as conn:
            sample_rows = conn.execute(text(
                f"SELECT DISTINCT TOP 2 [{col_name}] FROM {full_table} WHERE [{col_name}] IS NOT NULL"
            )).fetchall()
            stats["sample_values"] = [str(r[0]) for r in sample_rows]
    except Exception:
        pass

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


# ─────────────────────────────────────────────
# LLM Description Generation
# ─────────────────────────────────────────────

def generate_descriptions(table_name, columns, sample_rows, client):
    """
    Call LLM once per table to generate:
      - table_summary
      - column descriptions (one per column)
    """
    cols_text = "\n".join(f"- {name} ({dtype})" for name, dtype in columns)
    safe_samples = make_json_safe(sample_rows)

    prompt = f"""You are a cricket database expert. Analyze this table and describe it for someone writing SQL queries against a cricket match database.

Table: {table_name}

Columns:
{cols_text}

Sample data:
{json.dumps(safe_samples, indent=2)}

Return ONLY a valid JSON object with this exact structure:
{{
  "table_summary": "What this table stores and its purpose — be specific about cricket context",
  "columns": {{
    "column_name": "What this column stores, its data type context, role (primary key, foreign key, metric), and how it relates to other tables. Include warnings about common misuse."
  }}
}}

Important cricket context:
- This is a cricket statistics database with ball-by-ball, batting, bowling, and innings data
- Player names (batsman__fullname, bowler__fullname) may only exist in certain tables
- 'score' means different things in different tables (per-ball runs vs innings total)
- 'ball' and 'overs' are decimal format (5.3 = 5th over, 3rd ball)
- 'wickets' in innings tables means wickets LOST (not in hand)
- scoreboard S1 = bowling team, S2 = batting/chasing team

Return ONLY the JSON, no other text."""

    raw_output = ask_llm(prompt, client)
    data = extract_json(raw_output)

    if not data:
        data = {
            "table_summary": f"Table {table_name}",
            "columns": {name: dtype for name, dtype in columns}
        }

    return data


# ─────────────────────────────────────────────
# Auto-derive ❌ NO forbidden entries
# ─────────────────────────────────────────────

def derive_forbidden_entries(all_tables_columns):
    """
    Compare columns across tables. If a column exists in table A but not table B,
    and there's a similarly-named column in table B, generate a ❌ NO entry.

    Common confusions:
      - score vs score__runs
      - ball vs overs
      - bowler_id vs player_id
      - batsman__fullname only in fixtures__balls
    """
    # Known column aliases — maps wrong name → (correct name, which tables have the correct one)
    KNOWN_CONFUSIONS = {
        "score": {
            "fixtures__balls": ("score__runs", "Use score__runs in fixtures__balls"),
            "fixtures__bowling": ("runs", "Use 'runs' for runs conceded in fixtures__bowling"),
        },
        "score__runs": {
            "fixtures__batting": ("score", "Use 'score' in fixtures__batting"),
            "fixtures__runs": ("score", "Use 'score' in fixtures__runs"),
            "fixtures__bowling": ("runs", "Use 'runs' in fixtures__bowling"),
        },
        "runs": {
            "fixtures__balls": ("score__runs", "Use 'score__runs' in fixtures__balls"),
        },
        "overs": {
            "fixtures__balls": ("ball", "Use 'ball' in fixtures__balls"),
            "fixtures__batting": ("ball", "Use 'ball' in fixtures__batting"),
        },
        "ball": {
            "fixtures__bowling": ("overs", "Use 'overs' in fixtures__bowling"),
            "fixtures__runs": ("overs", "Use 'overs' in fixtures__runs"),
        },
        "batsman__fullname": {
            "fixtures__batting": (None, "Join to fixtures__balls for batsman names"),
            "fixtures__bowling": (None, "Join to fixtures__balls for batsman names"),
            "fixtures__runs": (None, "Join to fixtures__balls for player names"),
        },
        "bowler__fullname": {
            "fixtures__bowling": (None, "CRITICAL: Join to fixtures__balls for bowler names!"),
            "fixtures__runs": (None, "Join to fixtures__balls for player names"),
        },
        "batsman_id": {
            "fixtures__batting": ("player_id", "Use player_id in fixtures__batting"),
            "fixtures__bowling": (None, "Join to fixtures__balls for batsman information"),
        },
        "bowler_id": {
            "fixtures__bowling": ("player_id", "Use player_id in fixtures__bowling"),
        },
        "wickets": {
            "fixtures__balls": ("score__is_wicket", "Use score__is_wicket or score__out for individual balls, or join to fixtures__runs for total"),
        },
    }

    forbidden_per_table = {}

    for wrong_col, table_rules in KNOWN_CONFUSIONS.items():
        for table_short, (correct_col, message) in table_rules.items():
            # Check if the wrong column actually doesn't exist in this table
            full_table = f"history2.{table_short}"
            if full_table in all_tables_columns:
                actual_cols = {c[0] for c in all_tables_columns[full_table]}
                if wrong_col not in actual_cols:
                    forbidden_per_table.setdefault(full_table, {})[f"❌ NO '{wrong_col}'"] = message

    return forbidden_per_table


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    engine = get_engine()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        import getpass
        api_key = getpass.getpass("Enter OpenAI API key: ")
    client = OpenAI(api_key=api_key)

    # Tables to skip (not relevant to cricket stats)
    SKIP_TABLES = {"continents", "countries"}

    # Step 1: Auto-discover ALL tables in the database
    print("Step 1: Discovering tables from DB...")
    tables = []
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """))
        for schema, table in result:
            if table in SKIP_TABLES:
                print(f"  Skipping: {schema}.{table}")
                continue
            tables.append((schema, table))
    print(f"  Found {len(tables)} tables (skipped {len(SKIP_TABLES)})")

    # Read columns for all tables
    print("\nStep 1b: Reading schema from DB...")
    all_tables_columns = {}
    for schema, table in tables:
        full_name = f"{schema}.{table}"
        columns = get_columns(engine, schema, table)
        all_tables_columns[full_name] = columns
        print(f"  {full_name}: {len(columns)} columns")

    # Step 2: Auto-derive ❌ NO entries
    print("\nStep 2: Deriving forbidden column entries...")
    forbidden_map = derive_forbidden_entries(all_tables_columns)
    for table, entries in forbidden_map.items():
        print(f"  {table}: {len(entries)} forbidden entries")

    # Step 3: Build metadata per table
    print(f"\nStep 3: Building rich metadata for {len(tables)} tables...")
    metadata = {}

    for schema, table in tables:
        full_name = f"{schema}.{table}"
        columns = all_tables_columns[full_name]
        print(f"\n  Processing: {full_name}")

        # Get sample rows
        sample_rows = get_sample_rows(engine, schema, table)
        print(f"    Sample rows: {len(sample_rows)}")

        # Get LLM descriptions
        print(f"    Generating LLM descriptions...")
        llm_data = generate_descriptions(full_name, columns, sample_rows, client)

        # Build column entries with stats
        column_entries = {}
        for col_name, data_type in columns:
            # Skip internal/system columns (but keep 'id' — it's the PK for teams, fixtures, players)
            if col_name.startswith("_") or col_name in ("resource", "sort", "active"):
                continue

            entry = {
                "type": data_type,
                "description": llm_data.get("columns", {}).get(col_name, ""),
            }

            # Add stats (min/max/samples)
            stats = get_column_stats(engine, schema, table, col_name, data_type)
            entry.update(stats)

            column_entries[col_name] = entry

        # Add ❌ NO entries
        if full_name in forbidden_map:
            for key, value in forbidden_map[full_name].items():
                column_entries[key] = value

        # Filter sample rows to only important columns
        important_sample = []
        skip_cols = {"resource", "sort", "active", "updated_at", "_id", "_create_at"}
        for row in sample_rows:
            filtered = {k: v for k, v in row.items()
                        if k.lower() not in skip_cols and not k.startswith("_") and not k.endswith("__idx")}
            important_sample.append(filtered)

        metadata[full_name] = {
            "table_summary": llm_data.get("table_summary", f"Table {full_name}"),
            "columns": column_entries,
            "sample_rows": important_sample,
        }

        # Save after each table (incremental)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"    Saved to {OUTPUT_FILE}")

    print(f"\n{'='*60}")
    print(f"Metadata generation complete!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Tables: {len(metadata)}")
    total_cols = sum(len(t["columns"]) for t in metadata.values())
    print(f"Total column entries: {total_cols}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

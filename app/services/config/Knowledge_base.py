"""
Fast Database Schema Analyzer using OpenAI.

- Uses OpenAI API to query the model.
- Generates detailed table and column descriptions.
- Outputs JSON with table summary + column descriptions useful for SQL.
"""

import json
import re
from urllib.parse import quote
from openai import OpenAI
from sqlalchemy import create_engine, text


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_json_safe(obj): #5
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return str(obj)


def extract_json(text_output): #7
    """
    Cleans the output from Ollama and extracts valid JSON.
    """
    text_output = re.sub(r"```json|```", "", text_output).strip()
    match = re.search(r"\{.*\}", text_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def ask_openai(prompt, client, model="gpt-4o-mini"): #6
    """
    Calls OpenAI API to generate response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        usage = response.usage
        print(f"  Tokens — input: {usage.prompt_tokens}, output: {usage.completion_tokens}, total: {usage.total_tokens}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error calling OpenAI:", e)
        return ""


# ─────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────

def connect_database(): #1
    print("Supported DBs: sqlite, postgresql, mysql, mssql")
    db_type = input("Database type: ").strip().lower()

    if db_type == "sqlite":
        path = input("SQLite file path: ").strip()
        return create_engine(f"sqlite:///{path}"), db_type, "main"

    host = input("Host (default localhost): ").strip() or "localhost"
    port = input("Port (Enter for default): ").strip()
    database = input("Database name: ").strip()
    user = input("Username: ").strip()
    password = quote(input("Password: ").strip())

    if db_type == "postgresql":
        port = port or "5432"
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        schema = "public"

    elif db_type == "mysql":
        port = port or "3306"
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        schema = database

    elif db_type == "mssql":
        port = port or "1433"
        url = (
            f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
            "&Encrypt=no&TrustServerCertificate=yes"
        )
        schema = "dbo"

    else:
        raise ValueError("Unsupported DB")

    return create_engine(url), db_type, schema


# ─────────────────────────────────────────────
# Read schema
# ─────────────────────────────────────────────

def read_schema(engine, db_type, schema):#2
    tables = {}

    if db_type == "sqlite":
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ))
            for (table,) in result:
                cols = conn.execute(text(f"PRAGMA table_info('{table}')"))
                tables[f"main.{table}"] = [
                    {"name": c[1], "type": c[2]} for c in cols
                ]
        return tables

    query = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)

    with engine.connect() as conn:
        for table, col, dtype in conn.execute(query, {"schema": schema}):
            key = f"{schema}.{table}"
            tables.setdefault(key, []).append(
                {"name": col, "type": dtype}
            )

    return tables


# ─────────────────────────────────────────────
# Sample rows (LIMITED)
# ─────────────────────────────────────────────

def get_sample_rows(engine, db_type, table, limit=2):#3
    sql = {
        "sqlite": f"SELECT * FROM {table.split('.')[-1]} LIMIT {limit}",
        "postgresql": f"SELECT * FROM {table} LIMIT {limit}",
        "mysql": f"SELECT * FROM {table} LIMIT {limit}",
        "mssql": f"SELECT TOP {limit} * FROM {table}",
    }

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql[db_type]))
            keys = result.keys()
            return [dict(zip(keys, row)) for row in result]
    except Exception:
        return []


# ─────────────────────────────────────────────
# Generate table and column descriptions
# ─────────────────────────────────────────────

def describe_table_and_columns(table, columns, samples, client):#4
    """
    One OpenAI query per table to get:
    - Table summary
    - Detailed column descriptions
    """
    cols_text = "\n".join(f"- {c['name']} ({c['type']})" for c in columns)
    safe_samples = make_json_safe(samples)

    prompt = f"""You are a SQL database expert. Analyze this table and describe it for someone writing SQL queries.

Table: {table}

Columns:
{cols_text}

Sample data:
{json.dumps(safe_samples, indent=2)}

Return ONLY a valid JSON object with this exact structure:
{{
  "table_summary": "What this table stores and its purpose in the database",
  "columns": {{
    "column_name": "What this column stores, its role (primary key, foreign key, metric, etc), and how it can be used in SQL queries including possible JOINs with other tables"
  }}
}}

Example column description: "fixture_id denotes unique identifier of each match which can act as a foreign key to join multiple tables"

Return ONLY the JSON, no other text."""

    raw_output = ask_openai(prompt, client)
    data = extract_json(raw_output)

    if not data:
        # fallback: just store types
        data = {
            "table_summary": f"Summary of {table}",
            "columns": {c["name"]: c["type"] for c in columns}
        }

    return data


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    api_key = input("OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)

    engine, db_type, default_schema = connect_database()
    schema = input(f"Schema (default {default_schema}): ").strip() or default_schema
    output_file = input("Output file (default metadata.json): ").strip() or "metadata.json"

    schema_info = read_schema(engine, db_type, schema)
    metadata = {}

    for table, columns in schema_info.items():
        print(f"\nProcessing table: {table}")
        samples = get_sample_rows(engine, db_type, table)

        table_data = describe_table_and_columns(table, columns, samples, client)
        metadata[table] = table_data

        with open(output_file, "w") as f: #8
            json.dump(metadata, f, indent=2)

        print("Saved.")

    print(f"\nAll tables processed. Results saved to {output_file}")


if __name__ == "__main__":
    main()

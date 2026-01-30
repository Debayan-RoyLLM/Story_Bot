"""
Generic Database Schema Analyzer using Local Qwen LLM

Works with any database (SQL Server, PostgreSQL, MySQL, SQLite).
Asks the user for connection details, reads all tables and columns,
uses Qwen2.5-7B-Instruct to describe each column, and saves output
to a JSON file.
"""

import json
from urllib.parse import quote
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Load Qwen model ─────────────────────────────────────────────────
print("Loading Qwen2.5-7B-Instruct model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print("Model loaded.\n")


# ── Database connection builder ──────────────────────────────────────
DB_TEMPLATES = {
    "mssql": (
        "mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&Encrypt=no&TrustServerCertificate=yes&Connection Timeout=30"
    ),
    "postgresql": "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
    "mysql": "mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
    "sqlite": "sqlite:///{database}",
}

# SQL dialect-specific queries to fetch schema info
SCHEMA_QUERIES = {
    "mssql": """
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE,
               CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """,
    "postgresql": """
        SELECT table_schema, table_name, column_name, data_type,
               character_maximum_length, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
    """,
    "mysql": """
        SELECT table_schema, table_name, column_name, data_type,
               character_maximum_length, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
    """,
    "sqlite": None,  # handled separately
}

# Dialect-specific sample query syntax
SAMPLE_QUERY_TEMPLATES = {
    "mssql": "SELECT TOP {limit} * FROM {table}",
    "postgresql": "SELECT * FROM {table} LIMIT {limit}",
    "mysql": "SELECT * FROM {table} LIMIT {limit}",
    "sqlite": "SELECT * FROM {table} LIMIT {limit}",
}


def get_connection_from_user() -> tuple:
    """Prompt user for database connection details. Returns (engine, db_type, default_schema)."""
    print("Supported databases: mssql, postgresql, mysql, sqlite")
    db_type = input("Database type: ").strip().lower()

    if db_type not in DB_TEMPLATES:
        raise ValueError(f"Unsupported database type: {db_type}")

    if db_type == "sqlite":
        db_path = input("Path to SQLite file: ").strip()
        url = DB_TEMPLATES[db_type].format(database=db_path)
        return create_engine(url), db_type, "main"

    host = input("Host (default: localhost): ").strip() or "localhost"
    port = input("Port (default depends on db): ").strip()
    if not port:
        port = {"mssql": "1433", "postgresql": "5432", "mysql": "3306"}[db_type]
    database = input("Database name: ").strip()
    user = input("Username: ").strip()
    password = input("Password: ").strip()

    default_schema = {"mssql": "dbo", "postgresql": "public", "mysql": database}[db_type]

    url = DB_TEMPLATES[db_type].format(
        user=user, password=quote(password), host=host, port=port, database=database
    )
    return create_engine(url), db_type, default_schema


# ── Step 1: Read schema from database ───────────────────────────────
def get_schema_info(engine, db_type: str, schema: str) -> dict:
    """Query schema metadata for all tables and columns."""

    # SQLite doesn't have INFORMATION_SCHEMA
    if db_type == "sqlite":
        return _get_sqlite_schema(engine)

    query = text(SCHEMA_QUERIES[db_type])

    schema_info = {}
    with engine.connect() as conn:
        results = conn.execute(query, {"schema": schema})
        for row in results:
            table_key = f"{row[0]}.{row[1]}"
            if table_key not in schema_info:
                schema_info[table_key] = []
            schema_info[table_key].append({
                "column_name": row[2],
                "data_type": row[3],
                "max_length": row[4],
                "nullable": row[5],
            })

    return schema_info


def _get_sqlite_schema(engine) -> dict:
    """Read schema from SQLite using sqlite_master + PRAGMA."""
    schema_info = {}
    with engine.connect() as conn:
        tables = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ))
        for (table_name,) in tables:
            table_key = f"main.{table_name}"
            schema_info[table_key] = []
            cols = conn.execute(text(f"PRAGMA table_info('{table_name}')"))
            for col in cols:
                schema_info[table_key].append({
                    "column_name": col[1],
                    "data_type": col[2],
                    "max_length": None,
                    "nullable": "YES" if col[3] == 0 else "NO",
                })
    return schema_info


# ── Step 2: Fetch sample rows for context ────────────────────────────
def get_sample_rows(engine, db_type: str, table_name: str, limit: int = 3) -> list[dict]:
    """Fetch a few sample rows to help the LLM understand column content."""
    try:
        sql = SAMPLE_QUERY_TEMPLATES[db_type].format(table=table_name, limit=limit)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = []
            for r in result:
                row = {}
                for k, v in zip(columns, r):
                    row[k] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                rows.append(row)
            return rows
    except Exception as e:
        print(f"  Could not fetch samples from {table_name}: {e}")
        return []


# ── Step 3: Use Qwen to describe columns ────────────────────────────
def describe_columns(table_name: str, columns: list[dict], samples: list[dict]) -> dict:
    """Send table schema + samples to Qwen and get column descriptions."""
    columns_text = "\n".join(
        f"  - {c['column_name']} ({c['data_type']}, nullable={c['nullable']})"
        for c in columns
    )

    samples_text = json.dumps(samples, indent=2, default=str) if samples else "No sample data."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a pro Data Analyst. "
                "Your job is to define what each column means in a database table "
                "so that it can be helpful for SQL query generation. "
                "Return ONLY a valid JSON object where each key is the column name "
                "and the value is a 1-2 sentence description. No markdown, no code fences."
            )
        },
        {
            "role": "user",
            "content": (
                f"Table: {table_name}\n\n"
                f"Columns:\n{columns_text}\n\n"
                f"Sample rows:\n{samples_text}\n\n"
                "Describe each column. Return only the JSON object."
            )
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # Strip markdown code fences if present
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0]

    return json.loads(response)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    engine, db_type, default_schema = get_connection_from_user()

    schema = input(f"Enter schema name (default: {default_schema}): ").strip() or default_schema
    output_file = input("Output file name (default: generated_metadata.json): ").strip() or "generated_metadata.json"

    print(f"\nReading schema '{schema}' from database...")
    schema_info = get_schema_info(engine, db_type, schema)

    if not schema_info:
        print(f"No tables found in schema '{schema}'.")
        return

    print(f"Found {len(schema_info)} tables.\n")

    metadata = {}

    for table_name, columns in schema_info.items():
        col_names = [c["column_name"] for c in columns]
        print(f"Processing: {table_name} ({len(columns)} columns: {', '.join(col_names[:5])}...)")

        samples = get_sample_rows(engine, db_type, table_name)

        try:
            descriptions = describe_columns(table_name, columns, samples)
            metadata[table_name] = descriptions
            print(f"  Done.\n")
        except Exception as e:
            print(f"  LLM parsing error: {e}")
            metadata[table_name] = {
                c["column_name"]: f"{c['data_type']}, nullable={c['nullable']}"
                for c in columns
            }

    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_file}")
    print(f"Total tables: {len(metadata)}")
    print(f"Total columns described: {sum(len(v) for v in metadata.values())}")


if __name__ == "__main__":
    main()

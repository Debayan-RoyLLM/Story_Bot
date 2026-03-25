"""
Metadata, State, and Prompt Template

Loads configuration files that the LangGraph nodes need:
  - metadata.json          → table/column schema for validation + LLM context
  - enriched_metadata.json → auto-generated sample rows + value ranges
  - prompt_template.txt    → system prompt for SQL generation
  - State TypedDict        → data shape flowing through the graph
  - QueryOutput            → structured output format for LLM
"""

import json
from pathlib import Path
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm._config import config, logger


# ── Metadata ─────────────────────────────────────────────────────

def load_metadata():
    """
    Load database table metadata from JSON configuration file.

    Returns dict like:
        {"history2.fixtures__balls": {"fixture_id": "...", "❌ NO 'score'": "..."}}
    """
    try:
        with open(config.files.METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            logger.info(f"Metadata loaded from {config.files.METADATA_FILE}")
            return metadata
    except FileNotFoundError:
        logger.error(f"Metadata file not found at {config.files.METADATA_FILE}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        return {}


metadata = load_metadata()


# ── Enriched Metadata (sample rows + value ranges) ───────────────

ENRICHED_FILE = Path(config.files.METADATA_FILE).parent / "enriched_metadata.json"

def load_enriched_metadata():
    """Load auto-generated enriched metadata with sample rows and value ranges."""
    try:
        with open(ENRICHED_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Enriched metadata loaded from {ENRICHED_FILE}")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(f"Enriched metadata not found at {ENRICHED_FILE} — run enrich_metadata.py")
        return {}


enriched_metadata = load_enriched_metadata()


def build_data_context():
    """
    Build a context string from enriched metadata that gets injected into the prompt.
    Shows the LLM what real data looks like — sample rows + value ranges.

    This prevents:
      - "wickets > 9" when meaning "wickets in hand" (LLM sees wickets=4 means 4 LOST)
      - Using team names in batsman__fullname (LLM sees real player names)
      - Unrealistic results (LLM knows score max ~300 for T20)
    """
    if not enriched_metadata:
        return ""

    lines = [
        "",
        "------------------------------------",
        "REAL DATA CONTEXT (auto-generated from database)",
        "------------------------------------",
        "Use this to understand what values actually look like in each table.",
        "",
    ]

    for table_name, table_data in enriched_metadata.items():
        short_name = table_name.split(".")[-1]
        lines.append(f"--- {table_name} ---")

        # Sample rows (show 1-2 rows with key columns only)
        sample_rows = table_data.get("sample_rows", [])
        if sample_rows:
            lines.append("Sample row:")
            row = sample_rows[0]
            # Show only important columns, skip internal ones
            skip = {"resource", "updated_at", "_id", "_create_at", "id", "sort", "active"}
            skip.update(k for k in row if k.startswith("_") or k.endswith("__idx"))
            for k, v in row.items():
                if k.lower() not in skip and not k.startswith("_"):
                    lines.append(f"  {k} = {v}")

        # Value ranges for key numeric columns
        ranges = table_data.get("value_ranges", {})
        if ranges:
            lines.append("Value ranges (per-match, typical):")
            for col, r in ranges.items():
                # Skip internal columns
                if col.startswith("_") or col in ("id", "sort"):
                    continue
                min_v = r.get("min", "?")
                max_v = r.get("max", "?")
                avg_v = r.get("avg", "?")
                lines.append(f"  {col}: min={min_v}, max={max_v}, avg={avg_v}")

        lines.append("")

    lines.extend([
        "⚠️ IMPORTANT DATA NOTES:",
        "- wickets in fixtures__runs = wickets LOST (not in hand). Wickets in hand = 10 - wickets",
        "- batsman__fullname contains PLAYER names (e.g., 'Virat Kohli'), NOT team names",
        "- team__name contains TEAM names (e.g., 'Kolkata Knight Riders')",
        "- team_id and team__id are NUMERIC — never compare with string team names",
        "- score__runs in fixtures__balls is per-ball (0-6), score in fixtures__runs is innings total",
        "- The database contains multiple formats (T20, ODI, Test). Filter appropriately.",
        "- Some rows may have cumulative values. Always use per-match grouping (GROUP BY fixture_id).",
        "------------------------------------",
        "",
    ])

    return "\n".join(lines)


data_context = build_data_context()


# ── Value Ranges (used by sanity checker) ─────────────────────────

def get_sanity_ranges():
    """
    Extract realistic value ranges for sanity checking SQL results.
    Returns dict like: {"score": {"min": 0, "max": 300}, ...}
    """
    ranges = {}
    for table_name, table_data in enriched_metadata.items():
        for col, r in table_data.get("value_ranges", {}).items():
            if col not in ranges:
                ranges[col] = r
    return ranges


sanity_ranges = get_sanity_ranges()


# ── State TypedDict ──────────────────────────────────────────────

class State(TypedDict):
    """Data shape that flows through the LangGraph pipeline."""
    question: str       # "What is Kohli's strike rate?"
    query: str          # "SELECT ... FROM ..."
    result: str         # "45.6"
    answer: str         # "Kohli's strike rate is 45.6"
    table_info: dict    # metadata.json contents
    dialect: str        # "mssql"
    metadata: dict


# ── Prompt Template ──────────────────────────────────────────────

def load_prompt_template():
    """
    Load SQL generation prompt template and append auto-generated data context.
    """
    try:
        with open(config.files.PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
            logger.info(f"Prompt template loaded from {config.files.PROMPT_TEMPLATE_FILE}")
    except FileNotFoundError:
        logger.error(f"Prompt template file not found at {config.files.PROMPT_TEMPLATE_FILE}")
        template = "You are a SQL query generator for cricket statistics."
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        template = "You are a SQL query generator for cricket statistics."

    # Inject data context before the TABLE SCHEMA section
    if data_context and "TABLE SCHEMA:" in template:
        template = template.replace("TABLE SCHEMA:", data_context + "\nTABLE SCHEMA:")
    elif data_context:
        template += "\n" + data_context

    return template


custom_message = load_prompt_template()

query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", custom_message),
    ("human", "{input}")
])

logger.info(f"Template variables: {query_prompt_template.input_variables}")


# ── QueryOutput ──────────────────────────────────────────────────

class QueryOutput(TypedDict):
    """Structured output format for LLM SQL generation."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

"""
Metadata Helpers (Layer 1)

Reads metadata.json and extracts structured data for validation.
These functions convert the raw metadata dict into lookup tables
used by the validation checks.
"""

import re
import logging

logger = logging.getLogger(__name__)


def get_valid_columns(metadata: dict) -> dict[str, set]:
    """
    Extract valid column names per table from metadata.
    Skips entries starting with '❌' (those are hints, not real columns).

    Example output:
        {"history2.fixtures__balls": {"fixture_id", "score__runs", "ball", ...}}
    """
    table_columns = {}
    for table_name, columns in metadata.items():
        if isinstance(columns, dict):
            table_columns[table_name] = {
                col for col in columns.keys()
                if not col.startswith("❌")
            }
    return table_columns


def get_primary_tables(metadata: dict) -> list[str]:
    """
    Derive primary tables from metadata keys.

    Example output:
        ["history2.fixtures__balls", "history2.fixtures__batting", ...]
    """
    return list(metadata.keys())


def get_forbidden_columns(metadata: dict) -> dict[str, dict[str, str]]:
    """
    Extract forbidden column rules per table from metadata '❌ NO' entries.

    Parses keys like: "❌ NO 'overs'" → column name "overs"
    The value is the suggestion message.

    Example output:
        {"fixtures__bowling": {"bowler__fullname": "Join to fixtures__balls..."}}
    """
    forbidden = {}
    for table_name, columns in metadata.items():
        if not isinstance(columns, dict):
            continue
        table_forbidden = {}
        for col, desc in columns.items():
            if col.startswith("❌ NO"):
                match = re.search(r"'(\w+)'", col)
                if match:
                    table_forbidden[match.group(1)] = desc
        if table_forbidden:
            short_name = table_name.split(".")[-1] if "." in table_name else table_name
            forbidden[short_name] = table_forbidden
    return forbidden


def get_ambiguous_columns(metadata: dict) -> list[str]:
    """
    Find columns that appear in multiple tables.
    These need qualifying with table alias in JOIN queries.

    Example output:
        ["fixture_id", "scoreboard", "team_id", "ball", ...]
    """
    column_count = {}
    for table_name, columns in metadata.items():
        if not isinstance(columns, dict):
            continue
        for col in columns.keys():
            if col.startswith("❌"):
                continue
            column_count[col] = column_count.get(col, 0) + 1

    return [col for col, count in column_count.items() if count > 1]


def find_table_aliases(query_lower: str, table_short: str) -> list[str]:
    """
    Find all aliases used for a table in the query.

    Examples:
        "fixtures__balls AS b"  → returns ["b"]
        "fixtures__balls b"     → returns ["b"]
        "fixtures__balls"       → returns []
    """
    alias_pattern = rf'{re.escape(table_short)}\s+(?:AS\s+)?(\w+)'
    return re.findall(alias_pattern, query_lower, re.IGNORECASE)

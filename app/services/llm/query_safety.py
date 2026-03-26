"""
Query Safety Guards

Pre-execution checks that block dangerous or invalid SQL before it hits the database.

Handles:
    - Blocking non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
    - Blocking references to non-existent/forbidden tables
    - Whitelisting column names that contain SQL keywords (e.g., "updated_at")
"""

import re
import logging

logger = logging.getLogger(__name__)


# Tables that definitely don't exist in the database
FORBIDDEN_TABLES = ["HISTORY2.CONTINENTS", "HISTORY2.COUNTRIES", "HISTORY2.VENUES"]

# Destructive SQL keywords to block (matched as whole words)
FORBIDDEN_KEYWORDS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|MERGE|CREATE|EXEC)\b'
)

# Column names that contain SQL keywords but are safe (e.g., "updated_at" contains "UPDATE")
KEYWORD_WHITELIST = ["UPDATED_AT", "UPDATE_"]


def check_forbidden_tables(sql: str) -> str | None:
    """
    Check if the SQL references tables not in our database.
    Returns error message if forbidden table found, None if OK.
    """
    sql_upper = sql.upper()
    for table in FORBIDDEN_TABLES:
        if table in sql_upper:
            return f"Query references non-existent table: {table}"
    return None


def check_query_safety(query: str) -> str | None:
    """
    Block non-SELECT queries and destructive operations.

    Returns error message if query is unsafe, None if OK.
    """
    query_stripped = query.strip().upper()

    # Must start with SELECT or WITH (for CTEs)
    if not query_stripped.startswith(("SELECT", "WITH")):
        logger.error(f"BLOCKED non-SELECT query: {query[:80]}")
        return "Error: Only SELECT queries are allowed"

    # Check for destructive keywords
    match = FORBIDDEN_KEYWORDS.search(query_stripped)
    if match:
        match_pos = match.start()
        surrounding = query_stripped[max(0, match_pos - 5):match_pos + 20]
        # Allow if keyword is part of a safe column name
        if not any(safe in surrounding for safe in KEYWORD_WHITELIST):
            logger.error(f"BLOCKED destructive keyword '{match.group()}' in query: {query[:80]}")
            return f"Error: Query contains forbidden operation ({match.group()})"

    return None

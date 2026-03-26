"""
Query Safety Guards

Pre-execution checks that block dangerous or invalid SQL before it hits the database.

Handles:
    - Blocking non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
    - Blocking references to non-existent/forbidden tables
    - Blocking dangerous system procedures and commands
    - Stripping comments and string literals before analysis
"""

import re
import logging

logger = logging.getLogger(__name__)


# Tables that definitely don't exist in the database
FORBIDDEN_TABLES = ["HISTORY2.CONTINENTS", "HISTORY2.COUNTRIES", "HISTORY2.VENUES"]

# Destructive/dangerous SQL keywords to block (matched as whole words)
FORBIDDEN_KEYWORDS = re.compile(
    r'\b('
    r'INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|MERGE|CREATE'
    r'|EXEC|EXECUTE'
    r'|GRANT|REVOKE|DENY'
    r'|BACKUP|RESTORE|SHUTDOWN'
    r'|WAITFOR'
    r'|OPENROWSET|OPENQUERY|OPENDATASOURCE'
    r')\b'
)

# Dangerous system objects (matched case-insensitive, not as whole words)
FORBIDDEN_OBJECTS = re.compile(
    r'(xp_cmdshell|xp_regread|xp_regwrite|xp_servicecontrol'
    r'|sp_OACreate|sp_OAMethod|sp_configure'
    r'|INFORMATION_SCHEMA)',
    re.IGNORECASE
)

# Column/identifier names that contain SQL keywords but are safe
# e.g., "updated_at" contains "UPDATE", "created_at" contains "CREATE"
SAFE_IDENTIFIER_PATTERN = re.compile(r'[a-zA-Z0-9_]')


def _strip_comments(sql: str) -> str:
    """Remove SQL comments to prevent bypass via comment injection."""
    # Remove block comments /* ... */ (non-greedy, handles nested)
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    # Remove line comments -- ...
    sql = re.sub(r'--[^\n]*', ' ', sql)
    return sql


def _strip_string_literals(sql: str) -> str:
    """Remove string literals to avoid false positives on values like WHERE name = 'DROP'."""
    return re.sub(r"'[^']*'", "''", sql)


def _is_keyword_in_identifier(sql_upper: str, match: re.Match) -> bool:
    """Check if a matched keyword is part of a larger identifier (e.g., updated_at)."""
    start = match.start()
    end = match.end()
    char_before = sql_upper[start - 1] if start > 0 else ' '
    char_after = sql_upper[end] if end < len(sql_upper) else ' '
    return SAFE_IDENTIFIER_PATTERN.match(char_before) is not None or \
           SAFE_IDENTIFIER_PATTERN.match(char_after) is not None


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
    Block non-SELECT queries and dangerous operations.

    Returns error message if query is unsafe, None if OK.
    """
    # Strip comments and string literals for analysis
    cleaned = _strip_comments(query)
    cleaned_upper = _strip_string_literals(cleaned).strip().upper()

    # Must start with SELECT or WITH (for CTEs)
    if not cleaned_upper.startswith(("SELECT", "WITH")):
        logger.error(f"BLOCKED non-SELECT query: {query[:80]}")
        return "Error: Only SELECT queries are allowed"

    # Check for dangerous system objects (xp_cmdshell, sp_OA*, etc.)
    obj_match = FORBIDDEN_OBJECTS.search(cleaned)
    if obj_match:
        logger.error(f"BLOCKED dangerous system object '{obj_match.group()}' in query: {query[:80]}")
        return f"Error: Query references forbidden system object ({obj_match.group()})"

    # Check for destructive keywords
    match = FORBIDDEN_KEYWORDS.search(cleaned_upper)
    if match:
        if not _is_keyword_in_identifier(cleaned_upper, match):
            logger.error(f"BLOCKED destructive keyword '{match.group()}' in query: {query[:80]}")
            return f"Error: Query contains forbidden operation ({match.group()})"

    return None

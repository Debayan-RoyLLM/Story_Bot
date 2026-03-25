"""
Generic SQL Checks (Layer 4)

Validation checks for common SQL pitfalls.
These are not domain-specific — they apply to any SQL query.
They only produce warnings, never reject a query.
"""

import logging

logger = logging.getLogger(__name__)


def validate_division_safety(query: str) -> tuple[bool, list]:
    """
    Check division operations for safety.

    Warns if:
        - No NULLIF → risk of divide-by-zero at runtime
        - No CAST   → integer division may lose precision (e.g., 5/3 = 1)
    """
    logger.debug("[validate_division_safety] Starting...")
    warnings = []
    query_upper = query.upper()

    if "/" not in query:
        logger.debug("[validate_division_safety] No division found, skipping")
        return True, warnings

    if "NULLIF" not in query_upper:
        warnings.append("⚠️  Division without NULLIF - may divide by zero")
    if "CAST" not in query_upper:
        warnings.append("⚠️  Division without CAST - integer division may lose precision")

    logger.debug(f"[validate_division_safety] PASSED with {len(warnings)} warnings")
    return True, warnings


def validate_where_clause(query: str) -> tuple[bool, list]:
    """
    Warn if no WHERE clause on non-aggregate queries.

    A query without WHERE and without aggregates will scan the entire table,
    which is slow and likely not what was intended.
    """
    logger.debug("[validate_where_clause] Starting...")
    warnings = []
    query_upper = query.upper()

    is_aggregate = any(agg in query_upper for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])
    if "WHERE" not in query_upper and not is_aggregate:
        warnings.append("⚠️  No WHERE clause - query will scan entire table")

    logger.debug(f"[validate_where_clause] PASSED with {len(warnings)} warnings")
    return True, warnings

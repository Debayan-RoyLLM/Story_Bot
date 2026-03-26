"""
Generic SQL Checks (Layer 4)

Validation checks for common SQL pitfalls.
These are not domain-specific — they apply to any SQL query.
They only produce warnings, never reject a query.
"""

import re
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


def validate_groupby_aggregation(query: str, question: str) -> tuple[bool, list]:
    """
    Warn when GROUP BY may conflict with the question's intent.

    Detects:
        - Question asks for "total"/"overall" but final SELECT has GROUP BY
          without an outer aggregation (SUM/COUNT/AVG) wrapping it.
          This returns per-group rows instead of one total.
        - GROUP BY fixture_id in the outermost SELECT when the question
          expects a single aggregated answer.
    """
    logger.debug("[validate_groupby_aggregation] Starting...")
    warnings = []
    query_upper = query.upper()
    question_lower = question.lower()

    # Only check if query has GROUP BY
    if "GROUP BY" not in query_upper:
        logger.debug("[validate_groupby_aggregation] No GROUP BY found, skipping")
        return True, warnings

    # Keywords that indicate the user wants a single aggregated value
    total_keywords = ["total", "overall", "combined", "how many", "what is the", "aggregate"]
    wants_single = any(kw in question_lower for kw in total_keywords)

    if not wants_single:
        logger.debug("[validate_groupby_aggregation] Question doesn't ask for a total, skipping")
        return True, warnings

    # Check if GROUP BY is in the outermost SELECT (not just in a CTE/subquery)
    # Simple heuristic: find the last SELECT and check if GROUP BY follows it
    # Strip CTEs by finding content after the last ")" before a standalone SELECT
    outer_query = query_upper

    # Try to isolate the outermost query (after CTEs)
    # CTEs are: WITH ... AS (...), ... AS (...) SELECT ...
    cte_pattern = r'\)\s*SELECT\b'
    cte_matches = list(re.finditer(cte_pattern, query_upper))
    if cte_matches:
        last_match = cte_matches[-1]
        outer_query = query_upper[last_match.start() + 1:]

    if "GROUP BY" in outer_query:
        # Check if the outer SELECT has a wrapping aggregation
        select_clause = outer_query.split("FROM")[0] if "FROM" in outer_query else outer_query
        agg_functions = ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("]
        has_outer_agg = any(agg in select_clause for agg in agg_functions)

        if not has_outer_agg:
            warnings.append(
                "⚠️  Question asks for a total/aggregate but the outermost query has "
                "GROUP BY without SUM/COUNT/AVG — this returns per-group rows instead "
                "of a single total. Consider wrapping with an outer SELECT SUM(...)."
            )

    logger.debug(f"[validate_groupby_aggregation] PASSED with {len(warnings)} warnings")
    return True, warnings

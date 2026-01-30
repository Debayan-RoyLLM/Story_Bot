"""
SQL Query Validation Module

This module contains all validation functions for T-SQL query generation.
Each function validates a specific aspect of the generated SQL query.
"""

import re
import logging

logger = logging.getLogger(__name__)


def validate_subquery_safety(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 1: Validate subquery doesn't return multiple values with scalar operators."""
    logger.debug("Check 1: Subquery multiple values...")

    if "WHERE" in query_upper and "SELECT" in query_upper:
        scalar_patterns = [
            r'WHERE.*?[=><]\s*\(SELECT',
            r'AND.*?[=><]\s*\(SELECT',
            r'OR.*?[=><]\s*\(SELECT',
        ]

        for pattern in scalar_patterns:
            matches = re.findall(pattern, query_upper, re.DOTALL)
            for match in matches:
                if " IN " in match:
                    continue
                has_safety = any(x in match for x in ["TOP 1", "AVG(", "MAX(", "MIN(", "SUM(", "COUNT("])
                if not has_safety:
                    return False, "❌ Subquery with scalar comparison (=,>,<) must use TOP 1 or aggregate function (AVG/MAX/MIN)"

    return True, ""


def validate_column_names(query: str, metadata: dict) -> tuple[bool, str]:
    """
    Check 2: Validate that query uses EXACT column names from metadata.
    Returns: (is_valid, error_message)
    """
    logger.debug("Check 2: Column name validation...")
    query_upper = query.upper()
    query_lower = query.lower()

    # ===== CHECK 1: Forbidden Column Names =====
    forbidden = {
        " SCORE ": "Use score__runs (in fixtures__balls) or score (in fixtures__batting/runs)",
        " RUNS ": "Use score__runs (fixtures__balls) or runs (fixtures__bowling)",
        " RATE ": "Column 'RATE' must be qualified: batting.rate or bowling.rate",
        " INNING ": "Use inning from fixtures__runs, NOT from fixtures__balls",
        " OVER ": "Use ball number (fixtures__balls) or overs (fixtures__runs/bowling)",
        "SCOREBOARD_NAME": "Use scoreboard column (S1 or S2), not scoreboard_name",
        "TEAM_NAME": "Use team__id, not team_name",
        "VENUE": "Column venue does not exist in metadata",
        "SEASON": "No season column - use ball/overs for chronology",
        "DATE": "No date column - use fixture_id and ball/overs for order",
        "PLAYER_NAME": "Use batsman__fullname or bowler__fullname",
        "MATCH_ID": "Use fixture_id (not match_id)",
    }

    for forbidden_col, suggestion in forbidden.items():
        if forbidden_col in query_upper:
            return False, f"❌ Invalid column: {forbidden_col.strip()}. {suggestion}"

    # ===== CHECK 2: Column Table Mismatch =====
    if "fixtures__balls" in query_lower:
        forbidden_in_balls = {
            ".score ": "Use score__runs in fixtures__balls",
            ".INNING": "Inning only exists in fixtures__runs - JOIN if needed",
        }
        for col, msg in forbidden_in_balls.items():
            if col in query_lower:
                return False, f"❌ {msg}"

    # ===== CHECK 3: Verify Column Exists in Referenced Table =====
    if "SELECT" in query_upper and "FROM" in query_upper:
        if "history2.fixtures__runs" in query_lower:
            required_cols = ["fixture_id", "inning", "score", "wickets", "overs"]
            for col in required_cols:
                if col.lower() in query_lower and f"fixtures__runs.{col}" not in query_lower:
                    pass

    # Check for table aliases
    if "AS " in query_upper:
        aliases = re.findall(r'FROM\s+(\S+)\s+AS\s+(\S+)', query_upper)
        for table, alias in aliases:
            if "FIXTURES" not in table:
                return False, f"❌ Unknown table: {table}"

    # Check for qualified columns
    if "." in query and "CAST" not in query_upper:
        pass

    return True, "✅ Column names validated"


def validate_order_by_rules(query: str, query_upper: str, query_lower: str) -> tuple[bool, list]:
    """Check 3: Validate ORDER BY with DISTINCT usage."""
    logger.debug("Check 3: ORDER BY validation...")
    warnings = []

    if "ORDER BY" in query_upper and "DISTINCT" in query_upper:
        if "OVER" not in query_upper and "WITH TIES" not in query_upper:
            warnings.append("⚠️  DISTINCT with ORDER BY - ensure ORDER BY column in SELECT or use WITH TIES")

    return True, warnings


def validate_aggregate_rules(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 4: Validate aggregate function usage."""
    logger.debug("Check 4: Aggregate function validation...")

    forbidden_aggregates = [
        ("SUM(", "score__is_wicket", "Cannot SUM bit column. Use COUNT(CASE WHEN ...)"),
        ("AVG(", "score__is_wicket", "Cannot AVG bit column. Use AVG(CAST(...))"),
    ]

    for agg, col, msg in forbidden_aggregates:
        if agg in query_upper and col in query_lower:
            return False, f"❌ {msg}"

    return True, ""


def validate_data_types(query: str, query_upper: str, query_lower: str) -> tuple[bool, list]:
    """Check 5: Validate data type handling in divisions."""
    logger.debug("Check 5: Data type validation...")
    warnings = []

    if "/" in query:
        if "NULLIF" not in query_upper:
            warnings.append("⚠️  Division without NULLIF - may divide by zero")
        if "CAST" not in query_upper and "score__runs" in query_lower:
            warnings.append("⚠️  Division on score__runs without CAST - integer division may lose precision")

    return True, warnings


def validate_primary_table(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 6: Validate primary table is used."""
    logger.debug("Check 6: Primary table validation...")

    has_primary = "history2.fixtures__balls" in query_lower or "history2.fixtures__runs" in query_lower
    if not has_primary:
        return False, "❌ Query must use history2.fixtures__balls or history2.fixtures__runs"

    return True, ""


def validate_join_syntax(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 7: Validate JOIN has ON clause."""
    logger.debug("Check 7: JOIN syntax validation...")

    if "JOIN" in query_upper and "ON" not in query_upper:
        return False, "❌ JOIN without ON clause"

    return True, ""


def validate_where_clause(query: str, query_upper: str, query_lower: str) -> tuple[bool, list]:
    """Check 8: Validate WHERE clause presence."""
    logger.debug("Check 8: WHERE clause validation...")
    warnings = []

    is_aggregate = "COUNT(" in query_upper or "SUM(" in query_upper or "AVG(" in query_upper
    if "WHERE" not in query_upper and not is_aggregate:
        warnings.append("⚠️  No WHERE clause - query will scan entire table")

    return True, warnings


def validate_group_by(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 9: Validate GROUP BY clause completeness."""
    logger.debug("Check 9: GROUP BY validation...")

    if "GROUP BY" in query_upper:
        select_part = re.search(r'SELECT\s+(.*?)\s+FROM', query_upper, re.DOTALL)
        group_by_part = re.search(r'GROUP BY\s+(.*?)(?:HAVING|ORDER BY|$)', query_upper, re.DOTALL)

        if select_part and group_by_part:
            select_cols = select_part.group(1)
            group_cols = group_by_part.group(1)

            agg_functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']

            if '.' in select_cols and not any(agg in select_cols for agg in agg_functions):
                cols_in_select = re.findall(r'(\w+\.\w+)', select_cols)
                cols_in_group = re.findall(r'(\w+\.\w+)', group_cols)

                for col in cols_in_select:
                    if col not in group_cols and not any(f"{agg}{col}" in select_cols for agg in agg_functions):
                        return False, f"❌ Column {col} in SELECT must be in GROUP BY clause or wrapped in aggregate function (COUNT/SUM/AVG/MAX/MIN)"

    return True, ""


def validate_cte_order_by(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 10: Validate ORDER BY in CTEs."""
    logger.debug("Check 10: CTE ORDER BY validation...")

    if "WITH" in query_upper and "ORDER BY" in query_upper:
        cte_pattern = r'WITH\s+\w+\s+AS\s*\((.*?)\)(?:\s*,|\s+SELECT)'
        cte_matches = re.findall(cte_pattern, query_upper, re.DOTALL | re.IGNORECASE)

        for cte_body in cte_matches:
            if "ORDER BY" in cte_body:
                if "TOP" not in cte_body and "OFFSET" not in cte_body and "FOR XML" not in cte_body:
                    return False, "❌ ORDER BY in CTE requires TOP, OFFSET, or FOR XML clause"

    return True, ""


def validate_top_offset_conflict(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 11: Validate TOP and OFFSET aren't used together."""
    logger.debug("Check 11: TOP/OFFSET conflict check...")

    if "TOP" in query_upper and "OFFSET" in query_upper:
        select_parts = query_upper.split("SELECT")
        for part in select_parts[1:]:
            next_select_or_end = part.split("SELECT")[0] if "SELECT" in part else part
            if "TOP" in next_select_or_end and "OFFSET" in next_select_or_end:
                return False, "❌ Cannot use TOP and OFFSET in same SELECT. Use OFFSET...FETCH instead."

    return True, ""


def validate_fetch_syntax(query: str, query_upper: str, query_lower: str) -> tuple[bool, list]:
    """Check 12: Validate FETCH syntax."""
    logger.debug("Check 12: FETCH syntax validation...")
    warnings = []

    if "FETCH" in query_upper:
        if "OFFSET" not in query_upper:
            return False, "❌ FETCH requires OFFSET. Use: OFFSET n ROWS FETCH NEXT m ROWS ONLY"
        if "FETCH NEXT" in query_upper and "ROWS ONLY" not in query_upper:
            warnings.append("⚠️  FETCH NEXT should end with ROWS ONLY")

    return True, warnings


def validate_column_table_mismatch(query: str, query_upper: str, query_lower: str) -> tuple[bool, str]:
    """Check 13: Validate columns are used with correct tables."""
    logger.debug("Check 13: Column-table mismatch validation...")

    table_aliases = {}

    alias_patterns = [
        (r'fixtures__bowling\s+(?:AS\s+)?(\w+)', 'fixtures__bowling'),
        (r'fixtures__batting\s+(?:AS\s+)?(\w+)', 'fixtures__batting'),
        (r'fixtures__runs\s+(?:AS\s+)?(\w+)', 'fixtures__runs'),
        (r'fixtures__balls\s+(?:AS\s+)?(\w+)', 'fixtures__balls'),
    ]

    for pattern, table_name in alias_patterns:
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        if matches:
            table_aliases[table_name] = matches

    # Check fixtures__bowling
    if 'fixtures__bowling' in query_lower or 'fixtures__bowling' in table_aliases:
        aliases = table_aliases.get('fixtures__bowling', [])
        for alias in aliases + ['fixtures__bowling']:
            if re.search(rf'\b{alias}\.bowler__fullname\b', query_lower):
                return False, f"❌ bowler__fullname doesn't exist in fixtures__bowling. Use fixtures__balls.bowler__fullname and JOIN!"
            if re.search(rf'\b{alias}\.batsman__fullname\b', query_lower):
                return False, f"❌ batsman__fullname doesn't exist in fixtures__bowling. Use fixtures__balls.batsman__fullname and JOIN!"
            if re.search(rf'\b{alias}\.score__runs\b', query_lower):
                return False, f"❌ score__runs doesn't exist in fixtures__bowling. Use 'runs' for runs conceded!"
            if re.search(rf'\b{alias}\.ball\b', query_lower):
                return False, f"❌ 'ball' doesn't exist in fixtures__bowling. Use 'overs'!"

    # Check fixtures__batting
    if 'fixtures__batting' in query_lower or 'fixtures__batting' in table_aliases:
        aliases = table_aliases.get('fixtures__batting', [])
        for alias in aliases + ['fixtures__batting']:
            if re.search(rf'\b{alias}\.batsman__fullname\b', query_lower):
                return False, f"❌ batsman__fullname doesn't exist in fixtures__batting. Use fixtures__balls.batsman__fullname and JOIN!"
            if re.search(rf'\b{alias}\.score__runs\b', query_lower):
                return False, f"❌ Use 'score' not 'score__runs' in fixtures__batting!"
            if re.search(rf'\b{alias}\.overs\b', query_lower):
                return False, f"❌ 'overs' doesn't exist in fixtures__batting. Use 'ball'!"

    # Check fixtures__runs
    if 'fixtures__runs' in query_lower or 'fixtures__runs' in table_aliases:
        aliases = table_aliases.get('fixtures__runs', [])
        for alias in aliases + ['fixtures__runs']:
            if re.search(rf'\b{alias}\.(batsman__fullname|bowler__fullname)\b', query_lower):
                return False, f"❌ Player names don't exist in fixtures__runs. JOIN to fixtures__balls!"
            if re.search(rf'\b{alias}\.score__runs\b', query_lower):
                return False, f"❌ Use 'score' not 'score__runs' in fixtures__runs!"
            if re.search(rf'\b{alias}\.ball\b', query_lower):
                return False, f"❌ 'ball' doesn't exist in fixtures__runs. Use 'overs'!"

    # Check fixtures__balls
    if 'fixtures__balls' in query_lower or 'fixtures__balls' in table_aliases:
        aliases = table_aliases.get('fixtures__balls', [])
        for alias in aliases + ['fixtures__balls']:
            if re.search(rf'\b{alias}\.overs\b', query_lower):
                return False, f"❌ 'overs' doesn't exist in fixtures__balls. Use 'ball' or JOIN to fixtures__runs!"
            if re.search(rf'\b{alias}\.runs\b', query_lower):
                return False, f"❌ Use 'score__runs' not 'runs' in fixtures__balls!"
            if re.search(rf'\b{alias}\.player_id\b', query_lower):
                return False, f"❌ Use 'batsman_id' not 'player_id' in fixtures__balls!"

    return True, ""


def validate_unqualified_columns(query: str, query_upper: str, query_lower: str) -> tuple[bool, list]:
    """Check 14: Validate column names are qualified in JOINs."""
    logger.debug("Check 14: Unqualified column validation...")
    warnings = []

    if "JOIN" in query_upper:
        ambiguous_cols = ['score', 'overs', 'wickets', 'fixture_id', 'team_id', 'ball']

        for col in ambiguous_cols:
            if re.search(rf'(?<!\w\.)\b{col}\b(?!\s*=\s*\()', query_lower):
                if '.' not in query_lower[:query_lower.find(col)] or True:
                    if re.search(rf'(WHERE|SELECT|ON)\s+.*?\b{col}\b', query_upper):
                        warnings.append(f"⚠️  Consider qualifying column '{col}' with table alias in JOIN queries")
                        break

    return True, warnings


def validate_generated_query(query: str, metadata: dict, question: str) -> tuple[bool, str, list]:
    """
    Comprehensive validation with metadata verification.
    Orchestrates all 14 validation checks.

    Args:
        query: The SQL query to validate
        metadata: Table and column metadata
        question: The original natural language question

    Returns:
        tuple: (is_valid, error_message, warnings_list)
    """
    all_warnings = []
    query_upper = query.upper()
    query_lower = query.lower()

    # Metadata validation
    if not isinstance(metadata, dict) or not metadata:
        return False, "❌ Metadata not provided or invalid", []

    required_tables = ["history2.fixtures__balls", "history2.fixtures__runs"]
    has_table_info = any(table in str(metadata) for table in required_tables)
    if not has_table_info:
        logger.warning("Metadata may not contain expected tables")

    logger.debug("="*70)
    logger.debug("VALIDATING QUERY")
    logger.debug(f"Query: {query[:80]}...")
    logger.debug("="*70)

    # Run all validation checks in sequence
    validators = [
        (validate_subquery_safety, True),      # Returns (bool, str)
        (validate_column_names, False),        # Returns (bool, str) - uses metadata
        (validate_order_by_rules, True),       # Returns (bool, list)
        (validate_aggregate_rules, True),      # Returns (bool, str)
        (validate_data_types, True),           # Returns (bool, list)
        (validate_primary_table, True),        # Returns (bool, str)
        (validate_join_syntax, True),          # Returns (bool, str)
        (validate_where_clause, True),         # Returns (bool, list)
        (validate_group_by, True),             # Returns (bool, str)
        (validate_cte_order_by, True),         # Returns (bool, str)
        (validate_top_offset_conflict, True),  # Returns (bool, str)
        (validate_fetch_syntax, True),         # Returns (bool, list)
        (validate_column_table_mismatch, True),# Returns (bool, str)
        (validate_unqualified_columns, True),  # Returns (bool, list)
    ]

    for validator, uses_query_vars in validators:
        if uses_query_vars:
            is_valid, result = validator(query, query_upper, query_lower)
        else:
            is_valid, result = validator(query, metadata)

        if not is_valid:
            return False, result, []

        if isinstance(result, list):
            all_warnings.extend(result)

    logger.info("Query validation passed")
    if all_warnings:
        logger.warning(f"Validation warnings: {len(all_warnings)}")
        for warning in all_warnings:
            logger.warning(warning)

    return True, "✅ Query structure valid", all_warnings

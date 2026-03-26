"""
Template-Based SQL Fixes (No LLM)

Fast, regex-based corrections for common SQL generation errors.
These run before falling back to an LLM regeneration call.

Handles:
    - Column name mismatches across tables (score vs score__runs, ball vs overs)
    - PK column name fixes (team_id → id in lookup tables)
    - MySQL → T-SQL syntax (LIMIT → TOP)
    - Subquery cardinality (adding TOP 1)
    - Division safety (NULLIF wrapping)
"""

import re
import logging

logger = logging.getLogger(__name__)


# ── Column name mapping: {error_column: {table_context: (replacement, log_msg)}} ──
COLUMN_FIX_MAP = {
    "'score'": {
        "fixtures__balls":   ("score__runs", "score → score__runs (fixtures__balls)"),
        "fixtures__bowling": ("runs",        "score → runs (fixtures__bowling)"),
    },
    "'overs'": {
        "fixtures__balls": ("ball", "overs → ball (fixtures__balls)"),
    },
    "'runs'": {
        "fixtures__balls": ("score__runs", "runs → score__runs (fixtures__balls)"),
    },
    "'ball'": {
        "fixtures__bowling": ("overs", "ball → overs (fixtures__bowling)"),
    },
    "'score__runs'": {
        "fixtures__batting": ("score", "score__runs → score (fixtures__batting)"),
        "fixtures__runs":    ("score", "score__runs → score (fixtures__runs)"),
    },
}

# ── PK fixes: lookup tables where the PK is 'id' but LLM guesses wrong ──
PK_FIX_MAP = {
    "history2.teams":   ["team_id", "teams_id"],
    "history2.players": ["player_id", "players_id"],
}


def _try_column_fix(sql, error_lower):
    """Fix invalid column names using COLUMN_FIX_MAP."""
    sql_lower = sql.lower()

    for error_col, table_rules in COLUMN_FIX_MAP.items():
        if error_col not in error_lower:
            continue
        col_name = error_col.strip("'")
        for table_ctx, (replacement, log_msg) in table_rules.items():
            if table_ctx in sql_lower:
                logger.info(f"Template fix: {log_msg}")
                return re.sub(
                    rf'(?<!\w){re.escape(col_name)}(?!\w)',
                    replacement, sql, flags=re.IGNORECASE
                )
    return None


def _try_pk_fix(sql, error_lower):
    """Fix PK column names in lookup tables (e.g., team_id → id in teams)."""
    sql_lower = sql.lower()

    for table, wrong_names in PK_FIX_MAP.items():
        if table not in sql_lower:
            continue
        for wrong in wrong_names:
            if f"'{wrong}'" not in error_lower:
                continue
            logger.info(f"Template fix: {wrong} → id ({table})")
            return re.sub(
                rf'(?i)\bSELECT\s+(TOP\s+\d+\s+)?{re.escape(wrong)}\s+(FROM\s+{re.escape(table)})',
                lambda m: f'SELECT {m.group(1) or ""}id {m.group(2)}',
                sql
            )

    # Special case: fixture_id → id only in history2.fixtures (not fixtures__balls etc.)
    if "'fixture_id'" in error_lower and re.search(r'history2\.fixtures\b(?!_)', sql_lower):
        logger.info("Template fix: fixture_id → id (history2.fixtures)")
        return re.sub(
            r'(?i)\bSELECT\s+(TOP\s+\d+\s+)?(\w+\.)?fixture_id\s+(FROM\s+history2\.fixtures\b(?!_))',
            lambda m: f'SELECT {m.group(1) or ""}{m.group(2) or ""}id {m.group(3)}',
            sql
        )

    return None


def _try_syntax_fix(sql, error_lower):
    """Fix MySQL → T-SQL syntax issues and runtime errors."""

    # ── LIMIT → TOP ──
    limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
    if limit_match:
        n = limit_match.group(1)
        logger.info(f"Template fix: LIMIT {n} → TOP {n}")
        sql = re.sub(r'\s*LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'SELECT\s', f'SELECT TOP {n} ', sql, count=1, flags=re.IGNORECASE)
        return sql

    # ── Subquery returned more than 1 value → add TOP 1 ──
    if "subquery returned more than 1 value" in error_lower:
        logger.info("Template fix: adding TOP 1 to subquery")
        return re.sub(
            r'\(\s*SELECT\s+(?!TOP\s)',
            '(SELECT TOP 1 ',
            sql, count=1, flags=re.IGNORECASE
        )

    # ── Divide by zero → wrap denominator with NULLIF ──
    if "divide by zero" in error_lower:
        logger.info("Template fix: wrapping denominator with NULLIF")
        return re.sub(
            r'/\s*(\w+\.\w+|\w+)',
            r'/ NULLIF(\1, 0)',
            sql, count=1
        )

    return None


def try_template_fix(failed_sql, error_msg):
    """
    Attempt to fix common SQL errors with regex-based replacement.
    Tries fixes in order: column names → PK names → syntax.
    Returns fixed SQL or None if no template matches.
    """
    error_lower = error_msg.lower()

    if "invalid column name" in error_lower:
        # Try column name fix first
        fix = _try_column_fix(failed_sql, error_lower)
        if fix:
            return fix

        # Try PK column fix
        fix = _try_pk_fix(failed_sql, error_lower)
        if fix:
            return fix

    # Try syntax fixes (LIMIT, TOP 1, NULLIF)
    return _try_syntax_fix(failed_sql, error_lower)

"""
SQL Query Validation Package

Validates generated T-SQL queries using:
1. DB dry-run (SET NOEXEC ON) — lets SQL Server catch schema/syntax errors
2. Metadata-driven checks — business rules derived from metadata.json
3. Generic SQL checks — division safety, WHERE clause warnings

Structure:
    metadata_helpers.py  — Layer 1: Parse metadata.json into lookup tables
    db_validation.py     — Layer 2: SQL Server dry-run compilation
    metadata_checks.py   — Layer 3: Column-table mismatch, primary table, ambiguous cols
    sql_checks.py        — Layer 4: Division safety, WHERE clause
    __init__.py          — Layer 5: Orchestrator (this file)
"""

import logging

from app.services.validators.db_validation import validate_with_db
from app.services.validators.metadata_checks import (
    validate_column_table_mismatch,
    validate_primary_table,
    validate_unqualified_columns,
)
from app.services.validators.sql_checks import (
    validate_division_safety,
    validate_where_clause,
    validate_groupby_aggregation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CHECK RUNNERS (for consistent logging)
# ============================================================================

def _run_hard_check(check_name: str, check_fn, *args) -> tuple[bool, str]:
    """
    Run a validation check that can FAIL the query.
    Logs check name and result for debugging.
    """
    logger.info(f"[VALIDATOR] Running: {check_name}")
    is_valid, error_msg = check_fn(*args)

    if not is_valid:
        logger.error(f"[VALIDATOR] {check_name} → FAILED: {error_msg}")
    else:
        logger.info(f"[VALIDATOR] {check_name} → PASSED")

    return is_valid, error_msg


def _run_warning_check(check_name: str, check_fn, *args) -> list:
    """
    Run a validation check that only produces WARNINGS (never fails).
    Logs check name and any warnings for debugging.
    """
    logger.info(f"[VALIDATOR] Running: {check_name}")
    _, warnings = check_fn(*args)

    if warnings:
        for w in warnings:
            logger.warning(f"[VALIDATOR] {check_name} → {w}")
    else:
        logger.info(f"[VALIDATOR] {check_name} → PASSED (no warnings)")

    return warnings


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def validate_generated_query(query: str, metadata: dict, question: str,
                              engine=None) -> tuple[bool, str, list]:
    """
    Validate a generated SQL query.

    Runs checks in order:
      Hard checks (can reject query):
        1. DB dry-run
        2. Column-table mismatch
        3. Primary table usage
      Warning checks (always pass, collect warnings):
        4. Unqualified columns in JOINs
        5. Division safety
        6. WHERE clause

    Args:
        query: The SQL query to validate
        metadata: Table and column metadata (from metadata.json)
        question: The original natural language question
        engine: SQLAlchemy engine for DB dry-run (optional)

    Returns:
        tuple: (is_valid, error_message, warnings_list)
    """
    if not isinstance(metadata, dict) or not metadata:
        return False, "❌ Metadata not provided or invalid", []

    logger.info(f"[VALIDATOR] {'='*60}")
    logger.info(f"[VALIDATOR] Validating query: {query[:80]}...")
    logger.info(f"[VALIDATOR] Question: {question[:80]}...")
    logger.info(f"[VALIDATOR] {'='*60}")

    # ── Hard checks (reject on failure) ──────────────────────────

    if engine is not None:
        is_valid, error_msg = _run_hard_check("DB dry-run", validate_with_db, query, engine)
        if not is_valid:
            return False, error_msg, []

    is_valid, error_msg = _run_hard_check("Column-table mismatch", validate_column_table_mismatch, query, metadata)
    if not is_valid:
        return False, error_msg, []

    is_valid, error_msg = _run_hard_check("Primary table", validate_primary_table, query, metadata)
    if not is_valid:
        return False, error_msg, []

    # ── Warning checks (always pass) ─────────────────────────────

    all_warnings = []
    all_warnings += _run_warning_check("Unqualified columns", validate_unqualified_columns, query, metadata)
    all_warnings += _run_warning_check("Division safety", validate_division_safety, query)
    all_warnings += _run_warning_check("WHERE clause", validate_where_clause, query)
    all_warnings += _run_warning_check("GROUP BY aggregation", validate_groupby_aggregation, query, question)

    # ── Summary ──────────────────────────────────────────────────

    logger.info(f"[VALIDATOR] {'='*60}")
    logger.info(f"[VALIDATOR] RESULT: PASSED ({len(all_warnings)} warnings)")
    logger.info(f"[VALIDATOR] {'='*60}")

    return True, "✅ Query structure valid", all_warnings

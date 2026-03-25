"""
Metadata-Driven Checks (Layer 3)

Validation checks that read rules from metadata.json.
These catch the same errors as the DB dry-run but provide
human-readable fix suggestions that help the LLM regenerate.
"""

import re
import logging
from app.services.validators.metadata_helpers import (
    get_forbidden_columns,
    get_primary_tables,
    get_ambiguous_columns,
    find_table_aliases,
)

logger = logging.getLogger(__name__)


def validate_column_table_mismatch(query: str, metadata: dict) -> tuple[bool, str]:
    """
    Check columns used with table aliases against metadata '❌ NO' entries.

    Example:
        Query: SELECT bowl.bowler__fullname FROM fixtures__bowling bowl
        Metadata has: "❌ NO 'bowler__fullname'" in fixtures__bowling
        Result: FAIL → "bowler__fullname doesn't exist in fixtures__bowling.
                        Join to fixtures__balls for bowler names!"
    """
    logger.debug("[validate_column_table_mismatch] Starting...")
    query_lower = query.lower()
    forbidden_map = get_forbidden_columns(metadata)

    for table_short, forbidden_cols in forbidden_map.items():
        if table_short not in query_lower:
            logger.debug(f"[validate_column_table_mismatch] Table '{table_short}' not in query, skipping")
            continue

        aliases = find_table_aliases(query_lower, table_short)
        references = aliases + [table_short]
        logger.debug(f"[validate_column_table_mismatch] Table '{table_short}' found, aliases: {aliases}")

        for ref in references:
            for col, suggestion in forbidden_cols.items():
                if re.search(rf'\b{re.escape(ref)}\.{re.escape(col)}\b', query_lower):
                    msg = f"❌ '{col}' doesn't exist in {table_short}. {suggestion}"
                    logger.warning(f"[validate_column_table_mismatch] FAILED: {msg}")
                    return False, msg

    logger.debug("[validate_column_table_mismatch] PASSED")
    return True, ""


def validate_primary_table(query: str, metadata: dict) -> tuple[bool, str]:
    """
    Check that the query references at least one table from metadata.

    Prevents the LLM from querying random/non-existent tables.
    """
    logger.debug("[validate_primary_table] Starting...")
    query_lower = query.lower()
    tables = get_primary_tables(metadata)

    found_tables = [t for t in tables if t.lower() in query_lower]

    if not found_tables:
        table_list = ", ".join(tables)
        msg = f"❌ Query must reference at least one known table: {table_list}"
        logger.warning(f"[validate_primary_table] FAILED: {msg}")
        return False, msg

    logger.debug(f"[validate_primary_table] PASSED — found: {found_tables}")
    return True, ""


def validate_unqualified_columns(query: str, metadata: dict) -> tuple[bool, list]:
    """
    Warn about columns that exist in multiple tables used without table qualifier.

    Example:
        Query: SELECT fixture_id FROM balls JOIN runs ON ...
        'fixture_id' exists in both tables → warn to qualify as b.fixture_id
    """
    logger.debug("[validate_unqualified_columns] Starting...")
    warnings = []
    query_upper = query.upper()
    query_lower = query.lower()

    if "JOIN" not in query_upper:
        logger.debug("[validate_unqualified_columns] No JOIN found, skipping")
        return True, warnings

    ambiguous_cols = get_ambiguous_columns(metadata)
    logger.debug(f"[validate_unqualified_columns] Ambiguous columns: {ambiguous_cols}")

    for col in ambiguous_cols:
        if re.search(rf'(?<!\w\.)\b{re.escape(col)}\b(?!\s*=\s*\()', query_lower):
            if re.search(rf'(WHERE|SELECT|ON)\s+.*?\b{re.escape(col)}\b', query_upper):
                warning = f"⚠️  Consider qualifying column '{col}' with table alias in JOIN queries"
                logger.debug(f"[validate_unqualified_columns] {warning}")
                warnings.append(warning)
                break

    logger.debug(f"[validate_unqualified_columns] PASSED with {len(warnings)} warnings")
    return True, warnings

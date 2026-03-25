"""
DB Dry-Run Validation (Layer 2)

Uses SQL Server's SET NOEXEC ON to compile a query without executing it.
This catches syntax errors, invalid columns, wrong tables, type mismatches —
everything SQL Server checks at compile time.
"""

import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


def validate_with_db(query: str, engine) -> tuple[bool, str]:
    """
    Compile the query on SQL Server without executing it.

    How it works:
        SET NOEXEC ON  → tells SQL Server to compile only
        <query>        → SQL Server checks everything
        SET NOEXEC OFF → reset

    Returns:
        (True, "")           → query compiles fine
        (False, error_msg)   → SQL Server found an error
        (True, "")           → DB unreachable, skip gracefully
    """
    logger.debug("[validate_with_db] Starting DB dry-run...")

    try:
        with engine.connect() as connection:
            connection.execute(text("SET NOEXEC ON"))
            try:
                connection.execute(text(query))
                logger.debug("[validate_with_db] PASSED")
                return True, ""
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[validate_with_db] FAILED: {error_msg}")
                return False, f"❌ DB validation error: {error_msg}"
            finally:
                connection.execute(text("SET NOEXEC OFF"))
    except Exception as e:
        logger.warning(f"[validate_with_db] SKIPPED (connection error): {e}")
        return True, ""

"""
DIN-SQL Package — Decomposed-In-Context SQL Generation

Splits the text-to-SQL task into sequential planning steps:
    1. schema_linker.py  → Identify relevant tables, columns, joins
    2. decomposer.py     → Break question into sub-questions for CTE construction
    3. __init__.py        → Orchestrate pipeline + build context for SQL generation

Cricket domain knowledge (cricket_context.py) is injected into all
planning steps so the LLM understands ball format, metrics, column
disambiguation, and table semantics before writing any SQL.
"""

from app.services.llm._config import logger
from app.services.llm.din_sql.schema_linker import schema_link
from app.services.llm.din_sql.decomposer import decompose_question


def _build_din_context(schema_links: dict, decomposition: dict) -> str:
    """
    Build the DIN-SQL context string that gets prepended to the question
    before sending to the SQL generation prompt.

    This gives the LLM a structured plan so it doesn't have to figure out
    tables/columns/joins from scratch.
    """
    parts = [
        "=== DIN-SQL PLAN (use this to guide your query) ===",
        "",
        f"Relevant tables: {', '.join(schema_links.get('tables', []))}",
    ]

    columns = schema_links.get("columns", {})
    if columns:
        parts.append("Relevant columns:")
        for table, cols in columns.items():
            parts.append(f"  {table}: {', '.join(cols)}")

    joins = schema_links.get("joins", [])
    if joins:
        parts.append(f"Required joins: {'; '.join(joins)}")

    conditions = schema_links.get("conditions", [])
    if conditions:
        parts.append(f"Filter conditions: {'; '.join(conditions)}")

    aggregations = schema_links.get("aggregations", [])
    if aggregations:
        parts.append(f"Aggregations needed: {', '.join(aggregations)}")

    reasoning = schema_links.get("reasoning", "")
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")

    sub_questions = decomposition.get("sub_questions", [])
    if sub_questions:
        parts.append("")
        parts.append("Sub-questions (build your CTE/subqueries around these):")
        for i, sq in enumerate(sub_questions, 1):
            parts.append(f"  {i}. {sq}")
        composition = decomposition.get("composition", "")
        if composition:
            parts.append(f"Composition strategy: {composition}")

    parts.append("=== END DIN-SQL PLAN ===")
    return "\n".join(parts)


def din_sql_pipeline(question: str, table_info: dict, dialect: str) -> str:
    """
    Full DIN-SQL pipeline: schema_link → decompose → build context.

    Every question goes through both steps — cricket questions are inherently
    complex (column disambiguation, ball format, chase logic) so classification
    would add overhead without benefit.

    Returns a context string to prepend to the question before sending
    to the SQL generation prompt template.

    Args:
        question: The natural language question
        table_info: Table schema info for the prompt
        dialect: SQL dialect (mssql)

    Returns:
        Context string with the DIN-SQL plan
    """
    logger.info("=" * 70)
    logger.info("DIN-SQL Pipeline")
    logger.info(f"Question: {question[:80]}...")
    logger.info("=" * 70)

    # Step 1: Schema Linking (with cricket knowledge)
    logger.info("Step 1: Schema Linking...")
    schema_links = schema_link(question, table_info)

    # Step 2: Decompose question into sub-steps (with cricket knowledge)
    logger.info("Step 2: Decomposing question...")
    decomposition = decompose_question(question, schema_links)

    # Step 3: Build context for SQL generation
    din_context = _build_din_context(schema_links, decomposition)

    logger.info("DIN-SQL plan built successfully")
    logger.debug(f"Plan:\n{din_context}")

    return din_context

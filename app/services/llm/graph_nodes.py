"""
LangGraph Workflow Nodes

Three nodes that form the SQL generation pipeline:
    1. write_query     → LLM generates SQL (CHASE-SQL, 4 strategies)
    2. execute_query   → Run SQL on database, retry with LLM on error
    3. generate_answer → LLM converts SQL result to natural language

Also compiles the graph:
    START → write_query → execute_query → generate_answer → END
"""

import re

from sqlalchemy import text
from langgraph.graph import START, StateGraph, END

from app.services.llm._config import config, logger, engine, llm
from app.services.llm._cache import get_cached_query, cache_query
from app.services.llm.prompt import State, query_prompt_template, QueryOutput, sanity_ranges
from app.services.validators import validate_generated_query

MAX_RETRIES = 2  # Max regeneration attempts on error


# ── Helper: Template-based fix (no LLM) ───────────────────────

def _try_template_fix(failed_sql, error_msg):
    """
    Attempt to fix common SQL errors with string replacement.
    Returns fixed SQL or None if no template matches.
    """
    error_lower = error_msg.lower()
    sql = failed_sql

    # ── Invalid column name fixes ──
    if "invalid column name" in error_lower:
        if "'score'" in error_lower and "fixtures__balls" in sql.lower():
            logger.info("Template fix: score → score__runs (fixtures__balls)")
            return re.sub(r'(?<!\w)score(?!\w)', 'score__runs', sql, flags=re.IGNORECASE)

        if "'overs'" in error_lower and "fixtures__balls" in sql.lower():
            logger.info("Template fix: overs → ball (fixtures__balls)")
            return re.sub(r'(?<!\w)overs(?!\w)', 'ball', sql, flags=re.IGNORECASE)

        if "'runs'" in error_lower and "fixtures__balls" in sql.lower():
            logger.info("Template fix: runs → score__runs (fixtures__balls)")
            return re.sub(r'(?<!\w)runs(?!\w)', 'score__runs', sql, flags=re.IGNORECASE)

        if "'ball'" in error_lower and "fixtures__bowling" in sql.lower():
            logger.info("Template fix: ball → overs (fixtures__bowling)")
            return re.sub(r'(?<!\w)ball(?!\w)', 'overs', sql, flags=re.IGNORECASE)

        if "'score__runs'" in error_lower and "fixtures__batting" in sql.lower():
            logger.info("Template fix: score__runs → score (fixtures__batting)")
            return re.sub(r'(?<!\w)score__runs(?!\w)', 'score', sql, flags=re.IGNORECASE)

        if "'score__runs'" in error_lower and "fixtures__runs" in sql.lower():
            logger.info("Template fix: score__runs → score (fixtures__runs)")
            return re.sub(r'(?<!\w)score__runs(?!\w)', 'score', sql, flags=re.IGNORECASE)

        if "'score'" in error_lower and "fixtures__bowling" in sql.lower():
            logger.info("Template fix: score → runs (fixtures__bowling)")
            return re.sub(r'(?<!\w)score(?!\w)', 'runs', sql, flags=re.IGNORECASE)

    # ── LIMIT → TOP (MySQL syntax in T-SQL) ──
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

    return None  # No template matched


# ── Helper: Regenerate with LLM (fallback) ─────────────────────

def _regenerate_query(question, failed_sql, error_msg, table_info, dialect):
    """
    Try template fix first (free, instant). If that doesn't work,
    send the failed SQL + error to the LLM for regeneration.

    Returns: (new_sql, is_valid, validation_error, warnings)
    """
    # ── Step 1: Try template fix (no LLM call) ──
    template_fix = _try_template_fix(failed_sql, error_msg)
    if template_fix:
        is_valid, val_error, warnings = validate_generated_query(
            template_fix, table_info, question, engine=engine
        )
        if is_valid:
            logger.info("Template fix validated successfully — LLM call skipped")
            return template_fix, True, val_error, warnings
        else:
            logger.warning(f"Template fix failed validation: {val_error}")

    # ── Step 2: Template didn't work → ask LLM ──
    logger.info("No template fix available — calling LLM for regeneration")
    fix_prompt = f"""
The following SQL query FAILED. Fix it.

QUESTION: {question}

FAILED SQL:
{failed_sql}

ERROR:
{error_msg}

Generate a corrected SQL Server (T-SQL) query that fixes the error above.
Follow ALL schema, join, and restriction rules strictly.
"""

    prompt_input = {
        "input": fix_prompt,
        "table_info": table_info,
        "dialect": dialect,
    }

    messages = query_prompt_template.format_messages(**prompt_input)
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(messages)
    new_sql = result["query"]

    is_valid, val_error, warnings = validate_generated_query(
        new_sql, table_info, question, engine=engine
    )

    return new_sql, is_valid, val_error, warnings


# ── Node 1: Write Query (LLM Call #3) ───────────────────────────

def write_query(state: State):
    """
    Generate SQL using CHASE-SQL multi-path reasoning.

    Flow:
        1. Try each CHASE path until one passes validation
        2. If valid → return immediately
        3. If all 4 fail → send last error to LLM for regeneration (up to MAX_RETRIES)
        4. If still fails → return best-effort query
    """
    question = state["question"]
    table_info = state["table_info"]
    dialect = state.get("dialect", config.db.DIALECT)

    # ── Phase 0: Check cache ──────────────────────────────────
    cached = get_cached_query(question)
    if cached:
        logger.info(f"Cache HIT — skipping LLM generation for: {question[:60]}...")
        return {"query": cached}

    logger.info("=" * 70)
    logger.info("CHASE-SQL Multi-Path Generation")
    logger.info(f"Question: {question}")
    logger.info("=" * 70)

    last_failed_sql = None
    last_error_msg = None

    # ── Phase 1: Try CHASE-SQL paths ──────────────────────────
    for path in config.query.CHASE_REASONING_PATHS:
        chase_instruction = f"""
        Reasoning strategy: {path['name']}
        {path['instruction']}

        Generate ONE valid SQL Server (T-SQL) query.
        Follow ALL schema, join, and restriction rules strictly.
        """

        prompt_input = {
            "input": question + chase_instruction,
            "table_info": table_info,
            "dialect": dialect,
        }

        messages = query_prompt_template.format_messages(**prompt_input)
        structured_llm = llm.with_structured_output(QueryOutput)

        try:
            logger.info(f"Path '{path['name']}': Generating query...")
            result = structured_llm.invoke(messages)
            sql = result["query"]
            logger.debug(f"Generated: {sql[:70]}...")

            is_valid, error_msg, warnings = validate_generated_query(
                sql, table_info, question, engine=engine
            )

            if is_valid:
                logger.info(f"Query VALID! Strategy: {path['name']}")
                if warnings:
                    for w in warnings:
                        logger.warning(f"  {w}")
                return {"query": sql}
            else:
                logger.warning(f"Query INVALID: {error_msg}")
                last_failed_sql = sql
                last_error_msg = error_msg
                continue

        except Exception as e:
            logger.error(f"Query generation error: {str(e)}")
            continue

    # ── Phase 2: All paths failed → retry with error feedback ─
    logger.warning("All CHASE-SQL paths failed! Attempting regeneration...")

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Regeneration attempt {attempt}/{MAX_RETRIES}")
        logger.info(f"Feeding error back to LLM: {last_error_msg[:100]}...")

        try:
            new_sql, is_valid, val_error, warnings = _regenerate_query(
                question, last_failed_sql, last_error_msg, table_info, dialect
            )

            if is_valid:
                logger.info(f"Regeneration attempt {attempt} VALID!")
                if warnings:
                    for w in warnings:
                        logger.warning(f"  {w}")
                return {"query": new_sql}
            else:
                logger.warning(f"Regeneration attempt {attempt} INVALID: {val_error}")
                last_failed_sql = new_sql
                last_error_msg = val_error

        except Exception as e:
            logger.error(f"Regeneration error: {str(e)}")

    # ── Phase 3: All retries exhausted → return last attempt ──
    logger.error("All regeneration attempts failed. Returning last query.")
    return {"query": last_failed_sql or ""}


# ── Helper: Result Sanity Check ────────────────────────────────

# Realistic upper bounds for cricket stats (T20 context)
SANITY_LIMITS = {
    "score": 500,           # Max innings score (even ODI rarely exceeds 500)
    "runs": 500,
    "chase": 500,
    "total": 500,
    "wickets": 10,          # Max 10 wickets per innings
    "strike_rate": 700,     # Highest T20 SR ~400, leave margin
    "average": 500,         # Batting avg rarely above 100
    "economy": 50,          # Economy rate rarely above 20
    "percentage": 100,      # Can't exceed 100%
    "count": 50000,         # Sanity cap for match counts
}


def _check_result_sanity(result_text, question):
    """
    Check if the SQL result is realistic for cricket data.
    Returns (is_sane, reason) — reason explains why it's unrealistic.
    """
    if not result_text or result_text == "No results returned":
        return True, ""

    # Try to extract numeric value from result
    try:
        # Handle single value results
        value = float(result_text.replace(",", "").strip())
    except (ValueError, TypeError):
        # Multi-row or text result — can't sanity check
        return True, ""

    question_lower = question.lower()

    # Check against relevant limits based on question keywords
    for keyword, limit in SANITY_LIMITS.items():
        if keyword in question_lower:
            if abs(value) > limit:
                reason = (
                    f"Result {value} is unrealistic for '{keyword}' "
                    f"(expected max ~{limit}). "
                    f"Likely cause: query is aggregating across multiple matches "
                    f"without proper GROUP BY fixture_id, or missing WHERE filters."
                )
                logger.warning(f"Sanity check FAILED: {reason}")
                return False, reason

    # Generic check: if result is absurdly large
    if abs(value) > 100000:
        reason = (
            f"Result {value} is unrealistically large. "
            f"Likely aggregating across all matches without GROUP BY fixture_id."
        )
        logger.warning(f"Sanity check FAILED: {reason}")
        return False, reason

    return True, ""


# ── Node 2: Execute Query + Retry on Error ───────────────────

def execute_query(state: State):
    """
    Execute the generated SQL query on the database.
    If execution fails, sends the error + SQL back to LLM for regeneration.
    Retries up to MAX_RETRIES times.
    """
    query = state.get("query", "")
    question = state.get("question", "")
    table_info = state.get("table_info", {})
    dialect = state.get("dialect", config.db.DIALECT)

    if not query:
        return {"result": "Error: No query provided"}

    # ── Safety guard: block non-SELECT queries ──
    FORBIDDEN = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "MERGE", "CREATE", "EXEC")
    query_stripped = query.strip().upper()
    if not query_stripped.startswith(("SELECT", "WITH")):
        logger.error(f"BLOCKED non-SELECT query: {query[:80]}")
        return {"result": "Error: Only SELECT queries are allowed"}
    if any(keyword in query_stripped for keyword in FORBIDDEN):
        logger.error(f"BLOCKED destructive keyword in query: {query[:80]}")
        return {"result": "Error: Query contains forbidden operation (INSERT/UPDATE/DELETE/DROP)"}

    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0:
            logger.info("=" * 70)
            logger.info("EXECUTING QUERY")
            logger.info("=" * 70)
        else:
            logger.info(f"Execution retry {attempt}/{MAX_RETRIES}")

        logger.debug(f"Query: {query[:100]}...")

        try:
            with engine.connect() as connection:
                result = connection.execute(
                    text(query),
                    execution_options={"timeout": config.query.MAX_QUERY_TIMEOUT}
                )
                rows = result.fetchall()

                if not rows:
                    result_text = "No results returned"
                elif len(rows) == 1:
                    result_text = str(rows[0][0]) if rows[0] else "NULL"
                else:
                    result_text = "\n".join(
                        str(row) for row in rows[:config.query.MAX_RESULTS_DISPLAY]
                    )
                    if len(rows) > config.query.MAX_RESULTS_DISPLAY:
                        result_text += f"\n... ({len(rows) - config.query.MAX_RESULTS_DISPLAY} more rows)"

                logger.info("Execution successful")
                logger.info(f"Rows returned: {len(rows)}")

                # ── Sanity check result ──
                is_sane, sanity_reason = _check_result_sanity(result_text, question)
                if not is_sane:
                    # Treat as an error — regenerate
                    error_msg = f"Sanity check failed: {sanity_reason}"
                    logger.warning(error_msg)
                    if attempt < MAX_RETRIES:
                        try:
                            new_sql, is_valid, val_error, warnings = _regenerate_query(
                                question, query, error_msg, table_info, dialect
                            )
                            if is_valid:
                                logger.info("Regenerated query after sanity failure, retrying...")
                                query = new_sql
                                continue
                        except Exception as regen_e:
                            logger.error(f"Regeneration after sanity failure: {str(regen_e)}")
                    # Last attempt or regen failed — return with warning
                    return {"result": f"{result_text} (⚠️ {sanity_reason})", "query": query}

                cache_query(question, query)
                return {"result": result_text, "query": query}

        except TimeoutError:
            error_msg = f"Query execution timeout (>{config.query.MAX_QUERY_TIMEOUT} seconds)"
            logger.error(error_msg)
            return {"result": f"Error: {error_msg}"}

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"{error_type}: {error_msg}")

            # Don't retry on last attempt
            if attempt >= MAX_RETRIES:
                return {"result": f"Error: {error_type} - {error_msg}"}

            # ── Regenerate: send error + failed SQL to LLM ────
            logger.info(f"Sending execution error to LLM for regeneration...")
            try:
                new_sql, is_valid, val_error, warnings = _regenerate_query(
                    question, query, error_msg, table_info, dialect
                )

                if is_valid:
                    logger.info("Regenerated query passed validation, retrying execution...")
                    query = new_sql  # Use new query on next loop iteration
                else:
                    logger.warning(f"Regenerated query failed validation: {val_error}")
                    return {"result": f"Error: Regeneration failed validation - {val_error}"}

            except Exception as regen_e:
                logger.error(f"Regeneration failed: {str(regen_e)}")
                return {"result": f"Error: {error_type} - {error_msg}"}

    return {"result": f"Error: All {MAX_RETRIES} retry attempts exhausted"}


# ── Node 3: Generate Answer (LLM Call #4) ────────────────────

def generate_answer(state: State):
    """
    Convert SQL result into broadcast-style commentary.
    Skips LLM call if result is an error or empty.
    """
    question = state.get("question", "")
    query = state.get("query", "")
    result = state.get("result", "")

    logger.info("=" * 70)
    logger.info("GENERATING ANSWER")
    logger.info("=" * 70)

    if isinstance(result, str) and result.startswith("Error:"):
        logger.warning(f"Result is an error: {result}")
        return {"answer": f"Could not execute query: {result}"}

    if not result or result == "No results returned":
        logger.warning("No data returned from query")
        return {"answer": "No data found matching the query criteria."}

    try:
        prompt = (
            "Given the following user question, SQL query, and SQL result, "
            "answer the question in a clear, concise manner suitable for a cricket broadcast.\n\n"
            f'Question: {question}\n'
            f'SQL Query: {query}\n'
            f'SQL Result: {result}\n\n'
            "Format the answer to be interesting for broadcast commentary."
        )

        response = llm.invoke(prompt)
        answer = response.content

        logger.info(f"Answer generated: {answer[:100]}...")
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {"answer": f"Error generating answer: {str(e)}"}


# ── Compile Graph ────────────────────────────────────────────

graph_builder = StateGraph(State)

graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()

logger.info("LangGraph query workflow compiled successfully")

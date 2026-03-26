"""
LangGraph Workflow Nodes

Three nodes that form the SQL generation pipeline:
    1. write_query     → DIN-SQL planning + LLM generates SQL
    2. execute_query   → Run SQL on database, retry with LLM on error
    3. generate_answer → LLM converts SQL result to natural language (cricket-aware)

Also compiles the graph:
    START → write_query → execute_query → generate_answer → END

Supporting logic lives in dedicated modules:
    - template_fixes.py  → regex-based SQL error correction (no LLM)
    - sanity_check.py    → cricket domain validation of query results
    - query_safety.py    → pre-execution safety guards (block non-SELECT, forbidden tables)
"""

from sqlalchemy import text
from langgraph.graph import START, StateGraph, END

from app.services.llm._config import config, logger, engine, llm
from app.services.llm._cache import get_cached_query, cache_query
from app.services.llm.prompt import (
    State, query_prompt_template, QueryOutput,
    CRICKET_ANSWER_CONTEXT,
)
from app.services.llm.din_sql import din_sql_pipeline
from app.services.llm.template_fixes import try_template_fix
from app.services.llm.sanity_check import check_result_sanity
from app.services.llm.query_safety import check_forbidden_tables, check_query_safety
from app.services.validators import validate_generated_query

MAX_RETRIES = 2  # Max regeneration attempts on error


# ── Helper: Regenerate with LLM (fallback) ─────────────────────

def _regenerate_query(question, failed_sql, error_msg, table_info, dialect,
                      din_context="", warnings_from_validation=None):
    """
    Try template fix first (free, instant). If that doesn't work,
    send the failed SQL + error to the LLM for regeneration.

    Returns: (new_sql, is_valid, validation_error, warnings)
    """
    # ── Step 1: Try template fix (no LLM call) ──
    template_fix = try_template_fix(failed_sql, error_msg)
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

    # Build warning context so the LLM knows about detected issues
    warning_text = ""
    if warnings_from_validation:
        warning_text = "\nWARNINGS from validation (fix these too):\n"
        warning_text += "\n".join(f"- {w}" for w in warnings_from_validation)

    # Include DIN-SQL plan if available so the LLM has schema guidance
    plan_text = ""
    if din_context:
        plan_text = f"\n{din_context}\n"

    fix_prompt = f"""
The following SQL query FAILED. Fix it.
{plan_text}
QUESTION: {question}

FAILED SQL:
{failed_sql}

ERROR:
{error_msg}
{warning_text}
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


# ── Node 1: Write Query (DIN-SQL + LLM Generation) ──────────────

def write_query(state: State):
    """
    Generate SQL using DIN-SQL decomposed planning.

    Flow:
        1. Check cache
        2. Run DIN-SQL pipeline (schema link → decompose)
        3. Generate SQL with DIN plan as context
        4. Validate; if invalid → retry with error feedback (up to MAX_RETRIES)
    """
    question = state["question"]
    table_info = state["table_info"]
    dialect = state.get("dialect", config.db.DIALECT)

    # ── Phase 0: Check cache ──────────────────────────────────
    cached = get_cached_query(question)
    if cached:
        logger.info(f"Cache HIT — skipping LLM generation for: {question[:60]}...")
        return {"query": cached, "din_context": "", "query_warnings": []}

    logger.info("=" * 70)
    logger.info("DIN-SQL Query Generation")
    logger.info(f"Question: {question}")
    logger.info("=" * 70)

    last_failed_sql = None
    last_error_msg = None
    last_warnings = []

    # ── Phase 1: DIN-SQL Planning (schema link → decompose) ─────
    din_context = ""
    try:
        din_context = din_sql_pipeline(question, table_info, dialect)
    except Exception as e:
        logger.warning(f"DIN-SQL planning failed ({e}), generating without plan")
        din_context = ""

    # ── Phase 2: Generate SQL with DIN plan context ───────────
    generation_input = question
    if din_context:
        generation_input = f"{din_context}\n\n{question}"

    prompt_input = {
        "input": generation_input + "\n\nGenerate ONE valid SQL Server (T-SQL) query.\n"
                 "Follow ALL schema, join, and restriction rules strictly.",
        "table_info": table_info,
        "dialect": dialect,
    }

    messages = query_prompt_template.format_messages(**prompt_input)
    structured_llm = llm.with_structured_output(QueryOutput)

    try:
        logger.info("Generating SQL with DIN-SQL plan...")
        result = structured_llm.invoke(messages)
        sql = result["query"]
        logger.debug(f"Generated: {sql[:70]}...")

        # ── Pre-validation: check for forbidden tables ──
        forbidden_err = check_forbidden_tables(sql)
        if forbidden_err:
            logger.warning(f"Forbidden table detected: {forbidden_err}")
            last_failed_sql = sql
            last_error_msg = forbidden_err
            raise ValueError(forbidden_err)

        is_valid, error_msg, warnings = validate_generated_query(
            sql, table_info, question, engine=engine
        )

        if is_valid:
            logger.info("Query VALID on first attempt!")
            if warnings:
                for w in warnings:
                    logger.warning(f"  {w}")
            return {"query": sql, "din_context": din_context, "query_warnings": warnings}
        else:
            logger.warning(f"Query INVALID: {error_msg}")
            last_failed_sql = sql
            last_error_msg = error_msg
            last_warnings = warnings

    except Exception as e:
        logger.error(f"Query generation error: {str(e)}")
        last_failed_sql = ""
        last_error_msg = str(e)

    # ── Phase 3: Retry with error feedback ────────────────────
    logger.warning("Initial generation failed. Attempting regeneration...")

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Regeneration attempt {attempt}/{MAX_RETRIES}")
        logger.info(f"Feeding error back to LLM: {(last_error_msg or '')[:100]}...")

        try:
            new_sql, is_valid, val_error, warnings = _regenerate_query(
                question, last_failed_sql, last_error_msg, table_info, dialect,
                din_context=din_context, warnings_from_validation=last_warnings
            )

            if is_valid:
                logger.info(f"Regeneration attempt {attempt} VALID!")
                if warnings:
                    for w in warnings:
                        logger.warning(f"  {w}")
                return {"query": new_sql, "din_context": din_context, "query_warnings": warnings}
            else:
                logger.warning(f"Regeneration attempt {attempt} INVALID: {val_error}")
                last_failed_sql = new_sql
                last_error_msg = val_error
                last_warnings = warnings

        except Exception as e:
            logger.error(f"Regeneration error: {str(e)}")

    # ── Phase 4: All retries exhausted → return last attempt ──
    logger.error("All regeneration attempts failed. Returning last query.")
    return {"query": last_failed_sql or "", "din_context": din_context, "query_warnings": last_warnings}


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
    din_context = state.get("din_context", "")

    if not query:
        return {"result": "Error: No query provided"}

    # ── Safety guard ──
    safety_error = check_query_safety(query)
    if safety_error:
        return {"result": safety_error}

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
                col_names = list(result.keys())

                if not rows:
                    result_text = "No results returned"
                elif len(rows) == 1:
                    # Single row — show "column = value" so the answer LLM
                    # knows what the number represents
                    if len(col_names) == 1:
                        result_text = f"{col_names[0]} = {rows[0][0]}"
                    else:
                        result_text = ", ".join(
                            f"{col}={val}" for col, val in zip(col_names, rows[0])
                        )
                else:
                    # Multi-row — add a header line so the answer LLM can
                    # distinguish per-match breakdowns from totals
                    header = " | ".join(col_names)
                    data_rows = [
                        " | ".join(str(v) for v in row)
                        for row in rows[:config.query.MAX_RESULTS_DISPLAY]
                    ]
                    result_text = f"Columns: {header}\n" + "\n".join(data_rows)
                    if len(rows) > config.query.MAX_RESULTS_DISPLAY:
                        result_text += f"\n... ({len(rows) - config.query.MAX_RESULTS_DISPLAY} more rows)"

                logger.info("Execution successful")
                logger.info(f"Rows returned: {len(rows)}")

                # ── Sanity check result ──
                is_sane, sanity_reason = check_result_sanity(result_text, question)
                if not is_sane:
                    error_msg = f"Sanity check failed: {sanity_reason}"
                    logger.warning(error_msg)
                    if attempt < MAX_RETRIES:
                        try:
                            new_sql, is_valid, val_error, warnings = _regenerate_query(
                                question, query, error_msg, table_info, dialect,
                                din_context=din_context, warnings_from_validation=[]
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
                    question, query, error_msg, table_info, dialect,
                    din_context=din_context, warnings_from_validation=[]
                )

                if is_valid:
                    logger.info("Regenerated query passed validation, retrying execution...")
                    query = new_sql
                else:
                    logger.warning(f"Regenerated query failed validation: {val_error}")
                    return {"result": f"Error: Regeneration failed validation - {val_error}"}

            except Exception as regen_e:
                logger.error(f"Regeneration failed: {str(regen_e)}")
                return {"result": f"Error: {error_type} - {error_msg}"}

    return {"result": f"Error: All {MAX_RETRIES} retry attempts exhausted"}


# ── Node 3: Generate Answer (LLM Call) ────────────────────

def generate_answer(state: State):
    """
    Convert SQL result into broadcast-style commentary.
    Skips LLM call if result is an error or empty.
    """
    question = state.get("question", "")
    query = state.get("query", "")
    result = state.get("result", "")
    query_warnings = state.get("query_warnings", [])

    logger.info("=" * 70)
    logger.info("GENERATING ANSWER")
    logger.info("=" * 70)

    if isinstance(result, str) and result.startswith("Error:"):
        logger.warning(f"Result is an error: {result}")
        return {"answer": f"Could not execute query: {result}"}

    if not result or result == "No results returned":
        logger.warning("No data returned from query")
        return {"answer": "No data found matching the query criteria."}

    # Surface any validation warnings so the answer LLM knows about potential issues
    warning_text = ""
    if query_warnings:
        warning_text = "\nQUERY WARNINGS (the SQL may have issues — factor these into your answer):\n"
        warning_text += "\n".join(f"- {w}" for w in query_warnings)
        warning_text += "\n"

    try:
        prompt = (
            "You are a cricket analytics expert. Provide concise, data-driven insights, not play-by-play commentary.\n\n"
            f"{CRICKET_ANSWER_CONTEXT}\n\n"
            "CRITICAL PLAUSIBILITY RULES — check BEFORE answering:\n"
            "- A single bowler CANNOT take more than 10 wickets in a match (typically max 4-5 in T20).\n"
            "- A team CANNOT lose more than 10 wickets per innings (max 20 per match across both innings).\n"
            "- Batting average below 3.0 is almost certainly a data/query error.\n"
            "- Economy rate above 36 or below 0 is impossible.\n"
            "- Strike rate above 700 is impossible.\n"
            "- If a value is NULL, missing, or clearly anomalous, say so explicitly — do NOT fabricate analysis around missing data.\n"
            "- If the SQL result looks like multiple rows (e.g., per-match breakdown) but the question asks for a single total, note the mismatch.\n"
            "- If any value violates cricket logic, flag it: 'This result appears unreliable due to [reason]. The query may need correction.'\n\n"
            f"Question: {question}\n"
            f"SQL Query: {query}\n"
            f"SQL Result: {result}\n"
            f"{warning_text}\n"
            "Provide a calm, fact-based answer that focuses on key performance indicators (e.g., strike rate, economy, run rate, wickets efficiency, chase context). "
            "Avoid storytelling or flamboyant commentary language. "
            "Give a clear conclusion and one or two actionable observations."
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

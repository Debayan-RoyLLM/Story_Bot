"""
DIN-SQL Step 1: Schema Linking

LLM identifies which tables, columns, join conditions, and WHERE filters
are needed for a given question. Cricket domain knowledge is injected
so the LLM resolves column disambiguation correctly.
"""

import json
from app.services.llm._config import logger, llm
from app.services.llm.prompt import metadata
from app.services.llm.din_sql.cricket_context import CRICKET_PLANNING_CONTEXT


_SCHEMA_LINK_TEMPLATE = """You are a cricket database schema expert.

{cricket_context}

CRITICAL RULES FOR TABLE SELECTION:
1. For BOWLER stats (economy, dot balls, wickets taken per ball):
   Use fixtures__balls ALONE filtered by bowler__fullname.
   Economy = CAST(SUM(score__runs) AS FLOAT) * 6.0 / NULLIF(COUNT(*), 0) — NO JOIN needed.
   Dot balls = COUNT(*) WHERE score__runs = 0 AND bowler__fullname = 'X' — NO JOIN needed.
2. For BATSMAN stats (strike rate, runs, average):
   Use fixtures__balls ALONE filtered by batsman__fullname. NO JOIN needed.
3. Player names (batsman__fullname, bowler__fullname) exist ONLY in fixtures__balls.
4. Team names (team__name) exist in fixtures__balls — use that for team filtering.
5. Only JOIN fixtures__bowling/fixtures__batting when you need columns SPECIFIC to those tables
   (e.g., fixtures__bowling.wickets for cumulative bowler wickets, fixtures__batting.score for
   batsman innings total). Most questions can be answered from fixtures__balls alone.
6. NEVER join fixtures__runs with fixtures__balls unless absolutely necessary — different granularity
   (2 rows vs 120+ rows per match) causes row multiplication that inflates all aggregates.
7. Use fixtures__runs directly (without joining) when you ONLY need innings totals (score, wickets, overs).

Prefer FEWER tables — use the minimum set needed.

Given the metadata and question below, identify EXACTLY which tables, columns,
join conditions, and WHERE filters are needed. Think step by step.

DATABASE METADATA:
{metadata_summary}

QUESTION: {question}

Respond in STRICT JSON (no markdown, no explanation):
{json_template}
"""

_JSON_TEMPLATE = """{
  "tables": ["history2.table_name"],
  "columns": {"history2.table_name": ["col1", "col2"]},
  "joins": ["history2.A.col = history2.B.col"],
  "conditions": ["col = value"],
  "aggregations": ["SUM(col)"],
  "reasoning": "one-line explanation of why these tables/columns"
}"""


def schema_link(question: str, table_info: dict) -> dict:
    """
    LLM identifies which tables, columns, and conditions are relevant.
    Cricket knowledge is injected so the LLM knows column disambiguation.

    Returns dict with tables, columns, joins, conditions, aggregations.
    """
    # Build a compact metadata summary (table → columns list)
    meta_lines = []
    for table_name, table_data in metadata.items():
        cols = table_data.get("columns", table_data) if isinstance(table_data, dict) else {}
        if isinstance(cols, dict):
            col_names = [c for c in cols.keys() if not c.startswith("❌")]
            meta_lines.append(f"{table_name}: {', '.join(col_names)}")
    metadata_summary = "\n".join(meta_lines)

    prompt = _SCHEMA_LINK_TEMPLATE.format(
        cricket_context=CRICKET_PLANNING_CONTEXT,
        metadata_summary=metadata_summary,
        question=question,
        json_template=_JSON_TEMPLATE,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        schema_links = json.loads(content)
        logger.info(f"Schema link: tables={schema_links.get('tables', [])}")
        return schema_links

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Schema linking failed ({e}), using fallback")
        return {
            "tables": ["history2.fixtures__balls"],
            "columns": {},
            "joins": [],
            "conditions": [],
            "aggregations": [],
            "reasoning": "fallback — schema linking failed",
        }

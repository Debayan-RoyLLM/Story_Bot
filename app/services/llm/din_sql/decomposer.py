"""
DIN-SQL Step 3: Question Decomposition

For HARD queries only — LLM breaks a complex question into 2-4 simpler
sub-questions that can each be answered with a single CTE or subquery.
Cricket knowledge ensures decomposition respects domain logic.
"""

import json
from app.services.llm._config import logger, llm
from app.services.llm.din_sql.cricket_context import CRICKET_PLANNING_CONTEXT


DECOMPOSE_PROMPT = """You are a cricket statistics analyst who breaks down complex questions.

{cricket_context}

Break this complex question into 2-4 simpler sub-questions that can each
be answered with a single SQL query or CTE. The sub-questions should
build on each other logically.

QUESTION: {question}
SCHEMA LINKS: {schema_links}

Respond in STRICT JSON (no markdown):
{{
  "sub_questions": [
    "sub-question 1 (simplest, foundational data)",
    "sub-question 2 (builds on sub-question 1)",
    ...
  ],
  "composition": "How to combine sub-answers into the final answer"
}}
"""


def decompose_question(question: str, schema_links: dict) -> dict:
    """
    Break a HARD question into sub-questions.
    Cricket knowledge ensures decomposition respects domain logic
    (e.g., chase = inning 2, wickets = lost not in hand).
    """
    prompt = DECOMPOSE_PROMPT.format(
        cricket_context=CRICKET_PLANNING_CONTEXT,
        schema_links=json.dumps(schema_links, indent=2),
        question=question,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        result = json.loads(content)
        logger.info(f"Decomposed into {len(result.get('sub_questions', []))} sub-questions")
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Decomposition failed ({e}), treating as single question")
        return {
            "sub_questions": [question],
            "composition": "Answer directly",
        }

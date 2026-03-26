"""
Question Generation

Uses OpenAI to generate statistical questions from match narrative.
Single LLM call — generates clear, SQL-queryable questions directly.
Cricket domain knowledge is injected so questions are answerable from the DB.
"""

import re
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm._config import llm
from app.services.llm.prompt import CRICKET_QUESTION_CONTEXT


def generate_questions(narrative):
    """
    Single LLM call: Generate 8 clear, SQL-queryable questions from narrative.

    Input:  "Kohli and Rahul batting, 45 runs needed from 30 balls..."
    Output: List of question strings
    """
    messages = [
        SystemMessage(content=(
            "You are a live cricket match statistic analyser working in a broadcasting company.\n\n"
            f"{CRICKET_QUESTION_CONTEXT}"
        )),
        HumanMessage(content=f"""{narrative}
    I have a database for the ball by ball information of every cricket match of every tournament in the past 20 years, so any numbers for any stat can be extracted.

    Give 8 appropriate statistical questions to fetch from the data which will interest the audience based on the context of the match.

    Rules:
    - Questions must be clear, unambiguous, and directly queryable from a SQL database
    - Stats should be relevant to what is happening in the game
    - Involve the players in the game preferably but keep it general as well
    - Give ONLY the questions, nothing else — no headings, no stats, no SQL, no numbering prefix
    - Frame questions using metrics that can be computed: strike rate, economy, run rate, batting average, wickets, boundaries (4s and 6s), dot balls
    - For player stats, use full player names (e.g., 'Virat Kohli')
    - For team stats, reference team names (e.g., 'Kolkata Knight Riders')
    - Questions about "this match" should reference specific players/teams in the narrative
    """),
    ]
    response = llm.invoke(messages)
    return response.content.split('\n')


def stat_questions(sentences):
    """
    Clean up question list:
    - Remove empty lines
    - Strip number prefixes like "1. ", "2) ", "- "
    - Remove formatting artifacts
    """
    cleaned = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Remove "1. ", "2) ", "- ", "* " prefixes
        s = re.sub(r'^[\d]+[\.\)\-\:\s]+', '', s).strip()
        s = re.sub(r'^[\-\*\•]\s+', '', s).strip()
        if s:
            cleaned.append(s)
    return cleaned

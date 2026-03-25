"""
Question Generation

Uses OpenAI to generate statistical questions from match narrative.
Single LLM call — generates clear, SQL-queryable questions directly.
"""

from app.services.llm._config import client, config


def generate_questions(narrative):
    """
    Single LLM call: Generate 8 clear, SQL-queryable questions from narrative.

    Input:  "Kohli and Rahul batting, 45 runs needed from 30 balls..."
    Output: List of question strings
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a live cricket match statistic analyser working in a broadcasting company."},
            {"role": "user", "content": f"""{narrative}
    I have a database for the ball by ball information of every cricket match of every tournament in the past 20 years, so any numbers for any stat can be extracted.

    Give 8 appropriate statistical questions to fetch from the data which will interest the audience based on the context of the match.

    Rules:
    - Questions must be clear, unambiguous, and directly queryable from a SQL database
    - Stats should be relevant to what is happening in the game
    - Involve the players in the game preferably but keep it general as well
    - Give ONLY the questions, nothing else — no headings, no stats, no SQL, no numbering prefix
    """}
        ],
        model=config.query.DEFAULT_LLM_MODEL
    )
    return chat_completion.choices[0].message.content.split('\n')


def stat_questions(sentences):
    """
    Clean up question list — remove empty lines and formatting artifacts.
    """
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned

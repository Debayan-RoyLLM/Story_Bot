"""
Metadata, State, and Prompt Template

Loads configuration files that the LangGraph nodes need:
  - metadata.json    → table/column schema for validation + LLM context
  - prompt_template.txt → system prompt for SQL generation
  - State TypedDict   → data shape flowing through the graph
  - QueryOutput       → structured output format for LLM
"""

import json
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm._config import config, logger


# ── Metadata ─────────────────────────────────────────────────────

def load_metadata():
    """
    Load database table metadata from JSON configuration file.

    Returns dict like:
        {"history2.fixtures__balls": {"fixture_id": "...", "❌ NO 'score'": "..."}}
    """
    try:
        with open(config.files.METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            logger.info(f"Metadata loaded from {config.files.METADATA_FILE}")
            return metadata
    except FileNotFoundError:
        logger.error(f"Metadata file not found at {config.files.METADATA_FILE}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        return {}


metadata = load_metadata()


# ── State TypedDict ──────────────────────────────────────────────

class State(TypedDict):
    """Data shape that flows through the LangGraph pipeline."""
    question: str       # "What is Kohli's strike rate?"
    query: str          # "SELECT ... FROM ..."
    result: str         # "45.6"
    answer: str         # "Kohli's strike rate is 45.6"
    table_info: dict    # metadata.json contents
    dialect: str        # "mssql"
    metadata: dict


# ── Prompt Template ──────────────────────────────────────────────

def load_prompt_template():
    """
    Load SQL generation prompt template from configuration file.
    Falls back to a minimal prompt if file not found.
    """
    try:
        with open(config.files.PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
            logger.info(f"Prompt template loaded from {config.files.PROMPT_TEMPLATE_FILE}")
            return template
    except FileNotFoundError:
        logger.error(f"Prompt template file not found at {config.files.PROMPT_TEMPLATE_FILE}")
        return "You are a SQL query generator for cricket statistics."
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        return "You are a SQL query generator for cricket statistics."


custom_message = load_prompt_template()

query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", custom_message),
    ("human", "{input}")
])

logger.info(f"Template variables: {query_prompt_template.input_variables}")


# ── QueryOutput ──────────────────────────────────────────────────

class QueryOutput(TypedDict):
    """Structured output format for LLM SQL generation."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

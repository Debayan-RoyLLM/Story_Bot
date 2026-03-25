"""
Shared Singletons

All shared state lives here: DB engine, OpenAI client, LLM instance,
config, logger. Other modules in the llm package import from here
instead of creating their own connections.
"""

import os
import logging
from urllib.parse import quote
from openai import OpenAI
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI

from app.services.config.settings import Config

# ── Config ───────────────────────────────────────────────────────
config = Config()

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("app.services.llm")

# ── API Keys ─────────────────────────────────────────────────────
api_key_openai = config.api.get_openai_api_key()
password = config.api.get_sql_password()

# ── OpenAI Client (for direct API calls) ─────────────────────────
client = OpenAI(api_key=api_key_openai)

# ── Database Engine (shared by query execution + validator dry-run)
engine = create_engine(config.db.get_connection_string(password))
logger.info("Database connection established successfully")

# ── LangChain LLM (for structured output in graph nodes) ─────────
llm = ChatOpenAI(model=config.query.DEFAULT_LLM_MODEL, api_key=api_key_openai)

# ── LangSmith Client ─────────────────────────────────────────────
if not os.environ.get("LANGSMITH_API_KEY"):
    logger.warning("LANGSMITH_API_KEY not set in environment")

from langsmith import Client
ls_client = Client()

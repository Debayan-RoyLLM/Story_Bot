"""
Shared Singletons

All shared state lives here: DB engine, LLM instance, config, logger.
Other modules in the llm package import from here instead of creating
their own connections.
"""

import os
import logging
from langchain_openai import ChatOpenAI

from app.services.config.settings import Config
from app.db.database import engine  # single shared engine

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
logger.info("Database connection established successfully")

# ── LangChain LLM (for structured output in graph nodes) ─────────
llm = ChatOpenAI(model=config.query.DEFAULT_LLM_MODEL, api_key=api_key_openai)

# ── LangSmith Client ─────────────────────────────────────────────
if not os.environ.get("LANGSMITH_API_KEY"):
    logger.warning("LANGSMITH_API_KEY not set in environment")

from langsmith import Client
ls_client = Client()

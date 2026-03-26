"""
LLM Package — Cricket Statistics Query System

Generates SQL queries from natural language questions using OpenAI GPT-4o-mini
with DIN-SQL decomposed planning, DB dry-run validation, and metadata-driven checks.

Public API:
    run_statements(narrative, game_state_dict, graph, metadata)
    graph       — compiled LangGraph workflow
    metadata    — table/column schema from metadata.json

Package structure:
    _config.py       — Shared singletons (engine, llm, config)
    question_gen.py  — OpenAI question generation (cricket-aware)
    ml_filter.py     — BERT + PCA + ML classification (no LLM)
    prompt.py        — Metadata, State, prompt template, QueryOutput
    din_sql.py       — DIN-SQL pipeline: schema link, classify, decompose
    graph_nodes.py   — LangGraph nodes: write_query, execute_query, generate_answer
    orchestrator.py  — Main entry point: run_statements
"""

from app.services.llm.orchestrator import run_statements
from app.services.llm.graph_nodes import graph
from app.services.llm.prompt import flat_metadata as metadata

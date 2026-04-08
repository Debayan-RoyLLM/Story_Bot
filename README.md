# AI Storyboard — Cricket Statistics Engine

An AI-powered system that generates live cricket match narratives and answers statistical questions in real time using LLM-driven SQL generation. Designed to power live T20 cricket broadcasting by automatically producing relevant match statistics and natural language commentary.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Detail](#pipeline-detail)
  - [1. Narrative Generation](#1-narrative-generation)
  - [2. Question Generation](#2-question-generation)
  - [3. ML-Based Question Filtering](#3-ml-based-question-filtering)
  - [4. DIN-SQL Planning](#4-din-sql-planning)
  - [5. SQL Generation & Validation](#5-sql-generation--validation)
  - [6. Result Sanity Checking](#6-result-sanity-checking)
  - [7. Natural Language Answer](#7-natural-language-answer)
- [Database Schema](#database-schema)
- [Cricket Domain Knowledge](#cricket-domain-knowledge)
- [Configuration](#configuration)
- [Setup & Installation](#setup--installation)
- [API Usage](#api-usage)
- [Output Format](#output-format)

---

## Overview

AI Storyboard simulates a T20 cricket match ball-by-ball, and every 5 balls it:

1. Builds a **live match narrative** (current batsmen, scores, run rates, wickets)
2. Generates **8 statistical questions** relevant to the match context
3. Filters questions through an **ML ensemble classifier** (BERT + PCA + Logistic Regression/Random Forest/XGBoost)
4. Plans SQL generation using **DIN-SQL** (schema linking, classification, decomposition)
5. Generates **T-SQL queries** via GPT-4o-mini with multi-layer validation
6. Executes queries against a **SQL Server** cricket database
7. Converts results into **natural language answers**

All outputs are logged to a CSV file for downstream use.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  FastAPI     │────▶│  Orchestrator    │────▶│  Question Gen    │
│  /fixtures   │     │  (every 5 balls) │     │  (OpenAI GPT-4o) │
└─────────────┘     └──────────────────┘     └──────────────────┘
                            │                         │
                            ▼                         ▼
                    ┌──────────────────┐     ┌──────────────────┐
                    │  CSV Writer      │     │  ML Filter       │
                    │  (llm_outputs)   │     │  BERT+PCA+Ensemble│
                    └──────────────────┘     └──────────────────┘
                            ▲                         │
                            │                         ▼
                    ┌──────────────────┐     ┌──────────────────┐
                    │  LangGraph       │◀────│  DIN-SQL Planner │
                    │  Workflow        │     │  (Schema Link +  │
                    │                  │     │   Decompose)     │
                    │  write_query ──▶ │     └──────────────────┘
                    │  execute_query ─▶│
                    │  generate_answer │
                    └──────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  SQL Server      │
                    │  (sportmonk DB)  │
                    └──────────────────┘
```

---

## Tech Stack

| Category        | Technology                                    |
|-----------------|-----------------------------------------------|
| Web Framework   | FastAPI + Uvicorn                             |
| Database        | Microsoft SQL Server (ODBC Driver 18)         |
| ORM             | SQLAlchemy                                    |
| LLM             | OpenAI GPT-4o-mini                            |
| Orchestration   | LangChain, LangGraph, LangSmith              |
| ML / Embeddings | Hugging Face Transformers (BERT), PyTorch     |
| ML Classifiers  | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Serialization   | joblib                                        |
| Config          | python-dotenv                                 |

---

## Project Structure

```
ai_storyboard/
├── app/
│   ├── main.py                              # FastAPI application entry point
│   ├── db/
│   │   └── database.py                      # SQLAlchemy engine & session factory
│   ├── functions/
│   │   └── fixtures_functions.py            # Helper functions for fixture data retrieval
│   ├── Queries/
│   │   └── fixture_queries.py               # Pre-written SQL queries (SQLAlchemy text)
│   └── services/
│       ├── fixtures_services.py             # API router, game state computation, CSV output
│       ├── config/
│       │   ├── settings.py                  # Configuration classes (DB, ML, API, Query)
│       │   ├── metadata.json                # Table/column schema with descriptions & samples
│       │   ├── prompt_template.txt          # System prompt for SQL generation
│       │   ├── cricket_knowledge.txt        # Domain rules (ball format, metrics, formulas)
│       │   └── enrich_metadata.py           # Script to regenerate metadata.json
│       ├── llm/
│       │   ├── orchestrator.py              # Main entry: question gen → filter → graph
│       │   ├── graph_nodes.py               # LangGraph nodes: write, execute, answer
│       │   ├── prompt.py                    # Metadata loading & prompt templates
│       │   ├── question_gen.py              # LLM generates questions from narrative
│       │   ├── ml_filter.py                 # BERT + PCA + ensemble classification
│       │   ├── template_fixes.py            # Regex-based SQL error corrections
│       │   ├── sanity_check.py              # Domain validation of query results
│       │   ├── query_safety.py              # Block non-SELECT queries
│       │   ├── _config.py                   # Shared singletons (LLM, logger; reuses DB engine from database.py)
│       │   ├── _cache.py                    # Query result cache by question text
│       │   └── din_sql/
│       │       ├── schema_linker.py         # LLM-based table/column selection
│       │       ├── decomposer.py            # Break complex questions into sub-questions
│       │       └── cricket_context.py       # Shared cricket domain context
│       └── validators/
│           ├── __init__.py                  # Validator orchestrator
│           ├── db_validation.py             # Dry-run query compilation (SET NOEXEC ON)
│           ├── metadata_checks.py           # Column-table mismatch detection
│           ├── metadata_helpers.py          # Parse metadata.json into lookup tables
│           └── sql_checks.py               # Division safety, GROUP BY, WHERE checks
├── data/
│   └── llm_outputs.csv                      # Pipeline output (questions, queries, answers)
└── req.txt                                  # Python dependencies
```

---

## Pipeline Detail

### 1. Narrative Generation

Every 5 balls, the system computes the current match state from the database and builds a narrative string:

- **Current batsmen**: names, individual runs, balls faced, strike rates
- **Current bowler**: name, wickets taken
- **Match situation**: runs required, balls remaining, required run rate, current run rate
- **Recent history**: outcomes of last 2 balls (dot, boundary, wicket, etc.)
- **Wickets in hand**: 10 minus wickets lost

Example narrative:
> *"Rahul and Kohli are batting with 45 runs required off 30 balls. Kohli is on 38 off 22 (SR 172.7), Rahul on 15 off 12 (SR 125.0). Bumrah has 2 wickets. RRR: 9.0, CRR: 8.5. Wickets in hand: 8. Last 2 balls: dot, 4."*

### 2. Question Generation

A single OpenAI call transforms the narrative into **8 statistical questions** relevant to the current match context. Cricket domain knowledge is injected to ensure questions are answerable given the database schema.

### 3. ML-Based Question Filtering

Each generated question passes through a multi-stage classifier:

1. **BERT Embedding** — encode question text into a 768-dimensional vector (`bert-base-uncased`)
2. **PCA Reduction** — reduce to 59 dimensions using a pre-trained PCA model
3. **Game State Features** — concatenate 11 match features (required runs, balls remaining, wickets, run rates, etc.) → 70-dimensional input
4. **Ensemble Classification** — three classifiers vote:
   - Logistic Regression
   - Random Forest (primary gate — **probability > 0.6 required**)
   - XGBoost
5. Questions failing the threshold are discarded (typically ~3–4 of 8)

**Graceful degradation**: If ML model files are not present, the filter is skipped entirely and all questions pass through unfiltered. This is logged as an ERROR at startup so the disabled state is clearly visible. Questions returned without filtering have `probabilities: None` so downstream consumers can distinguish filtered from unfiltered results.

### 4. DIN-SQL Planning

Valid questions go through the **DIN-SQL** (Decomposed In-context SQL) pipeline:

| Step | Component | Purpose |
|------|-----------|---------|
| Schema Linking | `schema_linker.py` | LLM identifies required tables, columns, joins, and WHERE conditions |
| Classification | Built-in | LLM classifies question difficulty as EASY or HARD |
| Decomposition | `decomposer.py` | (HARD only) Break into 2–4 sub-questions with composition strategy |

### 5. SQL Generation & Validation

**Generation**: GPT-4o-mini writes T-SQL queries using the DIN-SQL plan, table metadata, and cricket domain knowledge.

**Multi-layer validation** catches errors before execution:

| Layer | Component | What It Catches |
|-------|-----------|-----------------|
| Template Fixes | `template_fixes.py` | Common column name errors (`score` → `score__runs`), syntax (`LIMIT` → `TOP`) |
| DB Dry-Run | `db_validation.py` | Syntax errors, invalid columns, type mismatches (`SET NOEXEC ON`) |
| Metadata Checks | `metadata_checks.py` | Column-table mismatches (column used with wrong table) |
| SQL Checks | `sql_checks.py` | Division by zero, missing WHERE clauses, GROUP BY consistency |
| Safety Guards | `query_safety.py` | Block dangerous operations (see below) |

**Safety guards** block queries before execution by:
- Stripping SQL comments (`/* */`, `--`) and string literals to prevent bypass
- Requiring queries start with `SELECT` or `WITH`
- Blocking destructive keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `MERGE`, `CREATE`, `EXEC/EXECUTE`, `GRANT`, `REVOKE`, `DENY`, `BACKUP`, `RESTORE`, `SHUTDOWN`, `WAITFOR`, `OPENROWSET`, `OPENQUERY`
- Blocking dangerous system objects: `xp_cmdshell`, `sp_OACreate`, `sp_configure`, `INFORMATION_SCHEMA`
- Smart identifier detection to avoid false positives (e.g., `updated_at` containing `UPDATE`)

**Error recovery**: failed validation → try template fix → if still failing, regenerate with LLM (up to 2 retries, error message included in prompt).

**Execution**: validated queries run against SQL Server with a 30-second timeout, results limited to 10 rows.

### 6. Result Sanity Checking

Query results are validated against cricket domain bounds:

- Score < 500, wickets ≤ 10, strike rate < 700, economy < 50
- Per-player limits (e.g., bowler max 10 wickets per match)
- Impossible results are rejected

### 7. Natural Language Answer

The LLM converts the SQL result into a cricket-aware natural language answer, incorporating the original question, query, result set, and domain rules.

---

## Database Schema

The system queries a SQL Server database (`sportmonk`) with tables in the `history2` schema:

| Table | Description | Granularity |
|-------|-------------|-------------|
| `fixtures` | Match metadata (league, season, teams, status, result) | 1 row per match |
| `fixtures__balls` | Ball-by-ball data (batsman, bowler, runs, wickets) | 1 row per ball (120 per T20) |
| `fixtures__batting` | Cumulative batsman stats | Per batsman per ball |
| `fixtures__bowling` | Cumulative bowler stats | Per bowler per over |
| `fixtures__runs` | Innings totals | 2 rows per match |
| `fixtures__teams` | Team lookup | Per team per match |
| `fixtures__players` | Player lookup | Per player per match |
| `fixtures__leagues` | League lookup | Per league |

Key columns in `fixtures__balls`:
- `batsman__fullname`, `bowler__fullname`, `team__name`
- `score__runs` (runs scored on that ball)
- `ball` (decimal format: `5.3` = over 6, ball 3)
- `score__is_wicket` (boolean)
- `scoreboard` (`S1` = first innings, `S2` = second innings)

---

## Cricket Domain Knowledge

The system embeds cricket-specific rules to ensure accurate SQL generation:

**Ball format**: Stored as a decimal where `FLOOR(ball) * 6 + ROUND((ball % 1) * 10)` gives the absolute ball number. For example, `5.3` = ball 33.

**Key formulas**:
- **Strike Rate** = `SUM(runs) / COUNT(*) * 100`
- **Economy Rate** = `SUM(runs) * 6 / COUNT(*)`
- **Batting Average** = `SUM(runs) / wickets_lost`
- **Run Rate** = `SUM(runs) * 6 / COUNT(*)`

**Phase boundaries**:
- Powerplay (overs 1–6): `ball <= 5.6`
- Middle overs (7–15): `ball BETWEEN 6.1 AND 14.6`
- Death overs (16–20): `ball >= 15.1`

---

## Configuration

Configuration is centralized in `app/services/config/settings.py`:

| Config Class | Key Settings |
|-------------|--------------|
| `DatabaseConfig` | SQL Server host, port, database name, ODBC driver (all env-var overridable) |
| `MLModelConfig` | BERT model name, PCA dimensions (59), model file paths |
| `QueryConfig` | Query timeout (30s), max display rows (10), LLM model name |
| `StatementConfig` | Required valid statements (8), RF threshold (0.6), game state feature count (11) |
| `FileConfig` | Paths to metadata.json, prompt template, cricket knowledge |
| `APIConfig` | OpenAI API key, SQL password, LangSmith key |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Microsoft SQL Server with the `sportmonk` database populated
- ODBC Driver 18 for SQL Server
- Pre-trained ML models:
  - `/pca_model_59.pkl` — PCA reduction model
  - `/models/logistic_model.joblib` — Logistic Regression
  - `/models/forest_model.joblib` — Random Forest
  - `/models/xgb_model.joblib` — XGBoost

### Install Dependencies

```bash
pip install -r req.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
SQL_SERVER_PASSWORD=your-sql-server-password
LANGSMITH_API_KEY=your-langsmith-key          # optional, for tracing

# Optional database overrides (defaults shown):
SQL_SERVER_USERNAME=SA
SQL_SERVER_HOST=localhost
SQL_SERVER_PORT=1433
SQL_SERVER_DATABASE=sportmonk
SQL_SERVER_DRIVER=ODBC Driver 18 for SQL Server
```

### Generate Knowledge Base (Metadata)

Before running the server for the first time (or after any database schema changes), generate the metadata that powers SQL generation:

```bash
python -m app.services.config.enrich_metadata
```

This script:
1. Auto-discovers all tables in the `history2` schema from the database
2. Reads column names, types, and sample values
3. Calls GPT-4o-mini to generate table summaries and column descriptions
4. Auto-derives forbidden column entries (e.g., `score` vs `score__runs` disambiguation)
5. Outputs `app/services/config/metadata.json`

Re-run whenever tables/columns are added or the schema changes.

### Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## API Usage

### Get Fixture Stats

```
GET /fixtures/latest
```

The endpoint supports two lookup modes — pick whichever is more convenient:

**Name-based lookup** (preferred — human-friendly):

```
GET /fixtures/latest
    ?country_name={name}
    &league_code={code}
    &season_code={code}
    &localteam_code={code}
    &visitorteam_code={code}
    &round={round}            # optional
```

All five name/code parameters are required together. Resolves to the matching fixture via `country.name`, `league.code`, `season.code`, and the two team codes; `round` further disambiguates when multiple fixtures match.

**ID-based lookup** (legacy):

```
GET /fixtures/latest?country_id={id}&league_id={id}
```

Retrieves the latest fixture for the given country and league IDs.

**Direct fixture override**: passing `fixture_id={id}` skips lookup entirely and runs that specific fixture.

Once a fixture is resolved, the endpoint:

1. Iterates through all 120 balls of the T20 match
2. Every 5 balls, runs the full AI pipeline
3. Appends results to `data/llm_outputs_01.csv`
4. Returns the compiled statistics and any per-ball errors

**Error handling**:
- `400` if neither a complete name-based set nor `(country_id, league_id)` is provided
- `404` if no fixture matches the provided lookup
- Individual ball failures are caught and logged without aborting the match — errors are collected and returned in the response `errors` array.

---

## Output Format

Results are written to `data/llm_outputs.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `fixture_id` | Match identifier |
| `ball_no` | Ball number (1–120) |
| `over` | Over in decimal format (0.1–19.6) |
| `narrative` | Match context narrative |
| `sentence` | Generated statistical question |
| `query` | T-SQL query executed |
| `answer` | Natural language answer |

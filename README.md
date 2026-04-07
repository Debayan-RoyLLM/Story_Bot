# Story Bot

A FastAPI-based cricket match storytelling application that generates ball-by-ball narrative summaries and statistical question prompts using a combination of SQL match data and OpenAI-powered LLM reasoning.

## Overview

This project connects to a cricket match database, computes current game state metrics for a live fixture chase, and generates narrative descriptions plus analytic statements. The generated statements are validated and stored in `data/llm_outputs.csv`.

## Key Features

- `GET /fixtures/latest` endpoint for the latest fixture by `country_id` and `league_id`
- Ball-by-ball narrative generation for live chase scenarios
- Real-time game state computation: required runs, balls remaining, run rate, wickets in hand
- LLM-powered statistical question generation using OpenAI GPT models
- Persistence of outputs to `data/llm_outputs.csv`

## Project Structure

- `app/main.py` - FastAPI application entrypoint
- `app/services/fixtures_services.py` - Fixture simulation and narrative generation
- `app/services/LLM.py` - LLM orchestration, OpenAI integration, ML classification helpers
- `app/services/config/settings.py` - application configuration and environment settings
- `app/services/validators.py` - generated SQL validation utilities
- `app/functions/fixtures_functions.py` - helper methods for loading fixture data from the database
- `app/db/database.py` - SQLAlchemy database session provider
- `data/` - output files and data assets
- `models/` - pre-trained ML model files used for question classification

## Prerequisites

- Python 3.10+ (recommended)
- SQL Server instance accessible from the app
- OpenAI API key
- Required Python dependencies

## Dependencies

The repository includes a minimal dependency list in `req.txt`.

At minimum, install:

```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary openai pandas transformers torch joblib langchain_community langchain_openai langgraph
```

> The repository may depend on additional packages that are imported in code but not listed in `req.txt`.

## Tech Stack

- `FastAPI` for the HTTP API and request routing
- `Uvicorn` as the ASGI server
- `SQLAlchemy` for database session management and SQL execution
- `pyodbc` / `psycopg2-binary` for DB connectivity (SQL Server via ODBC in config)
- `OpenAI` Python client for GPT-based question and query generation
- `langchain_openai` and `langchain_community` for structured prompt workflows and SQL database utilities
- `langgraph` for stateful query generation and execution workflows
- `transformers`, `torch`, `joblib`, and `pandas` for local ML-based statement classification

## Validation and SQL Generation

### SQL Validator

The project uses a custom validation module in `app/services/validators.py`.

The validator performs multiple structured checks on generated SQL queries, including:

- subquery safety for scalar comparisons
- column name and table usage validation against metadata
- `ORDER BY` / `DISTINCT` rules
- aggregate rules and data type checks
- required base table usage (`history2.fixtures__balls` or `history2.fixtures__runs`)
- `JOIN` syntax and `WHERE` clause expectations
- `GROUP BY` completeness
- `CTE` and `FETCH` syntax rules for T-SQL
- table-specific column restrictions for `fixtures__balls`, `fixtures__batting`, `fixtures__bowling`, and `fixtures__runs`
- unqualified column warnings in joined queries

If a generated query fails validation, it is rejected and another candidate is attempted.

### SQL Generation Method

SQL generation is implemented in `app/services/LLM.py` using a CHASE-SQL multi-path reasoning approach:

- Natural language statistical questions are generated from match narrative and game state.
- The LLM is prompted with structured schema metadata and specific chase-focused reasoning instructions.
- Multiple reasoning paths are tried using the `config.query.CHASE_REASONING_PATHS` set in `app/services/config/settings.py`.
- Each generated SQL candidate is validated by `validate_generated_query` before selection.
- Candidates are scored using a chase-preference heuristic that rewards:
  - use of `history2.fixtures__balls`
  - chase-specific conditions like `scoreboard = 'S2'` and `inning = 2`
  - schema discipline and appropriate joins
- The best valid SQL candidate is selected and then executed via the LangGraph workflow.

### LangGraph Workflow

The SQL workflow uses `langgraph` to chain these stages:

1. `write_query` - generate and validate SQL
2. `execute_query` - execute the generated query against the database
3. `generate_answer` - convert query result into a broadcast-style answer

This workflow is compiled and executed through `app/services/LLM.py`.

## Configuration

### Environment Variables

Set the following environment variables for local execution:

- `OPENAI_API_KEY` - OpenAI API key used by the LLM module
- `SQL_SERVER_PASSWORD` - SQL Server password for the configured `SA` user
- `LANGSMITH_API_KEY` (optional) - LangSmith API key if used by future extensions

### Database Settings

The database configuration is defined in `app/services/config/settings.py`.

Default values:

- `USERNAME`: `SA`
- `SERVER`: `localhost`
- `DATABASE`: `sportmonk`
- `DRIVER`: `ODBC Driver 18 for SQL Server`
- `DIALECT`: `mssql`

Adjust these values in `app/services/config/settings.py` if your database environment differs.

### Model Files

ML model files are expected under `models/`:

- `forrest_model.joblib`
- `logistic_model.joblib`
- `xgb_model.joblib`
- `SVC_model.joblib`

If these files are missing, the app logs a warning and continues with reduced ML classification support.

## Running the Application

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r req.txt
```

3. Export required environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export SQL_SERVER_PASSWORD="your_sql_password"
```

4. Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Usage

### Latest Fixture Endpoint

Request:

```http
GET /fixtures/latest?country_id=1&league_id=1
```

Response fields:

- `fixture_id` - latest fixture identifier
- `balls_simulated` - number of balls processed
- `narratives_written` - number of generated narratives
- `game_states_written` - number of game state snapshots written
- `llm_outputs` - the latest array of generated question/SQL/answer results

### Output CSV

The app writes LLM outputs to `data/llm_outputs.csv` with columns:

- `fixture_id`
- `ball_no`
- `question`
- `sql_query`
- `answer`

## Notes

- `req.txt` currently contains a small dependency list plus sample SQL queries at the bottom.
- The application expects a ball-by-ball cricket schema and uses SQL Server for database access.
- OpenAI usage may incur costs depending on your subscription and model usage.

## Troubleshooting

- If the app cannot connect to the database, verify `SQL_SERVER_PASSWORD` and the SQL Server connection details in `app/services/config/settings.py`.
- If model loading fails, confirm that the file paths in `app/services/config/settings.py` match the actual `models/` file names.
- If OpenAI requests fail, verify `OPENAI_API_KEY` is valid and the network connection is available.

## License

This repository does not include a license. Add a `LICENSE` file if you wish to specify usage terms.

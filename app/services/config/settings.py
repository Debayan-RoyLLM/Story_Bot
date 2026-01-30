"""
Configuration Settings for Cricket Statistics Query System

This module contains all configuration constants and settings for the system.
"""

from pathlib import Path
import os

# Base directory for configuration files
CONFIG_DIR = Path(__file__).parent

# Project root directory (3 levels up from this file: config -> services -> app -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DatabaseConfig:
    """Database connection configuration."""

    USERNAME = "SA"
    SERVER = "localhost"
    DATABASE = "sportmonk"
    DRIVER = "ODBC Driver 18 for SQL Server"
    DIALECT = "mssql"

    @staticmethod
    def get_connection_string(password: str) -> str:
        """Generate SQL Server connection string."""
        from urllib.parse import quote
        encoded_pass = quote(password)
        return (
            f"mssql+pyodbc://{DatabaseConfig.USERNAME}:{encoded_pass}@"
            f"{DatabaseConfig.SERVER}/{DatabaseConfig.DATABASE}?"
            f"driver={DatabaseConfig.DRIVER}&TrustServerCertificate=yes"
        )


class MLModelConfig:
    """Machine Learning model configuration."""

    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_MAX_LENGTH = 128
    PCA_DIMENSIONS = 59

    # Model file paths (absolute paths from project root)
    MODELS_DIR = PROJECT_ROOT / "models"

    # ---- Model Paths ----
    PCA_MODEL_PATH = PROJECT_ROOT / "pca_model_59.pkl"
    LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_model.joblib"
    RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "forrest_model.joblib"
    XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
    SVC_MODEL_PATH = MODELS_DIR / "SVC_model.joblib"



class QueryConfig:
    """Query execution configuration."""

    MAX_QUERY_TIMEOUT = 30  # seconds
    MAX_RESULTS_DISPLAY = 10  # rows to display before truncation
    DEFAULT_LLM_MODEL = "gpt-4o-mini"

    # CHASE-SQL reasoning paths
    CHASE_REASONING_PATHS = [
        {
            "name": "event_first",
            "instruction": (
                "Start from ball-by-ball events and filter conditions early. "
                "Then aggregate if required."
            )
        },
        {
            "name": "aggregation_first",
            "instruction": (
                "Determine the required aggregation first, then identify the "
                "minimum tables and joins needed."
            )
        },
        {
            "name": "player_first",
            "instruction": (
                "Focus on player involvement first (batsman/bowler), then "
                "derive match-level constraints."
            )
        },
        {
            "name": "chase_context",
            "instruction": (
                "Assume this is a chase scenario. Explicitly reason using "
                "scoreboard = 'S2' and inning = 2 when applicable."
            )
        }
    ]


class StatementConfig:
    """Statement generation configuration."""

    REQUIRED_VALID_STATEMENTS = 8
    RANDOM_FOREST_THRESHOLD = 0.5  # Probability threshold for valid statements

    # Game state keys for BERT embedding
    GAME_STATE_KEYS = [
        'req_runs', 'balls_remaining', 'batsman_total_runs', 'batsman_balls_faced',
        'nonstriker_total_runs', 'nonstriker_balls_faced', 'team_run_rate',
        'req_run_rate', 'wickets', 'second_last_ball', 'last_ball'
    ]


class FileConfig:
    """File paths configuration."""

    METADATA_FILE = CONFIG_DIR / "metadata.json"
    PROMPT_TEMPLATE_FILE = CONFIG_DIR / "prompt_template.txt"


class APIConfig:
    """API and external service configuration."""

    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key from environment or prompt."""
        import getpass
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = getpass.getpass("Enter API key for OpenAI: ")
        return api_key

    @staticmethod
    def get_sql_password() -> str:
        """Get SQL Server password from environment or prompt."""
        import getpass
        password = os.environ.get("SQL_SERVER_PASSWORD")
        if not password:
            password = getpass.getpass("Enter SQL Server password: ")
        return password

    @staticmethod
    def get_langsmith_api_key() -> str:
        """Get LangSmith API key from environment."""
        return os.environ.get("LANGSMITH_API_KEY", "")


# Main configuration class
class Config:
    """Main configuration class aggregating all settings."""

    db = DatabaseConfig()
    ml = MLModelConfig()
    query = QueryConfig()
    statement = StatementConfig()
    files = FileConfig()
    api = APIConfig()

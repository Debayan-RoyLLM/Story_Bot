import os
from urllib.parse import quote
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

_password = quote(os.environ.get("SQL_SERVER_PASSWORD", ""))

DATABASE_URL = (
    f"mssql+pyodbc://SA:{_password}@127.0.0.1:1433/sportmonk"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Encrypt=no"
    "&TrustServerCertificate=yes"
    "&Connection Timeout=30"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
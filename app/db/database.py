from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = (
    "mssql+pyodbc://SA:Gitaa%402026@127.0.0.1:1433/sportmonk"
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
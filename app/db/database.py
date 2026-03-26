from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.config.settings import Config

_config = Config()
_password = _config.api.get_sql_password()

DATABASE_URL = _config.db.get_connection_string(_password)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

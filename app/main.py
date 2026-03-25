from fastapi import FastAPI
from app.services.fixtures_services import router as fixture_router

app = FastAPI()

app.include_router(fixture_router)

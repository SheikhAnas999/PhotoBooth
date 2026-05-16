from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import close_db, connect_db, get_database
from routes.generate import router as generate_router
from routes.template import router as template_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="ComfyUI Bridge API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(generate_router)
app.include_router(template_router)


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health_check():
    try:
        await get_database().command("ping")
        return {"status": "ok", "mongodb": "connected"}
    except Exception:
        return {"status": "degraded", "mongodb": "disconnected"}

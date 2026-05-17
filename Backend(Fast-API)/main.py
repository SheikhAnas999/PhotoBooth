import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx logs every poll to ComfyUI /history at INFO — suppress that noise only.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from database import close_db, connect_db, get_database
from routes.event import router as event_router
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
app.include_router(event_router)


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

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

from config import IMAGES_DIR
from database import close_db, connect_db, get_database
from routes.email import router as email_router
from routes.event import router as event_router
from routes.event_images import router as event_images_router
from routes.generate import router as generate_router
from routes.template import router as template_router
from services.event_images_hub import event_images_hub


@asynccontextmanager
async def lifespan(app: FastAPI):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    await connect_db()
    yield
    await event_images_hub.close_all()
    await close_db()


app = FastAPI(
    title="ComfyUI Bridge API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(generate_router)
app.include_router(template_router)
app.include_router(event_router)
app.include_router(email_router)
app.include_router(event_images_router)


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

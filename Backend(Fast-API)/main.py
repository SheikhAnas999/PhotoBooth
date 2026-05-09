from fastapi import FastAPI
from routes import router as generate_router

app = FastAPI(title="ComfyUI Bridge API", version="1.0.0")

app.include_router(generate_router)


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

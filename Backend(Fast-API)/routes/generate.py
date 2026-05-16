import json
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.generate import GenerateResponse

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent
    / "workflows"
    / "image_qwen_image_edit_2509 (3).json"
)

router = APIRouter(tags=["generate"])


def load_workflow() -> dict:
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(f"Workflow not found at {WORKFLOW_PATH}")
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_payload(image_name: str, prompt: Optional[str], seed: Optional[int]) -> dict:
    workflow = load_workflow()
    workflow["78"]["inputs"]["image"] = image_name

    if prompt is not None:
        workflow["434:348"]["inputs"]["prompt"] = prompt

    if seed is not None:
        workflow["434:340"]["inputs"]["seed"] = seed

    return {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }


async def upload_image_to_comfyui(client: httpx.AsyncClient, file: UploadFile) -> str:
    image_bytes = await file.read()

    upload_response = await client.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (file.filename, image_bytes, file.content_type or "image/png")},
        data={"overwrite": "true"},
    )
    upload_response.raise_for_status()

    result = upload_response.json()
    return result["name"]


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile = File(..., description="Image to edit"),
    prompt: Optional[str] = Form(default=None, description="Override the positive text prompt"),
    seed: Optional[int] = Form(default=None, description="Override the KSampler seed"),
):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            uploaded_filename = await upload_image_to_comfyui(client, image)
            payload = build_payload(uploaded_filename, prompt, seed)

            queue_response = await client.post(
                f"{COMFYUI_URL}/prompt",
                json=payload,
            )
            queue_response.raise_for_status()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to ComfyUI. Make sure it is running on port 8188.",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI returned an error: {exc.response.text}",
        )

    data = queue_response.json()
    prompt_id: str = data.get("prompt_id", "unknown")

    return GenerateResponse(
        prompt_id=prompt_id,
        uploaded_filename=uploaded_filename,
        message=f"Job queued successfully. Track it at {COMFYUI_URL}/history/{prompt_id}",
    )

import json
import uuid
import httpx
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path(__file__).parent / "workflows" / "image_qwen_image_edit_2509 (3).json"

router = APIRouter()


# ── Response schema ───────────────────────────────────────────────────────────
class GenerateResponse(BaseModel):
    prompt_id: str
    uploaded_filename: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_workflow() -> dict:
    """Load the workflow JSON from disk."""
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(f"Workflow not found at {WORKFLOW_PATH}")
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_payload(image_name: str, prompt: Optional[str], seed: Optional[int]) -> dict:
    """Inject request parameters into the workflow and wrap in ComfyUI payload."""
    workflow = load_workflow()

    # Node 78 – LoadImage: use the filename returned by ComfyUI after upload
    workflow["78"]["inputs"]["image"] = image_name

    # Node 434:348 – positive TextEncodeQwenImageEditPlus: override prompt if given
    if prompt is not None:
        workflow["434:348"]["inputs"]["prompt"] = prompt

    # Node 434:340 – KSampler: override seed if given
    if seed is not None:
        workflow["434:340"]["inputs"]["seed"] = seed

    return {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }


async def upload_image_to_comfyui(client: httpx.AsyncClient, file: UploadFile) -> str:
    """
    Upload the image to ComfyUI's /upload/image endpoint.
    Returns the filename ComfyUI assigned to the uploaded file.
    """
    image_bytes = await file.read()

    upload_response = await client.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (file.filename, image_bytes, file.content_type or "image/png")},
        data={"overwrite": "true"},
    )
    upload_response.raise_for_status()

    result = upload_response.json()
    # ComfyUI returns: {"name": "filename.png", "subfolder": "", "type": "input"}
    return result["name"]


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile = File(..., description="Image to edit"),
    prompt: Optional[str] = Form(default=None, description="Override the positive text prompt"),
    seed: Optional[int] = Form(default=None, description="Override the KSampler seed"),
):
    """
    Upload an image and queue a Qwen image-edit job on the local ComfyUI server.

    - Accepts multipart/form-data (image file + optional form fields)
    - Uploads the image to ComfyUI /upload/image
    - Queues the workflow with the uploaded filename injected into LoadImage node
    - ComfyUI must be running at http://127.0.0.1:8188
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:

            # Step 1 – upload image to ComfyUI
            uploaded_filename = await upload_image_to_comfyui(client, image)

            # Step 2 – build and queue the workflow
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

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pymongo import ReturnDocument

from config import COMFYUI_OUTPUT_DIR, PROJECT_ROOT
from database import get_database
from models.generate import GenerateResponse, PreviewImageResponse
from routes.event import _find_event_or_404, _resolve_event_folder
from routes.template import _find_template_or_404
from utils.event_images import notify_image_added

EVENTS_COLLECTION = "events"

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"
EDIT_WORKFLOW_PATH = WORKFLOWS_DIR / "image_qwen_image_edit_2509 (3).json"
DETECT_WORKFLOW_PATH = WORKFLOWS_DIR / "detect-person-in-image.json"

DETECT_IMAGE_NODE_ID = "692"
DETECT_TEXT_NODE_ID = "695"
EDIT_IMAGE_NODE_ID = "343"
EDIT_PROMPT_NODE_ID = "434:348"
EDIT_OUTPUT_NODE_ID = "342"
EDIT_SEED_NODE_ID = "344:434:340"

MAX_PEOPLE = 5
JOB_POLL_INTERVAL_SEC = 1.0
DETECT_JOB_TIMEOUT_SEC = 120.0
EDIT_JOB_TIMEOUT_SEC = 600.0

router = APIRouter(tags=["generate"])
logger = logging.getLogger(__name__)

_TEMPLATE_ID_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def load_workflow(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_edit_workflow() -> dict:
    return load_workflow(EDIT_WORKFLOW_PATH)


def load_detect_workflow() -> dict:
    return load_workflow(DETECT_WORKFLOW_PATH)


def build_detect_payload(image_name: str) -> dict:
    workflow = load_detect_workflow()
    workflow[DETECT_IMAGE_NODE_ID]["inputs"]["image"] = image_name
    return {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }


def build_edit_payload(
    image_name: str,
    prompt: str,
    seed: Optional[int] = None,
) -> dict:
    workflow = load_edit_workflow()
    workflow[EDIT_IMAGE_NODE_ID]["inputs"]["image"] = image_name
    workflow[EDIT_PROMPT_NODE_ID]["inputs"]["prompt"] = prompt

    if seed is not None:
        workflow[EDIT_SEED_NODE_ID]["inputs"]["seed"] = seed

    return {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }


def parse_person_count(detection_text: str) -> int:
    """Parse RT-DETR PreviewAny output and return the number of detected people."""
    text = detection_text.strip()
    if not text:
        return 0

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Detection output is not valid JSON")
        data = json.loads(text[start : end + 1])

    def count_people(value: Any) -> int:
        if isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                return sum(
                    1
                    for item in value
                    if item.get("label") == "person"
                )
            return sum(count_people(item) for item in value)
        if isinstance(value, dict):
            if "label" in value:
                return 1 if value.get("label") == "person" else 0
            return sum(count_people(item) for item in value.values())
        return 0

    return count_people(data)


def combine_prompts(base_prompt: str, people_prompt: str) -> str:
    base = base_prompt.strip()
    people = people_prompt.strip()
    if not base:
        return people
    if not people:
        return base
    return f"{people}\n{base}"


def normalize_people_prompts(people_prompts: Any) -> dict[str, str]:
    """Normalize MongoDB peoplePrompts object to string keys \"1\"..\"5\"."""
    if people_prompts is None:
        raise HTTPException(
            status_code=500,
            detail="Template is missing peoplePrompts",
        )

    if not isinstance(people_prompts, dict):
        raise HTTPException(
            status_code=500,
            detail="Template peoplePrompts has an invalid format",
        )

    alias_map = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
    }
    normalized: dict[str, str] = {}
    for key, value in people_prompts.items():
        str_key = alias_map.get(str(key).lower(), str(key))
        if value is not None:
            normalized[str_key] = str(value)

    return normalized


def prompt_count_for_detected(detected_count: int) -> int:
    """Map detected person count to a peoplePrompts key (capped at MAX_PEOPLE)."""
    return min(detected_count, MAX_PEOPLE)


def get_people_prompt_for_count(people_prompts: dict[str, str], person_count: int) -> str:
    key = str(prompt_count_for_detected(person_count))
    if key not in people_prompts:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Template is missing people prompt '{key}'. "
                f"Expected keys 1-{MAX_PEOPLE} in peoplePrompts."
            ),
        )
    return people_prompts[key]


def get_template_prompts(template: dict[str, Any]) -> tuple[str, dict[str, str]]:
    base_prompt = template.get("basePrompt") or template.get("base_prompt")
    if not base_prompt:
        raise HTTPException(
            status_code=500,
            detail="Template is missing basePrompt",
        )

    people_prompts = normalize_people_prompts(template.get("peoplePrompts"))
    return str(base_prompt), people_prompts


def extract_text_output(history_entry: dict[str, Any], node_id: str) -> str:
    node_outputs = history_entry.get("outputs", {}).get(node_id, {})
    texts = node_outputs.get("text", [])
    if not texts:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI did not return text output from node {node_id}",
        )

    item = texts[0]
    if isinstance(item, (list, tuple)):
        return str(item[0])
    return str(item)


def extract_image_output(history_entry: dict[str, Any], node_id: str) -> dict[str, str]:
    node_outputs = history_entry.get("outputs", {}).get(node_id, {})
    images = node_outputs.get("images", [])
    if not images:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI did not return an image from node {node_id}",
        )
    return images[0]


def _raise_comfyui_failure(history_entry: dict[str, Any]) -> None:
    status = history_entry.get("status", {})
    messages = status.get("messages", [])
    for message in messages:
        if isinstance(message, (list, tuple)) and len(message) >= 2:
            if message[0] == "execution_error":
                raise HTTPException(
                    status_code=502,
                    detail=f"ComfyUI execution failed: {message[1]}",
                )
    status_str = status.get("status_str", "error")
    raise HTTPException(
        status_code=502,
        detail=f"ComfyUI job failed with status: {status_str}",
    )


async def queue_comfyui_prompt(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    *,
    label: str = "workflow",
    request_id: str = "",
) -> str:
    prefix = f"[{request_id}] " if request_id else ""
    logger.info("%sQueueing ComfyUI job: %s", prefix, label)
    response = await client.post(f"{COMFYUI_URL}/prompt", json=payload)
    response.raise_for_status()
    data = response.json()
    prompt_id = data.get("prompt_id", "unknown")
    logger.info("%sComfyUI job queued: %s prompt_id=%s", prefix, label, prompt_id)
    return prompt_id


async def wait_for_prompt_completion(
    client: httpx.AsyncClient,
    prompt_id: str,
    timeout_sec: float,
    *,
    label: str = "workflow",
    request_id: str = "",
) -> dict[str, Any]:
    prefix = f"[{request_id}] " if request_id else ""
    deadline = time.monotonic() + timeout_sec
    started = time.monotonic()
    poll_count = 0

    logger.info(
        "%sWaiting for ComfyUI job: %s prompt_id=%s timeout=%ss",
        prefix,
        label,
        prompt_id,
        int(timeout_sec),
    )

    while time.monotonic() < deadline:
        poll_count += 1
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        data = response.json()

        if prompt_id not in data:
            if poll_count == 1 or poll_count % 15 == 0:
                logger.debug(
                    "%sStill waiting for %s prompt_id=%s (poll #%d)",
                    prefix,
                    label,
                    prompt_id,
                    poll_count,
                )
            await asyncio.sleep(JOB_POLL_INTERVAL_SEC)
            continue

        history_entry = data[prompt_id]
        status = history_entry.get("status", {})
        if status.get("status_str") == "error":
            logger.error(
                "%sComfyUI job failed: %s prompt_id=%s status=%s",
                prefix,
                label,
                prompt_id,
                status,
            )
            _raise_comfyui_failure(history_entry)

        if history_entry.get("outputs"):
            elapsed = time.monotonic() - started
            logger.info(
                "%sComfyUI job finished: %s prompt_id=%s elapsed=%.1fs polls=%d",
                prefix,
                label,
                prompt_id,
                elapsed,
                poll_count,
            )
            return history_entry

        if status.get("completed"):
            if history_entry.get("outputs"):
                elapsed = time.monotonic() - started
                logger.info(
                    "%sComfyUI job finished: %s prompt_id=%s elapsed=%.1fs polls=%d",
                    prefix,
                    label,
                    prompt_id,
                    elapsed,
                    poll_count,
                )
                return history_entry
            logger.error(
                "%sComfyUI job completed without outputs: %s prompt_id=%s",
                prefix,
                label,
                prompt_id,
            )
            _raise_comfyui_failure(history_entry)

        await asyncio.sleep(JOB_POLL_INTERVAL_SEC)

    logger.error(
        "%sComfyUI job timed out: %s prompt_id=%s after %ss",
        prefix,
        label,
        prompt_id,
        int(timeout_sec),
    )
    raise HTTPException(
        status_code=504,
        detail=f"ComfyUI job timed out after {int(timeout_sec)} seconds",
    )


async def upload_image_bytes_to_comfyui(
    client: httpx.AsyncClient,
    filename: str,
    image_bytes: bytes,
    content_type: str,
) -> str:
    upload_response = await client.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, image_bytes, content_type)},
        data={"overwrite": "true"},
    )
    upload_response.raise_for_status()
    return upload_response.json()["name"]


async def download_comfyui_image(
    client: httpx.AsyncClient,
    image_info: dict[str, str],
) -> bytes:
    params = {
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    }
    response = await client.get(f"{COMFYUI_URL}/view", params=params)
    response.raise_for_status()
    return response.content


def _comfyui_output_file_path(image_info: dict[str, str]) -> Path | None:
    filename = image_info.get("filename")
    if not filename:
        return None

    if image_info.get("type", "output") != "output":
        return None

    subfolder = image_info.get("subfolder", "")
    file_path = (COMFYUI_OUTPUT_DIR / subfolder / filename).resolve()
    output_root = COMFYUI_OUTPUT_DIR.resolve()

    try:
        file_path.relative_to(output_root)
    except ValueError:
        return None

    return file_path


def delete_comfyui_output_image(
    image_info: dict[str, str],
    *,
    request_id: str = "",
) -> None:
    prefix = f"[{request_id}] " if request_id else ""
    file_path = _comfyui_output_file_path(image_info)
    if file_path is None:
        logger.warning("%sSkipping ComfyUI output delete: invalid image info %s", prefix, image_info)
        return

    if file_path.is_file():
        file_path.unlink()
        logger.info("%sDeleted ComfyUI output file: %s", prefix, file_path)
    else:
        logger.warning("%sComfyUI output file not found for delete: %s", prefix, file_path)


def save_image_to_event_folder(
    event_folder: Path,
    image_bytes: bytes,
    image_info: dict[str, str],
    file_number: int,
) -> Path:
    suffix = Path(image_info.get("filename", "output.png")).suffix or ".png"
    event_folder.mkdir(parents=True, exist_ok=True)
    save_path = event_folder / f"{file_number}{suffix}"
    save_path.write_bytes(image_bytes)
    return save_path


async def increment_event_count(event_id: str) -> int:
    document = await get_database()[EVENTS_COLLECTION].find_one_and_update(
        {"eventId": event_id},
        {"$inc": {"count": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if document is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return int(document.get("count", 0))


def _validate_template_id_form_value(template_id: str) -> str:
    value = template_id.strip()
    if len(value) > 80 or " " in value:
        raise HTTPException(
            status_code=400,
            detail=(
                "template_id must be a template UUID from GET /get-templates or the event "
                "templates list — not the prompt text. For a custom prompt, use POST /preview-image."
            ),
        )
    if not _TEMPLATE_ID_UUID.match(value):
        raise HTTPException(
            status_code=400,
            detail=(
                "template_id must be a UUID like bbd81d6b-5b71-4b8a-a905-57ce323935fd. "
                "Find it via GET /show-events (templates[].templateId) or GET /get-templates."
            ),
        )
    return value


def _event_template_ids(event: dict[str, Any]) -> set[str]:
    templates = event.get("templates") or []
    return {
        item["templateId"]
        for item in templates
        if isinstance(item, dict) and item.get("templateId")
    }


async def count_people_in_image(
    client: httpx.AsyncClient,
    uploaded_filename: str,
    *,
    request_id: str = "",
) -> int:
    prefix = f"[{request_id}] " if request_id else ""
    logger.info("%sStarting person detection for image=%s", prefix, uploaded_filename)

    payload = build_detect_payload(uploaded_filename)
    prompt_id = await queue_comfyui_prompt(
        client,
        payload,
        label="person-detection",
        request_id=request_id,
    )
    history_entry = await wait_for_prompt_completion(
        client,
        prompt_id,
        DETECT_JOB_TIMEOUT_SEC,
        label="person-detection",
        request_id=request_id,
    )
    detection_text = extract_text_output(history_entry, DETECT_TEXT_NODE_ID)
    preview = detection_text[:200] + "..." if len(detection_text) > 200 else detection_text
    logger.debug("%sDetection raw output: %s", prefix, preview)

    person_count = parse_person_count(detection_text)
    logger.info("%sPerson detection complete: count=%d", prefix, person_count)
    return person_count


async def run_image_edit_workflow(
    client: httpx.AsyncClient,
    uploaded_filename: str,
    prompt: str,
    seed: Optional[int] = None,
    *,
    request_id: str = "",
) -> tuple[str, bytes, dict[str, str]]:
    prefix = f"[{request_id}] " if request_id else ""
    logger.info(
        "%sStarting image edit workflow image=%s prompt_len=%d seed=%s",
        prefix,
        uploaded_filename,
        len(prompt),
        seed if seed is not None else "default",
    )

    payload = build_edit_payload(uploaded_filename, prompt, seed)
    prompt_id = await queue_comfyui_prompt(
        client,
        payload,
        label="image-edit",
        request_id=request_id,
    )
    history_entry = await wait_for_prompt_completion(
        client,
        prompt_id,
        EDIT_JOB_TIMEOUT_SEC,
        label="image-edit",
        request_id=request_id,
    )
    image_info = extract_image_output(history_entry, EDIT_OUTPUT_NODE_ID)
    logger.info(
        "%sDownloading output image filename=%s subfolder=%s",
        prefix,
        image_info.get("filename"),
        image_info.get("subfolder", ""),
    )
    image_bytes = await download_comfyui_image(client, image_info)
    logger.info("%sOutput image downloaded: %d bytes", prefix, len(image_bytes))
    return prompt_id, image_bytes, image_info


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile = File(..., description="Image to edit"),
    template_id: str = Form(
        ...,
        description="Template UUID (e.g. bbd81d6b-5b71-4b8a-a905-57ce323935fd). Not the prompt text.",
    ),
    event_id: str = Form(
        ...,
        alias="eventId",
        description="Event ID to save the generated image under",
    ),
    seed: Optional[int] = Form(default=None, description="Override the KSampler seed"),
):
    request_id = uuid.uuid4().hex[:8]
    started = time.monotonic()

    image_bytes = await image.read()
    if not image_bytes:
        logger.warning("[%s] Rejected empty image upload", request_id)
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    template_id = _validate_template_id_form_value(template_id)

    filename = image.filename or "input.png"
    content_type = image.content_type or "image/png"
    logger.info(
        "[%s] POST /generate started event_id=%s template_id=%s file=%s size=%d bytes content_type=%s",
        request_id,
        event_id,
        template_id,
        filename,
        len(image_bytes),
        content_type,
    )

    try:
        event = await _find_event_or_404(event_id)
        event_folder = _resolve_event_folder(event.get("path", ""))
        if event_folder is None:
            raise HTTPException(
                status_code=400,
                detail=f"Event '{event_id}' has no valid image folder path",
            )
        if template_id not in _event_template_ids(event):
            raise HTTPException(
                status_code=400,
                detail=f"Template '{template_id}' is not assigned to event '{event_id}'",
            )

        template = await _find_template_or_404(template_id)
        logger.info(
            "[%s] Template loaded: name=%r templateId=%s",
            request_id,
            template.get("name"),
            template.get("templateId"),
        )
    except HTTPException as exc:
        logger.warning(
            "[%s] Template lookup failed template_id=%s status=%s detail=%s",
            request_id,
            template_id,
            exc.status_code,
            exc.detail,
        )
        raise
    except Exception as exc:
        logger.exception("[%s] Template lookup error template_id=%s", request_id, template_id)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load template: {exc}",
        ) from exc

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=EDIT_JOB_TIMEOUT_SEC)) as client:
            uploaded_filename = await upload_image_bytes_to_comfyui(
                client,
                filename,
                image_bytes,
                content_type,
            )
            logger.info(
                "[%s] Image uploaded to ComfyUI as %r",
                request_id,
                uploaded_filename,
            )

            person_count = await count_people_in_image(
                client,
                uploaded_filename,
                request_id=request_id,
            )
            if person_count < 1:
                logger.warning("[%s] No people detected in image", request_id)
                raise HTTPException(
                    status_code=400,
                    detail="No people detected in the uploaded image",
                )

            base_prompt, people_prompts = get_template_prompts(template)
            prompt_key = str(prompt_count_for_detected(person_count))
            people_prompt = get_people_prompt_for_count(
                people_prompts,
                person_count,
            )
            combined_prompt = combine_prompts(base_prompt, people_prompt)

            if person_count > MAX_PEOPLE:
                logger.info(
                    "[%s] Using capped people prompt: detected=%d prompt_key=%s",
                    request_id,
                    person_count,
                    prompt_key,
                )
            else:
                logger.info(
                    "[%s] Prompts resolved: detected=%d prompt_key=%s base_len=%d people_len=%d combined_len=%d",
                    request_id,
                    person_count,
                    prompt_key,
                    len(base_prompt),
                    len(people_prompt),
                    len(combined_prompt),
                )

            prompt_id, output_image_bytes, output_image_info = await run_image_edit_workflow(
                client,
                uploaded_filename,
                combined_prompt,
                seed,
                request_id=request_id,
            )

    except HTTPException as exc:
        logger.warning(
            "[%s] /generate failed status=%s detail=%s",
            request_id,
            exc.status_code,
            exc.detail,
        )
        raise
    except httpx.ConnectError as exc:
        logger.error("[%s] Cannot connect to ComfyUI on %s", request_id, COMFYUI_URL)
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to ComfyUI. Make sure it is running on port 8188.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[%s] ComfyUI HTTP error status=%s body=%s",
            request_id,
            exc.response.status_code,
            exc.response.text[:500],
        )
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI returned an error: {exc.response.text}",
        ) from exc
    except FileNotFoundError as exc:
        logger.error("[%s] Workflow file missing: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected error during /generate", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    delete_comfyui_output_image(output_image_info, request_id=request_id)

    new_event_count = await increment_event_count(event_id)
    saved_path = save_image_to_event_folder(
        event_folder,
        output_image_bytes,
        output_image_info,
        new_event_count,
    )
    saved_image_path = saved_path.relative_to(PROJECT_ROOT).as_posix()

    await notify_image_added(
        event_id,
        filename=saved_path.name,
        saved_image_path=saved_image_path,
        event_count=new_event_count,
    )

    output_image_base64 = base64.b64encode(output_image_bytes).decode("ascii")

    message = "Image generated successfully"
    if person_count > MAX_PEOPLE:
        message = (
            f"Detected {person_count} people; used {MAX_PEOPLE}-person prompt for generation"
        )

    elapsed = time.monotonic() - started
    logger.info(
        "[%s] POST /generate completed prompt_id=%s person_count=%d output_bytes=%d "
        "saved_image_path=%s event_count=%d total_elapsed=%.1fs",
        request_id,
        prompt_id,
        person_count,
        len(output_image_bytes),
        saved_image_path,
        new_event_count,
        elapsed,
    )

    return GenerateResponse(
        prompt_id=prompt_id,
        person_count=person_count,
        uploaded_filename=uploaded_filename,
        output_image_base64=output_image_base64,
        message=message,
        event_id=event_id,
        saved_image_path=saved_image_path,
        event_count=new_event_count,
    )


@router.post("/preview-image", response_model=PreviewImageResponse)
async def preview_image(
    image: UploadFile = File(..., description="Reference image to edit"),
    prompt: str = Form(..., description="Edit prompt for the Qwen image workflow"),
    seed: Optional[int] = Form(default=None, description="Override the KSampler seed"),
):
    request_id = uuid.uuid4().hex[:8]
    started = time.monotonic()

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    filename = image.filename or "preview-input.png"
    content_type = image.content_type or "image/png"
    logger.info(
        "[%s] POST /preview-image started file=%s prompt_len=%d",
        request_id,
        filename,
        len(prompt.strip()),
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=EDIT_JOB_TIMEOUT_SEC)) as client:
            uploaded_filename = await upload_image_bytes_to_comfyui(
                client,
                filename,
                image_bytes,
                content_type,
            )
            prompt_id, output_image_bytes, output_image_info = await run_image_edit_workflow(
                client,
                uploaded_filename,
                prompt.strip(),
                seed,
                request_id=request_id,
            )
    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to ComfyUI. Make sure it is running on port 8188.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI returned an error: {exc.response.text}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected error during /preview-image", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    delete_comfyui_output_image(output_image_info, request_id=request_id)
    output_image_base64 = base64.b64encode(output_image_bytes).decode("ascii")

    elapsed = time.monotonic() - started
    logger.info(
        "[%s] POST /preview-image completed prompt_id=%s output_bytes=%d elapsed=%.1fs",
        request_id,
        prompt_id,
        len(output_image_bytes),
        elapsed,
    )

    return PreviewImageResponse(
        prompt_id=prompt_id,
        output_image_base64=output_image_base64,
        message="Preview image generated successfully",
    )

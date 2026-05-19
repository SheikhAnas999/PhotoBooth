import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from models.event_images import EventImageItem, EventImagesResponse
from services.event_images_hub import event_images_hub
from utils.event_images import (
    get_event_folder,
    list_event_images,
    resolve_event_image_path,
)

router = APIRouter(tags=["event-images"])
logger = logging.getLogger(__name__)

SSE_HEARTBEAT_SEC = 25.0
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _snapshot_payload(event_id: str, path: str, images: list[dict]) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "path": path,
        "count": len(images),
        "images": images,
    }


async def _sse_event_stream(event_id: str, request: Request) -> AsyncIterator[str]:
    folder, _, relative_path = await get_event_folder(event_id)
    images = list_event_images(event_id, folder)
    yield _format_sse("snapshot", _snapshot_payload(event_id, relative_path, images))

    queue = await event_images_hub.subscribe(event_id)
    try:
        while True:
            if await request.is_disconnected():
                break

            try:
                message = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_SEC)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue

            yield _format_sse(message.get("type", "image_added"), message)
    finally:
        await event_images_hub.unsubscribe(event_id, queue)
        logger.debug("SSE client disconnected for event_id=%s", event_id)


@router.get("/event-images/{event_id}", response_model=EventImagesResponse)
async def get_event_images(event_id: str):
    folder, _, relative_path = await get_event_folder(event_id)
    images = list_event_images(event_id, folder)
    return EventImagesResponse(
        eventId=event_id,
        path=relative_path,
        count=len(images),
        images=[EventImageItem.model_validate(item) for item in images],
    )


@router.get("/event-images/{event_id}/stream")
async def stream_event_images(event_id: str, request: Request):
    await get_event_folder(event_id)

    return StreamingResponse(
        _sse_event_stream(event_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/event-images/{event_id}/files/{filename}")
async def get_event_image_file(event_id: str, filename: str):
    folder, _, _ = await get_event_folder(event_id)
    file_path = resolve_event_image_path(folder, filename)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{filename}' not found",
        )

    media_type = IMAGE_MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=file_path, media_type=media_type, filename=file_path.name)

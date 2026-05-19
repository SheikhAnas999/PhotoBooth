import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from config import IMAGES_DIR, PROJECT_ROOT
from database import get_database
from models.event import (
    DeleteEventResponse,
    EventCreate,
    EventListResponse,
    EventResponse,
)

router = APIRouter(tags=["events"])

COLLECTION = "events"
TEMPLATES_COLLECTION = "templates"
_INVALID_FOLDER_CHARS = '<>:"/\\|?*'


def _events_collection():
    return get_database()[COLLECTION]


def _templates_collection():
    return get_database()[TEMPLATES_COLLECTION]


def _document_to_response(document: dict[str, Any]) -> EventResponse:
    payload = {key: value for key, value in document.items() if key != "_id"}
    payload.setdefault("path", "")
    payload.setdefault("count", 0)
    return EventResponse.model_validate(payload)


async def _find_event_or_404(event_id: str) -> dict[str, Any]:
    document = await _events_collection().find_one({"eventId": event_id})
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event '{event_id}' not found",
        )
    return document


def _sanitize_folder_name(name: str) -> str:
    cleaned = "".join(char for char in name.strip() if char not in _INVALID_FOLDER_CHARS)
    return cleaned.rstrip(". ") or "event"


def _event_folder_path(event_name: str) -> Path:
    return IMAGES_DIR / _sanitize_folder_name(event_name)


def _relative_event_path(folder_path: Path) -> str:
    return folder_path.relative_to(PROJECT_ROOT).as_posix()


def _resolve_event_folder(path: str) -> Path | None:
    if not path:
        return None

    folder_path = (PROJECT_ROOT / path).resolve()
    images_root = IMAGES_DIR.resolve()

    try:
        folder_path.relative_to(images_root)
    except ValueError:
        return None

    return folder_path


async def _validate_templates_exist(template_ids: list[str]) -> None:
    unique_ids = list(dict.fromkeys(template_ids))
    found = await _templates_collection().count_documents(
        {"templateId": {"$in": unique_ids}}
    )
    if found != len(unique_ids):
        existing_cursor = _templates_collection().find(
            {"templateId": {"$in": unique_ids}},
            {"templateId": 1},
        )
        existing_docs = await existing_cursor.to_list(length=None)
        existing_ids = {doc["templateId"] for doc in existing_docs}
        missing = [tid for tid in unique_ids if tid not in existing_ids]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Templates not found: {', '.join(missing)}",
        )


@router.post(
    "/create-event",
    response_model=EventResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_event(body: EventCreate):
    template_ids = [item.templateId for item in body.templates]
    await _validate_templates_exist(template_ids)

    folder_path = _event_folder_path(body.name)
    if folder_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An event folder already exists for '{body.name}'",
        )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    folder_path.mkdir(parents=False, exist_ok=False)

    event_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    document = {
        "eventId": event_id,
        "name": body.name,
        "templates": [item.model_dump(by_alias=True) for item in body.templates],
        "path": _relative_event_path(folder_path),
        "count": 0,
        "createdAt": created_at,
    }

    try:
        await _events_collection().insert_one(document)
    except Exception:
        if folder_path.is_dir():
            folder_path.rmdir()
        raise

    return _document_to_response(document)


@router.get("/show-events", response_model=EventListResponse)
async def show_events():
    cursor = _events_collection().find().sort("createdAt", -1)
    documents = await cursor.to_list(length=None)
    events = [_document_to_response(doc) for doc in documents]
    return EventListResponse(events=events, count=len(events))


@router.delete("/delete-event/{event_id}", response_model=DeleteEventResponse)
async def delete_event(event_id: str):
    document = await _events_collection().find_one({"eventId": event_id})
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event '{event_id}' not found",
        )

    folder_path = _resolve_event_folder(document.get("path", ""))
    if folder_path is not None and folder_path.is_dir():
        shutil.rmtree(folder_path)

    await _events_collection().delete_one({"eventId": event_id})

    return DeleteEventResponse(
        eventId=event_id,
        message="Event deleted successfully",
    )

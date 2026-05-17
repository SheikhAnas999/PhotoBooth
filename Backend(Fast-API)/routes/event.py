import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status

from database import get_database
from models.event import EventCreate, EventListResponse, EventResponse

router = APIRouter(tags=["events"])

COLLECTION = "events"
TEMPLATES_COLLECTION = "templates"


def _events_collection():
    return get_database()[COLLECTION]


def _templates_collection():
    return get_database()[TEMPLATES_COLLECTION]


def _document_to_response(document: dict[str, Any]) -> EventResponse:
    payload = {key: value for key, value in document.items() if key != "_id"}
    return EventResponse.model_validate(payload)


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

    event_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    document = {
        "eventId": event_id,
        "name": body.name,
        "templates": [item.model_dump(by_alias=True) for item in body.templates],
        "createdAt": created_at,
    }

    await _events_collection().insert_one(document)
    return _document_to_response(document)


@router.get("/show-events", response_model=EventListResponse)
async def show_events():
    cursor = _events_collection().find().sort("createdAt", -1)
    documents = await cursor.to_list(length=None)
    events = [_document_to_response(doc) for doc in documents]
    return EventListResponse(events=events, count=len(events))

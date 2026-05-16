import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pymongo import ReturnDocument

from database import get_database
from models.template import (
    DeleteTemplateResponse,
    TemplateCreate,
    TemplateListResponse,
    TemplateResponse,
    TemplateUpdate,
)

router = APIRouter(tags=["templates"])

COLLECTION = "templates"


def _templates_collection():
    return get_database()[COLLECTION]


def _document_to_response(document: dict[str, Any]) -> TemplateResponse:
    payload = {key: value for key, value in document.items() if key != "_id"}
    return TemplateResponse.model_validate(payload)


async def _find_template_or_404(template_id: str) -> dict[str, Any]:
    document = await _templates_collection().find_one({"templateId": template_id})
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return document


@router.post(
    "/create-template",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_template(body: TemplateCreate):
    template_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    document = body.model_dump(by_alias=True)
    document["templateId"] = template_id
    document["createdAt"] = created_at
    document["updatedAt"] = None

    await _templates_collection().insert_one(document)
    return _document_to_response(document)


@router.get("/get-templates", response_model=TemplateListResponse)
async def get_templates():
    cursor = _templates_collection().find().sort("createdAt", -1)
    documents = await cursor.to_list(length=None)
    templates = [_document_to_response(doc) for doc in documents]
    return TemplateListResponse(templates=templates, count=len(templates))


@router.get("/get-single-template/{template_id}", response_model=TemplateResponse)
async def get_single_template(template_id: str):
    document = await _find_template_or_404(template_id)
    return _document_to_response(document)


@router.put("/edit-template/{template_id}", response_model=TemplateResponse)
async def edit_template(template_id: str, body: TemplateUpdate):
    updates = body.model_dump(by_alias=True, exclude_none=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field is required to update the template",
        )

    updates["updatedAt"] = datetime.now(timezone.utc)

    document = await _templates_collection().find_one_and_update(
        {"templateId": template_id},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return _document_to_response(document)


@router.delete("/delete-template/{template_id}", response_model=DeleteTemplateResponse)
async def delete_template(template_id: str):
    result = await _templates_collection().delete_one({"templateId": template_id})
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return DeleteTemplateResponse(
        templateId=template_id,
        message="Template deleted successfully",
    )

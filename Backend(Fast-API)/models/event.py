from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EventTemplateRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    templateId: str = Field(min_length=1)
    name: str = Field(min_length=1)


class EventCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    name: str = Field(min_length=1, description="Event display name")
    templates: list[EventTemplateRef] = Field(min_length=1)


class EventResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    eventId: str
    name: str
    templates: list[EventTemplateRef]
    path: str
    count: int = 0
    createdAt: datetime


class EventListResponse(BaseModel):
    events: list[EventResponse]
    count: int


class DeleteEventResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    eventId: str
    message: str

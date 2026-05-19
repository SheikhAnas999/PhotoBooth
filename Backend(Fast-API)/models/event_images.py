from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EventImageItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    filename: str
    url: str
    size: int
    created_at: datetime = Field(alias="createdAt")


class EventImagesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(alias="eventId")
    path: str
    count: int
    images: list[EventImageItem]

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Position(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)


class PeoplePrompts(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    one: str = Field(alias="1")
    two: str = Field(alias="2")
    three: str = Field(alias="3")
    four: str = Field(alias="4")
    five: str = Field(alias="5")


def validate_hex_color(value: str) -> str:
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError("textColor must be a hex color like #ffffff")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError("textColor must be a valid hex color") from exc
    return value.lower()


class TemplateFields(BaseModel):
    """Shared template fields used by create and update payloads."""

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    name: str = Field(min_length=1)
    basePrompt: str
    peoplePrompts: PeoplePrompts
    overlayText: str = ""
    fontFamily: str = "Arial"
    fontSize: int = Field(default=32, ge=12, le=96)
    textColor: str = "#ffffff"
    textPosition: Position
    logoUrl: str | None = None
    logoScale: float = Field(default=0.2, ge=0.08, le=0.45)
    logoLocked: bool = False
    logoPosition: Position

    @field_validator("textColor")
    @classmethod
    def check_text_color(cls, value: str) -> str:
        return validate_hex_color(value)


class TemplateCreate(TemplateFields):
    pass


class TemplateUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    name: str | None = Field(default=None, min_length=1)
    basePrompt: str | None = None
    peoplePrompts: PeoplePrompts | None = None
    overlayText: str | None = None
    fontFamily: str | None = None
    fontSize: int | None = Field(default=None, ge=12, le=96)
    textColor: str | None = None
    textPosition: Position | None = None
    logoUrl: str | None = None
    logoScale: float | None = Field(default=None, ge=0.08, le=0.45)
    logoLocked: bool | None = None
    logoPosition: Position | None = None

    @field_validator("textColor")
    @classmethod
    def check_text_color(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_hex_color(value)


class TemplateResponse(TemplateFields):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    templateId: str
    createdAt: datetime
    updatedAt: datetime | None = None


class TemplateListResponse(BaseModel):
    templates: list[TemplateResponse]
    count: int


class DeleteTemplateResponse(BaseModel):
    templateId: str
    message: str

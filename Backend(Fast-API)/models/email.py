from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ConnectGmailRequest(BaseModel):
    sender_gmail: EmailStr = Field(description="User Gmail address, e.g. you@gmail.com")
    sender_app_password: str = Field(
        min_length=1,
        description="Gmail app password for SMTP",
    )


class ConnectGmailResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sender_gmail: str
    message: str


class GmailConnectionData(BaseModel):
    sender_gmail: str


class ShowConnectGmailResponse(BaseModel):
    connected: bool
    data: GmailConnectionData | None = None


class SendEmailRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    receiver_email: EmailStr = Field(description="Recipient email address")
    message: str = Field(min_length=1, description="Plain-text email body")
    event_id: str = Field(
        min_length=1,
        alias="eventId",
        description="Event ID whose image folder will be zipped and attached",
    )


class SendEmailResponse(BaseModel):
    message: str
    zip_filename: str
    image_count: int
  
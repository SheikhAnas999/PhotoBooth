from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema
from pymongo import ReturnDocument

from database import get_database
from models.email import (
    ConnectGmailRequest,
    ConnectGmailResponse,
    GmailConnectionData,
    SendEmailRequest,
    SendEmailResponse,
    ShowConnectGmailResponse,
)
from utils.event_zip import (
    create_event_zip_archive,
    get_event_image_folder,
    zip_download_filename,
)

router = APIRouter(tags=["email"])

COLLECTION = "Email"
EMAIL_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "email_templates"
DEFAULT_EMAIL_SUBJECT = "Photo Booth"


def _email_collection():
    return get_database()[COLLECTION]


def _document_to_response(document: dict[str, Any], *, created: bool) -> ConnectGmailResponse:
    return ConnectGmailResponse(
        sender_gmail=document["sender_gmail"],
        message="Gmail connected successfully" if created else "Gmail credentials updated successfully",
    )


async def _get_connected_gmail_credentials() -> tuple[str, str]:
    document = await _email_collection().find_one(sort=[("_id", -1)])
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gmail is not connected. Call /connect-gmail first.",
        )

    sender_gmail = document.get("sender_gmail")
    sender_app_password = document.get("sender_app_password")
    if not sender_gmail or not sender_app_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stored Gmail credentials are incomplete. Reconnect with /connect-gmail.",
        )

    return str(sender_gmail), str(sender_app_password).replace(" ", "")


def _connection_config(sender_gmail: str, sender_app_password: str) -> ConnectionConfig:
    EMAIL_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    return ConnectionConfig(
        MAIL_USERNAME=sender_gmail,
        MAIL_PASSWORD=sender_app_password,
        MAIL_FROM=sender_gmail,
        MAIL_PORT=587,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True,
        TEMPLATE_FOLDER=EMAIL_TEMPLATES_DIR,
    )


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


@router.post("/connect-gmail", response_model=ConnectGmailResponse)
async def connect_gmail(body: ConnectGmailRequest, response: Response):
    sender_gmail = str(body.sender_gmail).lower()

    existing = await _email_collection().find_one({"sender_gmail": sender_gmail})
    created = existing is None

    set_fields: dict[str, Any] = {
        "sender_gmail": sender_gmail,
        "sender_app_password": body.sender_app_password,
    }

    document = await _email_collection().find_one_and_update(
        {"sender_gmail": sender_gmail},
        {"$set": set_fields},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save Gmail credentials",
        )

    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return _document_to_response(document, created=created)


@router.get("/show-connect-gmail", response_model=ShowConnectGmailResponse)
async def show_connect_gmail():
    document = await _email_collection().find_one(sort=[("_id", -1)])
    if document is None:
        return ShowConnectGmailResponse(connected=False, data=None)

    return ShowConnectGmailResponse(
        connected=True,
        data=GmailConnectionData(sender_gmail=document["sender_gmail"]),
    )


@router.get("/zip/{event_id}")
async def download_event_zip(event_id: str, background_tasks: BackgroundTasks):
    event_folder, event_name = await get_event_image_folder(event_id)
    zip_path = create_event_zip_archive(event_id, event_folder)
    filename = zip_download_filename(event_name)

    background_tasks.add_task(_remove_file, zip_path)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename,
    )


@router.post("/send-email", response_model=SendEmailResponse)
async def send_email(body: SendEmailRequest):
    event_folder, event_name = await get_event_image_folder(body.event_id)
    image_count = sum(1 for path in event_folder.iterdir() if path.is_file())
    zip_path = create_event_zip_archive(body.event_id, event_folder)
    zip_filename = zip_download_filename(event_name)

    sender_gmail, sender_app_password = await _get_connected_gmail_credentials()
    conf = _connection_config(sender_gmail, sender_app_password)

    message = MessageSchema(
        subject=DEFAULT_EMAIL_SUBJECT,
        recipients=[str(body.receiver_email)],
        body=body.message,
        subtype="plain",
        attachments=[str(zip_path)],
    )

    try:
        fm = FastMail(conf)
        await fm.send_message(message)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to send email: {exc}",
        ) from exc
    finally:
        _remove_file(zip_path)

    return SendEmailResponse(
        message="Email sent successfully",
        zip_filename=zip_filename,
        image_count=image_count,
    )

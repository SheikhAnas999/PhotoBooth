from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, status

from routes.event import _find_event_or_404, _resolve_event_folder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


async def get_event_folder(event_id: str) -> tuple[Path, str, str]:
    event = await _find_event_or_404(event_id)
    folder = _resolve_event_folder(event.get("path", ""))
    if folder is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Event '{event_id}' has no valid image folder path",
        )

    folder.mkdir(parents=True, exist_ok=True)
    event_name = str(event.get("name") or event_id)
    relative_path = str(event.get("path") or "")
    return folder, event_name, relative_path


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def build_image_url(event_id: str, filename: str) -> str:
    return f"/event-images/{event_id}/files/{filename}"


async def notify_image_added(
    event_id: str,
    *,
    filename: str,
    saved_image_path: str,
    event_count: int,
) -> None:
    from services.event_images_hub import event_images_hub

    await event_images_hub.publish(
        event_id,
        {
            "type": "image_added",
            "eventId": event_id,
            "filename": filename,
            "url": build_image_url(event_id, filename),
            "savedImagePath": saved_image_path,
            "eventCount": event_count,
        },
    )


def list_event_images(event_id: str, folder: Path) -> list[dict]:
    items: list[dict] = []
    for file_path in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime):
        if not _is_image_file(file_path):
            continue
        stat = file_path.stat()
        items.append(
            {
                "filename": file_path.name,
                "url": build_image_url(event_id, file_path.name),
                "size": stat.st_size,
                "createdAt": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            }
        )
    return items


def resolve_event_image_path(folder: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image filename",
        )

    file_path = (folder / safe_name).resolve()
    folder_resolved = folder.resolve()

    try:
        file_path.relative_to(folder_resolved)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image path",
        ) from exc

    if not _is_image_file(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{safe_name}' not found",
        )

    return file_path

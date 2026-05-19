import shutil
from pathlib import Path

from fastapi import HTTPException, status

from config import PROJECT_ROOT
from routes.event import _find_event_or_404, _resolve_event_folder

ZIP_CACHE_DIR = PROJECT_ROOT / "temp_zips"


def _sanitize_zip_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join(char for char in name.strip() if char not in invalid)
    return cleaned.rstrip(". ") or "event-images"


async def get_event_image_folder(event_id: str) -> tuple[Path, str]:
    event = await _find_event_or_404(event_id)
    folder = _resolve_event_folder(event.get("path", ""))
    if folder is None or not folder.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Event '{event_id}' has no valid image folder",
        )

    image_files = [path for path in folder.iterdir() if path.is_file()]
    if not image_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No images found for event '{event_id}'",
        )

    event_name = str(event.get("name") or event_id)
    return folder, event_name


def create_event_zip_archive(event_id: str, event_folder: Path) -> Path:
    ZIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ZIP_CACHE_DIR / f"{event_id}.zip"
    if zip_path.exists():
        zip_path.unlink()

    archive_base = ZIP_CACHE_DIR / event_id
    if Path(f"{archive_base}.zip").exists():
        Path(f"{archive_base}.zip").unlink()

    created_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=event_folder))
    return created_path


def zip_download_filename(event_name: str) -> str:
    return f"{_sanitize_zip_filename(event_name)}.zip"

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import settings

_client: AsyncIOMotorClient | None = None


async def connect_db() -> None:
    global _client
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    await _client.admin.command("ping")


async def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_database() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError("MongoDB is not connected")
    return _client[settings.mongodb_db_name]

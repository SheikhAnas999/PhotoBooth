from pydantic import BaseModel


class GenerateResponse(BaseModel):
    prompt_id: str
    uploaded_filename: str
    message: str

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    prompt_id: str
    person_count: int
    uploaded_filename: str
    output_image_base64: str
    message: str
    event_id: str
    saved_image_path: str
    event_count: int

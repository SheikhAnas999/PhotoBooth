from pydantic import BaseModel


class GenerateResponse(BaseModel):
    prompt_id: str
    person_count: int
    uploaded_filename: str
    output_image_base64: str
    message: str

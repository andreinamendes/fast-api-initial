from pydantic import BaseModel

class promptCreate(BaseModel):
  prompt: str = "a black cat"
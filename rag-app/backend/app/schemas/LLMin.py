from pydantic import BaseModel
from typing import List

class LLMin(BaseModel):
    query: str
    context: str

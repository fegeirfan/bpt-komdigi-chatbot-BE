from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    uploaded_at: str

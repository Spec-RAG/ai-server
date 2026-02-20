from pydantic import BaseModel
from typing import List, Optional


class HistoryMessage(BaseModel):
    role: str  # "human" or "ai"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryMessage]] = []


class ChatResponse(BaseModel):
    answer: str


class SourceDocument(BaseModel):
    index: int
    source_url: str
    page_content: str


class RagResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

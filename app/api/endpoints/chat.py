from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse, RagResponse
from app.services.example import get_answer
from app.services.rag_chain import get_rag_answer

router = APIRouter()

@router.post("/example", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = get_answer(request.message)
    return ChatResponse(answer=answer)

@router.post("/rag", response_model=RagResponse)
async def rag_chat(request: ChatRequest):
    result = get_rag_answer(request.message, request.history)
    return RagResponse(**result)


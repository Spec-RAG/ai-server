from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.example import get_answer

router = APIRouter()

@router.post("/example", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = get_answer(request.message)
    return ChatResponse(answer=answer)

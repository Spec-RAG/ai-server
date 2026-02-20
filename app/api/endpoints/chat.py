from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse, RagResponse
from app.services.example import get_answer
from app.services.rag_chain import get_rag_answer_async, get_rag_answer_stream_with_sources_async
import json

router = APIRouter()

@router.post("/example", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = get_answer(request.message)
    return ChatResponse(answer=answer)

@router.post("/rag", response_model=RagResponse)
async def rag_chat(request: ChatRequest):
    result = await get_rag_answer_async(request.message, request.history)
    return RagResponse(**result)

@router.post("/rag/stream")
async def rag_chat_stream(request: ChatRequest):
    async def event_generator():
        async for event in get_rag_answer_stream_with_sources_async(request.message, request.history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

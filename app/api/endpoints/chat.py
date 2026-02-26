from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse, RagResponse
from app.services.example import get_answer
from app.services.rag_cache_processor import (
    get_rag_answer_cached,
    get_rag_answer_cached_singleflight_in_process,
)
from app.services.history_mapper import build_history_messages
from app.services.query_processor import rewrite_query, resolve_search_query
from app.services.rag_chain import get_rag_answer_async, get_rag_answer_stream_with_sources_async
import json

router = APIRouter()

@router.post("/example", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = get_answer(request.message)
    return ChatResponse(answer=answer)


@router.post("/rag/cache", response_model=RagResponse)
async def rag_chat_cache(request: ChatRequest):
    result = await get_rag_answer_cached(request.message, request.history)
    return RagResponse(**result)


@router.post("/rag/cache-singleflight-inprocess", response_model=RagResponse)
async def rag_chat_cache_singleflight_inprocess(request: ChatRequest):
    result = await get_rag_answer_cached_singleflight_in_process(request.message, request.history)
    return RagResponse(**result)


@router.post("/rag", response_model=RagResponse)
async def rag_chat_raw(request: ChatRequest):
    history_messages = build_history_messages(request.history)
    search_query = await resolve_search_query(request.message, history_messages)
    result = await get_rag_answer_async(
        request.message,
        search_query,
        history_messages,
    )
    return RagResponse(**result)

@router.post("/rag/stream")
async def rag_chat_stream(request: ChatRequest):
    async def event_generator():
        history_messages = build_history_messages(request.history)
        search_query=await resolve_search_query(request.message, history_messages)

        async for event in get_rag_answer_stream_with_sources_async(
            request.message,
            search_query,
            history_messages,
        ):
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

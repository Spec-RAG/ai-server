import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Pinecone Vector Store
# ---------------------------------------------------------------------------
_vectorstore: PineconeVectorStore | None = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )


def get_vectorstore() -> PineconeVectorStore:
    """Return (or lazily create) a PineconeVectorStore connected to the existing index."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    _vectorstore = PineconeVectorStore(
        index=index,
        embedding=_get_embeddings(),
        namespace=settings.PINECONE_NAMESPACE,
        text_key="content",
    )
    return _vectorstore


# ---------------------------------------------------------------------------
# 2. RAG Chain helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    logger.info("=" * 60)
    logger.info(f"Retrieved {len(docs)} documents from Pinecone")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        logger.info(f"  [{i}] project={meta.get('project', 'unknown')}")
        logger.info(f"       heading={meta.get('heading', 'unknown')}")
        logger.info(f"       path={meta.get('path', 'unknown')}")
        logger.info(f"       source_url={meta.get('source_url', 'unknown')}")
        logger.info(f"       preview: {doc.page_content[:200]}...")
    logger.info("=" * 60)
    return "\n\n---\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))


def _build_history_messages(history: List["HistoryMessage"]) -> list:
    """Convert HistoryMessage list to LangChain HumanMessage/AIMessage objects."""
    messages = []
    for h in history:
        role = h.role if hasattr(h, 'role') else h.get('role')
        content = h.content if hasattr(h, 'content') else h.get('content')
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def _build_rag_prompt_and_llm():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_CHAT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 Spring Projects 전문가입니다.\n"
         "아래 참고 문서(context)를 바탕으로 사용자의 질문에 한국어로 답변해주세요.\n"
         "각 문단의 내용을 설명한 후, 해당 문단이 참조한 문서 번호를 문단 마지막에 [1], [2] 형식으로 반드시 표기하세요.\n"
         "참고 문서에 없는 내용이라면 \"해당 내용은 제공된 문서에서 확인할 수 없습니다.\"라고 안내하세요.\n\n"
         "[참고 문서]\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    return prompt, llm


async def _rewrite_query(question: str, history_messages: list) -> str:
    """대화 history를 반영해 벡터 검색에 적합한 독립적인 쿼리로 재작성한다.

    history가 없는 경우(첫 질문)에는 이 함수를 호출하지 않는다.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_CHAT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "아래 대화 기록과 마지막 사용자 질문을 참고하여, "
         "벡터 DB 검색에 사용할 독립적인 한 줄 쿼리를 생성하세요.\n"
         "대화 맥락을 반영해 질문의 의도를 명확히 드러내야 합니다.\n"
         "쿼리 문장만 출력하고 다른 말은 절대 하지 마세요."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    rewritten = await chain.ainvoke({"history": history_messages, "question": question})
    logger.info(f"[QueryRewrite] original='{question}' → rewritten='{rewritten}'")
    return rewritten


# ---------------------------------------------------------------------------
# 3. Public helpers (sync / async / stream)
# ---------------------------------------------------------------------------
def get_rag_answer(question: str, history: list = []) -> dict:
    history_messages = _build_history_messages(history)

    if history_messages:
        import asyncio
        search_query = asyncio.get_event_loop().run_until_complete(
            _rewrite_query(question, history_messages)
        ).strip()
    else:
        search_query = question

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = retriever.invoke(search_query)
    context = _format_docs(docs)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "history": history_messages, "question": question})

    sources = [
        {
            "index": i,
            "source_url": doc.metadata.get("source_url", ""),
            "page_content": doc.page_content,
        }
        for i, doc in enumerate(docs, 1)
    ]
    return {"answer": answer, "sources": sources}


async def get_rag_answer_async(question: str, history: list = []) -> dict:
    history_messages = _build_history_messages(history)

    if history_messages:
        search_query = (await _rewrite_query(question, history_messages)).strip()
    else:
        search_query = question

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = await retriever.ainvoke(search_query)
    context = _format_docs(docs)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "history": history_messages, "question": question})

    sources = [
        {
            "index": i,
            "source_url": doc.metadata.get("source_url", ""),
            "page_content": doc.page_content,
        }
        for i, doc in enumerate(docs, 1)
    ]
    return {"answer": answer, "sources": sources}


def get_rag_answer_stream(question: str, history: list = []):
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = retriever.invoke(question)
    context = _format_docs(docs)
    history_messages = _build_history_messages(history)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()
    for chunk in chain.stream({"context": context, "history": history_messages, "question": question}):
        yield chunk


async def get_rag_answer_stream_async(question: str, history: list = []):
    history_messages = _build_history_messages(history)

    if history_messages:
        search_query = (await _rewrite_query(question, history_messages)).strip()
    else:
        search_query = question

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = await retriever.ainvoke(search_query)
    context = _format_docs(docs)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"context": context, "history": history_messages, "question": question}):
        yield chunk


async def get_rag_answer_stream_with_sources_async(question: str, history: list = []):
    """SSE용 async generator.

    history가 있을 경우 먼저 쿼리를 재작성(Query Rewriting)하여 retrieval 품질을 높인다.

    Yields:
        dict: {"type": "chunk", "content": "<text>"}    — LLM 텍스트 청크
        dict: {"type": "answer", "content": "<full>"}   — 전체 답변 (청크 합산)
        dict: {"type": "sources", "sources": [...]}     — 참고 문서 목록
    """
    history_messages = _build_history_messages(history)

    # history가 있을 때만 쿼리 재작성 (첫 질문은 그대로 사용)
    if history_messages:
        search_query = (await _rewrite_query(question, history_messages)).strip()
    else:
        search_query = question

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = await retriever.ainvoke(search_query)
    context = _format_docs(docs)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()

    full_answer = ""
    async for chunk in chain.astream({"context": context, "history": history_messages, "question": question}):
        full_answer += chunk
        yield {"type": "chunk", "content": chunk}

    yield {"type": "answer", "content": full_answer}

    sources = [
        {
            "index": i,
            "source_url": doc.metadata.get("source_url", ""),
            "page_content": doc.page_content,
        }
        for i, doc in enumerate(docs, 1)
    ]
    yield {"type": "sources", "sources": sources}

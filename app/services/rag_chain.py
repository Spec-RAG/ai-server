import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )


def get_vectorstore() -> PineconeVectorStore:
    """Create a fresh PineconeVectorStore per call to avoid shared async session races."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    return PineconeVectorStore(
        index=index,
        embedding=_get_embeddings(),
        namespace=settings.PINECONE_NAMESPACE,
        text_key="content",
    )


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

# ---------------------------------------------------------------------------
# 3. Public helpers (sync / async / stream)
# ---------------------------------------------------------------------------
async def get_rag_answer(question: str, search_query: str, history_messages: list=[]) -> dict:
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


async def get_rag_answer_async(question: str, search_query: str, history_messages: list=[]) -> dict:
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


def get_rag_answer_stream(question: str, search_query: str, history_messages: list=[]):
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = retriever.invoke(search_query)
    context = _format_docs(docs)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()
    for chunk in chain.stream({"context": context, "history": history_messages, "question": question}):
        yield chunk


async def get_rag_answer_stream_async(question: str, search_query: str, history_messages: list=[]):
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.PINECONE_TOP_K})
    docs = await retriever.ainvoke(search_query)
    context = _format_docs(docs)

    prompt, llm = _build_rag_prompt_and_llm()
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"context": context, "history": history_messages, "question": question}):
        yield chunk


async def get_rag_answer_stream_with_sources_async(
    question: str, search_query: str, history_messages: list=[]
):
    """SSE용 async generator.

    Yields:
        dict: {"type": "chunk", "content": "<text>"}    — LLM 텍스트 청크
        dict: {"type": "answer", "content": "<full>"}   — 전체 답변 (청크 합산)
        dict: {"type": "sources", "sources": [...]}     — 참고 문서 목록
    """
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

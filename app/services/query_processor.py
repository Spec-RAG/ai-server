from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.query_normalizer import normalize_query

logger = logging.getLogger(__name__)


async def rewrite_query(question: str, history_messages: list) -> str:
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
    return rewritten.strip()


async def build_search_query(question: str, history_messages: list) -> str:
    """rewrite + normalize 후 최종 검색용 쿼리 반환."""
    if history_messages:
        search_query = (await rewrite_query(question, history_messages)).strip()
    else:
        search_query = question

    normalized = normalize_query(search_query)
    logger.info("[QueryNormalize] raw='%s' normalized='%s'", search_query, normalized)
    return normalized or search_query

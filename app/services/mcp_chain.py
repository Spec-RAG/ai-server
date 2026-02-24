import json
import logging
from typing import AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from app.core.config import settings

logger = logging.getLogger(__name__)

@tool
async def spring_docs_search(query: str) -> list[dict]:
    """Search the Spring official documentation (docs.spring.io) for the given query.
    Use this tool whenever you need to find factual information, guides, or API details about Spring Framework, Spring Boot, or any other Spring projects.
    """
    search_query = f"{query} site:docs.spring.io".strip()
    logger.info(f"🤖 [Agent Decision] Calling Tavily Search with query: {search_query}")

    try:
        tavily_tool = TavilySearch(
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=settings.TAVILY_API_KEY),
            max_results=5,
            include_raw_content=True,
            include_domains=["docs.spring.io"],
        )
        results = await tavily_tool.ainvoke({"query": search_query})
        
        if not results:
            logger.info("🤖 [Agent Observation] Tavily Search returned empty results.")
            return []
            
        logger.info(f"🤖 [Agent Observation] Tavily Search returned {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"[Tavily Search Error] {e}")
        return []

def _build_llm_with_tools():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_CHAT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )
    tools = [spring_docs_search]
    # bind_tools를 통해 LLM이 도구를 인식하고 호출을 결정할 수 있도록 합니다.
    return llm.bind_tools(tools), tools

async def get_tavily_rag_answer_stream_with_sources_async(
    question: str, search_query: str, history_messages: list=[]
) -> AsyncGenerator[dict, None]:
    """LangChain의 bind_tools를 이용해 순수 Python 루프로 도구 호출과 스트리밍을 구현합니다."""
    
    llm_with_tools, tools = _build_llm_with_tools()
    tool_map = {tool.name: tool for tool in tools}
    
    system_prompt = SystemMessage(content=(
        "당신은 친절한 Spring Projects 전문가 챗봇입니다.\n"
        "사용자의 질문에 한국어로 답변해주세요.\n"
        "Spring과 관련된 구체적인 정보, 사용법, API나 최신 변경점 등에 대한 답변을 할 때는 "
        "반드시 `spring_docs_search` 도구를 사용하여 공식 문서를 검색한 후 그 결괏값을 바탕으로 답변하세요.\n"
        "도구를 사용했다면, 답변의 각 문단 끝에 해당 내용이 참조한 문서 번호를 [1], [2] 형식으로 표시해주세요.\n"
        "단순한 인사나 검색이 필요 없는 일반적인 대화라면 도구를 사용하지 않고 바로 답변하셔도 됩니다."
    ))

    # LLM에게 전달할 전체 메시지 목록 구성
    messages = [system_prompt] + history_messages + [HumanMessage(content=question)]
    sources_collected = []
    
    logger.info("🤖 [Model] Processing user request...")

    # 1단계: 모델에게 질문을 전달하여 답변 또는 도구 호출 여부를 결정
    response: AIMessage = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # 2단계: 도구 호출(Tool Calls)이 있다면 실행하여 결과를 덧붙이고 다시 모델 호출
    if response.tool_calls:
        for tool_call in response.tool_calls:
            selected_tool = tool_map[tool_call["name"]]
            
            # 비동기로 도구 실행
            tool_result = await selected_tool.ainvoke(tool_call["args"])
            
            # 소스 수집용 (TavilySearch는 dict를 반환하고, 내부에 'results' 리스트를 가집니다)
            # 기존 TavilySearchResults는 list[dict]를 반환했었음.
            if isinstance(tool_result, dict) and "results" in tool_result:
                results_list = tool_result["results"]
            elif isinstance(tool_result, list):
                results_list = tool_result
            else:
                results_list = []
                
            for idx, doc in enumerate(results_list, start=len(sources_collected) + 1):
                sources_collected.append({
                    "index": idx,
                    "source_url": doc.get("url", ""),
                    "page_content": doc.get("content", "")[:1000],
                })

            # 모델에게 도구 실행 결과를 알려주기 위한 메시지 추가
            tool_message = ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_message)
        
        # 3단계: 도구 실행 결과(Observation)를 바탕으로 최종 답변을 스트리밍
        full_answer = ""
        # StrOutputParser를 파이프에 추가하면 안전한 스트링만 넘어옵니다
        final_chain = llm_with_tools | StrOutputParser()
        
        async for chunk_text in final_chain.astream(messages):
            if chunk_text:
                full_answer += chunk_text
                yield {"type": "chunk", "content": chunk_text}
        
        yield {"type": "answer", "content": full_answer}

    else:
        # 도구 호출이 필요 없는 경우(단순 인사 등)
        text_content = ""
        if isinstance(response.content, str):
            text_content = response.content
        elif isinstance(response.content, list):
            for item in response.content:
                if isinstance(item, str):
                    text_content += item
                elif isinstance(item, dict) and "text" in item:
                    text_content += item["text"]
        else:
            text_content = str(response.content)

        yield {"type": "chunk", "content": text_content}
        yield {"type": "answer", "content": text_content}

    # 수집된 출처 목록 리턴
    yield {"type": "sources", "sources": sources_collected}
    logger.info("🤖 [Model Stream Completed]")

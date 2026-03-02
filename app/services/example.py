from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

def get_gemini_chain():
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        "당신은 Spring Framework 전문가입니다. 다음 질문에 한국어로 답변해주세요: {question}"
    )
    
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0
    )
    
    # Define the output parser
    parser = StrOutputParser()
    
    # Create the chain
    chain = prompt | llm | parser
    
    return chain

def get_answer(question: str) -> str:
    chain = get_gemini_chain()
    return chain.invoke({"question": question})

async def get_answer_async(question: str) -> str:
    chain = get_gemini_chain()
    return await chain.ainvoke({"question": question})

def get_answer_stream(question: str):
    chain = get_gemini_chain()
    for chunk in chain.stream({"question": question}):
        yield chunk

async def get_answer_stream_async(question: str):
    chain = get_gemini_chain()
    async for chunk in chain.astream({"question": question}):
        yield chunk

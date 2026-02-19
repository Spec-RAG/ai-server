from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from typing import List, Optional

# List of Spring Projects
SPRING_PROJECTS = [
    "spring boot", "spring framework", "spring data", "spring cloud", "spring cloud data flow",
    "spring security", "spring authorization server", "spring for graphql", "spring session",
    "spring integration", "spring hateoas", "spring modulith", "spring rest docs", "spring ai",
    "spring batch", "spring amqp", "spring for apache kafka", "spring ldap",
    "spring for apache pulsar", "spring shell", "spring statemachine", "spring web flow",
    "spring web services"
]

def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0
    )

def classify_project(question: str) -> str:
    """Classify the user question into one of the Spring Projects."""
    llm = get_gemini_model()
    
    prompt = ChatPromptTemplate.from_template(
        """
        Classify the following question into one of these Spring projects:
        {projects}
        
        If the question is not related to any specific project, or if you are unsure, return 'spring framework'.
        Return ONLY the project name in lowercase.
        
        Question: {question}
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    
    project = chain.invoke({"projects": ", ".join(SPRING_PROJECTS), "question": question})
    return project.strip().lower()

def get_system_prompt(project_name: str) -> str:
    if project_name not in SPRING_PROJECTS:
        project_name = "spring framework"
    return f"당신은 {project_name} 전문가입니다. 다음 질문에 한국어로 답변해주세요."

def get_answer(question: str, history: List = []) -> str:
    # 1. Determine System Prompt
    if history:
        # If history exists, we assume the context is already set or we use a generic fallback
        # that doesn't override the existing conversation flow. 
        # However, the user request says: "history가 있는 경우에는 해당 system prompt를 따름"
        # Since we don't store the system prompt in the history list (usually), 
        # we will use a generic "Spring Expert" prompt to avoid re-classifying and changing topics abruptly,
        # relying on the history messages to provide the specific context.
        system_content = "당신은 Spring 전문가입니다. 이전 대화의 맥락을 고려하여 한국어로 답변해주세요."
    else:
        # No history, so we classify the intent
        project_name = classify_project(question)
        system_content = get_system_prompt(project_name)

    # 2. Create the Chat Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    llm = get_gemini_model()
    chain = prompt | llm | StrOutputParser()
    
    # 3. Invoke
    return chain.invoke({"question": question, "history": history})

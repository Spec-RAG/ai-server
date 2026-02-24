from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Spring QnA Chatbot"
    API_STR: str = "/api"
    # OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_CHAT_MODEL: str = "gemini-3-flash-preview"
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "spring-docs"
    PINECONE_NAMESPACE: str
    PINECONE_TOP_K: int = 4
    TAVILY_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

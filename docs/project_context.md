# Project Context: FastAPI Spring QnA Chatbot

## Project Goal
To build a chatbot server using FastAPI and LangChain that can answer questions about the Spring Framework. The system aims to eventually incorporate RAG (Retrieval-Augmented Generation) using Spring's official documentation.

## Tech Stack
- **Language**: Python 3.10+
- **Web Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **LLM Provider**: Google Gemini / OpenAI (GPT-3.5/4)
- **Configuration**: Pydantic Settings
- **Server**: Uvicorn

## Architecture
The project follows a standard service-layer architecture:
- **`app/api/`**: Contains API route handlers.
- **`app/core/`**: Contains core configuration and settings.
- **`app/schemas/`**: Contains Pydantic models for data validation and serialization.
- **`app/services/`**: Contains the business logic, specifically the LangChain interaction code.

## Key Features
1.  **Chat Endpoint**: `/api/chat` - Accepts a message and returns an answer.
2.  **Spring QnA**: Specialized in answering Spring Framework related questions.
3.  **RAG (Planned)**: Will index Spring documentation for more accurate answers.

## Directory Structure
- `app/main.py`: Application entry point.
- `app/api/endpoints/chat.py`: The main chat handler.
- `app/services/chain.py`: Logic for constructing and invoking the LangChain chain.

## Environment Variables
Required environment variables in `.env`:
- `OPENAI_API_KEY`: API key for OpenAI.
- `GEMINI_API_KEY`: API key for Google Gemini.
- `PROJECT_NAME`: Name of the project.
- `API_STR`: API prefix (default: `/api`).

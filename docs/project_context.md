# Project Context: FastAPI Spring QnA Chatbot

## Project Goal
To build a chatbot server using FastAPI and LangChain that can answer questions about the Spring Framework.
The system uses RAG (Retrieval-Augmented Generation) with Spring's official documentation stored in a Pinecone vector database.

## Tech Stack
- **Language**: Python 3.11.11 (managed via `uv`)
- **Web Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **LLM Provider**: Google Gemini (`gemini-3-flash-preview` for chat, `gemini-embedding-001` for embeddings)
- **Vector DB**: Pinecone
- **Configuration**: Pydantic Settings (`pydantic-settings`)
- **Package Manager**: `uv` (see `pyproject.toml` / `uv.lock`)
- **Server**: Uvicorn

## Architecture
The project follows a standard service-layer architecture:
- **`app/api/`**: API route handlers and router registration.
- **`app/core/`**: Core configuration and settings (`config.py`).
- **`app/schemas/`**: Pydantic models for request/response validation.
- **`app/services/`**: Business logic and LangChain chain implementations.

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/example` | Basic LangChain chain (non-RAG) |
| POST | `/api/chat/rag` | RAG chain with Pinecone retrieval and source citations |
| POST | `/api/chat/rag/stream` | RAG chain via SSE streaming (chunk → answer → sources) |

### Request Schema (`ChatRequest`)
```json
{
  "message": "Spring Security란?",
  "history": [
    { "role": "human", "content": "이전 질문" },
    { "role": "ai", "content": "이전 답변" }
  ]
}
```

### RAG Response Schema (`RagResponse`)
```json
{
  "answer": "Spring Security는 인증과 인가를 제공합니다. [1] 필터 체인 기반으로 동작합니다. [2]",
  "sources": [
    { "index": 1, "source_url": "https://docs.spring.io/...", "page_content": "..." },
    { "index": 2, "source_url": "https://docs.spring.io/...", "page_content": "..." }
  ]
}
```
- `answer` 내 문단 끝에 `[n]` 형식으로 참조 문서 번호 표기
- `sources` 리스트로 각 번호가 어떤 문서를 참조하는지 확인 가능

### SSE Stream 이벤트 형식 (`/rag/stream`)
```
data: {"type": "chunk",   "content": "Spring"}          ← 텍스트 청크 (반복)
data: {"type": "answer",  "content": "Spring Security..."}  ← 전체 답변 (1회)
data: {"type": "sources", "sources": [...]}              ← 참고 문서 목록 (1회)
data: [DONE]
```
- `chunk`: LLM이 생성하는 텍스트 조각, 화면에 순차적으로 append
- `answer`: 모든 chunk를 합산한 최종 완성 답변
- `sources`: `index`, `source_url`, `page_content` 포함

## Directory Structure
```
app/
├── main.py                    # Application entry point
├── api/
│   ├── api.py                 # Router registration (/chat prefix)
│   └── endpoints/
│       └── chat.py            # /example, /rag endpoint handlers
├── core/
│   └── config.py              # Settings via pydantic-settings
├── schemas/
│   └── chat.py                # ChatRequest, ChatResponse, SourceDocument, RagResponse
└── services/
    ├── chain.py               # Basic LangChain chain (non-RAG)
    ├── example.py             # Example chain usage
    └── rag_chain.py           # RAG chain: Pinecone retrieval + Gemini + citation
```

## Key Service: `rag_chain.py`
- **Vector store**: Pinecone (`PineconeVectorStore`), `text_key="content"`, top-k=4
- **Embeddings**: `gemini-embedding-001`
- **LLM**: `gemini-3-flash-preview`, `temperature=0`
- **Chat history**: Supports `HumanMessage` / `AIMessage` via `MessagesPlaceholder`
- **Citation**: Each retrieved doc is numbered `[1]...[n]` in the context; the prompt instructs the LLM to cite sources inline per paragraph
- **Functions**:
  - `get_rag_answer(question, history)` → `dict`
  - `get_rag_answer_async(question, history)` → `dict`
  - `get_rag_answer_stream(question, history)` → generator
  - `get_rag_answer_stream_async(question, history)` → async generator

## Environment Variables
Required in `.env`:
| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `spring-docs`) |
| `PINECONE_NAMESPACE` | Pinecone namespace (required, no default) |
| `PROJECT_NAME` | Project name (default: `Spring QnA Chatbot`) |
| `API_STR` | API prefix (default: `/api`) |

## Development
```bash
# Install dependencies
uv sync

# Run dev server
uv run uvicorn app.main:app --reload
```

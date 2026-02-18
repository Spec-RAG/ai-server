# FastAPI Spring QnA Chatbot

A FastAPI-based chatbot server that uses LangChain to answer questions about the Spring Framework.

## Quick Start

1.  **Install uv** (if not installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

3.  **Add Libraries**:
    To add a new library (e.g. numpy), run:
    ```bash
    uv add numpy
    ```
    To add a development library (e.g. pytest), run:
    ```bash
    uv add --dev pytest
    ```

4.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env and set your OPENAI_API_KEY
    ```

4.  **Run Server**:
    ```bash
    uv run uvicorn app.main:app --reload
    ```

4.  **API Documentation**:
    - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure

```bash
.
├── app/
│   ├── api/               # API Endpoints & Router
│   │   ├── endpoints/     # 실제 비즈니스 로직과 연결된 API 핸들러 (예: chat.py)
│   │   └── api.py         # 라우터 통합 및 설정
│   ├── core/              # 프로젝트 핵심 설정 (환경변수, 공통 유틸리티)
│   ├── schemas/           # Pydantic 데이터 모델 (요청/응답 검증 및 문서화)
│   ├── services/          # 비즈니스 로직 및 외부 서비스 연동 (LangChain 등)
│   └── main.py            # FastAPI 애플리케이션 진입점 및 설정
├── docs/                  # 프로젝트 문서 및 AI 컨텍스트 파일
├── pyproject.toml         # 프로젝트 설정 및 의존성 관리 (uv)
├── uv.lock                # 의존성 잠금 파일 (버전 고정)
├── .python-version        # Python 버전 명시 (3.11)
├── .env.example           # 환경 변수 템플릿
└── README.md              # 프로젝트 설명 및 실행 가이드
```

## Project Context for AI
For detailed project context, architecture, and design decisions, please refer to [docs/project_context.md](docs/project_context.md).

# SpringDocsAI


## Overview

SpringDocsAI는 Spring 공식 문서를 기반으로 한국어 질의응답을 제공하는 RAG 서비스입니다.  
사용자는 Spring Boot, Spring Framework, Spring Data 등 공식 문서에 대해 자연어로 질문할 수 있고, 시스템은 관련 문서를 검색·재정렬한 뒤 근거와 함께 답변을 생성합니다.  

---

## Core Features

- Spring 공식 문서 기반 한국어 질의응답
- 근거 문서 기반 응답 생성
- 문서 수집, 정제, 청킹, 임베딩 파이프라인 구축
- 벡터 검색 및 재정렬 기반 RAG 응답 생성
- 캐시 및 동시성 제어를 통한 요청 최적화
- MCP 기반 외부 도구 연동 구조 적용

---

## UI/UX
### 채팅 인터페이스
<img width="1909" height="920" alt="Image" src="https://github.com/user-attachments/assets/3e8ab87b-2e78-4070-8c21-32929937d449" /></br>
### 응답 링크 클릭 시 화면
<img width="1910" height="923" alt="Image" src="https://github.com/user-attachments/assets/f3d4cb94-bb9c-4ae5-aaec-df71cbbdc7d6" />


---

## Tech Stack

### Backend
- Python
- FastAPI

### AI / RAG
- LangChain
- Gemini
- Pinecone

### Database
- redis

### Infra / Monitoring
- Docker
- Prometheus
- Grafana
- k6

### Protocol / Tooling
- MCP
- JSON-RPC 2.0

---



### Components

- AI Server  
  질문 정규화, 검색어 생성, 벡터 검색, 재정렬, 답변 생성을 담당

- Redis  
  캐시와 동시성 제어를 위한 저장소

- Pinecone  
  임베딩된 문서 벡터 저장 및 유사도 검색을 담당

- Gemini API  
  질의 재작성, 임베딩 생성, 답변 생성을 담당



---

## Backend Engineering Highlights

### 1. Spring 공식 GitHub 문서 기반 RAG 파이프라인 구축
- Spring Boot, Spring Framework, Spring Data 공식 GitHub 저장소를 수집 대상으로 선정
- 공식 문서를 수집하고 검색 가능한 청크 단위로 분할
- 메타데이터와 함께 임베딩 후 Pinecone에 저장
- 벡터 검색, 재정렬, 답변 생성으로 이어지는 RAG 파이프라인 구성
- 결과
  - Spring 공식 문서 근거 기반의 한국어 질의응답 서비스 구현
  - 단순 생성형 응답이 아닌 검색 기반 답변 흐름 구축

### 2. 다층 캐시 전략으로 반복 질의 비용 절감
- 동일하거나 유사한 질문에 대해 임베딩, 검색, 생성이 반복 수행되는 구조
- Answer Cache, Retrieval Cache, Embedding Cache로 캐시 계층 분리
- canonical query 기반 키를 사용해 질의 표현 차이에도 캐시 재사용 가능하도록 구성
- 결과
  - 반복 질의에 대한 외부 API 호출 수와 응답 지연 감소
  - 고비용 RAG 연산의 중복 실행 완화

### 3. Redis 분산락으로 캐시 생성 구간 중복 실행 제어
- 캐시 미스 상황에서 동일 질문이 동시에 유입되면 동일한 RAG 연산이 중복 수행될 수 있음
- Redis 기반 분산락을 도입해 동일 키에 대한 캐시 생성 작업을 단일화
- 결과
  - 동일 질문 동시 유입 시 중복 생성과 중복 외부 호출 억제


### 4. In-Process Singleflight와 세마포어 기반 백프레셔 적용
- 서로 다른 질문이 동시에 몰리면 분산락만으로는 전체 실행량 폭증을 제어하기 어려움
- 프로세스 내부에서는 Future 기반 singleflight로 동일 작업 대기를 공유
- 전체 RAG 실행 구간에는 asyncio.Semaphore를 적용해 동시 실행 상한을 강제
- 결과
  - 프로세스 내부 중복 실행을 추가로 감소
  - 처리 가능한 범위를 넘는 스파이크 요청에서 서버 안정성 확보

### 5. 부하 테스트와 모니터링 기반 병목 분석 체계 구축
- k6로 동시 요청 수와 arrival rate를 단계적으로 증가시키며 부하 테스트 수행
- p95, 오류율, 처리량, CPU, 메모리 사용량을 함께 분석
- Prometheus와 Grafana를 활용해 애플리케이션 및 인프라 지표를 시각화
- 결과
  - 병목 구간과 안정 처리 가능 범위를 수치 기반으로 판단
  - 최적화 전후 차이를 정량적으로 비교할 수 있는 환경 마련

### 6. MCP 기반 외부 도구 연동 구조 적용
- MCP는 AI 모델과 외부 데이터, 파일, API, 도구를 연결하기 위한 표준 프로토콜
- LLM과 도구 간 통신 방식을 표준화해 도구별 개별 연동 부담을 줄임
- JSON-RPC 2.0 기반 요청과 응답 구조를 사용해 도구 목록 조회와 호출 흐름을 구성
- 클라이언트가 `tools/list`로 사용 가능한 도구와 입력 스키마를 확인하고 `tools/call`로 필요한 도구를 실행하도록 설계
- 결과
  - 외부 도구 연동 방식을 표준화해 확장성과 유지보수성 향상
  - 도구별 입출력 형식을 일관되게 관리할 수 있는 구조 마련

---

## Project Structure

```text
ai-server/
├── app/                # FastAPI 애플리케이션, API 엔드포인트, RAG 실행 로직
├── data/               # 수집·가공된 문서 데이터와 임베딩 결과 저장
├── data_pipeline/      # 문서 수집, 정제, 청킹, 임베딩 파이프라인
├── docs/               # 프로젝트 문서
└── docker-compose.yml  # AI 서버와 Redis 인프라 실행 설정

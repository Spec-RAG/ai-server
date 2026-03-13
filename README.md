# 어슬렁 (Eoseulleong) — AI 여행 추천 웹 서비스

> AI 기반 여행 추천 및 일정 생성 플랫폼
> 사용자의 취향과 입력 정보를 바탕으로 여행지를 추천하고, 개인 맞춤형 여행 계획을 생성하는 웹 서비스입니다.

---

## 🚀 Overview

어슬렁은 **AI 추천 시스템과 대화형 인터페이스**를 결합하여
사용자가 쉽고 빠르게 여행 계획을 세울 수 있도록 돕는 서비스입니다.

화면설계서(https://www.notion.so/2f9136f77e8a803e892bf59cd0dc22aa?v=2f9136f77e8a80d38e9e000ca883d22c&source=copy_link)

API명세서(https://www.notion.so/API-2f4136f77e8a80559f62f50d1aacc770?source=copy_link)

### 핵심 기능

* ✨ AI 기반 여행지 추천
* 🗺️ 개인 맞춤 여행 일정 생성
* 🤖 자연어 기반 챗봇 인터페이스
* 👤 사용자 계정 및 플랜 관리

---

## 🧩 Tech Stack

### Frontend

* **React + TypeScript**
* **TanStack Router**
* **axios + ts-rest**
* **zod**
* **TanStack Query (React Query)**
* **Tailwind CSS**
* **shadcn/ui**
* **Vite**

### Architecture

* REST API 기반 클라이언트–서버 구조
* 상태 관리: React Query 중심의 서버 상태 관리
* 컴포넌트 기반 UI 아키텍처
* 재사용 가능한 디자인 시스템 구축

---

## 🏗️ Frontend Architecture

src/
├── components/     # 공통 UI 컴포넌트
├── pages/          # 라우트 기반 페이지
├── hooks/          # 커스텀 훅
├── stores/         # 클라이언트 상태 관리
├── apis/           # API 클라이언트
└── types/          # 타입 정의

주요 설계 목표:

* 확장 가능한 컴포넌트 구조
* 명확한 책임 분리
* 타입 안정성 보장
* 유지보수 친화적인 코드 구조

---

## ⚙️ Frontend Engineering Highlights (Architecture & Implementation)

### 1) 계약 기반 API 설계 (ts-rest + Zod)

프론트–서버 간 계약(Contract)을 **ts-rest**로 정의하고, 요청/응답 스키마는 **Zod**로 검증하여
**타입 안정성 + 런타임 안정성**을 동시에 확보했습니다.

- API contract를 단일 소스로 관리 (endpoint, params, body, response schema)
- DTO 변경 시 컴파일 타임에 영향 범위를 즉시 확인 가능
- Zod 스키마로 런타임 데이터 shape 방어 (예상치 못한 응답/결측값 대응)

> 결과: "API 문서/타입/검증"이 분리되지 않고 한 흐름으로 유지되며, 리팩토링 비용이 크게 감소합니다.

---

### 2) Axios 인스턴스 계층화 + 큐 기반 요청 제어

네트워크 계층을 axios instance로 분리하고, 인증/재발급 시나리오에서 흔히 발생하는
**동시 요청 폭주 / 토큰 재발급 중복 호출** 문제를 **큐(Queue) 기반**으로 제어했습니다.

- 역할별 axios instance 분리 (일반 API / 인증 전용 / 필요 시 SSE 등)
- interceptor에서 401/만료 대응 → refresh/reissue 단일화
- 재발급 진행 중 들어오는 요청은 큐에 적재 후, 재발급 성공 시 순차 재시도
- 재발급 실패 시 큐 일괄 실패 처리 및 사용자 상태 초기화

> 결과: 인증 경계에서 "한 번만 재발급하고 나머지는 대기"하는 안정적인 네트워크 흐름을 보장합니다.

---

### 3) 서버 상태 관리 전략 (TanStack Query)

서버 상태는 TanStack Query(React Query)로 일원화하여
캐싱/무효화/재시도/로딩 표현을 예측 가능하게 구성했습니다.

- 도메인 단위 Query Key 네이밍 전략 (`PLAN_QK`, `QK.me()` 등)
- mutation 성공 시 invalidate/refetch 범위를 명확히 분리
- keepPreviousData 등으로 리스트 UX 안정화
- 에러 시 공통 토스트/리다이렉트 정책 적용 (사용자 경험 일관성)

> 결과: 화면별로 흩어지는 fetch 로직을 줄이고, "데이터의 생명주기"가 코드로 드러나게 됩니다.

---

### 4) 단계 기반 UX 상태 유지 (sessionStorage + Context Provider)

추천 플로우(입력 → 추천 결과 → 선택 → 일정 생성)는 페이지 전환이 잦아
사용자 입력/선택 상태가 쉽게 유실됩니다. 이를 해결하기 위해:

- 단계별 입력/선택 상태를 **sessionStorage 기반 store**로 영속화  
  (새로고침/뒤로가기/재진입에도 UX 유지)
- 페이지 레이아웃 단위로 **Context Provider**를 두어 단계 흐름에서 필요한 상태/액션을 주입
- "UI 단계"와 "저장된 상태"를 분리해, 각 화면은 자신의 책임(표현/검증/전환)만 담당

예시(개념):
- `ModelInputStore` : 지역/기간/예산/취향 등 입력 상태 보존
- `ModelHistoryStore` : 최근 추천/선택 이력
- Layout Provider : 현재 단계에서 필요한 핸들러/전환 로직 제공

> 결과: 복잡한 다단계 플로우에서도 상태 유실 없이 자연스러운 전환과 복구가 가능합니다.

---

### 5) 라우팅 구조 설계 (TanStack Router)

TanStack Router의 중첩 라우팅을 활용해,
단계 플로우와 공통 레이아웃을 분리했습니다.

- Layout(상위)에서 공통 컨텍스트/상태 주입
- Page(하위)는 화면 단위 책임 유지
- 라우트 params 기반으로 "플랜 상세/모델 진행" 등 화면을 명확히 구분

> 결과: 화면이 100개 이상으로 확장되더라도 라우트/레이아웃/상태 책임이 흐트러지지 않습니다.

---

### 6) UI 시스템화 (Tailwind + shadcn/ui + 공통 컴포넌트)

디자인 시스템 토큰과 공통 컴포넌트를 기반으로 UI 일관성을 유지했습니다.

- `Column/Row` 같은 레이아웃 컴포넌트로 화면 구조 표준화
- 공통 Button/Text/Badge/ImageBox 등 재사용 컴포넌트 구축
- shadcn/ui 기반으로 접근성/인터랙션 품질 확보
- 로딩/빈 상태/토스트 등 "상태 UI" 패턴 통일

> 결과: 새로운 화면 추가 시 "조립" 중심으로 개발 가능하며, UI 품질이 균일해집니다.

---

## 📌 Future Improvements

* 추천 알고리즘 고도화
* 사용자 경험 개선
* 성능 최적화
* 모바일 UX 강화

# LLM의 함수 호출(Tool Calling) 작동 원리

LLM(대규모 언어 모델)은 근본적으로 '텍스트 예측 모델'이기 때문에 혼자서 인터넷 검색을 하거나 파이썬 함수를 직접 실행할 수 없습니다. 대신 **'특수한 프롬프트 구조'**와 **'앱(LangChain 등)과 LLM 간의 핑퐁(Ping-Pong)'**이라는 방식을 통해 도구를 호출하고 그 결과를 답변에 반영합니다.

LangChain의 `bind_tools`를 사용했을 때 실제로 어떤 JSON 메시지들이 오가는지 단계별로 살펴보겠습니다.

---

## 1. 기본 세팅: 시스템 프롬프트 주입

파이썬 코드에서 `llm.bind_tools([spring_docs_search])`를 실행하면, LangChain은 모든 사용자의 질문(프롬프트) 최상단에 우리가 연결한 도구(함수)의 이름, 설명, 매개변수 형식을 JSON 형태로 숨겨서 LLM에게 보냅니다.

**[서버 ➡️ LLM] 시스템 프롬프트 주입**
```json
{
  "role": "system",
  "content": "You are a helpful assistant. You have access to the following tools:\n\n1. Name: spring_docs_search\n   Description: Search the Spring official documentation (docs.spring.io) for the given query. Use this tool whenever you need to find factual information, guides, or API details about Spring Framework, Spring Boot, or any other Spring projects.\n   Parameters: {'type': 'object', 'properties': {'query': {'type': 'string'}}}"
}
```
*💡 LLM은 이 시스템 프롬프트를 보고 자기 자신에게 `spring_docs_search`라는 기능이 있다는 사실과 그 사용법(query 파라미터 구조)을 인지하게 됩니다.*

---

## 2. 케이스별 프롬프트 구조와 동작 과정

### Case A. 일상 대화일 때 (도구 미사용)

**1️⃣ 사용자의 질문 전송**
```json
{ "role": "user", "content": "안녕? 밥은 먹었어?" }
```

**2️⃣ LLM의 추론 및 응답**
LLM은 "안녕이라는 인사에는 스프링 공식 문서 검색(`spring_docs_search`)이 필요 없네. 그냥 일상어로 대답해야겠다." 라고 판단합니다.

```json
{
  "role": "assistant",
  "content": "안녕하세요! 저는 AI라서 밥을 먹지는 않지만, 당신의 하루가 든든하기를 바랍니다. 무엇을 도와드릴까요?",
  "tool_calls": [] 
}
```
*💡 `tool_calls` 배열이 비어있으므로, 파이썬 서버도 "도구 사용이 필요 없구나" 하고 `content` 텍스트만 사용자에게 반환하고 종료합니다.*

---

### Case B. 검색이 필요할 때 (도구 호출 및 재실행)

**1️⃣ 사용자의 질문 전송**
```json
{ "role": "user", "content": "스프링 부트 액추에이터 커스텀 엔드포인트 알려줘" }
```

**2️⃣ 첫 번째 LLM 응답 (도구 실행 요청)**
질문을 읽은 LLM은 "이건 내가 알고 있는 도구를 써야 정확해!"라고 예측하고, 답변 텍스트를 생성하는 대신 **도구를 실행해 달라는 특수 JSON 포맷**을 반환합니다.

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_abc12345",
      "type": "function",
      "function": {
        "name": "spring_docs_search",
        "arguments": "{\"query\": \"Spring Boot Actuator custom endpoint\"}"
      }
    }
  ]
}
```
*💡 여기서 LLM의 역할은 잠시 멈춥니다. 자기가 직접 검색을 못 하니, 파이썬 코드(서버)에게 "내 대신 저 인자(`arguments`)로 검색 기능 좀 실행해 줘!"라고 텍스트만 던져준 것입니다.*

**3️⃣ 파이썬 서버의 개입 (Local Execution)**
우리의 애플리케이션(파이썬 서버) 코드는 위 `tool_calls` 결과를 보고, 자신의 로컬 메모리에 있는 `spring_docs_search("Spring Boot Actuator custom endpoint")` 파이썬 함수를 실제로 실행하여 인터넷에서 문서를 가져옵니다.

**4️⃣ 두 번째 프롬프트 전송 (검색 결과 주입 후 재질문 ⭐️)**
파이썬 서버는 **[최초 질문 + LLM의 도구 요청 기록 + 실제 파이썬이 받아온 검색 결과]**를 하나의 채팅 기록으로 이어서 다시 LLM에게 **통째로** 전송합니다.

**[서버 ➡️ LLM] 두 번째 질의 (전체 대화 내역)**
```json
[
  { "role": "system", "content": "(도구 설명...)" },
  { "role": "user", "content": "스프링 부트 액추에이터 커스텀 엔드포인트 알려줘" },
  { 
    "role": "assistant", 
    "content": "", 
    "tool_calls": [{"id": "call_abc12345", "name": "spring_docs_search", "arguments": "..."}] 
  },
  {
    "role": "tool",
    "tool_call_id": "call_abc12345",
    "content": "[{\"url\": \"https://docs.spring.io/...\", \"content\": \"액추에이터에서 커스텀 엔드포인트를 만들려면 클래스에 @Endpoint를 붙이세요...\"}]"
  }
]
```

**5️⃣ LLM의 최종 텍스트 생성 (스트리밍)**
다시 프롬프트를 건네받은 LLM은 전체 문맥을 처음부터 다시 읽습니다. 
*"아, 내가 검색해 달라고 한 거에 대한 결과(`role: tool`)가 드디어 도착했구나! 이 내용을 바탕으로 사용자에게 줄 한국어 답변을 예쁘게 만들어야지!"* 

그리고 마침내 아래처럼 텍스트를 출력하기 시작합니다.
```json
// 최종 출력 텍스트 청크들
{"role": "assistant", "content": "스프링 부트 액추에이터에서 "}
{"role": "assistant", "content": "커스텀 엔드포인트를 열기 위해서는 "}
{"role": "assistant", "content": "`@Endpoint` 어노테이션을 사용하면 됩니다. ..."}
```

---

## 요약

이처럼 **"에이전트(Agent)"** 방식은 파이프라인이 한 번에 끝나지 않습니다. 
LangChain의 `llm.bind_tools()`를 엮으면, 복잡한 if/else 처리나 프롬프트 조립 과정을 LangChain이 백그라운드에서 표준화된 JSON 메시지 규약(system, user, assistant, tool)으로 맞춰서 구글/OpenAI 서버와 통신해 줍니다. 결국 사용자는 한 번 질문했지만 백그라운드에서는 **[LLM의 검색 결심 -> 앱의 함수 대리 실행 -> 결과를 포함한 2차 프롬프트 재전송 -> 최종 대답 완성]** 이라는 핑퐁 과정이 순식간에 일어나는 것입니다.

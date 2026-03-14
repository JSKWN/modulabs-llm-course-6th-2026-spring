***

# 📚 LLM 교육과정 1주차 학습 내용 정리

> **교육 기간**: 2026-03-10 ~ 2026-03-14 (5일차)
> **핵심 라이브러리**: `openai`, `langchain-openai`, `langchain-core`, `gradio`

***

## 전체 학습 흐름

| 단계 | 날짜 | 핵심 내용 |
| :-- | :-- | :-- |
| 1일차 | 03-10 | Python 기초 + LangChain 첫 접촉 (`ChatOpenAI.invoke`) [^4] |
| 2일차 | 03-11 | Python 문법 심화 (for/if/dict/함수/파일I/O) + API 파라미터 이해 [^2] |
| 3일차 | 03-12 | OpenAI API 직접 호출 + temperature/top_p/penalty 실험 + 스트리밍 + 대화 유지 [^1] |
| 4일차 | 03-13 | LangChain 심화 (invoke/stream/batch, PromptTemplate, OutputParser, bind) [^3] |
| 5일차 | 03-14 | LCEL 체인 패턴 (Runnable*) + Gradio UI + LLM 챗봇 완성 [^5] |


***

## 🔑 STAGE 1: OpenAI API 직접 호출

### 환경 설정 및 API 키 로드

실습에서는 Google Colab Secrets를 통해 API 키를 안전하게 불러왔습니다.[^4][^1]

```python
from google.colab import userdata
import os

api_key = userdata.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key  # 환경변수로 등록하면 OpenAI() 자동 인식
```

> **핵심**: `os.environ["OPENAI_API_KEY"]`를 먼저 실행하면, `OpenAI()`나 `ChatOpenAI()` 초기화 시 `api_key` 인자 없이도 자동으로 키를 가져옴.

### 기본 API 호출 및 응답 구조

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "안녕하세요."}]
)

print(response.choices[^0].message.content)   # 텍스트만 출력
print(response.usage.total_tokens)           # 사용 토큰 수
print(response.choices[^0].finish_reason)     # stop / length / content_filter
```

`response` 객체 주요 필드:[^1]

- `response.choices[^0].message.content` — 실제 응답 텍스트
- `response.choices[^0].finish_reason` — `stop`(정상), `length`(max_tokens 초과)
- `response.usage.total_tokens` — prompt + completion 토큰 합계


### role 시스템

```python
messages=[
    {"role": "system",    "content": "당신은 10년차 ML/AI 엔지니어입니다."},
    {"role": "user",      "content": "역할 부여가 왜 성능에 좋은가요?"},
    {"role": "assistant", "content": "(이전 AI 응답)"}  # 대화 유지에 사용
]
```


### 생성 파라미터[^1]

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.7,       # 0~2: 낮을수록 결정론적
    top_p=0.9,             # 상위 p% 토큰만 선택 (Nucleus Sampling)
    max_tokens=500,        # 초과 시 finish_reason='length'
    frequency_penalty=0.5, # 반복 단어 억제
    presence_penalty=0.5,  # 새로운 주제 장려
    stop=["6."],           # 이 문자열이 나오면 생성 중단
    seed=42,               # 재현 가능한 출력
)
```


### 스트리밍 및 다중 턴 대화 유지

OpenAI API는 **Stateless**이므로 대화 이력을 messages 리스트에 직접 누적해야 합니다.[^1]

```python
# 스트리밍
response = client.chat.completions.create(model="gpt-4o-mini", messages=[...], stream=True)
for chunk in response:
    text = chunk.choices[^0].delta.content
    if text:
        print(text, end="", flush=True)

# 다중 턴: AI 응답도 messages에 추가
messages.append({"role": "assistant", "content": answer1})
```


***

## 🔗 STAGE 2: LangChain 활용

### OpenAI API vs LangChain 비교

| 항목 | OpenAI 직접 호출 | LangChain |
| :-- | :-- | :-- |
| 클라이언트 | `OpenAI()` | `ChatOpenAI(model="gpt-4o-mini")` |
| 호출 | `client.chat.completions.create(...)` | `llm.invoke(messages)` |
| 응답 추출 | `response.choices[^0].message.content` | `response.content` |
| 스트리밍 | `chunk.choices[^0].delta.content` | `chunk.content` |
| 프롬프트 | 직접 dict 구성 | `ChatPromptTemplate` |
| 체인 | 수동 코드 | `prompt \| llm \| parser` |

### invoke / stream / batch[^3]

```python
# invoke: 전체 응답 한 번에
response = llm.invoke("LangChain이란?")
print(response.content)

# stream: 실시간 스트리밍
for chunk in llm.stream("머신러닝 3가지 종류"):
    print(chunk.content, end="", flush=True)

# batch: 여러 입력 병렬 처리
results = llm.batch(["Python이란?", "Java란?", "Rust란?"])
for answer in results:
    print(answer.content)
```


### LCEL 체인 핵심 패턴[^5]

```
chain = prompt | llm | parser
         ↓        ↓       ↓
      입력 포맷  LLM 호출  출력 변환
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "{role}입니다. {style} 스타일로 답변하세요."),
    ("human",  "{question}")
])
llm    = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain  = prompt | llm | parser

result = chain.invoke({"role": "IT 전문가", "style": "간결한", "question": "클라우드란?"})
print(result)  # 문자열 직접 반환
```


### Runnable 클래스 4종[^5]

| 클래스 | 용도 | 사용 예 |
| :-- | :-- | :-- |
| `RunnableSequence` | 순차 실행 | `A \| B \| C` |
| `RunnableParallel` | 병렬 실행, dict 반환 | 요약 + 키워드 동시 추출 |
| `RunnablePassthrough` | 입력 그대로 전달 | 입력값 유지하며 다음 단계로 |
| `RunnableBranch` | 조건 분기 | 질문 종류에 따라 다른 체인 실행 |

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: route_logic(x) == 'technical', tech_chain),
    (lambda x: route_logic(x) == 'billing',   billing_chain),
    general_chain  # 기본값 (default)
)
```


***

## 🎨 STAGE 3: Gradio UI 구축

Gradio는 3가지 방식으로 UI를 만들 수 있습니다.[^5]

```python
# gr.Interface — 단순 함수 래핑
gr.Interface(fn=func, inputs=gr.Textbox(), outputs=gr.Textbox()).launch()

# gr.ChatInterface — 챗봇 UI (핵심 패턴)
gr.ChatInterface(fn=chat_fn, type="messages").launch()

# gr.Blocks — 커스텀 레이아웃 (Row/Column 자유 배치)
with gr.Blocks() as demo:
    with gr.Row():
        input_text = gr.Textbox()
        btn = gr.Button("전송")
    btn.click(fn=process, inputs=input_text, outputs=output_text)
```


### LLM 챗봇 최종 완성 패턴 (LangChain + Gradio)[^5]

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini")

def chat_with_llm(message, history):
    messages = [SystemMessage(content="당신은 친절한 어시스턴트입니다.")]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))
    return llm.invoke(messages).content

demo = gr.ChatInterface(
    type="messages",   # 최신 권장 방식 (tuples 형식은 deprecated)
    fn=chat_with_llm,
    title="AI 어시스턴트",
)
demo.launch(share=True)
```

> **주의**: `type="messages"` 설정을 빠뜨리면 `UserWarning: The tuples format is deprecated` 경고가 발생합니다.

***

`.md` 파일(`1주차 학습 내용 정리.md`)이 다운로드 가능한 파일로 저장되어 있으니 확인해보세요 . 추가로 특정 부분을 더 자세히 보충하거나, 4일차 `PydanticOutputParser`나 `bind()` 관련 내용을 별도로 깊이 있게 정리해 드릴 수도 있습니다.

<div align="center">⁂</div>

[^1]: 2026-03-12-LLM-gyoyuggwajeong-1juca-3casi-silseub.ipynb

[^2]: 2026-03-11-LLM-gyoyuggwajeong-1juca-2casi-silseub-5.ipynb

[^3]: 2026-03-13-LLM-gyoyuggwajeong-1juca-4casi-silseub-2.ipynb

[^4]: 2026-03-10-LLM-gyoyuggwajeong-1juca-1casi-silseub-4.ipynb

[^5]: 2026-03-14-LLM-gyoyuggwajeong-1juca-5casi-silseub-3.ipynb


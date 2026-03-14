<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 처음 네가 말해줬던 1주차 내용 정리에, 새로운 응답 내용을 보충해서 종합본을 md파일로 작성해줘

아래 내용을 그대로 복사해서 `1주차 학습 내용 정리 (종합본).md` 파일로 저장하시면 됩니다.

***

````markdown
# 📚 LLM 교육과정 1주차 학습 내용 정리 (종합본)

> **교육 기간**: 2026-03-10 ~ 2026-03-14 (5일차)  
> **사용 환경**: Google Colab (Python 3.12)  
> **핵심 라이브러리**: `openai`, `langchain-openai`, `langchain-core`, `gradio`

---

## 📌 전체 학습 흐름 요약

| 단계 | 날짜 | 핵심 내용 |
|------|------|-----------|
| 1일차 | 03-10 | Python 기초 복습 + LangChain 첫 접촉 (`ChatOpenAI.invoke`) |
| 2일차 | 03-11 | Python 문법 심화 (for/if/dict/함수/파일I/O) + API 파라미터 이해 |
| 3일차 | 03-12 | OpenAI API 직접 호출 + temperature/top_p/penalty 실험 + 스트리밍 + 대화 유지 |
| 4일차 | 03-13 | LangChain 심화 (invoke/stream/batch, PromptTemplate, OutputParser, bind) |
| 5일차 | 03-14 | LCEL 체인 패턴 (Runnable*) + Gradio UI 구축 + LLM 챗봇 완성 |

---

## 🔑 STAGE 1: OpenAI API 직접 호출

### 1-1. 환경 설정 및 API 키 로드

```python
# 방법 1: Google Colab Secrets 사용 (권장)
# - Colab 왼쪽 사이드바 🔑 아이콘에서 OPENAI_API_KEY 등록 후 사용
from google.colab import userdata
import os

api_key = userdata.get('OPENAI_API_KEY')  # Colab Secrets에서 키 불러오기
os.environ["OPENAI_API_KEY"] = api_key    # 환경변수로 등록 → OpenAI() 자동 인식

# 방법 2: .env 파일 사용 (로컬 개발 환경)
from dotenv import load_dotenv
load_dotenv()                              # .env 파일에서 OPENAI_API_KEY 자동 로드
api_key = os.getenv("OPENAI_API_KEY")
```

> **핵심**: `os.environ["OPENAI_API_KEY"] = api_key`를 먼저 실행하면,
> `OpenAI()`나 `ChatOpenAI()` 초기화 시 `api_key` 인자 없이도 자동으로 키를 인식함.

---

### 1-2. OpenAI 클라이언트 생성 및 기본 호출

```python
from openai import OpenAI

# 클라이언트 생성 (OPENAI_API_KEY 환경변수가 있으면 인자 불필요)
client = OpenAI()

# Chat Completion API 호출
# - model: 사용할 GPT 모델명
# - messages: role + content 형태의 대화 목록
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "안녕하세요, 자기소개 부탁드립니다."}
    ]
)

# 응답 추출 방법
print(response.choices.message.content)  # 텍스트만 출력
print(response.usage.total_tokens)          # 사용 토큰 수 (비용 계산에 사용)
print(response.choices.finish_reason)    # 종료 이유
```

**`response` 객체 주요 구조:**

| 필드 | 설명 |
|------|------|
| `response.choices[0].message.content` | 실제 응답 텍스트 |
| `response.choices[0].finish_reason` | `stop`(정상), `length`(max_tokens 초과), `content_filter` |
| `response.usage.total_tokens` | prompt + completion 토큰 합계 |
| `response.id` | 요청 고유 ID |

---

### 1-3. role 시스템

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",      # AI에게 페르소나/행동 지침을 주는 메시지
            "content": "당신은 10년차 ML/AI 엔지니어입니다."
        },
        {
            "role": "user",        # 사용자 입력
            "content": "역할을 부여하면 왜 LLM 성능이 좋아지나요?"
        }
    ]
)
```

**role 종류:**

| role | 설명 |
|------|------|
| `system` | AI에게 페르소나/행동 지침을 주는 메시지 |
| `user` | 사용자의 입력 |
| `assistant` | AI의 이전 응답 (다중 턴 대화 유지에 사용) |
| `tool` | 함수 호출 결과 (Function Calling) |

---

### 1-4. 생성 파라미터 실험

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],

    temperature=0.7,         # 0~2: 낮을수록 결정론적, 높을수록 창의적/무작위
    top_p=0.9,               # Nucleus Sampling: 상위 p% 확률 토큰만 선택
    max_tokens=500,          # 응답 최대 토큰 수 (초과 시 finish_reason='length')
    frequency_penalty=0.5,   # 0~2: 같은 단어 반복 억제
    presence_penalty=0.5,    # 0~2: 새로운 주제/단어 장려
    stop=["6."],             # 이 문자열이 나오면 즉시 생성 중단
    seed=42,                 # 동일 seed → 유사한 출력 (재현성 확보)
)
```

**Temperature 개념 (Softmax 기반):**

| Temperature | 특성 | 사용 시나리오 |
|-------------|------|--------------|
| 0 | 항상 같은 답 (결정론적) | 데이터 추출, 번역 |
| 0.7 | 균형 잡힌 창의성 | 일반 대화, 설명 |
| 2.0 | 매우 무작위적 | 창작, 브레인스토밍 |

> `o1` 계열 모델은 temperature, top_p가 1.0으로 고정되어 변경 불가

---

### 1-5. 스트리밍 출력

```python
# stream=True: 응답을 토큰 단위로 실시간 전송 (ChatGPT처럼 타이핑 효과)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Python의 장점 3가지"}],
    stream=True
)

full_answer = ""
for chunk in response:
    # delta.content: 이번 청크에 포함된 텍스트 조각
    text = chunk.choices.delta.content
    if text:
        print(text, end="", flush=True)  # end="": 줄바꿈 없이 이어 출력
        full_answer += text              # 전체 응답 누적
```

---

### 1-6. 다중 턴 대화 유지 (Multi-turn)

OpenAI API는 **Stateless** (상태 없음) → 대화 이력을 직접 `messages` 리스트에 누적해야 함.

```python
# 대화 시작: system 메시지로 역할 설정
messages = [
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."}
]

# ── 턴 1 ──────────────────────────────
question1 = "파이썬이란 무엇인가요?"
messages.append({"role": "user", "content": question1})  # 사용자 질문 추가

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
answer1 = response.choices.message.content
messages.append({"role": "assistant", "content": answer1})  # ★ AI 응답도 저장해야 맥락 유지

# ── 턴 2 (이전 맥락이 유지됨) ─────────
question2 = "그렇다면 어떻게 설치하나요?"
messages.append({"role": "user", "content": question2})
# ... 이후 반복
```

---

### 1-7. 토크나이저 (tiktoken / BPE)

```python
import tiktoken

# GPT-4o에서 사용하는 BPE 인코딩 불러오기
encoding = tiktoken.get_encoding("o200k_base")

text = "Hello, world! 안녕하세요."
tokens = encoding.encode(text)  # 텍스트 → 토큰 ID 리스트

print(f"텍스트: {text}")
print(f"토큰 ID: {tokens}")
print(f"토큰 수: {len(tokens)}")  # 토큰 수 = API 호출 비용과 직결

# 각 토큰 ID → 원래 텍스트 조각 확인
for t in tokens:
    print(f"ID {t:6} → {encoding.decode([t])}")
```

**BPE(Byte Pair Encoding) 핵심 개념:**

| 특성 | 설명 |
|------|------|
| 방식 | 단어/문자 단위가 아닌 **서브워드** 단위 토크나이징 |
| OOV 해결 | 미등록어도 서브워드로 분해하여 처리 가능 |
| 사용처 | GPT, Llama, Mistral 등 대부분의 LLM |

---

## 📦 핵심 라이브러리 역할 이해

### 3개 라이브러리의 관계와 역할

```
[사용자 입력 dict]
       ↓
ChatPromptTemplate   ← "어떤 형식으로 LLM에게 물어볼지" 정의
       ↓
   ChatOpenAI        ← "실제로 GPT API를 호출해서 응답 받기"
       ↓
 StrOutputParser     ← "LLM의 복잡한 응답 객체에서 텍스트만 뽑기"
       ↓
[최종 문자열 출력]
```

---

### `langchain_openai` — ChatOpenAI

```python
from langchain_openai import ChatOpenAI
```

**역할**: OpenAI의 GPT 모델을 LangChain 방식으로 감싼(wrapping) 클래스.

- `openai` 패키지의 `client.chat.completions.create(...)` 를 내부적으로 대신 처리
- LangChain의 `|` 체인 연산자에 연결 가능한 형태로 만들어줌
- `invoke`, `stream`, `batch` 같은 **통일된 인터페이스** 제공

> **비유**: 레거시 전화기(OpenAI 직접 호출) → 스마트폰 앱(ChatOpenAI). 기능은 같지만 다른 것들과 연동이 훨씬 쉬워짐.

---

### `langchain_core.prompts` — ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate
```

**역할**: LLM에게 보낼 메시지를 **재사용 가능한 템플릿**으로 관리하는 클래스.

- `{변수명}` 자리에 원하는 값을 넣어서 동적으로 프롬프트 생성
- system / human / assistant 역할 구조를 코드로 깔끔하게 관리
- 체인에서 LLM 앞단에 붙어 입력을 정제해주는 역할

> **비유**: 매번 손으로 편지를 쓰는 대신 → 양식지(template)를 만들어두고 빈칸만 채워 넣는 방식.

---

### `langchain_core.output_parsers` — StrOutputParser

```python
from langchain_core.output_parsers import StrOutputParser
```

**역할**: LLM이 반환하는 `AIMessage` 객체에서 **텍스트(`content`)만 추출**해주는 파서.

- `llm.invoke()`의 반환값은 `AIMessage` 객체 (텍스트 외에 메타데이터 등이 가득함)
- 체인 끝에 붙이면 자동으로 `.content` 텍스트만 꺼내줌

> **비유**: 택배 박스(AIMessage 전체) → 박스 개봉기(StrOutputParser) → 상품 본체(content 문자열)

---

## 🔗 STAGE 2: LangChain 활용

### 2-1. 설치 및 패키지 구조

```python
# !pip install langchain langchain-openai langchain-community
```

| 패키지 | 역할 |
|--------|------|
| `langchain` | 전체 프레임워크 (agent, chain 등) |
| `langchain-openai` | OpenAI 전용 래퍼 (`ChatOpenAI`) |
| `langchain-community` | 써드파티 통합 (다양한 LLM, 도구, 벡터DB 등) |
| `langchain-core` | 기본 추상화 클래스 (Message, Runnable, Parser 등) |

---

### 2-2. 3가지 실행 방식: invoke / stream / batch

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── invoke: 전체 응답을 한 번에 반환 ──────────────
response = llm.invoke("LangChain이란 무엇인가요?")
print(response.content)           # 텍스트만
print(response.usage_metadata)    # 토큰 사용량 확인

# ── stream: 실시간 스트리밍 ────────────────────────
# chunk 단위로 텍스트가 조금씩 전달됨 (ChatGPT 타이핑 효과)
for chunk in llm.stream("머신러닝 3가지 종류를 설명해줘"):
    print(chunk.content, end="", flush=True)

# ── batch: 여러 입력을 병렬 처리 ──────────────────
# 여러 질문을 한 번에 보내고 결과를 리스트로 받음 (처리 속도 빠름)
questions = ["Python이란?", "Java란?", "Rust란?"]
results = llm.batch(questions)
for answer in results:
    print(answer.content)
    print()
```

---

### 2-3. 메시지 클래스 활용

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

persona  = "MLOps 전문가"
question = "LangChain이 MLOps에서 어떻게 활용되나요?"

# 메시지 리스트 구성
# - SystemMessage: AI에게 역할/지침 부여
# - HumanMessage: 사용자 질문
# - AIMessage: AI의 이전 응답 (다중 턴 대화 유지 시 사용)
messages = [
    SystemMessage(content=f"당신은 {persona}입니다. 전문적으로 답변하세요."),
    HumanMessage(content=question)
]

response = llm.invoke(messages)
print(response.content)
print(response.response_metadata)  # 토큰 수, 모델명, 요청 ID 등
```

---

### 2-4. LCEL 체인 구성 (핵심 패턴)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# 1단계: 각 컴포넌트 초기화
# ─────────────────────────────────────────────

# LLM 초기화
# - model: 사용할 모델 (gpt-4o-mini = 빠르고 저렴)
# - temperature: 창의성 조절 (0~2)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 프롬프트 템플릿 정의
# - {role}, {style}, {question}: invoke() 시 채울 변수 자리
# - "system": AI 역할 지침, "human": 사용자 입력
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role}입니다. {style} 스타일로 답변하세요."),
    ("human",  "{question}")
])

# 출력 파서
# - llm 응답(AIMessage 객체)에서 .content 텍스트만 자동으로 뽑아줌
parser = StrOutputParser()

# ─────────────────────────────────────────────
# 2단계: | 연산자로 체인 연결 (LCEL)
# ─────────────────────────────────────────────

# LCEL (LangChain Expression Language)
# 데이터 흐름: dict → [prompt] → ChatPromptValue → [llm] → AIMessage → [parser] → str
chain = prompt | llm | parser

# ─────────────────────────────────────────────
# 3단계: 체인 실행
# ─────────────────────────────────────────────

# invoke()에 {변수명: 값} dict를 넘기면 prompt 변수에 자동 매핑
result = chain.invoke({
    "role":     "IT 전문가",
    "style":    "친근하고 간결한",
    "question": "클라우드 컴퓨팅이 무엇인가요?"
})
print(result)       # 순수 문자열(str) 반환
print(type(result)) # <class 'str'>
```

---

### 2-5. 파서 유무 비교

```python
# ❌ 파서 없는 체인
chain_no_parser = prompt | llm
result_raw = chain_no_parser.invoke({...})
# → AIMessage 객체 전체 반환
#   content='...' 외에도 usage_metadata, response_metadata 등 포함
print(type(result_raw))     # <class 'langchain_core.messages.ai.AIMessage'>
print(result_raw.content)   # 텍스트 꺼내려면 .content 수동 접근 필요

# ✅ 파서 있는 체인
chain_with_parser = prompt | llm | parser
result_clean = chain_with_parser.invoke({...})
# → 깔끔한 문자열만 반환
print(type(result_clean))   # <class 'str'>
print(result_clean)         # 바로 사용 가능
```

**컴포넌트별 입출력 정리:**

| 컴포넌트 | 받는 입력 | 반환하는 출력 | 없으면? |
|----------|-----------|--------------|---------|
| `ChatPromptTemplate` | `dict` | `ChatPromptValue` (메시지 리스트) | 프롬프트를 매번 수동 dict로 구성 |
| `ChatOpenAI` | `ChatPromptValue` 또는 메시지 리스트 | `AIMessage` (객체) | LLM 호출 불가 |
| `StrOutputParser` | `AIMessage` | `str` (텍스트만) | `.content` 매번 수동 접근 |

---

### 2-6. bind() — 파라미터 고정

```python
# bind(): LLM에 파라미터를 미리 고정해 두는 방법
# 체인에서 매번 파라미터를 넘기지 않아도 됨

# JSON 응답 강제
# - response_format={"type": "json_object"}: GPT가 반드시 JSON 형식으로 응답
# - ★ 주의: system 또는 user 메시지에 "json"이라는 단어가 반드시 포함되어야 함
llm_json = llm.bind(response_format={"type": "json_object"})

response = llm_json.invoke([
    SystemMessage(content="반드시 JSON 형식으로 응답하세요."),  # "json" 단어 포함 필수
    HumanMessage(content="주요 도시 3개와 인구를 알려주세요.")
])

import json
data = json.loads(response.content)  # 문자열 → Python dict로 파싱
```

---

### 2-7. Runnable 클래스 4종

```python
# LCEL의 핵심 구성 요소들
# chain = prompt | llm | parser 는 내부적으로 RunnableSequence로 동작

# ── RunnableParallel: 여러 작업 동시 실행 ────────
from langchain_core.runnables import RunnableParallel

# 같은 입력으로 요약과 키워드 추출을 동시에 실행
# 결과는 {'summary': ..., 'keywords': ...} 형태의 dict로 반환
parallel_chain = RunnableParallel(
    summary=ChatPromptTemplate.from_template('{text}를 한 줄로 요약해주세요.'),
    keywords=ChatPromptTemplate.from_template('{text}에서 키워드를 3개만 뽑아주세요.')
)
result = parallel_chain.invoke({"text": "LangChain은 LLM 기반 앱 개발 프레임워크입니다."})

# ── RunnableBranch: 조건에 따른 분기 실행 ────────
from langchain_core.runnables import RunnableBranch

tech_chain    = ChatPromptTemplate.from_template("기술지원팀입니다: {question}")
billing_chain = ChatPromptTemplate.from_template("요금 관리팀입니다: {question}")
general_chain = ChatPromptTemplate.from_template("일반 상담팀입니다: {question}")

def route_logic(x):
    text = x['question']
    if '오류' in text or '에러' in text:
        return 'technical'
    elif '가격' in text or '요금' in text:
        return 'billing'
    else:
        return 'general'

# (조건 lambda, 실행할 체인) 쌍으로 구성, 마지막은 기본값(default)
branch = RunnableBranch(
    (lambda x: route_logic(x) == 'technical', tech_chain),
    (lambda x: route_logic(x) == 'billing',   billing_chain),
    general_chain  # 조건에 맞는 게 없으면 여기로
)

questions = ["프린터 오류가 났습니다.", "월 요금이 얼마인가요?", "영업시간 알려주세요"]
for q in questions:
    print(f"Q: {q}")
    print(f"A: {branch.invoke({'question': q})}\n")
```

> **lambda 개념**:
> `lambda x: x + 1` ↔ `def add_one(x): return x + 1`
> 간단한 함수를 한 줄로 표현할 때 사용. RunnableBranch의 조건 함수처럼 일회성 함수에 유용.

**Runnable 4종 요약:**

| 클래스 | 용도 | 사용 예 |
|--------|------|---------|
| `RunnableSequence` | 순차 실행 | `A \| B \| C` |
| `RunnableParallel` | 병렬 실행, dict 반환 | 요약 + 키워드 동시 추출 |
| `RunnablePassthrough` | 입력 그대로 전달 | 입력값 유지하며 다음 단계로 |
| `RunnableBranch` | 조건 분기 | 질문 종류에 따라 다른 체인 실행 |

---

### 2-8. PydanticOutputParser — 구조화 출력

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 원하는 출력 구조를 Pydantic 모델로 정의
class BookReview(BaseModel):
    title:       str  = Field(description="책 제목")
    author:      str  = Field(description="저자명")
    rating:      int  = Field(description="평점 1-5", ge=1, le=5)
    summary:     str  = Field(description="한 줄 요약")
    recommended: bool = Field(description="추천 여부")

# 파서 생성
parser = PydanticOutputParser(pydantic_object=BookReview)

# 프롬프트에 {format_instructions} 포함 (파서가 자동으로 형식 지침 생성)
prompt = ChatPromptTemplate.from_messages([
    ("system", "도서 리뷰어입니다.\n{format_instructions}"),
    ("human",  "{book} 리뷰를 작성해주세요.")
])

chain = prompt | llm | parser

result = chain.invoke({
    "book": "레 미제라블",
    "format_instructions": parser.get_format_instructions()  # 형식 지침 자동 생성
})

# result는 BookReview 객체 → 필드로 바로 접근 가능
print(result.title)       # 레 미제라블
print(result.author)      # 빅토르 위고
print(result.rating)      # 5
print(result.recommended) # True
```

---

## 🎨 STAGE 3: Gradio UI 구축

### 3-1. gr.Interface — 단순 함수 인터페이스

```python
import gradio as gr

# fn: 입력 → 출력을 처리하는 일반 Python 함수
def greet(name):
    return f"안녕하세요, {name}님!"

demo1 = gr.Interface(
    fn=greet,                           # 실행할 함수
    inputs=gr.Textbox(label="이름 입력"),  # 입력 컴포넌트
    outputs=gr.Textbox(label="인사말"),   # 출력 컴포넌트
    title="인사 앱"
)

demo1.launch()   # Colab에서는 자동으로 share=True 설정됨
demo1.close()    # 서버 종료
```

---

### 3-2. gr.ChatInterface — 챗봇 UI

```python
# fn 함수는 반드시 (message, history) → str 시그니처를 가져야 함
def echobot(message, history):
    # message: 현재 사용자 입력 (str)
    # history: 이전 대화 목록 (list of dict)
    return f"Echo: {message}"

demo2 = gr.ChatInterface(
    fn=echobot,
    title="Echo 챗봇",
    examples=["안녕하세요", "날씨가 좋네요", "FAQ 문의"],
    type="messages"  # ★ 반드시 명시: OpenAI 스타일 role/content 형식
                     #   생략하면 deprecated tuples 형식으로 UserWarning 발생
)

demo2.launch()
```

---

### 3-3. gr.Blocks — 커스텀 레이아웃

```python
# Blocks: Row/Column으로 UI 자유 배치 + 이벤트 연결
with gr.Blocks(title="커스텀 레이아웃") as demo3:
    gr.Markdown("## 커스텀 레이아웃 데모")
    
    with gr.Row():                          # 가로로 나열
        with gr.Column(scale=2):            # 비중 2 (좌측, 더 넓게)
            input_text = gr.Textbox(label="입력")
            submit_btn = gr.Button("제출")
        with gr.Column(scale=1):            # 비중 1 (우측, 좁게)
            category_output = gr.Textbox(label="카테고리")
            output_text     = gr.Textbox(label="결과")

    def process(text):
        cat = "기술" if any(kw in text for kw in ["오류", "에러"]) else "일반"
        return cat, f"[{cat}] {text}"  # 반환값이 여러 개면 outputs에도 여러 개 지정

    # 버튼 클릭 이벤트 연결
    # - fn: 실행할 함수
    # - inputs: 함수에 넘길 컴포넌트
    # - outputs: 결과를 표시할 컴포넌트 (여러 개 가능)
    submit_btn.click(
        fn=process,
        inputs=input_text,
        outputs=[category_output, output_text]
    )

demo3.launch()
```

---

### 3-4. LLM + Gradio 챗봇 완성 (핵심 패턴)

```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─────────────────────────────────────────────
# 앱 시작 시 1번만 초기화 (요청마다 새로 만들면 비효율적)
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_with_llm(message, history):
    # ── history → LangChain 메시지 형식으로 변환 ──
    # history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages = [SystemMessage(content="당신은 친절한 어시스턴트입니다.")]
    
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # 현재 질문 추가
    messages.append(HumanMessage(content=message))
    
    # LLM 호출 후 텍스트만 반환
    return llm.invoke(messages).content

demo = gr.ChatInterface(
    type="messages",      # 최신 권장 방식 (tuples 형식은 deprecated)
    fn=chat_with_llm,
    title="AI 어시스턴트",
    examples=["안녕하세요", "파이썬이란?", "클라우드란?"],
)

demo.launch(share=True)  # share=True: 외부 접속 가능한 공개 URL 생성
```

---

### 3-5. LangChain 체인 + Gradio 챗봇 (FAQ 버전)

```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# 1단계: FAQ 지식 베이스 정의
# ─────────────────────────────────────────────
faq_data = [
    {"category": "계정",  "question": "비밀번호를 잊었어요?",   "answer": "it.company.com에서 재설정하세요."},
    {"category": "요금",  "question": "월 요금이 얼마인가요?",  "answer": "기본 5만원, IT팀 내선 1234."},
    {"category": "기기",  "question": "프린터가 안됩니다?",     "answer": "IT팀에 연락해 드라이버를 재설치하세요."},
    {"category": "보안",  "question": "2단계 인증 설정 방법?",  "answer": "Google Authenticator 앱을 설치하세요."},
]

# FAQ 리스트 → 하나의 문자열로 변환 (프롬프트에 삽입용)
faq_context = "\n".join(
    [f"Q: {item['question']} A: {item['answer']}" for item in faq_data]
)

# ─────────────────────────────────────────────
# 2단계: 체인 구성 (앱 시작 시 1번만)
# ─────────────────────────────────────────────
llm    = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 시스템 프롬프트에 FAQ 컨텍스트를 하드코딩
# {question}만 런타임에 채워짐
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""당신은 IT 헬프데스크 직원입니다.
아래 FAQ를 참고해 답변하세요. 모르는 내용은 IT팀(내선 1234)에 문의 안내하세요.

FAQ:
{faq_context}"""),
    ("human", "{question}")
])

chain = prompt | llm | parser  # dict → str 파이프라인

# ─────────────────────────────────────────────
# 3단계: Gradio 콜백 함수
# ─────────────────────────────────────────────
def chat_response(message, history):
    # chain.invoke()가 str을 반환하므로 바로 return
    return chain.invoke({"question": message})

# ─────────────────────────────────────────────
# 4단계: Gradio UI 생성 및 실행
# ─────────────────────────────────────────────
demo = gr.ChatInterface(
    type="messages",
    fn=chat_response,
    title="IT 헬프데스크",
    examples=["Wi-fi 비밀번호", "계정 잠금 해제", "2단계 인증 설정"],
    description="IT 관련 궁금한 점을 질문하세요!",
)

demo.launch(share=True)
```

---

## 🐍 Python 기초 문법 정리 (2일차)

### 자료형 및 변수

```python
name       = "홍길동"   # str (문자열)
age        = 30         # int (정수)
height     = 170.5      # float (실수)
is_student = False      # bool (불리언)

# 자주 쓰는 문자열 메서드
text = "  Hello World  "
text.strip()              # 양쪽 공백 제거 → "Hello World"
text.lstrip()             # 왼쪽 공백만 제거
text.split()              # 공백 기준 분리 → ['Hello', 'World']
text.replace("Hello", "Hi")  # 문자열 치환 → "Hi World"
len(text)                 # 길이 반환
```

### f-string (포맷 문자열)

```python
name  = "철수"
topic = "AI"
# {} 안에 변수나 표현식을 넣어 동적으로 문자열 생성
print(f"{name}님, {topic}에 대해 설명해드리겠습니다.")
# → "철수님, AI에 대해 설명해드리겠습니다."
```

### 리스트 & for/if

```python
models = ["gpt-4o", "gpt-4o-mini", "claude-3"]

# enumerate: 인덱스와 값을 동시에 가져옴
for i, item in enumerate(models):
    print(i, item)  # 0 gpt-4o / 1 gpt-4o-mini / ...

# zip: 두 리스트를 병렬로 순회
scores = [0.95, 0.87, 0.72]
for model, score in zip(models, scores):
    print(model, score)

# in 연산자: 포함 여부 확인
question = "LLM이란 무엇인가요?"
if "LLM" in question:
    print(question)
```

### 딕셔너리

```python
person = {"name": "홍길동", "age": 30}
print(person["name"])               # 직접 접근 → 없으면 KeyError
print(person.get("height", 170))    # 안전한 접근 → 없으면 기본값 반환

# messages 구조 = 딕셔너리 리스트 (OpenAI API 핵심 데이터 구조)
messages = [
    {"role": "system", "content": "IT 전문가입니다."},
    {"role": "user",   "content": "안녕하세요."},
]
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")
```

### 함수 정의

```python
# 기본 인자 (default argument)
def make_prompt(question, language="한국어"):
    prompt = f"{language}로 {question}에 답하세요."
    return prompt

# 반환값 여러 개 (tuple로 반환)
def ask_expert(persona, question):
    # ...
    return response.content, response.response_metadata["token_usage"]

answer, usage = ask_expert("전문가", "질문")  # 언패킹으로 받기
```

### 예외 처리

```python
# try/except: 에러가 발생해도 프로그램이 중단되지 않게 처리
try:
    result = 10 / 0
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")

# 딕셔너리에서 안전하게 값 가져오기
config = {"model": "gpt-4o-mini", "temperature": 0.7}
temperature = config.get("temperature", 0.5)  # 없으면 0.5 반환
```

### 파일 I/O & JSON

```python
import json, os

# 폴더 생성 (exist_ok=True: 이미 있어도 에러 안 남)
os.makedirs("sample_output", exist_ok=True)

# JSON 저장
messages = [{"role": "user", "content": "안녕하세요"}]
with open("sample_output/chat.json", "w", encoding="utf-8") as f:
    json.dump(
        messages, f,
        ensure_ascii=False,  # 한글을 유니코드 이스케이프 없이 그대로 저장
        indent=2             # 보기 좋게 들여쓰기
    )

# JSON 로드
with open("sample_output/chat.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)  # JSON 문자열 → Python 객체(list/dict)
```

---

## 📊 OpenAI API 직접 호출 vs LangChain 비교

| 항목 | OpenAI API 직접 호출 | LangChain |
|------|---------------------|-----------|
| **임포트** | `from openai import OpenAI` | `from langchain_openai import ChatOpenAI` |
| **클라이언트 생성** | `client = OpenAI()` | `llm = ChatOpenAI(model="gpt-4o-mini")` |
| **호출 방법** | `client.chat.completions.create(...)` | `llm.invoke(messages)` |
| **응답 텍스트 추출** | `response.choices[0].message.content` | `response.content` |
| **스트리밍** | `stream=True` + `chunk.choices[0].delta.content` | `for chunk in llm.stream(...)` + `chunk.content` |
| **프롬프트 관리** | 직접 dict 구성 | `ChatPromptTemplate` |
| **출력 파싱** | 직접 처리 | `StrOutputParser`, `PydanticOutputParser` |
| **체인 연결** | 수동 코드 작성 | `prompt \| llm \| parser` (LCEL) |
| **적합한 용도** | 단순 호출, 파라미터 세밀 제어 | 복잡한 체인, 프로덕션 앱 개발 |

---

## 🧠 핵심 개념 요약

### LCEL 패턴 암기

```
chain = prompt | llm | parser
          ↓        ↓       ↓
       입력 포맷  LLM 호출  출력 변환

dict → ChatPromptValue → AIMessage → str
```

### Temperature 빠른 참조

| 값 | 특성 | 추천 사용 시나리오 |
|----|------|------------------|
| 0 | 항상 같은 답 | 데이터 추출, 번역, FAQ |
| 0.7 | 균형 잡힌 창의성 | 일반 대화, 설명 |
| 1.5+ | 매우 무작위 | 창작, 아이디어 발산 |

### Gradio 컴포넌트 선택 기준

```
단순 함수 (입력 → 출력)    → gr.Interface
챗봇 UI                    → gr.ChatInterface  (type="messages" 필수)
복잡한 레이아웃             → gr.Blocks (Row/Column 자유 배치)
```

---

*본 문서는 2026-03-10 ~ 2026-03-14 교육과정 실습 파일을 바탕으로 작성되었습니다.*
````


# 🤖 ARA (AI Research Assistant)

**AI 기반 학술 논문 검색 및 분석 시스템**
사용자가 연구 질문을 입력하면, AI가 arXiv에서 관련 논문을 검색하고, PDF를 자동으로 처리하여 의미론적으로 가장 관련성 높은 정보를 찾아 종합적인 답변을 제공하는 지능형 연구 도우미입니다.
---


## 🎯 프로젝트 개요
### 핵심 가치
- **효율적인 문헌 조사**: 수백 개의 논문을 수동으로 검토할 필요 없음
- **지능형 정보 추출**: AI가 PDF 전문을 분석하여 핵심 내용만 추출
- **대화형 인터페이스**: 사용자가 중간에 개입하여 검색 결과의 정확성 향상
- **실시간 처리**: 질문부터 답변까지 평균 2-5분

### 대상 사용자
- 🎓 대학원생 및 박사 과정 학생
- 📚 학술 연구원
- 🔬 산업 R&D 팀
- 📖 문헌 조사가 필요한 모든 사람
---

## 🏗️ 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                      사용자 인터페이스                        │
│                    (Gradio 6.3.0)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 대화형 검색  │  │  빠른 검색   │  │   정보 탭    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 워크플로우                       │
│  Human-in-the-Loop 기반 아젠트 오케스트레이션               │
│                                                              │
│  [질문 입력]                                                │
│      ↓                                                      │
│  [키워드 추출 분석] ──→ [사용자 확인] (INTERRUPT 1)         │
│      ↓                  │                                   │
│      └──── [재분석] ←─── [다시 선택]                        │
│      ↓                                                      │
│  [논문 수 선택] ──────→ [사용자 입력] (INTERRUPT 2)         │
│      ↓                                                      │
│  [arXiv 검색]                                              │
│      ↓                                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PDF 임베딩 파이프라인 (병렬 처리)                 │
│                                                              │
│  [PDF 다운로드] → [텍스트 추출] → [청킹] → [임베딩] → [저장] │
│                                                              │
│  ✓ 병렬 처리 (동시 5개 논문)                                │
│  ✓ 자동 재시도 및 오류 처리                                 │
│  ✓ 진행 상황 실시간 로깅                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            의미 기반 검색 (ChromaDB + Embeddings)            │
│                                                              │
│  [질문 벡터화]                                              │
│      ↓                                                      │
│  [코사인 유사도 검색]                                        │
│      ↓                                                      │
│  [상위 N개 청크 반환]                                        │
│      ↓                                                      │
│  [유사도 점수와 함께 반환]                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LLM 기반 답변 생성 (OpenAI GPT-4)               │
│                                                               │
│  [논문 요약 생성]                                            │
│      ↓                                                      │
│  [종합 답변 생성]                                            │
│      ↓                                                      │
│  [사용자에게 표시]                                           │
└─────────────────────────────────────────────────────────────┘
```
---
### 📊 기술 스택 한눈에 보기
 1. RAG 패턴 - arXiv API + PDF 처리 + ChromaDB + 의미 검색 + LLM
 2. LangGraph - 11개 노드, 20개 state 필드, 조건부 라우팅
 3. 임베딩 모델 - distiluse, 384차원, 다국어
 4. ChromaDB - 벡터 저장소, 코사인 유사도
 5. arXiv API - 논문 검색
 6. OpenAI API - GPT-4, GPT-4o
 7. 추가 기술 - pdfplumber, concurrent.futures, Gradio
---

## 🚀 설치 및 실행 방법 (상세 가이드)
### ⚡ 빠른 시작 (3분)
#### 사전 요구사항
- Python 3.10 이상
- pip (Python 패키지 관리자)
- OpenAI API 키 ([https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)에서 발급)
- 인터넷 연결
- 약 2GB의 여유 디스크 공간

#### Step 1: 저장소 클론
git clone https://github.com/soyoungju03/AI-hackathon-ARA.git
또는
```bash
# GitHub에서 프로젝트 다운로드
ZIP 파일로 다운로드:
- 우측 상단 "Code" → "Download ZIP" 클릭
- 압축 해제 후 폴더로 이동

```bash
cd AI-hackathon-ARA
```
---

#### Step 2: 가상 환경 생성
**Windows:**
Remove-Item -Recurse -Force venv

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install --upgrade pip

pip install -r requirements.txt

**Mac:**
rm -rf venv
python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

**Linux:**
rm -rf venv

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

cd ~/AI-hackathon-ARA #프로젝트 디렉토리 안에서 requirements.txt파일 설치 실행!!!

pip install -r requirements.txt

(venv)가 터미널에 보이면 성공



#### Step 3: 의존성 설치
```bash
# 모든 필요한 패키지 설치
pip install -r requirements.txt
```

**설치 시간**: 약 2-5분 (인터넷 속도에 따라 다름)

설치 완료 메시지:
```
Successfully installed gradio-6.3.0 langchain-0.1.0 langgraph-0.0.40 ...
```

---
#### Step 4: OpenAI API 키 설정
**Step 4-1: API 키 발급**
1. [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) 접속
2. "Create new secret key" 클릭
3. 키 복사 (예: `sk-proj-abc123...`)
**Step 4-2: .env 파일 생성**
프로젝트 루트 디렉토리(`ara-research-assistant/`)에 `.env` 파일 생성:
**Windows (PowerShell):**
```powershell
# .env 파일 생성
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```
**Windows (명령 프롬프트):**
```cmd
echo OPENAI_API_KEY=sk-your-api-key-here > .env
```
**Mac/Linux:**
```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```
**파일 확인:**
```bash
cat .env  # 또는 .env 파일을 텍스트 에디터로 열기
```

예시 내용:
```
OPENAI_API_KEY=sk-proj-abcdef123456789...
```

⚠️ **보안 주의**: 
- API 키는 절대 GitHub에 업로드하지 마세요
- `.gitignore`에 이미 `.env`가 포함되어 있습니다

---

#### Step 5: 애플리케이션 실행
```bash
python app.py
```
**실행 대기 시간**: 처음 실행 시 30초~2분 (모델 다운로드)
**성공 시 콘솔 출력:**
```
INFO:     Uvicorn running on http://127.0.0.1:7860
INFO:     Running on public URL: https://1db4c0cf0f19ddc8f2.gradio.live
```
---
#### Step 6: 브라우저에서 접속
위에서 나온 **공개 URL 복사**:
```
https://1db4c0cf0f19ddc8f2.gradio.live
```

이 URL을 **웹 브라우저**에 붙여넣고 엔터:
- 데스크톱: Chrome, Firefox, Safari 등
- 모바일: 모바일 브라우저에서도 접속 가능
---

### 🎮 사용 예시
#### 예시 1: 대화형 검색 (권장)

```
[Step 1] 질문 입력
사용자 입력: "Transformer 모델의 효율성 개선"

[Step 2] AI 분석
AI 응답:
"다음 키워드를 추출했습니다.
키워드: Transformer, efficiency, optimization

맞으면 '확인', 수정이 필요하면 '다시'라고 입력해주세요."

사용자 입력: 확인

[Step 3] 논문 수 선택
AI 요청: "검색할 논문의 개수를 선택해주세요 (1-10)"
사용자 입력: 5(숫자만 입력!!!)

[Step 4] 처리 진행 (자동)
진행 상황:
- arXiv 검색 중... ✓
- PDF 다운로드 중... ✓
- 텍스트 추출 중... ✓
- 임베딩 생성 중... ✓
- 의미 검색 중... ✓

[Step 5] 결과 수신
최종 답변:
"Transformer 모델의 효율성 개선을 위한 주요 기법들:

1. Knowledge Distillation
   - 큰 모델에서 작은 모델로 지식 전이
   - 성능 유지하면서 크기 90% 감소
   
2. Quantization
   - 모델 가중치를 낮은 정밀도로 저장
   - 메모리 사용량 70% 감소
   
... (더 많은 내용)"
```
---
## 📖 사용 흐름도
```
시작
  ↓
[Step 1] 질문 입력
"당신의 연구 질문을 입력하세요"
  ↓
[Step 2] AI가 자동으로 분석
- 키워드 추출
- 질문 의도 파악
- 연구 도메인 분류
  ↓
[Step 3] 사용자 확인 (INTERRUPT 1)
"추출된 키워드: [keyword1, keyword2]"
  ├─ 확인 → Step 4로
  └─ 다시 → Step 2로 (재분석)
  ↓
[Step 4] 논문 개수 선택 (INTERRUPT 2)
"1-10개 중 선택"
  ↓
[Step 5] 자동 처리 (모두 자동)
├─ arXiv 검색 (1-2초)
├─ PDF 다운로드 (20-30초)
├─ 텍스트 추출 (10-20초)
├─ 임베딩 생성 (20-30초)
├─ 의미 검색 (2-5초)
├─ 논문 요약 (10-20초)
└─ 최종 답변 생성 (5-10초)
  ↓
[Step 6] 결과 수신
"종합적인 분석 답변"
  ↓
끝
```
---
## 🔑 핵심 기능

### 1️⃣ 두 단계 Human-in-the-Loop
**첫 번째 Interrupt: 키워드 확인**
- AI가 사용자의 질문을 자동으로 분석
- 추출된 키워드를 사용자에게 제시
- 사용자가 "확인" 또는 "다시" 선택
- "다시" 선택 시 AI가 자동으로 재분석

**두 번째 Interrupt: 논문 수 선택**
- 1~10개 범위에서 검색할 논문 개수 선택
- 더 많은 논문 = 더 정확하지만 처리 시간 증가

### 2️⃣ PDF 임베딩 파이프라인
```
검색된 논문
    ↓
[PDF 다운로드] (병렬 처리, 최대 5개 동시)
    ↓
[텍스트 추출] (pdfplumber 사용)
    ↓
[청크 분할] (1800자 ~ 350자 오버랩, 약 450 토큰)
    ↓
[임베딩 생성] (Sentence Transformers: all-MiniLM-L6-v2)
    ↓
[ChromaDB 저장] (메타데이터 포함)
```

**특징:**
- 자동 재시도 및 오류 처리
- 진행 상황 실시간 로깅
- 배치 모드로 여러 논문 동시 처리
- 메모리 효율적인 스트리밍 처리

### 3️⃣ 의미 기반 검색 (Semantic Search)
- 사용자의 질문을 벡터로 변환
- ChromaDB에서 코사인 유사도로 상위 청크 검색
- 단순 키워드 매칭이 아닌 **의미론적 유사성** 기반
- 각 청크의 유사도 점수와 함께 반환

### 4️⃣ LLM 기반 답변 생성
- OpenAI GPT-4o를 사용한 고품질 응답
- 검색된 청크를 바탕으로 한 논거 기반 답변
- 논문 요약 및 연구 동향 분석
- 추가 학습을 위한 제안 포함


## 📈 성능 지표
### 처리 시간
| 단계 | 예상 시간 | 비고 |
|------|---------|------|
| 키워드 추출 | 3-5초 | GPT-4o 경량 모델 |
| arXiv 검색 | 2-3초 | 10-100개 논문 |
| PDF 처리 | 30-120초 | 논문당 15-30초 |
| 의미 검색 | 2-5초 | ChromaDB 쿼리 |
| 답변 생성 | 5-10초 | GPT-4o 이용 |
| **총 소요 시간** | **2-5분** | **3개 논문 기준** |


## 📁 프로젝트 구조
```
ara-research-assistant/
├── app.py                          # 메인 진입점 (Gradio UI)
├── requirements.txt                # 의존성 목록
├── README.md                       # 이 파일
├── .gitignore                      # Git 제외 파일
│
├── app/
│   ├── __init__.py
│   │
│   ├── config.py                   # 설정 및 환경 변수
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                # LangGraph 상태 정의
│   │   ├── workflow.py             # 워크플로우 및 ResearchAssistant
│   │   └── nodes.py                # 워크플로우 노드 구현
│   │
│   └── tools/
│       ├── __init__.py
│       ├── embeddings.py           # Sentence Transformers 래퍼
│       ├── vectorstore.py          # ChromaDB 통합
│       ├── pdf_embedding_pipeline_final.py  # PDF 처리 파이프라인
│       │
│       └── paper_search/
│           ├── __init__.py
│           └── arxiv_tool.py       # arXiv API 통합
│
└── data/
    ├── arxiv_vectorstore/          # ChromaDB 저장소
    ├── arxiv_chunks/               # 청크 파일
    └── logs/                       # 로그 파일
```

### 핵심 파일 설명

| 파일 | 설명 |
|------|------|
| `app.py` | Gradio 사용자 인터페이스 및 세션 관리 |
| `state.py` | LangGraph 상태 정의 및 데이터 모델 |
| `workflow.py` | 워크플로우 빌드 및 ResearchAssistant 클래스 |
| `nodes.py` | 10개의 워크플로우 노드 구현 |
| `embeddings.py` | Sentence Transformers 통합 |
| `vectorstore.py` | ChromaDB 벡터 저장소 관리 |
| `pdf_embedding_pipeline_final.py` | PDF 다운로드/처리/임베딩 파이프라인 |
| `arxiv_tool.py` | arXiv API 검색 래퍼 |

---

## 🔄 워크플로우 상세 흐름
### 상태 다이어그램

```
사용자 질문 입력
      ↓
[STAGE 0] receive_question_node
      ↓
[STAGE 0] analyze_question_node
  (키워드 추출, 질문 분석)
      ↓
[STAGE 0] request_keyword_confirmation_node
  ⚠️ INTERRUPT 1 - 사용자 대기
      │
      ├─ "다시" 선택
      │   ↓
      │ [STAGE 1] process_keyword_confirmation_response_node
      │   ↓
      │ is_reanalyzing = True
      │   ↓
      │ analyze_question_node (재분석)
      │   ↓
      │ request_keyword_confirmation_node (다시 확인 요청)
      │   ↓
      │ ⚠️ INTERRUPT 1 - 다시 대기
      │
      └─ "확인" 선택
          ↓
      [STAGE 1] process_keyword_confirmation_response_node
          ↓
      [STAGE 1] request_paper_count_node
      ⚠️ INTERRUPT 2 - 사용자 대기
          ↓
      [STAGE 2] process_paper_count_response_node
          ↓
      [STAGE 2] search_papers_node
      (arXiv 검색 + PDF 처리)
          ↓
      [조건부 분기] check_search_results
          ├─ 검색 실패
          │   ↓
          │ generate_response_node (오류 메시지)
          │
          └─ 검색 성공
              ↓
          [STAGE 2] evaluate_relevance_node
          (ChromaDB 의미 검색)
              ↓
          [STAGE 2] summarize_papers_node
          (논문 요약)
              ↓
          [STAGE 2] generate_response_node
          (최종 답변 생성)
              ↓
          END

```

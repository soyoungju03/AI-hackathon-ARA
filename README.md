# 🤖 ARA (AI Research Assistant)

**AI 기반 학술 논문 검색 및 분석 시스템**

> 자연스러운 언어로 연구 질문을 입력하면, AI가 arXiv에서 관련 논문을 검색하고, PDF를 자동으로 처리하여 의미론적으로 가장 관련성 높은 정보를 찾아 종합적인 답변을 제공하는 지능형 연구 도우미입니다.

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
│                                                              │
│  [논문 요약 생성]                                            │
│      ↓                                                      │
│  [종합 답변 생성]                                            │
│      ↓                                                      │
│  [사용자에게 표시]                                           │
└─────────────────────────────────────────────────────────────┘
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

---

## 📊 기술 스택

| 계층 | 기술 | 설명 |
|------|------|------|
| **UI** | Gradio 6.3.0 | 웹 기반 사용자 인터페이스 |
| **오케스트레이션** | LangGraph | 워크플로우 및 상태 관리 |
| **LLM** | OpenAI GPT-4o | 질문 분석 및 답변 생성 |
| **검색** | arXiv API | 학술 논문 검색 |
| **임베딩** | Sentence Transformers | 텍스트 벡터화 (all-MiniLM-L6-v2) |
| **벡터 DB** | ChromaDB | 임베딩된 청크 저장 및 검색 |
| **PDF 처리** | pdfplumber | PDF 텍스트 추출 |
| **배포** | Hugging Face Spaces | 클라우드 호스팅 |

---

## 🚀 사용 방법

### 1. 대화형 검색 (권장)

**Step 1: 질문 입력**
```
당신의 연구 질문을 입력하세요.
예: "Transformer 모델의 attention 메커니즘 설명"
```

**Step 2: 키워드 확인**
```
추출된 키워드:
- Transformer
- Attention mechanism
- Deep learning

맞으면 "확인", 수정이 필요하면 "다시"를 입력하세요.
```

**Step 3: 논문 수 선택**
```
검색할 논문의 개수를 선택해주세요 (1-10):
3
```

**Step 4: 결과 수신**
```
검색 결과:
- 검색된 논문: 3개
- 처리된 청크: 150개
- 상위 관련 청크: 6개

최종 답변:
[논문 기반의 종합적인 답변]
```

### 2. 빠른 검색

- 논문 개수를 미리 선택
- 질문을 입력
- Interrupt 없이 자동으로 진행
- 사용자 입력 없이 빠른 결과 제공

---

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

### 리소스 사용
- **메모리**: 2-4GB (실행 중)
- **디스크**: 500MB-2GB (저장된 임베딩)
- **네트워크**: 50-200MB (PDF 다운로드)

---

## 🔧 설치 및 실행

### 로컬 개발 환경

**1. 저장소 클론**
```bash
git clone https://huggingface.co/spaces/Jusoyoung/ara-research-assistant
cd ara-research-assistant
```

**2. 가상 환경 생성**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

**3. 의존성 설치**
```bash
pip install -r requirements.txt
```

**4. 환경 변수 설정**
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

**5. 애플리케이션 실행**
```bash
python app.py
```

**6. 브라우저에서 접속**
```
http://localhost:7860
```

### Hugging Face Spaces 배포

**공개 URL**: [https://huggingface.co/spaces/Jusoyoung/ara-research-assistant](https://huggingface.co/spaces/Jusoyoung/ara-research-assistant)

---

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

출력: final_response
```

---

## 🎨 사용자 인터페이스

### 대화형 검색 탭
- Chatbot 형식의 대화형 인터페이스
- 메시지 히스토리 저장
- 자동 스크롤
- 예시 질문 제공

### 빠른 검색 탭
- 논문 개수 슬라이더 (1-10)
- 질문 입력 텍스트박스
- 검색 버튼
- 마크다운 형식의 결과 표시

### 정보 탭
- API 상태 확인
- 시스템 정보
- 기술 스택
- 사용 가이드

---

## 🐛 문제 해결

### "OPENAI_API_KEY가 설정되지 않았습니다"
```bash
# Hugging Face Spaces 사용 시:
# Settings → Repository secrets에서 OPENAI_API_KEY 추가

# 로컬 실행 시:
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### "arXiv에서 관련 논문을 찾지 못했습니다"
- 키워드를 더 구체적으로 입력
- 검색할 논문 수를 늘림
- 다른 표현으로 재시도

### "PDF 처리 중 오류"
- 인터넷 연결 확인
- arXiv 서버 상태 확인
- 디스크 공간 확인

### "메모리 부족"
- 검색할 논문 수를 줄임
- 청크 크기 조정 (`pdf_embedding_pipeline_final.py`)

---

## 📝 로깅

모든 처리 과정은 상세하게 로깅됩니다.

```bash
# 터미널 로그
[STAGE 0] 새 질문 시작: ...
[ANALYZE_QUESTION] 질문 분석 시작
  추출된 키워드: ['keyword1', 'keyword2']
  질문 의도: 최신 연구 동향
  연구 도메인: computer science
[REQUEST_KEYWORD_CONFIRMATION] 사용자 확인 대기
...
```

---

## 🤝 기여 및 피드백

이 프로젝트는 AI Hackathon 프로젝트입니다.

**피드백 및 개선 제안은 환영합니다!**

---

## 📄 라이선스

MIT License

---

## 👨‍💻 개발자 정보

**프로젝트**: AI Research Assistant (ARA)
**개발자**: Soyoung JU
**개발 기간**: 2025년 1월 (AI Hackathon)
**버전**: 2.3 (Gradio 6.3.0 완전 호환)

---

## 🙏 감사의 말

- OpenAI (GPT-4o API)
- arXiv (논문 검색 API)
- LangChain & LangGraph (워크플로우)
- ChromaDB (벡터 저장소)
- Hugging Face (배포 플랫폼)
- Gradio (사용자 인터페이스)

---

**마지막 업데이트**: 2026년 1월 15일
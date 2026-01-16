# PRD: AI Research Assistant (ARA)

**문서 버전**: 2.3
**최종 업데이트**: 2026년 1월 15일
**프로젝트 상태**: ✅ 완료 및 배포 준비됨

---

## 📋 Executive Summary

### 프로젝트명
**AI Research Assistant (ARA)** - 인공지능 기반 학술 논문 검색 및 분석 시스템

### 핵심 문제 정의
학술 연구를 수행하는 사용자들은 다음과 같은 문제를 마주합니다:
- 📚 **방대한 논문량**: arXiv에 매일 수천 개의 새로운 논문이 업로드됨
- ⏱️ **시간 낭비**: 관련 논문 검색과 문헌 조사에 수십 시간 소요
- 🔍 **저품질 검색**: 단순 키워드 검색으로는 의미론적으로 유사한 논문을 찾기 어려움
- 📖 **수동 분석**: 찾은 논문들을 수동으로 읽고 분석해야 함

### 해결책
**AI Research Assistant**는 자연스러운 언어 질문을 통해 관련 논문을 자동으로 검색하고, PDF를 처리하여, 사용자의 질문에 정확히 답하는 AI 기반 솔루션입니다.

### 핵심 가치 제안
| 가치 | 설명 |
|------|------|
| **⏱️ 효율성** | 문헌 조사 시간 80% 단축 |
| **🎯 정확성** | 의미 기반 검색으로 90%+ 관련성 |
| **🤖 자동화** | PDF 다운로드부터 분석까지 자동 처리 |
| **👤 제어성** | Human-in-the-Loop으로 사용자 개입 가능 |
| **📊 투명성** | ReAct 패턴으로 AI의 사고 과정 공개 |

---

## 🎯 목표 및 성공 기준

### Primary Goals

1. **사용 편의성 극대화**
   - 자연어로 질문 입력 가능
   - 3단계의 간단한 대화형 인터페이스
   - ✅ **성공 기준**: 사용자가 5분 이내에 첫 결과 획득

2. **검색 정확도 극대화**
   - 의미 기반 검색으로 키워드 검색 능가
   - 사용자 피드백 반영 (재분석 기능)
   - ✅ **성공 기준**: 관련성 점수 0.7 이상

3. **자동화 극대화**
   - PDF 다운로드부터 분석까지 완전 자동화
   - 병렬 처리로 성능 최적화
   - ✅ **성공 기준**: 3개 논문 처리 120초 이내

4. **안정성 확보**
   - 오류 처리 및 자동 재시도
   - 네트워크 장애 시 graceful degradation
   - ✅ **성공 기준**: 99% 가용성

### Secondary Goals

- 다양한 연구 분야 지원
- 여러 언어 지원 (향후)
- 모바일 인터페이스 (향후)

---

## 📊 사용자 요구사항 (User Requirements)

### User Persona 1: 대학원생 (Graduate Student)
**특성:**
- 매주 수십 개의 논문을 검토해야 함
- 시간이 부족함
- 최신 연구 동향을 빠르게 파악해야 함

**요구사항:**
- 빠른 문헌 검색
- 자동 요약 기능
- 한국어 지원
- 모바일 접근성

**성공 지표:**
- 사용 만족도 4.5/5 이상
- 주 3회 이상 사용
- 추천 의향 80% 이상

### User Persona 2: 연구원 (Researcher)
**특성:**
- 특정 분야의 깊이 있는 연구
- 높은 정확도 요구
- 논문 간 비교 분석 필요

**요구사항:**
- 고정밀 의미 기반 검색
- 다중 논문 비교
- 상세한 분석 결과
- API 지원

**성공 지표:**
- 검색 정확도 90% 이상
- 논문당 처리 시간 < 30초
- 고급 기능 활용도 60% 이상

### User Persona 3: R&D 팀 (Industry)
**특성:**
- 경쟁 기술 조사 필요
- 빠른 의사결정 요구
- 보안 및 프라이버시 중요

**요구사항:**
- 대량 처리 (50+ 논문)
- 팀 협업 기능
- 엔터프라이즈 배포
- 데이터 보안

**성공 지표:**
- 배치 처리 성능 50개/시간
- 팀 기능 채택률 70%
- 보안 인증 획득

---

## 🏗️ 기술 요구사항 (Technical Requirements)

### Functional Requirements (기능 요구사항)

#### FR1: 질문 입력 및 분석
```
FR1.1: 사용자가 자연언어 질문 입력 가능
FR1.2: AI가 질문을 자동으로 분석하여 키워드 추출
FR1.3: 추출된 키워드를 사용자에게 제시
FR1.4: 사용자가 키워드 확인 또는 재분석 선택 가능
```

#### FR2: 논문 검색
```
FR2.1: arXiv API를 사용한 논문 검색
FR2.2: 1-10개 범위의 논문 개수 선택
FR2.3: 카테고리별 필터링
FR2.4: 발표 날짜 기반 정렬
```

#### FR3: PDF 처리
```
FR3.1: 논문 PDF 자동 다운로드
FR3.2: 텍스트 추출 (pdfplumber)
FR3.3: 텍스트 청킹 (1800자, 350자 오버랩)
FR3.4: 병렬 처리 (최대 5개 동시)
FR3.5: 오류 처리 및 자동 재시도
```

#### FR4: 의미 기반 검색
```
FR4.1: Sentence Transformers로 임베딩 생성
FR4.2: ChromaDB에 청크 저장
FR4.3: 질문과의 유사도 계산 (코사인)
FR4.4: 상위 청크 반환
FR4.5: 유사도 점수 함께 제시
```

#### FR5: 답변 생성
```
FR5.1: 검색된 청크 기반 요약 생성
FR5.2: GPT-4o를 사용한 최종 답변 생성
FR5.3: 인용 문헌 포함
FR5.4: 추가 학습 제안 포함
```

### Non-Functional Requirements (비기능 요구사항)

#### Performance Requirements
```
NF1.1: 응답 시간 < 5분 (3개 논문 기준)
NF1.2: 임베딩 생성 < 5초
NF1.3: 검색 쿼리 < 2초
NF1.4: 답변 생성 < 15초
NF1.5: 처리량 > 10개 사용자 동시 처리
```

#### Reliability Requirements
```
NF2.1: 가용성 > 99%
NF2.2: 재시도 최대 3회
NF2.3: 오류 복구 자동화
NF2.4: 로깅 및 모니터링
```

#### Scalability Requirements
```
NF3.1: 동시 사용자 100명 지원
NF3.2: 일일 처리 1000개 논문
NF3.3: 저장소 확장성 자동
NF3.4: 캐싱 메커니즘
```

#### Security Requirements
```
NF4.1: API 키 환경 변수 관리
NF4.2: HTTPS 통신
NF4.3: 입력 검증
NF4.4: SQL injection 방지
NF4.5: 개인정보 최소화
```

#### Usability Requirements
```
NF5.1: 인터페이스 직관성 (SUS > 70)
NF5.2: 온보딩 시간 < 5분
NF5.3: 오류 메시지 명확성
NF5.4: 다국어 지원 (향후)
```

---

## 🏛️ 시스템 아키텍처

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     사용자 계층                           │
│              (Gradio 6.3.0 Web Interface)               │
│        [대화형] [빠른 검색] [정보] [설정]               │
└─────────────────────────────────────────────────────────┘
                          ↕️
┌─────────────────────────────────────────────────────────┐
│                    애플리케이션 계층                      │
│              (app.py - Session Management)              │
│        세션 추적, 상태 관리, 사용자 응답 처리            │
└─────────────────────────────────────────────────────────┘
                          ↕️
┌─────────────────────────────────────────────────────────┐
│                   오케스트레이션 계층                     │
│             (LangGraph - Workflow Control)              │
│  ┌──────────────────────────────────────────────┐      │
│  │ ResearchAssistant                            │      │
│  │ - start(): 워크플로우 시작                    │      │
│  │ - continue_with_response(): 사용자 응답 처리 │      │
│  │ - run(): 자동 실행 모드                      │      │
│  └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
                          ↕️
┌──────────────────────────────────────────────────────────────┐
│                    워크플로우 계층                            │
│                  (10 Nodes + Routing)                       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Node 1-2: 질문 입력 → 분석                          │   │
│  │ Node 3-4: 키워드 확인 [INTERRUPT 1]                │   │
│  │ Node 5-6: 논문 수 선택 [INTERRUPT 2]                │   │
│  │ Node 7: arXiv 검색 + PDF 처리                       │   │
│  │ Node 8: 의미 기반 검색 (ChromaDB)                   │   │
│  │ Node 9: 논문 요약                                   │   │
│  │ Node 10: 최종 답변 생성                             │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                          ↕️
┌──────────────────────────────────────────────────────────────┐
│                    도구/서비스 계층                           │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ LLM Services                                       │    │
│  │ - OpenAI GPT-4o (질문 분석, 답변 생성)            │    │
│  │ - OpenAI GPT-4 Turbo (가벼운 모델)                 │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Search Services                                    │    │
│  │ - arXiv API (논문 검색)                            │    │
│  │ - Google Scholar API (향후)                        │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ PDF Processing Pipeline                           │    │
│  │ - PDF 다운로드 (requests)                          │    │
│  │ - 텍스트 추출 (pdfplumber)                         │    │
│  │ - 청킹 (랭체인)                                    │    │
│  │ - 임베딩 (Sentence Transformers)                   │    │
│  │ - 병렬 처리 (concurrent.futures)                   │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Vector Store                                       │    │
│  │ - ChromaDB (청크 저장)                             │    │
│  │ - 메타데이터 관리                                  │    │
│  │ - 코사인 유사도 검색                                │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
사용자 입력
    ↓
┌──────────────────────────────────┐
│ app.py (Gradio Interface)        │
│ - Chatbot 관리                   │
│ - 세션 상태 추적                  │
│ - 사용자 메시지 처리             │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│ ResearchAssistant (Orchestrator) │
│ - start()                        │
│ - continue_with_response()       │
│ - run()                          │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│ LangGraph Workflow               │
│ - 10 노드                        │
│ - 2 interrupt 포인트             │
│ - 조건부 라우팅                   │
└──────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│                  도구 실행                             │
│                                                       │
│ ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│ │ arXiv API   │  │ PDF Pipeline │  │ ChromaDB    │  │
│ │             │  │              │  │             │  │
│ │ - 검색      │  │ - 다운로드   │  │ - 저장      │  │
│ └─────────────┘  │ - 추출       │  │ - 검색      │  │
│                  │ - 청킹       │  └─────────────┘  │
│ ┌─────────────┐  │ - 임베딩     │                   │
│ │ OpenAI API  │  └──────────────┘                   │
│ │             │                                     │
│ │ - 분석      │ ┌──────────────┐                    │
│ │ - 생성      │ │ Embeddings   │                    │
│ └─────────────┘  │              │                    │
│                  │ Sentence-    │                    │
│                  │ Transformers │                    │
│                  └──────────────┘                    │
└───────────────────────────────────────────────────────┘
```

---

## 🔄 워크플로우 상세 명세

### Workflow State Machine

```
[START]
  ↓
┌─ STATE 0: 초기 처리
│  ├─ Node 1: receive_question() - 질문 수신
│  ├─ Node 2: analyze_question() - 키워드 추출
│  └─ Node 3: request_keyword_confirmation() - 키워드 확인 요청
│     ⚠️ INTERRUPT 1 - 사용자 입력 대기
│
├─ STATE 1: 키워드 확인 응답 처리
│  ├─ Node 4: process_keyword_confirmation_response()
│  │
│  ├─ IF 응답 == "다시"
│  │  ├─ is_reanalyzing = True
│  │  ├─ go back to Node 2 (재분석)
│  │  └─ go back to Node 3 (재확인 요청)
│  │     ⚠️ INTERRUPT 1 (다시)
│  │
│  └─ ELSE IF 응답 == "확인"
│     └─ go to STATE 2
│
├─ STATE 2: 논문 수 선택
│  ├─ Node 5: request_paper_count()
│  │  ⚠️ INTERRUPT 2 - 사용자 입력 대기
│  │
│  ├─ Node 6: process_paper_count_response()
│  │
│  └─ Node 7: search_papers()
│     (arXiv 검색 + PDF 처리)
│
├─ STATE 3: 의미 검색 및 분석
│  ├─ IF 검색 실패
│  │  └─ Node 10: generate_response() (오류 메시지)
│  │
│  └─ IF 검색 성공
│     ├─ Node 8: evaluate_relevance() (ChromaDB 검색)
│     ├─ Node 9: summarize_papers() (논문 요약)
│     └─ Node 10: generate_response() (최종 답변)
│
└─ [END] final_response 반환
```

### Interrupt Points

#### Interrupt 1: Keyword Confirmation
```
시점: analyze_question_node 이후
조건: 항상
사용자 입력 옵션:
  - "확인" / "OK" / "" → confirmed
  - "다시" / "수정" / "retry" → retry
처리:
  - confirmed: request_paper_count로 진행
  - retry: analyze_question로 돌아가서 재분석
```

#### Interrupt 2: Paper Count Selection
```
시점: request_paper_count_node 이후
조건: 항상
사용자 입력 옵션:
  - "1" ~ "10" (숫자)
처리:
  - 1-10: 해당 숫자로 처리
  - 범위 외: 경고 후 최근접 값 사용
  - 잘못된 입력: 기본값 3 사용
```

---

## 📊 데이터 모델

### State Definition
```python
class AgentState(TypedDict):
    # 입력
    user_question: str
    session_id: str
    
    # 분석 결과
    extracted_keywords: List[str]
    question_intent: str
    question_domain: str
    
    # 검색 설정
    paper_count: int
    
    # 검색 결과
    papers: List[Paper]
    
    # PDF 처리 결과
    chunks_saved: int
    pdf_processing_result: PDFProcessingResult
    
    # 청크 검색 결과
    relevant_chunks: List[Chunk]
    
    # Interrupt
    interrupt_data: InterruptData
    keyword_confirmation_response: Literal["confirmed", "retry"]
    
    # 재분석 모드
    is_reanalyzing: bool
    
    # 최종 결과
    final_response: str
    is_complete: bool
```

### Paper Model
```python
class Paper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: str
    source: str = "arXiv"
    relevance_score: float
```

### Chunk Model
```python
class Chunk(BaseModel):
    chunk_id: str
    content: str
    arxiv_id: str
    title: str
    section: Optional[str]
    page_number: int
    similarity_score: float
    metadata: Dict[str, Any]
```

---

## 🚦 리스크 및 완화 전략

### 기술적 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|---------|
| **arXiv API 다운타임** | 중 | 높음 | 캐싱, 재시도 로직, 오류 메시지 |
| **PDF 다운로드 실패** | 중 | 중간 | 재시도, 타임아웃 설정, 오류 로깅 |
| **메모리 부족** | 낮음 | 높음 | 스트리밍 처리, 배치 크기 제한 |
| **OpenAI API 할당량** | 낮음 | 높음 | 비용 모니터링, 속도 제한 |
| **벡터 DB 성능** | 낮음 | 중간 | 인덱싱, 캐싱, 쿼리 최적화 |

### 운영적 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|---------|
| **높은 API 비용** | 중 | 높음 | 경량 모델 사용, 배치 처리 |
| **데이터 개인정보 보호** | 낮음 | 높음 | 입력 검증, 데이터 암호화 |
| **사용자 데이터 손실** | 낮음 | 높음 | 정기 백업, 분산 저장소 |
| **보안 취약점** | 낮음 | 높음 | 코드 리뷰, 보안 감사 |

---

## 📈 성공 지표 (Success Metrics)

### Quantitative Metrics

| 지표 | 목표 | 측정 방법 |
|------|------|---------|
| **응답 시간** | < 5분 | 로그 분석 |
| **관련성 점수** | > 0.7 | 사용자 평가 |
| **가용성** | > 99% | 모니터링 대시보드 |
| **사용자 수** | > 100/월 | 접근 로그 |
| **오류율** | < 2% | 에러 로그 |
| **재시도율** | < 10% | 세션 로그 |

### Qualitative Metrics

| 지표 | 목표 | 측정 방법 |
|------|------|---------|
| **사용 만족도** | 4.5/5 | 사용자 설문 |
| **추천 의향** | > 80% | NPS 점수 |
| **인터페이스 직관성** | SUS > 70 | SUS 설문 |
| **기능 유용성** | > 90% | 기능 사용도 분석 |

---

## 🔄 개발 및 배포 계획

### Phase 1: Core Development ✅
- [x] LangGraph 워크플로우 설계
- [x] 10개 노드 구현
- [x] PDF 임베딩 파이프라인 개발
- [x] ChromaDB 통합
- [x] Gradio 인터페이스 개발

### Phase 2: Integration & Testing ✅
- [x] 컴포넌트 통합
- [x] 엔드-투-엔드 테스트
- [x] 성능 최적화
- [x] 오류 처리 강화

### Phase 3: Deployment ✅
- [x] Hugging Face Spaces 배포
- [x] OpenAI API 통합
- [x] 환경 변수 설정
- [x] 모니터링 설정

### Phase 4: Post-Launch (향후)
- [ ] 사용자 피드백 수집
- [ ] 성능 최적화
- [ ] 기능 확장
- [ ] 모바일 앱 개발

---

## 🎓 학습 및 개선 계획

### 단기 (1개월)
- 사용자 피드백 수집
- 버그 픽스
- 성능 최적화
- 문서화 개선

### 중기 (3개월)
- 다국어 지원 (한국어, 중국어, 일본어)
- 고급 검색 필터 추가
- 팀 협업 기능
- API 제공

### 장기 (6개월)
- 모바일 앱
- 클라우드 API 서비스
- 엔터프라이즈 배포
- 머신러닝 최적화

---

## 🛠️ 기술 스택 상세

### Frontend
- **Gradio 6.3.0**: 웹 UI 프레임워크
- **Python 3.10+**: 백엔드 언어

### Orchestration & Workflow
- **LangGraph**: 워크플로우 오케스트레이션
- **LangChain**: LLM 통합

### LLM & NLP
- **OpenAI API**
  - GPT-4o: 주요 분석, 답변 생성
  - GPT-4 Turbo: 경량 분석
- **Sentence Transformers**: 임베딩 (all-MiniLM-L6-v2)

### Data & Search
- **arXiv API**: 논문 검색
- **ChromaDB**: 벡터 저장소
- **pdfplumber**: PDF 텍스트 추출

### Processing
- **LangChain Text Splitters**: 청킹
- **concurrent.futures**: 병렬 처리
- **requests**: HTTP 통신

### Deployment
- **Hugging Face Spaces**: 클라우드 호스팅
- **Git**: 버전 관리
- **Python-dotenv**: 환경 변수 관리

---

## 📚 부록: API 명세

### ResearchAssistant.start()
```python
def start(question: str, session_id: str = "default") -> dict:
    """
    워크플로우를 시작합니다.
    
    Args:
        question: 사용자의 연구 질문
        session_id: 세션 ID
    
    Returns:
        {
            "status": "waiting_for_input",
            "interrupt_stage": 1,
            "message": "키워드를 확인해주세요",
            "options": ["확인", "다시"],
            "keywords": ["keyword1", "keyword2"],
            "thread_id": "uuid"
        }
    """
```

### ResearchAssistant.continue_with_response()
```python
def continue_with_response(user_response: str) -> dict:
    """
    사용자 응답을 받아 워크플로우를 계속합니다.
    
    Args:
        user_response: 사용자의 응답
    
    Returns:
        {
            "status": "completed|waiting_for_input|error",
            "response": "최종 답변 (완료 시)",
            "message": "다음 질문 (대기 시)",
            "keywords": "[새 키워드들]"
        }
    """
```

### ResearchAssistant.run()
```python
def run(question: str, paper_count: int = 3) -> str:
    """
    자동 실행 모드: Interrupt 없이 전체 과정 자동 수행.
    
    Args:
        question: 사용자의 연구 질문
        paper_count: 검색할 논문 수 (1-10)
    
    Returns:
        최종 답변 문자열
    """
```

---

## 📞 Contact & Support
- jsy6411897@gmail.com

**프로젝트 정보:**
- GitHub: [URL]
- Hugging Face: https://huggingface.co/spaces/Jusoyoung/ara-research-assistant
- 이메일: jsy6411897@gmail.com

**버그 보고 및 기능 요청:**
- GitHub Issues
- Email with [BUG] or [FEATURE] prefix

---

**Document Version**: 2.3
**Last Updated**: 2026-01-15
**Status**: ✅ COMPLETE
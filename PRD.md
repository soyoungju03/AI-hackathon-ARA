# AI Research Assistant PRD (Product Requirements Document)

**버전**: 2.0  
**작성일**: 2026-01-11  
**작성자**: 주소영

---

## 1. 프로젝트 개요

### 1.1 프로젝트명
**[ARA] AI Research Assistant** - 학술 논문 기반 지능형 연구 도우미

### 1.2 배경 및 목적

#### 배경
- 학술 논문 사이트(국내/해외)가 분산되어 있어 원하는 자료를 찾기 어려움
- 논문을 찾고 이해하는 데 많은 시간이 소요됨
- 대학생, 연구자, 기술직 현업자들이 공통으로 겪는 문제

#### 목적
사용자의 전공 및 기술 관련 질문을 분석하여, 국내외 다양한 학술 오픈소스에서 관련 논문을 검색하고, AI가 핵심 내용을 요약 정리하여 제공하는 지능형 연구 도우미 시스템 구축

### 1.3 핵심 목표
1. **정확한 질문 분석**: ReAct 패턴을 통한 질문 의도 파악 및 핵심 키워드 추출
2. **다양한 논문 소스 지원**: arXiv 외 국내외 다양한 학술 DB 연동
3. **Human-in-the-Loop**: 사용자 개입을 통한 검색 논문 수 결정 및 품질 향상
4. **높은 연관성**: 사용자 질문과 높은 연관성을 가진 논문만 선별
5. **지속적 대화 지원**: Short-term/Long-term Memory를 통한 맥락 유지

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              사용자 인터페이스                                 │
│                    (FastAPI + Gradio / React Frontend)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LangGraph Orchestrator                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ReAct Agent Loop                              │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │   │
│  │   │ Thought  │ → │  Action  │ → │Observation│ → │ Decision │        │   │
│  │   │ (분석)   │   │ (실행)   │   │ (관찰)   │   │ (결정)   │        │   │
│  │   └──────────┘   └──────────┘   └──────────┘   └──────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌───────────────────────────────────┼───────────────────────────────────┐  │
│  │              Human-in-the-Loop (Interrupt)                            │  │
│  │  • 검색할 논문 수 결정                                                  │  │
│  │  • 검색 결과 확인 및 선택                                               │  │
│  │  • 추가 검색 여부 결정                                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   Tool: 논문 검색    │  │  Tool: 웹 검색      │  │  Tool: 요약 생성    │
│  ┌───────────────┐  │  │  (Tavily/Serper)    │  │  (LLM Summary)      │
│  │ arXiv API     │  │  └─────────────────────┘  └─────────────────────┘
│  │ Semantic S.   │  │
│  │ PubMed        │  │
│  │ CORE          │  │
│  │ DOAJ          │  │
│  │ DBpia(한국)   │  │
│  │ RISS(한국)    │  │
│  └───────────────┘  │
└─────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG Pipeline                                    │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Document Loader   │ →│   Text Splitter     │ →│   Embedding Model   │  │
│  │   (PDF, Abstract)   │  │   (Chunk 생성)      │  │   (all-MiniLM-L6)  │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                      │                                       │
│                                      ▼                                       │
│                          ┌─────────────────────┐                            │
│                          │   Vector Database   │                            │
│                          │     (Weaviate)      │                            │
│                          └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Memory System                                   │
│  ┌────────────────────────────┐  ┌────────────────────────────────────────┐ │
│  │    Short-term Memory       │  │         Long-term Memory                │ │
│  │  • 현재 세션 대화 내용      │  │  • 과거 질문/응답 히스토리              │ │
│  │  • 임시 검색 결과          │  │  • 사용자 관심 주제                     │ │
│  │  • 진행 중인 분석 상태      │  │  • 자주 검색하는 키워드                 │ │
│  │  (In-Memory / Redis)       │  │  (PostgreSQL / Weaviate)               │ │
│  └────────────────────────────┘  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 워크플로우 (LangGraph State Machine)

```
                                    START
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   1. 질문 수신 노드     │
                         │   (receive_question)   │
                         └────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   2. 질문 분석 노드     │
                         │   (analyze_question)   │
                         │   - ReAct: Thought     │
                         └────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  3. 키워드 추출 노드    │
                         │  (extract_keywords)    │
                         └────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   4. 사용자 확인 노드 (INTERRUPT)    │
                    │   - 추출된 키워드 확인               │
                    │   - 검색할 논문 수 선택              │
                    │   - 검색 소스 선택                   │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   5. 논문 검색 노드     │
                         │   (search_papers)      │
                         │   - ReAct: Action      │
                         └────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   6. 연관성 평가 노드   │
                         │   (evaluate_relevance) │
                         │   - ReAct: Observation │
                         └────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   7. 결과 확인 노드 (INTERRUPT)     │
                    │   - 검색 결과 미리보기               │
                    │   - 추가 검색 또는 진행 선택         │
                    └─────────────────────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
              (추가 검색 필요)              (진행)
                   │                              │
                   ▼                              ▼
          [5. 논문 검색으로]        ┌────────────────────────┐
                                   │   8. 논문 요약 노드     │
                                   │   (summarize_papers)   │
                                   └────────────────────────┘
                                              │
                                              ▼
                                   ┌────────────────────────┐
                                   │   9. 응답 생성 노드     │
                                   │   (generate_response)  │
                                   │   - ReAct: Decision    │
                                   └────────────────────────┘
                                              │
                                              ▼
                                   ┌────────────────────────┐
                                   │  10. 메모리 저장 노드   │
                                   │   (save_to_memory)     │
                                   └────────────────────────┘
                                              │
                                              ▼
                                            END
```

---

## 3. 상세 기능 명세

### 3.1 질문 분석 및 키워드 추출

| 항목 | 설명 |
|------|------|
| 기능명 | Question Analysis & Keyword Extraction |
| 입력 | 사용자의 자연어 질문 |
| 출력 | 핵심 키워드 리스트 (2-5개), 질문 의도 분류 |
| 처리 방식 | LLM + Function Calling |
| ReAct 단계 | Thought |

**Function Calling 스키마**:
```json
{
  "name": "analyze_question",
  "parameters": {
    "keywords": ["string"],
    "intent": "string",
    "domain": "string",
    "complexity": "simple|moderate|complex"
  }
}
```

### 3.2 Human-in-the-Loop (Interrupt)

| 항목 | 설명 |
|------|------|
| 기능명 | User Confirmation Interrupt |
| 트리거 | 키워드 추출 완료 후 / 검색 결과 확인 시 |
| 사용자 선택 | 논문 수 (1-10), 검색 소스 선택, 진행/수정 |

**Interrupt State**:
```python
class InterruptState(TypedDict):
    extracted_keywords: List[str]
    suggested_paper_count: int
    available_sources: List[str]
    user_selection: Optional[Dict]
    should_continue: bool
```

### 3.3 다중 소스 논문 검색

| 소스명 | 유형 | API/방식 | 특징 |
|--------|------|----------|------|
| arXiv | 해외 | REST API | 물리, CS, 수학 분야 |
| Semantic Scholar | 해외 | REST API | AI 기반 연관성 분석 |
| PubMed | 해외 | E-utilities API | 의학/생명과학 |
| CORE | 해외 | REST API | 오픈액세스 논문 |
| DOAJ | 해외 | REST API | 오픈액세스 저널 |
| DBpia | 국내 | 웹 스크래핑* | 국내 학술 논문 |
| RISS | 국내 | 웹 스크래핑* | 국내 학위논문 |

*주의: 웹 스크래핑 시 이용약관 확인 필요

### 3.4 연관성 평가 알고리즘

```python
def evaluate_relevance(query: str, paper: Paper) -> float:
    """
    논문의 연관성을 0-1 사이 점수로 평가
    
    평가 요소:
    1. 키워드 매칭 점수 (0.3)
    2. 의미적 유사도 (Embedding Cosine Similarity) (0.4)
    3. 출판 연도 최신성 (0.1)
    4. 인용 수/영향력 (0.2)
    """
    keyword_score = calculate_keyword_match(query, paper)
    semantic_score = calculate_semantic_similarity(query, paper.abstract)
    recency_score = calculate_recency(paper.published_date)
    impact_score = calculate_impact(paper.citations)
    
    final_score = (
        keyword_score * 0.3 +
        semantic_score * 0.4 +
        recency_score * 0.1 +
        impact_score * 0.2
    )
    return final_score
```

### 3.5 논문 요약 생성

| 항목 | 설명 |
|------|------|
| 기능명 | Paper Summarization |
| 입력 | 논문 제목, 초록, (선택적) 본문 |
| 출력 | 구조화된 요약 |

**요약 구조**:
```markdown
## 논문 요약

### 1. 핵심 아이디어
[논문의 주요 기여점]

### 2. 연구 배경 및 문제점
[해결하고자 하는 문제]

### 3. 제안 방법론
[문제 해결 접근법]

### 4. 실험 및 성능 지표
[주요 실험 결과]

### 5. 한계점 및 향후 연구
[알려진 제한사항]
```

### 3.6 웹 검색 지원

| 항목 | 설명 |
|------|------|
| 기능명 | Web Search Integration |
| 목적 | 논문 외 최신 정보 보완 |
| API 옵션 | Tavily API / Serper API |
| 사용 시점 | 논문만으로 부족한 경우 / 최신 뉴스 필요 시 |

### 3.7 Memory 시스템

#### Short-term Memory
```python
class ShortTermMemory:
    """현재 세션 내 임시 저장"""
    
    def __init__(self):
        self.current_session_id: str
        self.conversation_history: List[Message]
        self.search_results_cache: Dict[str, List[Paper]]
        self.analysis_state: Dict
        
    def add_message(self, message: Message):
        """대화 메시지 추가"""
        
    def get_context(self, window_size: int = 5) -> str:
        """최근 대화 맥락 반환"""
```

#### Long-term Memory
```python
class LongTermMemory:
    """영구 저장소 (Vector DB + SQL)"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_store: Weaviate
        self.metadata_store: PostgreSQL
        
    def save_interaction(self, interaction: Interaction):
        """질문-응답 쌍 저장"""
        
    def retrieve_similar(self, query: str, k: int = 5) -> List[Interaction]:
        """유사한 과거 상호작용 검색"""
        
    def get_user_preferences(self) -> UserPreferences:
        """사용자 선호도/관심사 반환"""
```

---

## 4. 데이터 구축 전략

### 4.1 데이터 파이프라인

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  논문 검색   │ →  │  메타데이터   │ →  │   텍스트     │ →  │   임베딩     │
│   API 호출   │    │    추출      │    │   청킹       │    │    생성      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────┐
                                                           │ Vector DB    │
                                                           │   저장       │
                                                           └──────────────┘
```

### 4.2 청킹 전략

| 파라미터 | 값 | 이유 |
|----------|-----|------|
| Chunk Size | 512 tokens | 논문 섹션 단위 유지 |
| Overlap | 50 tokens | 문맥 연결성 보장 |
| Splitter | RecursiveCharacterTextSplitter | 구조적 분할 |

### 4.3 메타데이터 스키마

```json
{
  "paper_id": "string",
  "title": "string",
  "authors": ["string"],
  "abstract": "string",
  "published_date": "date",
  "source": "arXiv|SemanticScholar|PubMed|...",
  "url": "string",
  "keywords": ["string"],
  "citations": "integer",
  "embedding_model": "string",
  "chunk_index": "integer",
  "indexed_at": "datetime"
}
```

---

## 5. 임베딩 모델

### 5.1 모델 선택

| 모델명 | 차원 | 특징 | 선택 이유 |
|--------|------|------|----------|
| **all-MiniLM-L6-v2** | 384 | 빠른 속도, 경량 | 기본 추천 |
| all-mpnet-base-v2 | 768 | 높은 정확도 | 정확도 중시 시 |
| text-embedding-3-small | 1536 | OpenAI 모델 | API 비용 허용 시 |

### 5.2 추천 모델
**sentence-transformers/all-MiniLM-L6-v2**
- 무료, 로컬 실행 가능
- 학술 텍스트에 적합한 성능
- Hugging Face에서 쉽게 사용

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```

---

## 6. Vector Database 선택

### 6.1 Chroma vs Weaviate 비교

| 항목 | Chroma | Weaviate |
|------|--------|----------|
| 설치 복잡도 | ★☆☆ (매우 쉬움) | ★★☆ (Docker 필요) |
| 초기 구축 시간 | 빠름 (수 분) | 중간 (30분-1시간) |
| 확장성 | 제한적 | 높음 |
| 하이브리드 검색 | 기본 지원 안함 | 기본 지원 |
| 필터링 | 기본적 | 고급 필터링 |
| 프로덕션 준비도 | 개발/프로토타입 | 프로덕션 레디 |
| 데이터 양 | ~100K 문서 | 수백만 문서 |

### 6.2 추천: **Weaviate**

**선택 이유**:
1. 논문 검색 프로젝트는 데이터량이 증가할 가능성이 높음
2. 하이브리드 검색(Semantic + Keyword)이 논문 검색에 필수
3. Docker 환경이 이미 준비됨
4. 메타데이터 필터링이 강력함 (연도, 출처별 필터링)

### 6.3 Weaviate Docker 설정

```yaml
# docker-compose.yml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.23.0
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

---

## 7. 개발 환경 및 기술 스택

### 7.1 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.10+ |
| 프레임워크 | FastAPI + Gradio |
| AI/LLM | OpenAI GPT-4o / Claude API |
| 오케스트레이션 | LangGraph 0.2+ |
| Vector DB | Weaviate |
| 캐시/세션 | Redis |
| 영구 저장 | PostgreSQL |
| 임베딩 | sentence-transformers |
| 웹 검색 | Tavily API |
| 컨테이너 | Docker + Docker Compose |
| 배포 | Hugging Face Spaces |

### 7.2 Python 패키지

```txt
# requirements.txt
fastapi>=0.109.0
uvicorn>=0.27.0
gradio>=4.14.0
langchain>=0.1.0
langgraph>=0.0.40
langchain-openai>=0.0.5
langchain-community>=0.0.16
weaviate-client>=4.4.0
sentence-transformers>=2.2.2
arxiv>=2.1.0
requests>=2.31.0
python-dotenv>=1.0.0
redis>=5.0.0
psycopg2-binary>=2.9.9
pydantic>=2.5.0
httpx>=0.26.0
tavily-python>=0.3.0
```

### 7.3 환경 변수

```bash
# .env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/research_assistant
```

---

## 8. 프로젝트 구조

```
AI-Research-Assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI + Gradio 진입점
│   ├── config.py               # 설정 관리
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── research_agent.py   # 메인 ReAct 에이전트
│   │   └── prompts.py          # 프롬프트 템플릿
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py            # LangGraph State 정의
│   │   ├── nodes.py            # 그래프 노드들
│   │   ├── edges.py            # 조건부 엣지
│   │   └── workflow.py         # 전체 워크플로우
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── paper_search/
│   │   │   ├── __init__.py
│   │   │   ├── arxiv_tool.py
│   │   │   ├── semantic_scholar_tool.py
│   │   │   ├── pubmed_tool.py
│   │   │   └── korean_sources.py  # DBpia, RISS
│   │   ├── web_search.py
│   │   └── summarizer.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py       # 세션 메모리
│   │   └── long_term.py        # 영구 메모리
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── weaviate_client.py
│   │   └── embeddings.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── paper.py            # Paper 데이터 모델
│   │   ├── user.py             # User 모델
│   │   └── conversation.py     # 대화 모델
│   │
│   └── utils/
│       ├── __init__.py
│       ├── relevance.py        # 연관성 평가
│       └── formatters.py       # 출력 포매팅
│
├── ui/
│   ├── gradio_app.py           # Gradio UI
│   └── components/
│       ├── chat.py
│       └── settings.py
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_memory.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── scripts/
│   ├── setup_weaviate.py       # Weaviate 스키마 설정
│   └── seed_data.py            # 초기 데이터
│
├── .env.example
├── requirements.txt
├── README.md
└── PRD.md
```

---

## 9. 개발 로드맵

### Phase 1: 기반 구축 (1주차)
- [ ] 프로젝트 구조 설정
- [ ] LangGraph 기본 워크플로우 구현
- [ ] arXiv 검색 도구 개선
- [ ] 기본 ReAct 패턴 적용

### Phase 2: 핵심 기능 (2주차)
- [ ] Human-in-the-Loop (Interrupt) 구현
- [ ] 다중 논문 소스 추가 (Semantic Scholar, PubMed)
- [ ] 연관성 평가 알고리즘 구현
- [ ] Weaviate 연동

### Phase 3: 메모리 & 고도화 (3주차)
- [ ] Short-term Memory 구현
- [ ] Long-term Memory 구현
- [ ] 웹 검색 통합
- [ ] Function Calling 고도화

### Phase 4: UI & 배포 (4주차)
- [ ] FastAPI + Gradio UI 완성
- [ ] Docker 구성
- [ ] Hugging Face Spaces 배포
- [ ] 테스트 및 버그 수정

---

## 10. 현재 문제점 및 해결 방안

### 10.1 Hugging Face 배포 에러
**문제**: `OPENAI_API_KEY` 환경 변수 미설정
**해결**: Hugging Face Spaces의 Settings > Repository secrets에 API 키 등록

### 10.2 Human-in-the-Loop 미적용
**문제**: 사용자 개입 없이 자동 진행
**해결**: LangGraph의 `interrupt_before` 기능 활용

### 10.3 낮은 논문 연관성
**문제**: 2,3번째 논문이 주제와 무관
**해결**: 
1. Semantic similarity 기반 필터링 추가
2. 검색 결과에 대한 LLM 재평가
3. 사용자 피드백 반영

### 10.4 제한된 논문 소스
**문제**: arXiv만 사용
**해결**: 다중 소스 통합 (Semantic Scholar, PubMed, CORE 등)

---

## 11. 부록

### A. ReAct 패턴 상세

```
Thought: 사용자의 질문을 분석한 결과, "자율주행 LiDAR" 관련 
         최신 기술 동향을 알고 싶어하는 것으로 파악됩니다.
         핵심 키워드: "autonomous driving", "LiDAR", "point cloud"

Action: search_papers(
    keywords=["autonomous driving LiDAR point cloud"],
    sources=["arxiv", "semantic_scholar"],
    max_results=5
)

Observation: 5개의 논문을 찾았습니다:
    1. "PointPillars: Fast Encoders..." (relevance: 0.92)
    2. "LiDAR-based 3D Object Detection..." (relevance: 0.88)
    ...

Thought: 상위 2개 논문이 높은 연관성을 보입니다. 
         사용자에게 검색 결과를 확인받겠습니다.

[INTERRUPT: 사용자 확인 대기]
```

### B. API 비용 추정

| 항목 | 예상 사용량 | 단가 | 월 예상 비용 |
|------|------------|------|-------------|
| GPT-4o (요약) | 100K tokens | $5/1M | ~$0.5 |
| GPT-4o-mini (분석) | 500K tokens | $0.15/1M | ~$0.08 |
| Embedding | 1M tokens | $0.02/1M | ~$0.02 |
| Tavily | 1000 calls | $0.01/call | ~$10 |

### C. 참고 자료
- LangGraph 공식 문서: https://langchain-ai.github.io/langgraph/
- Weaviate 공식 문서: https://weaviate.io/developers/weaviate
- arXiv API: https://info.arxiv.org/help/api/
- Semantic Scholar API: https://api.semanticscholar.org/
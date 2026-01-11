# 📚 AI Research Assistant

학술 논문 기반 지능형 연구 도우미 - LangGraph + ReAct Pattern + Human-in-the-Loop

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Jusoyoung/AI-Research-Assistant)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 프로젝트 소개

**AI Research Assistant**는 사용자의 기술/학술 질문을 분석하여 관련 논문을 검색하고, AI가 핵심 내용을 요약 정리해주는 지능형 연구 도우미입니다.

### 주요 특징

- **🧠 ReAct 패턴**: Thought-Action-Observation 구조로 투명한 AI 사고 과정
- **👤 Human-in-the-Loop**: 사용자가 검색 과정에 개입하여 결과 품질 향상
- **🔄 LangGraph 워크플로우**: 복잡한 작업을 단계별로 관리
- **📊 연관성 평가**: 검색 결과를 평가하여 관련성 높은 논문만 제공

## 🛠️ 기술 스택

| 구분 | 기술 |
|------|------|
| 오케스트레이션 | LangGraph |
| AI 패턴 | ReAct (Reasoning + Acting) |
| LLM | OpenAI GPT-4o |
| 논문 검색 | arXiv API |
| 웹 인터페이스 | Gradio |
| 언어 | Python 3.10+ |

## 🔄 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 질문 입력                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [Thought] 질문 분석                                        │
│  "자율주행 LiDAR 기술에 대한 최신 연구 동향을 알고 싶어함"    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [Action] 키워드 추출                                       │
│  Keywords: ["autonomous driving", "LiDAR", "point cloud"]   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  🛑 [INTERRUPT] Human-in-the-Loop                          │
│  "추출된 키워드를 확인해주세요. 검색할 논문 수를 선택하세요"  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [Action] 논문 검색 (arXiv)                                 │
│  5개 논문 검색 중...                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [Observation] 연관성 평가                                  │
│  논문 1: 0.92 ✓ | 논문 2: 0.85 ✓ | 논문 3: 0.45 ✗          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [Action] 요약 생성                                         │
│  선별된 논문들의 핵심 내용을 구조화된 형식으로 요약          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    최종 응답 제공                            │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### 2. 가상 환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 OpenAI API 키를 설정하세요:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 5. 앱 실행

```bash
python app.py
```

브라우저에서 `http://localhost:7860` 접속

## 📁 프로젝트 구조

```
AI-Research-Assistant/
│
├── app.py                      # 메인 진입점 (Hugging Face 배포용)
├── requirements.txt            # Python 의존성
├── .env.example               # 환경 변수 템플릿
│
├── app/
│   ├── __init__.py
│   ├── config.py              # 설정 관리
│   │
│   ├── graph/
│   │   ├── state.py           # LangGraph State 정의
│   │   ├── nodes.py           # 워크플로우 노드들
│   │   └── workflow.py        # 전체 워크플로우 구성
│   │
│   ├── tools/
│   │   └── paper_search/
│   │       └── arxiv_tool.py  # arXiv 검색 도구
│   │
│   ├── memory/                # (Phase 3)
│   ├── models/                # (Phase 3)
│   └── utils/                 # (Phase 3)
│
└── ui/
    └── gradio_app.py          # Gradio UI (대체 버전)
```

## 💡 사용 방법

### 대화형 검색 (Human-in-the-Loop)

1. **질문 입력**: 연구하고 싶은 주제를 자연어로 입력
2. **키워드 확인**: AI가 추출한 키워드를 확인
3. **논문 수 선택**: 검색할 논문 수를 1-10 사이에서 선택
4. **결과 확인**: 요약된 논문 정보 확인

### 빠른 검색

1. **질문 입력**: 검색 주제 입력
2. **논문 수 설정**: 슬라이더로 선택
3. **검색 버튼 클릭**: 바로 결과 확인

## 🔧 Hugging Face Spaces 배포

### Secrets 설정

1. Space의 Settings 탭으로 이동
2. "Repository secrets" 섹션 찾기
3. 다음 secret 추가:
   - Name: `OPENAI_API_KEY`
   - Value: `sk-your-api-key`

## 🗺️ 로드맵

### Phase 1 ✅ (현재)
- [x] LangGraph 워크플로우 구현
- [x] ReAct 패턴 적용
- [x] Human-in-the-Loop (Interrupt)
- [x] arXiv 검색 도구 개선

### Phase 2 (예정)
- [ ] 다중 논문 소스 (Semantic Scholar, PubMed)
- [ ] Weaviate Vector DB 연동
- [ ] 연관성 평가 알고리즘 고도화

### Phase 3 (예정)
- [ ] Short-term / Long-term Memory
- [ ] 웹 검색 통합 (Tavily)
- [ ] 국내 논문 검색 (DBpia, RISS)

## 📝 주요 개념 설명

### ReAct 패턴이란?

ReAct는 **Reasoning + Acting**의 약자로, AI가 문제를 해결하는 과정을 명시적으로 표현합니다:

1. **Thought (생각)**: 현재 상황을 분석하고 다음 행동을 계획
2. **Action (행동)**: 도구를 사용하거나 작업을 실행
3. **Observation (관찰)**: 행동의 결과를 관찰하고 기록

이 과정을 반복하여 복잡한 문제를 단계적으로 해결합니다.

### Human-in-the-Loop이란?

AI 워크플로우 중간에 사용자가 개입하여 방향을 조정할 수 있는 구조입니다. 이를 통해:

- 검색 키워드의 정확성 확인
- 사용자 의도에 맞는 결과 보장
- AI의 판단 오류 방지

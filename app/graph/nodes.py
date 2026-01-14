# -*- coding: utf-8 -*-
"""
수정된 LangGraph 노드들 (두 단계 Human-in-the-Loop 버전)

주요 변경 사항:
1. request_user_confirmation_node를 두 개로 분리
   - request_keyword_confirmation_node: 첫 번째 Interrupt (키워드 확인)
   - request_paper_count_node: 두 번째 Interrupt (논문 수 선택)

2. process_user_response_node를 두 개로 분리
   - process_keyword_confirmation_response: 키워드 확인 응답 처리
   - process_paper_count_response: 논문 수 응답 처리

3. 각 노드가 하나의 명확한 목적을 가짐 (단일 책임 원칙)
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import (
    AgentState, 
    Paper, 
    ReActStep, 
    InterruptData,
    add_react_step
)
from app.tools.paper_search.arxiv_tool import search_arxiv
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def get_llm(model: str = None):
    """LLM 인스턴스를 생성합니다."""
    return ChatOpenAI(
        model=model or settings.default_model,
        api_key=settings.openai_api_key,
        temperature=0.3
    )


# ============================================
# 노드 1: 질문 수신 (receive_question)
# ============================================

def receive_question_node(state: AgentState) -> dict:
    """사용자 질문을 수신합니다."""
    user_question = state["user_question"]
    
    thought_content = f'사용자 질문을 수신했습니다: "{user_question}"\n이제 질문을 분석하여 핵심 키워드와 의도를 파악해야 합니다.'
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "react_steps": [new_step]
    }


# ============================================
# 노드 2: 질문 분석 (analyze_question)
# ============================================

QUESTION_ANALYSIS_PROMPT = """
당신은 학술 연구 질문을 분석하는 전문가입니다.
사용자의 질문을 분석하여 다음 정보를 추출해주세요.

## 사용자 질문
{question}

## 분석해야 할 항목

1. **핵심 키워드**: 논문 검색에 사용할 핵심 기술 키워드 2-5개
   - 영어로 변환해주세요
   - 구체적이고 검색 효과가 좋은 키워드를 선택해주세요

2. **질문 의도**: 사용자가 알고 싶어하는 것이 무엇인지
   - "최신 연구 동향" / "특정 기술 설명" / "비교 분석" / "응용 사례" 등

3. **연구 도메인**: 어떤 학문 분야에 해당하는지
   - "computer science" / "physics" / "mathematics" / "biology" 등

## 응답 형식 (반드시 이 형식을 따라주세요)
KEYWORDS: keyword1, keyword2, keyword3
INTENT: 질문 의도 설명
DOMAIN: 연구 도메인
"""


def analyze_question_node(state: AgentState) -> dict:
    """사용자 질문을 분석하여 키워드를 추출합니다."""
    user_question = state["user_question"]
    
    logger.info(f"질문 분석 시작: {user_question}")
    
    try:
        llm = get_llm(settings.light_model)
        prompt = QUESTION_ANALYSIS_PROMPT.format(question=user_question)
        
        logger.info("LLM에 요청 전송 중...")
        
        response = llm.invoke([
            SystemMessage(content="당신은 학술 연구 질문 분석 전문가입니다."),
            HumanMessage(content=prompt)
        ])
        
        logger.info(f"LLM 응답 수신")
        
        response_text = response.content
        keywords = []
        intent = ""
        domain = ""
        
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line.startswith("KEYWORDS:"):
                keywords_str = line.replace("KEYWORDS:", "").strip()
                keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            elif line.startswith("INTENT:"):
                intent = line.replace("INTENT:", "").strip()
            elif line.startswith("DOMAIN:"):
                domain = line.replace("DOMAIN:", "").strip()
        
        logger.info(f"추출된 키워드: {keywords}")
        
        observation_content = f"질문 분석 완료:\n- 추출된 키워드: {keywords}\n- 질문 의도: {intent}\n- 연구 도메인: {domain}"
        
        new_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "extracted_keywords": keywords,
            "question_intent": intent,
            "question_domain": domain,
            "react_steps": [new_step]
        }
        
    except Exception as e:
        logger.error(f"질문 분석 중 오류: {str(e)}")
        return {
            "extracted_keywords": [],
            "question_intent": "",
            "question_domain": "",
            "error_message": str(e),
            "react_steps": [ReActStep(step_type="observation", content=f"분석 실패: {str(e)}")]
        }


# ============================================
# 노드 3: 키워드 확인 요청 (request_keyword_confirmation)
# 첫 번째 Interrupt 지점
# ============================================

def request_keyword_confirmation_node(state: AgentState) -> dict:
    """
    추출된 키워드가 맞는지 사용자에게 확인받습니다.
    
    첫 번째 Human-in-the-Loop Interrupt 지점입니다.
    사용자가 "확인" 또는 "다시"라고 응답할 때까지 대기합니다.
    """
    keywords = state.get("extracted_keywords", [])
    
    message = f"""
추출된 키워드를 확인해주세요.

키워드: {', '.join(keywords) if keywords else '없음'}

맞으면 "확인"을, 수정이 필요하면 "다시"라고 입력해주세요.
    """.strip()
    
    interrupt_data = InterruptData(
        interrupt_type="confirm_keywords",
        message=message,
        options=["확인", "다시"],
        default_value="확인",
        metadata={
            "keywords": keywords,
            "stage": 1
        }
    )
    
    thought_content = "키워드 추출이 완료되었습니다. 사용자에게 확인을 요청합니다."
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "interrupt_data": interrupt_data,
        "waiting_for": "keyword_confirmation",
        "interrupt_stage": 1,
        "waiting_for_user": True,
        "react_steps": [new_step]
    }


# ============================================
# 노드 4: 키워드 확인 응답 처리
# ============================================

def process_keyword_confirmation_response_node(state: AgentState) -> dict:
    """
    사용자의 키워드 확인 응답을 처리합니다.
    
    사용자가 "확인"이라고 하면 논문 수 선택 단계로 진행합니다.
    사용자가 "다시"라고 하면 분석 단계로 돌아갑니다.
    """
    user_response = state.get("user_response", "").strip().lower()
    
    # 응답이 "다시" 또는 재분석을 의미하면, 분석 단계로 돌아가야 함
    if user_response in ["다시", "retry", "다시하기", "수정"]:
        observation_content = "사용자가 키워드 재분석을 요청했습니다. 질문 분석 단계로 돌아갑니다."
        
        new_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "keyword_confirmation_response": "retry",
            "waiting_for": None,
            "waiting_for_user": False,
            "interrupt_data": None,
            "react_steps": [new_step],
            "user_response": None
        }
    
    # 그 외의 경우 "확인"으로 처리
    observation_content = f"사용자가 키워드를 확인했습니다. 키워드: {', '.join(state.get('extracted_keywords', []))}"
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "keyword_confirmation_response": "confirmed",
        "waiting_for": None,
        "waiting_for_user": False,
        "interrupt_data": None,
        "interrupt_stage": 1,  # 1단계 완료
        "react_steps": [new_step],
        "user_response": None
    }


# ============================================
# 노드 5: 논문 수 선택 요청 (request_paper_count)
# 두 번째 Interrupt 지점
# ============================================

def request_paper_count_node(state: AgentState) -> dict:
    """
    몇 개의 논문을 검색할지 사용자에게 선택받습니다.
    
    두 번째 Human-in-the-Loop Interrupt 지점입니다.
    사용자가 1-10 중 하나를 선택할 때까지 대기합니다.
    """
    message = """
검색할 논문의 개수를 선택해주세요.

1부터 10 사이의 숫자를 입력해주세요.
(기본값: 3개)
    """.strip()
    
    interrupt_data = InterruptData(
        interrupt_type="select_paper_count",
        message=message,
        options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default_value="3",
        metadata={
            "stage": 2
        }
    )
    
    thought_content = "키워드 확인이 완료되었습니다. 이제 검색할 논문 수를 사용자에게 선택받습니다."
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "interrupt_data": interrupt_data,
        "waiting_for": "paper_count_selection",
        "interrupt_stage": 2,
        "waiting_for_user": True,
        "react_steps": [new_step]
    }


# ============================================
# 노드 6: 논문 수 응답 처리
# ============================================

def process_paper_count_response_node(state: AgentState) -> dict:
    """
    사용자가 선택한 논문 수를 처리합니다.
    """
    user_response = state.get("user_response", "3")
    
    try:
        paper_count = int(user_response)
        paper_count = max(1, min(10, paper_count))  # 1-10 범위로 제한
    except ValueError:
        paper_count = 3
    
    observation_content = f"사용자가 논문 수를 선택했습니다: {paper_count}개"
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "paper_count": paper_count,
        "waiting_for": None,
        "waiting_for_user": False,
        "interrupt_data": None,
        "interrupt_stage": 2,  # 2단계 완료
        "react_steps": [new_step],
        "user_response": None
    }


# ============================================
# 노드 7: 논문 검색 (search_papers)
# ============================================

def search_papers_node(state: AgentState) -> dict:
    """설정된 키워드와 옵션으로 논문을 검색합니다."""
    keywords = state.get("extracted_keywords", [])
    paper_count = state.get("paper_count", 3)
    domain = state.get("question_domain", None)
    
    action_content = f"논문 검색을 실행합니다:\n- 키워드: {keywords}\n- 검색 수: {paper_count}\n- 도메인: {domain or '전체'}\n- 소스: arXiv"
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    try:
        papers = search_arxiv(
            keywords=keywords,
            max_results=paper_count,
            domain=domain
        )
        
        observation_content = f"검색 완료: {len(papers)}개의 논문을 찾았습니다."
        for i, paper in enumerate(papers, 1):
            observation_content += f"\n{i}. {paper.title[:50]}... (연관성: {paper.relevance_score})"
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "papers": papers,
            "react_steps": [action_step, observation_step],
            "error_message": None
        }
        
    except Exception as e:
        logger.error(f"논문 검색 중 오류: {str(e)}")
        error_step = ReActStep(
            step_type="observation",
            content=f"검색 중 오류 발생: {str(e)}"
        )
        
        return {
            "papers": [],
            "react_steps": [action_step, error_step],
            "error_message": str(e)
        }


# ============================================
# 노드 8: 연관성 평가 (evaluate_relevance)
# ============================================
# nodes.py의 evaluate_relevance_node 함수 수정

from app.tools.embeddings import calculate_semantic_similarity

def evaluate_relevance_node(state: AgentState) -> dict:
    """
    검색된 논문들의 연관성을 재평가하고 필터링합니다.
    
    Sentence Transformers를 사용하여 사용자 질문과
    각 논문의 의미적 유사도를 계산합니다.
    """
    papers = state.get("papers", [])
    user_question = state.get("user_question", "")
    threshold = getattr(settings, 'relevance_threshold', 0.6)
    
    logger.info(f"의미 기반 연관성 평가 시작: {len(papers)}개 논문")
    
    # 각 논문에 대해 의미적 유사도를 계산합니다
    for paper in papers:
        # 제목과 초록을 결합하여 문서 텍스트를 만듭니다
        # 제목에 더 큰 가중치를 두기 위해 제목을 두 번 포함합니다
        document_text = f"{paper.title} {paper.title} {paper.abstract[:300]}"
        
        # 의미적 유사도 계산
        similarity = calculate_semantic_similarity(user_question, document_text)
        
        # 기존 점수와 결합 (기존 점수가 있다면)
        if paper.relevance_score > 0:
            # 의미 유사도 70%, 기존 점수 30%로 가중 평균
            paper.relevance_score = 0.7 * similarity + 0.3 * paper.relevance_score
        else:
            paper.relevance_score = similarity
        
        logger.info(f"  {paper.title[:50]}... → 유사도: {paper.relevance_score:.3f}")
    
    # 유사도 기준으로 정렬 (높은 것부터)
    papers.sort(key=lambda p: p.relevance_score, reverse=True)
    
    # 임계값 이상인 논문만 선택
    relevant_papers = [p for p in papers if p.relevance_score >= threshold]
    
    # 만약 임계값을 넘는 논문이 없다면, 상위 3개는 포함
    if not relevant_papers and papers:
        relevant_papers = papers[:min(3, len(papers))]
        logger.info(f"임계값을 넘는 논문이 없어 상위 {len(relevant_papers)}개를 선택합니다")
    
    thought_content = f"""의미 기반 연관성 평가 완료:
- 전체 검색 결과: {len(papers)}개
- 임계값({threshold}) 이상: {len(relevant_papers)}개
- 최고 유사도: {papers[0].relevance_score:.3f} ({papers[0].title[:40]}...)
- 최저 유사도: {papers[-1].relevance_score:.3f} ({papers[-1].title[:40]}...)"""
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "relevant_papers": relevant_papers,
        "react_steps": [new_step]
    }
#nodes.py의 evaluate_relevance_node 함수를 수정
#기존 relevance_score를 임계값과 비교-->사용자 질문과의 의미적 유사도도로 대체


# ============================================
# 노드 9: 논문 요약 생성 (summarize_papers)
# ============================================

SUMMARIZE_PROMPT = """
다음 논문의 초록을 읽고 핵심 내용을 한국어로 요약해주세요.

## 논문 제목
{title}

## 초록
{abstract}

## 요약 형식

### 핵심 아이디어
논문의 주요 기여점을 2-3문장으로 설명해주세요.

### 연구 배경 및 문제점
해결하고자 하는 문제를 설명해주세요.

### 제안 방법론
문제 해결 접근법을 설명해주세요.

### 주요 성과
실험 결과나 달성한 성과를 설명해주세요.
"""


def summarize_papers_node(state: AgentState) -> dict:
    """선별된 논문들의 요약을 생성합니다."""
    relevant_papers = state.get("relevant_papers", [])
    
    if not relevant_papers:
        return {
            "react_steps": [ReActStep(
                step_type="observation",
                content="요약할 논문이 없습니다."
            )]
        }
    
    try:
        llm = get_llm()
        
        action_step = ReActStep(
            step_type="action",
            content=f"{len(relevant_papers)}개 논문의 요약을 생성합니다."
        )
        
        summarized_papers = []
        
        for paper in relevant_papers:
            try:
                prompt = SUMMARIZE_PROMPT.format(
                    title=paper.title,
                    abstract=paper.abstract[:500] if paper.abstract else "초록 없음"
                )
                
                response = llm.invoke([
                    HumanMessage(content=prompt)
                ])
                
                paper.summary = response.content
                summarized_papers.append(paper)
                
            except Exception as e:
                logger.warning(f"논문 요약 실패: {str(e)}")
                paper.summary = "요약 생성 실패"
                summarized_papers.append(paper)
        
        observation_step = ReActStep(
            step_type="observation",
            content=f"{len(summarized_papers)}개 논문의 요약이 완료되었습니다."
        )
        
        return {
            "relevant_papers": summarized_papers,
            "react_steps": [action_step, observation_step]
        }
        
    except Exception as e:
        logger.error(f"요약 생성 중 오류: {str(e)}")
        return {
            "relevant_papers": relevant_papers,
            "react_steps": [ReActStep(step_type="observation", content=f"요약 생성 오류: {str(e)}")]
        }


# ============================================
# 노드 10: 최종 응답 생성 (generate_response)
# ============================================

FINAL_RESPONSE_PROMPT = """
사용자의 질문에 대해 검색된 논문들을 바탕으로 종합적인 답변을 생성해주세요.

## 사용자 질문
{question}

## 검색된 논문들
{papers_info}

## 답변 형식

친절하고 자세한 답변을 작성해주세요:
1. 질문에 대한 직접적인 답변
2. 관련 연구 동향 요약
3. 각 논문의 요약
4. 추가 탐구 제안

답변은 한국어로 작성해주세요.
"""


def generate_response_node(state: AgentState) -> dict:
    """검색 결과를 종합하여 최종 응답을 생성합니다."""
    user_question = state["user_question"]
    relevant_papers = state.get("relevant_papers", [])
    error_message = state.get("error_message")
    
    if error_message:
        return {
            "final_response": f"검색 중 오류가 발생했습니다: {error_message}",
            "is_complete": True
        }
    
    if not relevant_papers:
        return {
            "final_response": "입력하신 질문과 관련된 논문을 찾지 못했습니다.\n더 구체적인 키워드로 다시 시도해주세요.",
            "is_complete": True
        }
    
    try:
        papers_info = ""
        for i, paper in enumerate(relevant_papers, 1):
            authors = ', '.join(paper.authors[:3]) if paper.authors else "미상"
            papers_info += f"\n\n논문 {i}: {paper.title}\n저자: {authors}\n요약: {paper.summary[:200] if paper.summary else '요약 없음'}..."
        
        llm = get_llm()
        prompt = FINAL_RESPONSE_PROMPT.format(
            question=user_question,
            papers_info=papers_info
        )
        
        response = llm.invoke([
            SystemMessage(content="당신은 친절하고 전문적인 학술 연구 어시스턴트입니다."),
            HumanMessage(content=prompt)
        ])
        
        final_response = response.content
        
    except Exception as e:
        logger.error(f"최종 응답 생성 중 오류: {str(e)}")
        final_response = f"검색된 {len(relevant_papers)}개의 논문을 찾았습니다.{papers_info}"
    
    decision_step = ReActStep(
        step_type="thought",
        content="최종 응답 생성이 완료되었습니다."
    )
    
    return {
        "final_response": final_response,
        "is_complete": True,
        "react_steps": [decision_step]
    }
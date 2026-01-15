# -*- coding: utf-8 -*-
"""
수정된 LangGraph 노드들 (단순화 버전)

주요 변경사항:
- analyze_question_node에서 자동 승인 로직 제거
- 재분석이든 아니든 항상 사용자에게 키워드 확인 요청
- is_reanalyzing 플래그는 유지하되, 메시지 표시용으로만 사용
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import (
    AgentState, 
    ReActStep, 
    InterruptData
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
# 노드 1: 질문 수신
# ============================================

def receive_question_node(state: AgentState) -> dict:
    """사용자의 질문을 수신하고 처리를 시작합니다."""
    user_question = state.get("user_question", "")
    
    logger.info("="*60)
    logger.info("[RECEIVE_QUESTION] 사용자 질문 수신")
    logger.info("="*60)
    logger.info(f"질문: {user_question}")
    
    thought_content = f'사용자 질문을 수신했습니다: "{user_question}"\n이제 질문을 분석하여 핵심 키워드와 의도를 파악해야 합니다.'
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "react_steps": [new_step]
    }


# ============================================
# 노드 2: 질문 분석 (단순화 버전)
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
    """
    사용자 질문을 분석하여 핵심 키워드를 추출합니다.
    
    🔑 단순화: 재분석이든 아니든 항상 사용자에게 확인 요청
    """
    
    user_question = state.get("user_question", "")
    is_reanalyzing = state.get("is_reanalyzing", False)
    
    logger.info("="*60)
    logger.info("[ANALYZE_QUESTION] 질문 분석 시작")
    logger.info(f"  재분석 모드: {is_reanalyzing}")
    logger.info("="*60)
    logger.info(f"분석 대상: {user_question[:50]}...")
    
    try:
        llm = get_llm(settings.light_model)
        prompt = QUESTION_ANALYSIS_PROMPT.format(question=user_question)
        
        logger.info("LLM에 질문 분석 요청 전송...")
        
        response = llm.invoke([
            SystemMessage(content="당신은 학술 연구 질문 분석 전문가입니다."),
            HumanMessage(content=prompt)
        ])
        
        logger.info("✓ LLM 응답 수신")
        
        # LLM 응답 파싱
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
        
        logger.info(f"✓ 분석 완료")
        logger.info(f"  추출된 키워드: {keywords}")
        logger.info(f"  질문 의도: {intent}")
        logger.info(f"  연구 도메인: {domain}")
        
        # 🔑 단순화: 재분석 여부와 관계없이 동일한 메시지
        if is_reanalyzing:
            observation_content = f"""질문 재분석 완료:
- 새로운 키워드: {', '.join(keywords)}
- 질문 의도: {intent}
- 연구 도메인: {domain}

새로운 키워드를 확인해주세요."""
            logger.info("  → 재분석 완료: 사용자 확인 대기")
        else:
            observation_content = f"""질문 분석 완료:
- 추출된 키워드: {', '.join(keywords)}
- 질문 의도: {intent}
- 연구 도메인: {domain}"""
        
        new_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        # 🔑 핵심 수정: 자동 승인 제거, 항상 None으로 설정
        return {
            "extracted_keywords": keywords,
            "question_intent": intent,
            "question_domain": domain,
            "is_reanalyzing": False,  # 플래그 초기화
            "keyword_confirmation_response": None,  # 항상 사용자 확인 필요
            "react_steps": [new_step]
        }
        
    except Exception as e:
        logger.error(f"질문 분석 중 오류: {str(e)}", exc_info=True)
        return {
            "extracted_keywords": ["research"],
            "question_intent": "general research",
            "question_domain": "computer science",
            "is_reanalyzing": False,
            "keyword_confirmation_response": None,
            "error_message": str(e),
            "react_steps": [ReActStep(step_type="observation", content=f"분석 실패, 기본값 사용: {str(e)}")]
        }


# ============================================
# 노드 3: 키워드 확인 요청
# ============================================

def request_keyword_confirmation_node(state: AgentState) -> dict:
    """
    추출된 키워드가 맞는지 사용자에게 확인받습니다.
    
    첫 번째 Human-in-the-Loop Interrupt 지점입니다.
    """
    
    keywords = state.get("extracted_keywords", [])
    
    logger.info("[REQUEST_KEYWORD_CONFIRMATION] 사용자 확인 대기 시작")
    
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
    
    - "다시" → is_reanalyzing=True 설정, 질문 분석 단계로 돌아감
    - 그 외 ("확인" 등) → is_reanalyzing=False, 논문 수 선택 단계로 진행
    """
    
    user_response = state.get("user_response", "").strip().lower()
    
    logger.info("[PROCESS_KEYWORD_CONFIRMATION] 사용자 응답 처리")
    logger.info(f"  응답: {user_response}")
    
    # "다시" 응답 확인
    if user_response in ["다시", "retry", "다시하기", "수정", "다시해", "reanalyze"]:
        logger.info("  → '다시' 선택: 재분석 모드 활성화")
        
        observation_content = "사용자가 키워드 재분석을 요청했습니다. 질문을 다시 분석합니다."
        
        new_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "keyword_confirmation_response": "retry",
            "is_reanalyzing": True,  # 재분석 모드 활성화
            "waiting_for": None,
            "waiting_for_user": False,
            "interrupt_data": None,
            "react_steps": [new_step],
            "user_response": None
        }
    
    # 그 외의 경우 "확인"으로 처리
    logger.info("  → '확인' 선택: 논문 수 선택 단계로 이동")
    
    observation_content = f"사용자가 키워드를 확인했습니다. 키워드: {', '.join(state.get('extracted_keywords', []))}"
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "keyword_confirmation_response": "confirmed",
        "is_reanalyzing": False,
        "waiting_for": None,
        "waiting_for_user": False,
        "interrupt_data": None,
        "interrupt_stage": 1,
        "react_steps": [new_step],
        "user_response": None
    }


# ============================================
# 노드 5: 논문 수 선택 요청
# ============================================

def request_paper_count_node(state: AgentState) -> dict:
    """몇 개의 논문을 검색할지 사용자에게 선택받습니다."""
    
    logger.info("[REQUEST_PAPER_COUNT] 사용자 선택 대기 시작")
    
    message = """
검색할 논문의 개수를 선택해주세요.

1부터 10 사이의 숫자를 입력해주세요.
(기본값: 3개)

더 많은 논문을 선택할수록 처리 시간이 길어집니다.
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
    """사용자가 선택한 논문 수를 처리합니다."""
    
    user_response = state.get("user_response", "3")
    
    logger.info("[PROCESS_PAPER_COUNT] 사용자 응답 처리")
    logger.info(f"  응답: {user_response}")
    
    try:
        paper_count = int(user_response)
        paper_count = max(1, min(10, paper_count))
        logger.info(f"  → 해석됨: {paper_count}개")
    except ValueError:
        logger.warning(f"  → 유효하지 않은 입력, 기본값 3 사용")
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
        "interrupt_stage": 2,
        "react_steps": [new_step],
        "user_response": None
    }


# ============================================
# 노드 7-10: 나머지 노드들은 기존과 동일
# ============================================

def search_papers_node(state: AgentState) -> dict:
    """arXiv에서 논문을 검색합니다."""
    # 기존 코드와 동일
    # 여기서는 간략하게 표시
    return {
        "papers": [],
        "error_message": "search_papers_node 구현 필요"
    }


def evaluate_relevance_node(state: AgentState) -> dict:
    """의미 기반 관련성을 평가합니다."""
    return {
        "relevant_chunks": [],
        "error_message": "evaluate_relevance_node 구현 필요"
    }


def summarize_papers_node(state: AgentState) -> dict:
    """논문을 요약합니다."""
    return {
        "summarized_content": "summarize_papers_node 구현 필요"
    }


def generate_response_node(state: AgentState) -> dict:
    """최종 응답을 생성합니다."""
    return {
        "final_response": "generate_response_node 구현 필요",
        "is_complete": True
    }
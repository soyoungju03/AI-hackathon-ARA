# -*- coding: utf-8 -*-
"""
LangGraph State 정의 (수정 버전)
==================================

이 파일은 두 단계의 Human-in-the-Loop을 지원하도록 수정된 상태 정의입니다.

주요 변경 사항:
1. waiting_for 필드 추가: "무엇을" 대기 중인지 명시
2. interrupt_stage 필드 추가: 현재 어느 단계의 Interrupt인지 추적
3. keyword_confirmation_response 필드 추가: 사용자의 키워드 확인 응답 저장
"""

from typing import TypedDict, List, Optional, Annotated, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import operator


# ============================================
# 기본 데이터 모델 정의
# ============================================

class Paper(BaseModel):
    """
    논문 정보를 담는 데이터 모델입니다.
    """
    title: str = Field(..., description="논문 제목")
    authors: List[str] = Field(default_factory=list, description="저자 목록")
    abstract: str = Field("", description="논문 초록")
    url: str = Field("", description="논문 URL")
    published_date: str = Field("", description="출판일")
    source: str = Field("arXiv", description="논문 출처")
    relevance_score: float = Field(0.0, description="연관성 점수 (0-1)")
    summary: Optional[str] = Field(None, description="AI가 생성한 요약")


class ReActStep(BaseModel):
    """
    ReAct 패턴의 각 단계를 기록합니다.
    """
    step_type: Literal["thought", "action", "observation"] = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)


class InterruptData(BaseModel):
    """
    Human-in-the-Loop Interrupt 시 필요한 데이터입니다.
    """
    interrupt_type: Literal[
        "confirm_keywords",      # 첫 번째 Interrupt: 키워드 확인 요청
        "select_paper_count",    # 두 번째 Interrupt: 논문 수 선택 요청
        "review_results",        # 검색 결과 검토 요청
        "confirm_continue"       # 계속 진행 여부 확인
    ] = Field(...)
    
    message: str = Field(...)
    options: Optional[List[str]] = Field(None)
    default_value: Optional[str] = Field(None)
    metadata: dict = Field(default_factory=dict)


# ============================================
# LangGraph 상태 정의 (수정)
# ============================================

class AgentState(TypedDict):
    """
    수정된 LangGraph 상태입니다.
    
    주요 추가 필드:
    - waiting_for: 무엇을 대기 중인지 명시 ("keyword_confirmation", "paper_count_selection", None)
    - interrupt_stage: 현재 Interrupt 단계 (1 = 키워드 확인, 2 = 논문 수 선택)
    - keyword_confirmation_response: 사용자의 키워드 확인 응답 ("confirmed" 또는 "retry")
    """
    
    # === 입력 데이터 ===
    user_question: str
    session_id: str
    
    # === 분석 결과 ===
    extracted_keywords: List[str]
    question_intent: str
    question_domain: str
    
    # === 검색 설정 ===
    paper_count: int
    selected_sources: List[str]
    
    # === 검색 결과 ===
    papers: Annotated[List[Paper], operator.add]
    relevant_papers: List[Paper]
    
    # === ReAct 단계 기록 ===
    react_steps: Annotated[List[ReActStep], operator.add]
    
    # === 새로운 Interrupt 관련 필드 ===
    
    # 현재 Interrupt 데이터
    interrupt_data: Optional[InterruptData]
    
    # 무엇을 대기 중인가?
    # None = 대기 없음
    # "keyword_confirmation" = 사용자의 키워드 확인 대기 중
    # "paper_count_selection" = 사용자의 논문 수 선택 대기 중
    waiting_for: Optional[Literal["keyword_confirmation", "paper_count_selection"]]
    
    # 현재 Interrupt 단계 (추적용)
    # 1 = 키워드 확인 단계
    # 2 = 논문 수 선택 단계
    interrupt_stage: int
    
    # 사용자의 일반적인 응답 (모든 Interrupt에서 사용)
    user_response: Optional[str]
    
    # 사용자의 키워드 확인 응답 ("confirmed" 또는 "retry")
    keyword_confirmation_response: Optional[Literal["confirmed", "retry"]]
    
    # 사용자가 대기 중인지 여부 (하위 호환성용)
    waiting_for_user: bool
    
    # === 출력 ===
    final_response: str
    error_message: Optional[str]
    is_complete: bool


def create_initial_state(
    user_question: str,
    session_id: str = "default"
) -> AgentState:
    """
    초기 상태를 생성합니다.
    """
    return AgentState(
        # 입력
        user_question=user_question,
        session_id=session_id,
        
        # 분석 결과
        extracted_keywords=[],
        question_intent="",
        question_domain="",
        
        # 검색 설정
        paper_count=3,
        selected_sources=["arxiv"],
        
        # 검색 결과
        papers=[],
        relevant_papers=[],
        
        # ReAct 기록
        react_steps=[],
        
        # Interrupt (초기값)
        interrupt_data=None,
        waiting_for=None,
        interrupt_stage=0,
        user_response=None,
        keyword_confirmation_response=None,
        waiting_for_user=False,
        
        # 출력
        final_response="",
        error_message=None,
        is_complete=False
    )


def add_react_step(
    state: AgentState,
    step_type: Literal["thought", "action", "observation"],
    content: str
) -> dict:
    """
    ReAct 단계를 상태에 추가합니다.
    """
    new_step = ReActStep(
        step_type=step_type,
        content=content
    )
    
    return {"react_steps": [new_step]}
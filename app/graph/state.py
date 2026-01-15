# -*- coding: utf-8 -*-
"""
LangGraph State 정의 (PDF 임베딩 파이프라인 통합 버전)
=======================================================

이 파일은 두 단계의 Human-in-the-Loop과 PDF 임베딩 파이프라인을 
지원하도록 수정된 상태 정의입니다.

주요 기능:
1. 두 단계 Interrupt (키워드 확인 + 논문 수 선택)
2. PDF 임베딩 파이프라인 결과 저장
3. 청크 기반 검색 결과 추적
4. ReAct 패턴 기록
5. 재분석 모드 지원 (is_reanalyzing 플래그)

데이터 흐름:
user_question → extracted_keywords → papers → chunks_saved → 
relevant_chunks → summarized_content → final_response
"""

from typing import TypedDict, List, Optional, Annotated, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import operator


# ============================================
# 기본 데이터 모델 정의
# ============================================

class Paper(BaseModel):
    """
    논문 정보를 담는 데이터 모델입니다.
    
    이 모델은 arXiv 검색 결과로부터 생성됩니다.
    """
    title: str = Field(..., description="논문 제목")
    authors: List[str] = Field(default_factory=list, description="저자 목록")
    abstract: str = Field("", description="논문 초록")
    url: str = Field("", description="논문 URL")
    published_date: str = Field("", description="출판일")
    source: str = Field("arXiv", description="논문 출처")
    relevance_score: float = Field(0.0, description="연관성 점수 (0-1)")
    summary: Optional[str] = Field(None, description="AI가 생성한 요약")


class Chunk(BaseModel):
    """
    PDF에서 추출한 청크 정보입니다.
    
    논문을 의미 있는 크기의 조각으로 나누어 저장한 것입니다.
    각 청크는 벡터 임베딩을 가지고 있으며, ChromaDB에 저장됩니다.
    """
    chunk_id: str = Field(..., description="청크 고유 ID")
    content: str = Field(..., description="청크의 실제 텍스트 내용")
    arxiv_id: str = Field(..., description="논문의 arXiv ID")
    title: str = Field("", description="논문 제목")
    section: Optional[str] = Field(None, description="논문의 섹션 (예: Introduction, Methods)")
    page_number: int = Field(1, description="논문에서의 예상 페이지 번호")
    chunk_index: str = Field("", description="논문 내에서의 청크 순서")
    authors: str = Field("", description="논문 저자")
    
    # 의미 기반 검색 결과
    similarity_score: float = Field(0.0, description="질문과의 유사도 점수 (0-1)")
    
    # 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


class ReActStep(BaseModel):
    """
    ReAct 패턴의 각 단계를 기록합니다.
    
    Thought - Action - Observation의 세 단계를 추적하여,
    AI의 사고 과정을 투명하게 보여줍니다.
    """
    step_type: Literal["thought", "action", "observation"] = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)


class InterruptData(BaseModel):
    """
    Human-in-the-Loop Interrupt 시 필요한 데이터입니다.
    
    사용자에게 선택을 요청할 때 필요한 모든 정보를 담고 있습니다.
    """
    interrupt_type: Literal[
        "confirm_keywords",      # 첫 번째 Interrupt: 키워드 확인 요청
        "select_paper_count",    # 두 번째 Interrupt: 논문 수 선택 요청
        "review_results",        # 검색 결과 검토 요청
        "confirm_continue"       # 계속 진행 여부 확인
    ] = Field(...)
    
    message: str = Field(..., description="사용자에게 보여줄 메시지")
    options: Optional[List[str]] = Field(None, description="선택 가능한 옵션들")
    default_value: Optional[str] = Field(None, description="기본값")
    metadata: dict = Field(default_factory=dict, description="추가 정보")


class PDFProcessingResult(BaseModel):
    """
    PDF 처리 파이프라인의 결과를 담는 모델입니다.
    
    여러 논문을 배치로 처리한 결과를 추적합니다.
    """
    total: int = Field(..., description="처리한 논문 총 개수")
    successful: int = Field(..., description="성공적으로 처리된 논문 개수")
    failed: int = Field(..., description="처리 실패한 논문 개수")
    total_chunks: int = Field(..., description="생성된 총 청크 개수")
    time: float = Field(..., description="처리에 소요된 시간 (초)")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="각 논문의 처리 결과")
    message: str = Field(..., description="처리 결과 메시지")


# ============================================
# LangGraph 상태 정의
# ============================================

class AgentState(TypedDict):
    """
    LangGraph 워크플로우의 상태입니다.
    
    이 상태는 질문 입력부터 최종 답변까지의 전체 과정을 추적합니다.
    
    필드 분류:
    1. 입력 데이터: user_question, session_id
    2. 분석 결과: extracted_keywords, question_intent, question_domain
    3. 검색 설정: paper_count, selected_sources
    4. 검색 결과: papers (arXiv 검색 결과)
    5. PDF 처리: chunks_saved, pdf_processing_result
    6. 청크 검색: relevant_chunks (의미 기반 검색 결과)
    7. 요약: summarized_content
    8. Interrupt: interrupt_data, waiting_for, interrupt_stage
    9. 재분석 모드: is_reanalyzing (새로 추가)
    10. 기록: react_steps
    11. 출력: final_response, error_message, is_complete
    """
    
    # ========== 입력 데이터 ==========
    user_question: str
    session_id: str
    
    # ========== 분석 결과 ==========
    extracted_keywords: List[str]
    question_intent: str
    question_domain: str
    
    # ========== 검색 설정 ==========
    paper_count: int
    selected_sources: List[str]
    
    # ========== 검색 결과 (arXiv) ==========
    papers: Annotated[List[Paper], operator.add]
    relevant_papers: List[Paper]
    
    # ========== PDF 처리 결과 ==========
    chunks_saved: int
    pdf_processing_result: Optional[PDFProcessingResult]
    
    # ========== 청크 검색 결과 (의미 기반) ==========
    relevant_chunks: Annotated[List[Chunk], operator.add]
    
    # ========== 평가 결과 ==========
    evaluation_result: Dict[str, Any]
    
    # ========== 요약 및 분석 ==========
    summarized_content: str
    
    # ========== ReAct 패턴 기록 ==========
    react_steps: Annotated[List[ReActStep], operator.add]
    
    # ========== Interrupt 관련 필드 ==========
    interrupt_data: Optional[InterruptData]
    waiting_for: Optional[Literal["keyword_confirmation", "paper_count_selection"]]
    interrupt_stage: int
    user_response: Optional[str]
    keyword_confirmation_response: Optional[Literal["confirmed", "retry"]]
    waiting_for_user: bool
    
    # ========== 재분석 모드 (새로 추가) ==========
    # "다시" 선택 시 키워드 재추출 후 자동으로 다음 단계로 진행하기 위한 플래그
    # True: 재분석 중이므로 키워드 확인을 건너뜀
    # False: 일반 모드, 사용자에게 키워드 확인 요청
    is_reanalyzing: bool
    
    # ========== 출력 ==========
    final_response: str
    error_message: Optional[str]
    is_complete: bool


def create_initial_state(
    user_question: str,
    session_id: str = "default"
) -> AgentState:
    """
    워크플로우의 초기 상태를 생성합니다.
    
    사용자 질문과 세션 ID를 받아서, 모든 필드가 초기값으로 설정된
    상태 객체를 반환합니다.
    
    Args:
        user_question: 사용자의 연구 질문
        session_id: 대화 세션의 고유 ID
    
    Returns:
        초기값으로 설정된 AgentState 객체
    """
    return AgentState(
        # === 입력 데이터 ===
        user_question=user_question,
        session_id=session_id,
        
        # === 분석 결과 ===
        extracted_keywords=[],
        question_intent="",
        question_domain="",
        
        # === 검색 설정 ===
        paper_count=3,
        selected_sources=["arxiv"],
        
        # === 검색 결과 ===
        papers=[],
        relevant_papers=[],
        
        # === PDF 처리 결과 ===
        chunks_saved=0,
        pdf_processing_result=None,
        
        # === 청크 검색 결과 ===
        relevant_chunks=[],
        
        # === 평가 결과 ===
        evaluation_result={},
        
        # === 요약 및 분석 ===
        summarized_content="",
        
        # === ReAct 기록 ===
        react_steps=[],
        
        # === Interrupt 관련 필드 ===
        interrupt_data=None,
        waiting_for=None,
        interrupt_stage=0,
        user_response=None,
        keyword_confirmation_response=None,
        waiting_for_user=False,
        
        # === 재분석 모드 ===
        is_reanalyzing=False,  # 초기값은 False (일반 모드)
        
        # === 출력 ===
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
    ReAct 패턴의 새로운 단계를 상태에 추가합니다.
    
    이 함수는 노드에서 ReAct 단계를 기록할 때 사용됩니다.
    Thought, Action, Observation의 각 단계를 시간과 함께 기록합니다.
    
    Args:
        state: 현재 AgentState
        step_type: 단계의 유형 ("thought", "action", 또는 "observation")
        content: 단계의 내용
    
    Returns:
        상태 업데이트 딕셔너리 (react_steps 필드 업데이트)
    """
    new_step = ReActStep(
        step_type=step_type,
        content=content
    )
    
    return {"react_steps": [new_step]}
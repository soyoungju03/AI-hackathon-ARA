"""
LangGraph State 정의
====================

이 파일은 LangGraph 워크플로우에서 사용되는 상태(State)를 정의합니다.

상태(State)란?
--------------
LangGraph에서 상태는 워크플로우가 진행되는 동안 유지되고 전달되는 데이터입니다.
마치 릴레이 경주에서 바통을 넘기는 것처럼, 각 노드는 상태를 받아서 작업을 수행하고,
수정된 상태를 다음 노드에 전달합니다.

Human-in-the-Loop을 위한 Interrupt
-----------------------------------
LangGraph는 특정 노드에서 워크플로우를 "일시 정지"할 수 있습니다.
이것을 Interrupt라고 부르며, 사용자의 입력을 기다릴 때 사용합니다.
예를 들어:
1. 키워드 추출 후 → 사용자에게 확인 요청 (Interrupt)
2. 사용자가 확인/수정 → 워크플로우 재개
3. 검색 결과 후 → 사용자에게 선택 요청 (Interrupt)
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
    
    Pydantic의 BaseModel을 사용하면 데이터 검증이 자동으로 이루어집니다.
    예를 들어, title이 없으면 에러가 발생합니다.
    """
    title: str = Field(..., description="논문 제목")
    authors: List[str] = Field(default_factory=list, description="저자 목록")
    abstract: str = Field("", description="논문 초록")
    url: str = Field("", description="논문 URL")
    published_date: str = Field("", description="출판일")
    source: str = Field("arXiv", description="논문 출처 (arXiv, Semantic Scholar 등)")
    relevance_score: float = Field(0.0, description="연관성 점수 (0-1)")
    summary: Optional[str] = Field(None, description="AI가 생성한 요약")


class ReActStep(BaseModel):
    """
    ReAct 패턴의 각 단계를 기록합니다.
    
    ReAct는 Reasoning + Acting의 약자로, AI가 생각(Thought)하고 
    행동(Action)하고 관찰(Observation)하는 과정을 명시적으로 기록합니다.
    
    이렇게 기록하면:
    1. AI의 사고 과정을 추적할 수 있습니다
    2. 문제가 생겼을 때 어디서 잘못되었는지 파악하기 쉽습니다
    3. 사용자에게 AI가 어떻게 결론에 도달했는지 설명할 수 있습니다
    """
    step_type: Literal["thought", "action", "observation"] = Field(
        ..., description="단계 유형"
    )
    content: str = Field(..., description="단계 내용")
    timestamp: datetime = Field(default_factory=datetime.now, description="시간")


class InterruptData(BaseModel):
    """
    Human-in-the-Loop Interrupt 시 필요한 데이터입니다.
    
    Interrupt가 발생하면 이 데이터가 사용자에게 전달되고,
    사용자의 응답을 기다립니다.
    """
    interrupt_type: Literal[
        "confirm_keywords",      # 키워드 확인 요청
        "select_paper_count",    # 논문 수 선택 요청
        "review_results",        # 검색 결과 검토 요청
        "confirm_continue"       # 계속 진행 여부 확인
    ] = Field(..., description="Interrupt 유형")
    
    message: str = Field(..., description="사용자에게 보여줄 메시지")
    options: Optional[List[str]] = Field(None, description="선택 가능한 옵션들")
    default_value: Optional[str] = Field(None, description="기본값")
    metadata: dict = Field(default_factory=dict, description="추가 메타데이터")


# ============================================
# LangGraph 상태 정의
# ============================================

class AgentState(TypedDict):
    """
    LangGraph 워크플로우의 메인 상태입니다.
    
    TypedDict를 사용하면 각 필드의 타입을 명시할 수 있습니다.
    이렇게 하면 IDE에서 자동 완성이 잘 작동하고,
    타입 에러를 미리 잡을 수 있습니다.
    
    Annotated와 operator.add:
    -------------------------
    일부 필드는 Annotated[List[...], operator.add]로 정의되어 있습니다.
    이것은 "이 필드는 덮어쓰기 대신 추가(append)한다"는 의미입니다.
    
    예를 들어, react_steps 필드에 새 단계를 추가하면,
    기존 단계들이 지워지지 않고 새 단계가 추가됩니다.
    """
    
    # === 입력 데이터 ===
    # 사용자의 원본 질문
    user_question: str
    
    # 세션 ID (대화 식별용)
    session_id: str
    
    # === 분석 결과 ===
    # 추출된 키워드 목록
    extracted_keywords: List[str]
    
    # 질문의 의도 (예: "최신 연구 동향", "특정 기술 설명" 등)
    question_intent: str
    
    # 질문의 도메인 (예: "computer science", "physics" 등)
    question_domain: str
    
    # === 검색 설정 (Human-in-the-Loop으로 결정됨) ===
    # 검색할 논문 수
    paper_count: int
    
    # 사용할 검색 소스 목록
    selected_sources: List[str]
    
    # === 검색 결과 ===
    # 검색된 논문들 (리스트에 추가되는 방식)
    papers: Annotated[List[Paper], operator.add]
    
    # 연관성이 높은 논문들 (필터링 후)
    relevant_papers: List[Paper]
    
    # === ReAct 단계 기록 ===
    # AI의 사고-행동-관찰 과정 기록
    react_steps: Annotated[List[ReActStep], operator.add]
    
    # === Interrupt 관련 ===
    # 현재 Interrupt 데이터 (없으면 None)
    interrupt_data: Optional[InterruptData]
    
    # 사용자의 Interrupt 응답
    user_response: Optional[str]
    
    # Interrupt 대기 중인지 여부
    waiting_for_user: bool
    
    # === 출력 ===
    # 최종 응답 텍스트
    final_response: str
    
    # 에러 메시지 (있는 경우)
    error_message: Optional[str]
    
    # 워크플로우 완료 여부
    is_complete: bool


def create_initial_state(
    user_question: str,
    session_id: str = "default"
) -> AgentState:
    """
    초기 상태를 생성합니다.
    
    새로운 사용자 질문이 들어오면 이 함수를 호출하여
    워크플로우의 시작 상태를 만듭니다.
    
    Args:
        user_question: 사용자의 질문
        session_id: 세션 식별자 (대화 추적용)
    
    Returns:
        AgentState: 초기화된 상태 딕셔너리
    
    Example:
        >>> state = create_initial_state(
        ...     user_question="자율주행 LiDAR 최신 기술",
        ...     session_id="user123_session1"
        ... )
        >>> print(state["user_question"])
        "자율주행 LiDAR 최신 기술"
    """
    return AgentState(
        # 입력
        user_question=user_question,
        session_id=session_id,
        
        # 분석 결과 (아직 비어있음)
        extracted_keywords=[],
        question_intent="",
        question_domain="",
        
        # 검색 설정 (기본값, 나중에 사용자가 변경 가능)
        paper_count=3,
        selected_sources=["arxiv"],  # 기본적으로 arXiv만 사용
        
        # 검색 결과 (아직 비어있음)
        papers=[],
        relevant_papers=[],
        
        # ReAct 기록 (아직 비어있음)
        react_steps=[],
        
        # Interrupt (아직 없음)
        interrupt_data=None,
        user_response=None,
        waiting_for_user=False,
        
        # 출력 (아직 없음)
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
    
    이 함수는 노드에서 상태를 업데이트할 때 사용합니다.
    반환값을 노드의 반환값에 포함시키면 LangGraph가 자동으로
    상태를 업데이트합니다.
    
    Args:
        state: 현재 상태
        step_type: 단계 유형 ("thought", "action", "observation")
        content: 단계 내용
    
    Returns:
        dict: 상태 업데이트 딕셔너리
    
    Example:
        >>> def analyze_node(state):
        ...     # 분석 로직 수행
        ...     return add_react_step(
        ...         state, 
        ...         "thought", 
        ...         "사용자가 자율주행 기술에 대해 물어보고 있습니다."
        ...     )
    """
    new_step = ReActStep(
        step_type=step_type,
        content=content
    )
    
    return {"react_steps": [new_step]}

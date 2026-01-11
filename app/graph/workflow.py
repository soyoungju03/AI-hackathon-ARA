"""
LangGraph 워크플로우 정의
========================

이 파일은 전체 워크플로우 그래프를 구성합니다.

LangGraph 워크플로우란?
-----------------------
LangGraph는 LangChain 팀이 만든 라이브러리로,
복잡한 AI 워크플로우를 "그래프" 형태로 정의할 수 있게 해줍니다.

그래프의 구성 요소:
1. 노드(Node): 각 처리 단계 (함수)
2. 엣지(Edge): 노드 간의 연결
3. 상태(State): 노드들 사이에서 전달되는 데이터

워크플로우 흐름:
---------------
START → receive_question → analyze_question → request_user_confirmation
                                                        ↓
                                              [INTERRUPT: 사용자 입력 대기]
                                                        ↓
        process_user_response → search_papers → evaluate_relevance
                                                        ↓
                        summarize_papers → generate_response → END
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 로컬 모듈 임포트
from app.graph.state import AgentState, create_initial_state
from app.graph.nodes import (
    receive_question_node,
    analyze_question_node,
    request_user_confirmation_node,
    process_user_response_node,
    search_papers_node,
    evaluate_relevance_node,
    summarize_papers_node,
    generate_response_node
)


def should_continue_after_interrupt(state: AgentState) -> Literal["process_response", "wait"]:
    """
    Interrupt 후 다음 단계를 결정하는 조건부 엣지 함수입니다.
    
    사용자가 응답했으면 진행하고, 아직 대기 중이면 계속 대기합니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        str: 다음 노드 이름 또는 "wait"
    """
    if state.get("waiting_for_user", False):
        return "wait"
    return "process_response"


def check_search_results(state: AgentState) -> Literal["evaluate", "error"]:
    """
    검색 결과를 확인하고 다음 단계를 결정합니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        str: "evaluate" (정상) 또는 "error" (에러 발생 시)
    """
    if state.get("error_message"):
        return "error"
    return "evaluate"


def build_research_workflow() -> StateGraph:
    """
    연구 어시스턴트 워크플로우 그래프를 구축합니다.
    
    이 함수는 전체 워크플로우의 구조를 정의합니다.
    각 노드를 추가하고, 노드 간의 연결(엣지)을 설정합니다.
    
    Returns:
        StateGraph: 컴파일된 워크플로우 그래프
    
    워크플로우 다이어그램:
    
        ┌─────────────────┐
        │      START      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │receive_question │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │analyze_question │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │request_confirm  │ ← INTERRUPT HERE
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │process_response │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ search_papers   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │evaluate_relevance│
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │summarize_papers │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │generate_response│
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │       END       │
        └─────────────────┘
    """
    
    # StateGraph 초기화
    # AgentState를 상태 타입으로 사용합니다
    workflow = StateGraph(AgentState)
    
    # ============================================
    # 노드 추가
    # ============================================
    
    # 각 노드를 그래프에 추가합니다
    # 노드 이름은 문자열이고, 두 번째 인자는 실제 함수입니다
    
    workflow.add_node("receive_question", receive_question_node)
    workflow.add_node("analyze_question", analyze_question_node)
    workflow.add_node("request_confirmation", request_user_confirmation_node)
    workflow.add_node("process_response", process_user_response_node)
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)
    workflow.add_node("summarize_papers", summarize_papers_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # ============================================
    # 엣지 추가 (노드 간 연결)
    # ============================================
    
    # 시작점 설정
    # 워크플로우는 "receive_question" 노드에서 시작합니다
    workflow.set_entry_point("receive_question")
    
    # 순차적 엣지 추가
    # add_edge(A, B)는 "A 노드 다음에 B 노드가 실행된다"를 의미합니다
    
    workflow.add_edge("receive_question", "analyze_question")
    workflow.add_edge("analyze_question", "request_confirmation")
    
    # Human-in-the-Loop: request_confirmation 후에는 
    # 사용자 응답을 기다려야 하므로 process_response로 연결합니다
    # 실제 Interrupt는 워크플로우 실행 시 처리됩니다
    workflow.add_edge("request_confirmation", "process_response")
    
    workflow.add_edge("process_response", "search_papers")
    
    # 조건부 엣지: 검색 결과에 따라 분기
    workflow.add_conditional_edges(
        "search_papers",
        check_search_results,
        {
            "evaluate": "evaluate_relevance",
            "error": "generate_response"  # 에러 시 바로 응답 생성으로
        }
    )
    
    workflow.add_edge("evaluate_relevance", "summarize_papers")
    workflow.add_edge("summarize_papers", "generate_response")
    
    # 종료점 설정
    workflow.add_edge("generate_response", END)
    
    return workflow


def create_research_agent(checkpointer=None):
    """
    연구 어시스턴트 에이전트를 생성합니다.
    
    이 함수는 워크플로우를 빌드하고 컴파일하여
    실행 가능한 에이전트를 반환합니다.
    
    Args:
        checkpointer: 상태 저장을 위한 체크포인터 (선택)
                     Human-in-the-Loop을 위해 필요합니다.
    
    Returns:
        CompiledGraph: 컴파일된 워크플로우 (실행 가능)
    
    Example:
        >>> agent = create_research_agent()
        >>> result = agent.invoke(initial_state)
    """
    
    # 워크플로우 빌드
    workflow = build_research_workflow()
    
    # 체크포인터가 없으면 메모리 기반 체크포인터 사용
    # 체크포인터는 워크플로우의 상태를 저장하여
    # Interrupt 후에도 상태를 유지할 수 있게 해줍니다
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # 워크플로우 컴파일
    # interrupt_before를 설정하면 해당 노드 실행 전에 워크플로우가 중지됩니다
    # 이것이 Human-in-the-Loop의 핵심입니다!
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_response"]  # 사용자 응답 처리 전에 중지
    )
    
    return compiled


class ResearchAssistant:
    """
    연구 어시스턴트 클래스입니다.
    
    이 클래스는 워크플로우 실행을 관리하고,
    Human-in-the-Loop 상호작용을 처리합니다.
    
    사용법:
    
    1. 기본 사용 (Interrupt 없이 자동 실행):
        >>> assistant = ResearchAssistant()
        >>> response = assistant.run("자율주행 LiDAR 기술")
        >>> print(response)
    
    2. Human-in-the-Loop 사용:
        >>> assistant = ResearchAssistant()
        >>> 
        >>> # 첫 번째 실행 (Interrupt까지)
        >>> result = assistant.start("자율주행 LiDAR 기술")
        >>> print(result.interrupt_message)  # 사용자에게 보여줄 메시지
        >>> 
        >>> # 사용자 응답으로 계속 실행
        >>> final = assistant.continue_with_response("5")  # 5개 논문 검색
        >>> print(final.response)
    """
    
    def __init__(self):
        """어시스턴트를 초기화합니다."""
        self.checkpointer = MemorySaver()
        self.agent = create_research_agent(self.checkpointer)
        self.current_thread_id = None
        self.current_state = None
    
    def run(
        self, 
        question: str, 
        paper_count: int = 3,
        session_id: str = "default"
    ) -> str:
        """
        질문에 대해 자동으로 전체 워크플로우를 실행합니다.
        
        이 메서드는 Human-in-the-Loop 없이 자동으로 진행됩니다.
        기본 논문 수(paper_count)를 사용합니다.
        
        Args:
            question: 사용자의 질문
            paper_count: 검색할 논문 수 (기본값: 3)
            session_id: 세션 식별자
        
        Returns:
            str: 최종 응답 텍스트
        """
        # 초기 상태 생성
        initial_state = create_initial_state(question, session_id)
        initial_state["paper_count"] = paper_count
        initial_state["user_response"] = str(paper_count)
        initial_state["waiting_for_user"] = False
        
        # 고유한 thread_id 생성 (상태 추적용)
        import uuid
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # 워크플로우 실행
        try:
            # stream 대신 invoke 사용 (전체 실행)
            final_state = self.agent.invoke(initial_state, config)
            return final_state.get("final_response", "응답을 생성할 수 없습니다.")
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
    
    def start(self, question: str, session_id: str = "default") -> dict:
        """
        워크플로우를 시작하고 첫 번째 Interrupt에서 멈춥니다.
        
        Human-in-the-Loop을 사용할 때 이 메서드로 시작합니다.
        Interrupt 지점에서 멈추고, 사용자에게 보여줄 정보를 반환합니다.
        
        Args:
            question: 사용자의 질문
            session_id: 세션 식별자
        
        Returns:
            dict: Interrupt 정보 (메시지, 옵션 등)
        """
        import uuid
        
        # 초기 상태 생성
        initial_state = create_initial_state(question, session_id)
        
        # 고유한 thread_id 생성
        self.current_thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        # 워크플로우 실행 (Interrupt까지)
        try:
            # stream을 사용하여 Interrupt 지점까지 실행
            for event in self.agent.stream(initial_state, config):
                self.current_state = event
            
            # 현재 상태에서 Interrupt 정보 추출
            # LangGraph의 get_state를 사용하여 현재 상태 확인
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            if current_values.get("interrupt_data"):
                interrupt_data = current_values["interrupt_data"]
                return {
                    "status": "waiting_for_input",
                    "message": interrupt_data.message,
                    "options": interrupt_data.options,
                    "keywords": current_values.get("extracted_keywords", []),
                    "thread_id": self.current_thread_id
                }
            else:
                # Interrupt 없이 완료된 경우
                return {
                    "status": "completed",
                    "response": current_values.get("final_response", ""),
                    "thread_id": self.current_thread_id
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "thread_id": self.current_thread_id
            }
    
    def continue_with_response(self, user_response: str) -> dict:
        """
        사용자 응답을 받아 워크플로우를 계속 실행합니다.
        
        start() 메서드 이후 호출되어야 합니다.
        사용자의 응답을 상태에 추가하고 나머지 워크플로우를 실행합니다.
        
        Args:
            user_response: 사용자의 응답 (예: 논문 수 선택)
        
        Returns:
            dict: 최종 결과 또는 다음 Interrupt 정보
        """
        if not self.current_thread_id:
            return {
                "status": "error",
                "message": "먼저 start()를 호출해주세요."
            }
        
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            # 사용자 응답을 상태에 업데이트
            self.agent.update_state(
                config,
                {
                    "user_response": user_response,
                    "waiting_for_user": False
                }
            )
            
            # 워크플로우 계속 실행
            for event in self.agent.stream(None, config):
                self.current_state = event
            
            # 최종 상태 확인
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            if current_values.get("is_complete"):
                return {
                    "status": "completed",
                    "response": current_values.get("final_response", ""),
                    "papers": current_values.get("relevant_papers", []),
                    "react_steps": current_values.get("react_steps", [])
                }
            else:
                # 또 다른 Interrupt가 있는 경우
                return {
                    "status": "waiting_for_input",
                    "message": current_values.get("interrupt_data", {}).get("message", ""),
                    "thread_id": self.current_thread_id
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


# 모듈 레벨에서 기본 어시스턴트 인스턴스 생성
default_assistant = None

def get_assistant() -> ResearchAssistant:
    """
    기본 어시스턴트 인스턴스를 반환합니다.
    
    싱글톤 패턴을 사용하여 하나의 인스턴스만 유지합니다.
    """
    global default_assistant
    if default_assistant is None:
        default_assistant = ResearchAssistant()
    return default_assistant

# app/graph/workflow.py
# -*- coding: utf-8 -*-
"""
완전히 수정된 LangGraph 워크플로우 (노드 기반 라우팅 + 의미 기반 검색)
================================================================================

이 워크플로우는 학술 논문 검색을 위한 복잡한 상호작용을 관리합니다.
핵심 특징은 다음과 같습니다:

1. 노드 기반 라우팅: 조건부 함수 대신 명시적인 라우팅 노드를 사용하여
   각 결정 지점을 명확하게 추적할 수 있습니다.

2. 두 단계 Human-in-the-Loop: 사용자가 키워드 확인과 논문 수 선택 단계에서
   개입하여 검색의 정확도를 높입니다.

3. 의미 기반 검색: Sentence Transformers를 사용하여 논문의 관련성을
   정량적으로 평가합니다. 이것은 evaluate_relevance_node에서 수행됩니다.

4. ReAct 패턴: 모든 노드가 Thought-Action-Observation 패턴을 따라
   AI의 사고 과정을 투명하게 보여줍니다.
"""

from typing import Literal
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState, create_initial_state
from app.graph.nodes import (
    receive_question_node,
    analyze_question_node,
    request_keyword_confirmation_node,
    process_keyword_confirmation_response_node,
    request_paper_count_node,
    process_paper_count_response_node,
    search_papers_node,
    evaluate_relevance_node,  # 이 노드가 의미 기반 검색을 수행합니다
    summarize_papers_node,
    generate_response_node
)

logger = logging.getLogger(__name__)


# ============================================
# 라우팅 노드들 (별도로 관리됨)
# ============================================

def check_keyword_confirmation_status_node(state: AgentState) -> dict:
    """
    키워드 확인 후 상태를 검사하고 다음 단계를 결정합니다.
    
    이 노드는 사용자의 첫 번째 응답을 처리한 후 호출됩니다.
    사용자가 "확인"을 선택했는지 "다시"를 선택했는지에 따라
    워크플로우의 경로가 달라집니다.
    
    "다시"를 선택한 경우, analyze_question 노드로 돌아가서
    질문을 다시 분석합니다. 이렇게 하면 AI가 다른 관점에서
    키워드를 추출할 수 있습니다.
    
    "확인"을 선택한 경우, request_paper_count 노드로 진행하여
    사용자에게 논문 개수를 선택받습니다.
    """
    
    # 현재 상태를 자세히 로깅합니다. 이것은 디버깅과 모니터링에 유용합니다.
    waiting_for = state.get("waiting_for")
    waiting_for_user = state.get("waiting_for_user")
    keyword_response = state.get("keyword_confirmation_response")
    
    logger.info("=" * 60)
    logger.info("[CHECK_KEYWORD_CONFIRMATION_STATUS] 상태 검사 시작")
    logger.info(f"  waiting_for: {waiting_for}")
    logger.info(f"  waiting_for_user: {waiting_for_user}")
    logger.info(f"  keyword_confirmation_response: {keyword_response}")
    logger.info("=" * 60)
    
    # 사용자의 응답을 검증하고 다음 노드를 결정합니다
    if keyword_response == "retry":
        logger.info("  → 사용자가 '다시'를 선택했습니다")
        logger.info("  → 다음 노드: analyze_question (재분석)")
        return {
            "next_node": "analyze_question"
        }
    
    elif keyword_response == "confirmed":
        logger.info("  → 사용자가 '확인'을 선택했습니다")
        logger.info("  → 다음 노드: request_paper_count (논문 수 선택)")
        return {
            "next_node": "request_paper_count"
        }
    
    else:
        # 예상치 못한 응답이 들어온 경우, 안전하게 진행합니다
        logger.warning(f"  ⚠️ 예상치 못한 응답: {keyword_response}")
        logger.info("  → 기본값으로 처리: request_paper_count로 진행")
        return {
            "next_node": "request_paper_count"
        }


def check_paper_count_status_node(state: AgentState) -> dict:
    """
    논문 수 선택 후 상태를 검사하고 다음 단계를 결정합니다.
    
    이 노드는 사용자의 두 번째 응답을 처리한 후 호출됩니다.
    사용자가 선택한 논문 수가 유효한 범위(1-10)에 있는지 확인하고,
    그렇지 않으면 기본값(3)을 사용합니다.
    
    모든 Human-in-the-Loop 단계가 완료되었으므로, 이제 실제
    논문 검색 단계로 진행할 준비가 되었습니다.
    """
    
    # 현재 상태를 자세히 로깅합니다
    waiting_for = state.get("waiting_for")
    waiting_for_user = state.get("waiting_for_user")
    paper_count = state.get("paper_count")
    user_response = state.get("user_response")
    
    logger.info("=" * 60)
    logger.info("[CHECK_PAPER_COUNT_STATUS] 상태 검사 시작")
    logger.info(f"  waiting_for: {waiting_for}")
    logger.info(f"  waiting_for_user: {waiting_for_user}")
    logger.info(f"  paper_count: {paper_count}")
    logger.info(f"  user_response: {user_response}")
    logger.info("=" * 60)
    
    # 논문 수가 유효한 범위에 있는지 검증합니다
    if paper_count is None or paper_count < 1 or paper_count > 10:
        logger.warning(f"  ⚠️ 유효하지 않은 논문 수: {paper_count}")
        logger.info("  → 기본값 3으로 설정")
        return {
            "paper_count": 3,
            "next_node": "search_papers"
        }
    
    logger.info(f"  ✓ 논문 수 확인됨: {paper_count}개")
    logger.info("  → 다음 노드: search_papers (논문 검색)")
    
    return {
        "next_node": "search_papers"
    }


# ============================================
# 워크플로우 빌드
# ============================================

def build_research_workflow() -> StateGraph:
    """
    노드 기반 라우팅을 사용하는 연구 어시스턴트 워크플로우를 구축합니다.
    
    이 워크플로우는 복잡한 상호작용을 체계적으로 관리합니다.
    각 노드는 명확한 책임을 가지고 있으며, 노드들 사이의 전환은
    명시적인 엣지로 정의됩니다.
    
    워크플로우 구조:
    
    START
      ↓
    receive_question (질문 수신)
      ↓
    analyze_question (질문 분석 및 키워드 추출)
      ↓
    request_keyword_confirmation ← [INTERRUPT 1: 사용자에게 키워드 확인 요청]
      ↓
    process_keyword_confirmation_response (사용자 응답 처리)
      ↓
    check_keyword_confirmation_status ← [라우팅: 다음 경로 결정]
      ↓
    [분기]
    ├─ analyze_question (사용자가 '다시' 선택 시)
    └─ request_paper_count ← [INTERRUPT 2: 논문 수 선택 요청]
      ↓
    process_paper_count_response (논문 수 응답 처리)
      ↓
    check_paper_count_status ← [라우팅: 검색 준비 완료]
      ↓
    search_papers (arXiv에서 논문 검색)
      ↓
    [조건부 분기]
    ├─ generate_response (검색 실패 시)
    └─ evaluate_relevance (검색 성공 시 - 의미 기반 평가)
      ↓
    summarize_papers (선별된 논문 요약 생성)
      ↓
    generate_response (최종 답변 생성)
      ↓
    END
    
    의미 기반 검색의 핵심은 evaluate_relevance 노드입니다.
    이 노드에서 Sentence Transformers를 사용하여 각 논문의
    관련성을 정량적으로 평가하고, 임계값 이상인 논문만 선별합니다.
    """
    
    # StateGraph 인스턴스를 생성합니다. AgentState 타입을 사용합니다.
    workflow = StateGraph(AgentState)
    
    # ============================================
    # 노드 추가
    # ============================================
    
    # 각 노드는 app/graph/nodes.py에 정의되어 있습니다.
    # 노드는 상태를 입력받아 업데이트된 상태를 반환하는 함수입니다.
    
    workflow.add_node("receive_question", receive_question_node)
    workflow.add_node("analyze_question", analyze_question_node)
    workflow.add_node("request_keyword_confirmation", request_keyword_confirmation_node)
    workflow.add_node("process_keyword_confirmation_response", process_keyword_confirmation_response_node)
    workflow.add_node("request_paper_count", request_paper_count_node)
    workflow.add_node("process_paper_count_response", process_paper_count_response_node)
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)  # 의미 기반 검색 노드
    workflow.add_node("summarize_papers", summarize_papers_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # 라우팅을 담당하는 특별한 노드들
    workflow.add_node("check_keyword_confirmation_status", check_keyword_confirmation_status_node)
    workflow.add_node("check_paper_count_status", check_paper_count_status_node)
    
    # ============================================
    # 엣지 추가
    # ============================================
    
    # 엣지는 노드들 사이의 전환을 정의합니다.
    # 단순 엣지는 항상 같은 경로로 진행하고,
    # 조건부 엣지는 상태에 따라 다른 경로로 진행합니다.
    
    # 시작점을 설정합니다
    workflow.set_entry_point("receive_question")
    
    # 초기 질문 처리 흐름
    workflow.add_edge("receive_question", "analyze_question")
    workflow.add_edge("analyze_question", "request_keyword_confirmation")
    workflow.add_edge("request_keyword_confirmation", "process_keyword_confirmation_response")
    
    # 첫 번째 라우팅 포인트
    workflow.add_edge("process_keyword_confirmation_response", "check_keyword_confirmation_status")
    
    # 키워드 확인 후 조건부 분기
    def route_after_keyword_check(state: AgentState) -> Literal["analyze_question", "request_paper_count"]:
        """
        check_keyword_confirmation_status 노드에서 설정한
        next_node 값을 기반으로 다음 경로를 결정합니다.
        
        이 함수는 LangGraph의 조건부 엣지에서 사용됩니다.
        반환값은 반드시 다음 노드의 이름이어야 합니다.
        """
        next_node = state.get("next_node")
        logger.info(f"[ROUTE_AFTER_KEYWORD_CHECK] 경로 결정: {next_node}")
        # 만약 next_node가 설정되지 않았다면 기본값을 사용합니다
        return next_node or "request_paper_count"
    
    workflow.add_conditional_edges(
        "check_keyword_confirmation_status",
        route_after_keyword_check,
        {
            "analyze_question": "analyze_question",
            "request_paper_count": "request_paper_count"
        }
    )
    
    # 논문 수 선택 흐름
    workflow.add_edge("request_paper_count", "process_paper_count_response")
    workflow.add_edge("process_paper_count_response", "check_paper_count_status")
    
    # 두 번째 라우팅 포인트 (항상 search_papers로 진행)
    workflow.add_edge("check_paper_count_status", "search_papers")
    
    # 검색 결과에 따른 조건부 분기
    def check_search_results(state: AgentState) -> Literal["evaluate_relevance", "generate_response"]:
        """
        검색 결과를 확인하고 다음 경로를 결정합니다.
        
        검색이 실패한 경우 (error_message가 있는 경우),
        바로 generate_response로 가서 에러 메시지를 사용자에게 전달합니다.
        
        검색이 성공한 경우, evaluate_relevance로 가서
        의미 기반 평가를 수행합니다. 이 단계에서 Sentence Transformers가
        각 논문의 관련성을 계산하고, 관련성 높은 논문만 선별합니다.
        """
        if state.get("error_message"):
            logger.info("[CHECK_SEARCH_RESULTS] 검색 실패 → generate_response")
            return "generate_response"
        logger.info("[CHECK_SEARCH_RESULTS] 검색 성공 → evaluate_relevance (의미 기반 평가)")
        return "evaluate_relevance"
    
    workflow.add_conditional_edges(
        "search_papers",
        check_search_results,
        {
            "evaluate_relevance": "evaluate_relevance",
            "generate_response": "generate_response"
        }
    )
    
    # 최종 응답 생성 흐름
    workflow.add_edge("evaluate_relevance", "summarize_papers")
    workflow.add_edge("summarize_papers", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow


# ============================================
# 에이전트 생성 및 실행
# ============================================

def create_research_agent(checkpointer=None):
    """
    연구 어시스턴트 에이전트를 생성합니다.
    
    이 함수는 워크플로우를 빌드하고 컴파일합니다.
    컴파일 과정에서 interrupt_before 파라미터를 사용하여
    Human-in-the-Loop 지점을 지정합니다.
    
    interrupt_before는 Interrupt가 발생할 노드 직전의 노드들을 지정합니다.
    따라서:
    - "process_keyword_confirmation_response" 직전에 멈춤
      → 사용자에게 키워드 확인 요청이 표시된 상태
    - "process_paper_count_response" 직전에 멈춤
      → 사용자에게 논문 수 선택 요청이 표시된 상태
    
    checkpointer는 워크플로우의 상태를 저장하고 복원하는 역할을 합니다.
    이것이 있어야 Interrupt에서 멈췄다가 나중에 다시 시작할 수 있습니다.
    """
    
    # 워크플로우를 빌드합니다
    workflow = build_research_workflow()
    
    # checkpointer가 제공되지 않았다면 기본값을 사용합니다
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # 워크플로우를 컴파일합니다
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "process_keyword_confirmation_response",  # 첫 번째 Interrupt
            "process_paper_count_response"             # 두 번째 Interrupt
        ]
    )
    
    logger.info("✓ 워크플로우 컴파일 완료 (의미 기반 검색 포함)")
    return compiled


# ============================================
# ResearchAssistant 클래스
# ============================================

class ResearchAssistant:
    """
    두 단계 Human-in-the-Loop과 의미 기반 검색을 지원하는 연구 어시스턴트입니다.
    
    이 클래스는 사용하기 쉬운 인터페이스를 제공합니다.
    내부적으로는 복잡한 LangGraph 워크플로우를 관리하지만,
    외부에서는 간단한 메서드 호출로 사용할 수 있습니다.
    
    사용 예:
        assistant = ResearchAssistant()
        
        # 대화형 모드 (Human-in-the-Loop)
        result = assistant.start("자율주행 기술")
        # ... 사용자가 응답 입력 ...
        result = assistant.continue_with_response("확인")
        # ... 사용자가 논문 수 입력 ...
        result = assistant.continue_with_response("5")
        
        # 또는 자동 모드 (빠른 검색)
        response = assistant.run("자율주행 기술", paper_count=5)
    """
    
    def __init__(self):
        """
        어시스턴트를 초기화합니다.
        
        checkpointer는 대화 상태를 메모리에 저장합니다.
        각 대화는 고유한 thread_id로 식별되므로,
        여러 사용자의 대화를 동시에 처리할 수 있습니다.
        """
        self.checkpointer = MemorySaver()
        self.agent = create_research_agent(self.checkpointer)
        self.current_thread_id = None
        self.current_state = None
        self.interrupt_count = 0
    
    def run(
        self, 
        question: str, 
        paper_count: int = 3,
        session_id: str = "default"
    ) -> str:
        """
        자동 실행 모드: Interrupt 없이 전체 워크플로우를 자동으로 실행합니다.
        
        이 모드는 빠른 검색에 사용됩니다. 사용자 개입 없이
        모든 단계를 자동으로 진행하고 최종 답변을 반환합니다.
        
        내부적으로는 키워드 확인과 논문 수 선택 단계를
        자동으로 "confirmed"와 지정된 paper_count로 설정합니다.
        
        Args:
            question: 사용자의 연구 질문
            paper_count: 검색할 논문의 개수 (1-10)
            session_id: 세션 식별자
            
        Returns:
            최종 답변 문자열
        """
        import uuid
        
        # 초기 상태를 생성합니다
        initial_state = create_initial_state(question, session_id)
        
        # Human-in-the-Loop을 우회하기 위한 설정
        initial_state["paper_count"] = paper_count
        initial_state["keyword_confirmation_response"] = "confirmed"
        initial_state["waiting_for_user"] = False
        
        # 고유한 thread_id를 생성합니다
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"[RUN MODE] 자동 실행 시작: {question[:50]}...")
            
            # 워크플로우를 한 번에 실행합니다 (invoke는 완료될 때까지 기다립니다)
            final_state = self.agent.invoke(initial_state, config)
            
            logger.info("[RUN MODE] 자동 실행 완료")
            return final_state.get("final_response", "응답을 생성할 수 없습니다.")
            
        except Exception as e:
            logger.error(f"[RUN MODE] 오류 발생: {str(e)}", exc_info=True)
            return f"오류가 발생했습니다: {str(e)}"
    
    def start(self, question: str, session_id: str = "default") -> dict:
        """
        워크플로우를 시작합니다.
        첫 번째 Interrupt (키워드 확인)에서 멈춥니다.
        
        이 메서드는 대화형 모드의 시작점입니다.
        질문을 받아 분석하고, 사용자에게 키워드 확인을 요청하는
        단계까지 진행한 후 멈춥니다.
        
        Args:
            question: 사용자의 연구 질문
            session_id: 세션 식별자
            
        Returns:
            상태 정보를 담은 딕셔너리:
            - status: "waiting_for_input" 또는 "error"
            - interrupt_stage: 현재 Interrupt 단계 (1 또는 2)
            - message: 사용자에게 보여줄 메시지
            - options: 선택 가능한 옵션들
            - keywords: 추출된 키워드 (첫 번째 Interrupt에만)
            - thread_id: 대화를 계속하기 위한 식별자
        """
        import uuid
        
        logger.info(f"[START MODE] 시작 - 질문: {question[:50]}...")
        
        # 초기 상태를 생성합니다
        initial_state = create_initial_state(question, session_id)
        
        # 새로운 thread_id를 생성합니다
        self.current_thread_id = str(uuid.uuid4())
        self.interrupt_count = 0
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            # 워크플로우를 스트리밍 방식으로 실행합니다
            # 첫 번째 Interrupt에서 자동으로 멈춥니다
            logger.info("[START MODE] 워크플로우 실행 중...")
            for event in self.agent.stream(initial_state, config):
                self.current_state = event
                logger.debug(f"  Event received: {list(event.keys())}")
            
            # 현재 상태를 가져옵니다
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            self.interrupt_count = 1
            logger.info("[START MODE] 첫 번째 Interrupt 도달")
            
            # Interrupt 데이터를 확인합니다
            if current_values.get("interrupt_data"):
                interrupt_data = current_values["interrupt_data"]
                logger.info(f"  Interrupt Type: {interrupt_data.interrupt_type}")
                logger.info(f"  Keywords: {current_values.get('extracted_keywords', [])}")
                
                return {
                    "status": "waiting_for_input",
                    "interrupt_stage": 1,
                    "message": interrupt_data.message,
                    "options": interrupt_data.options,
                    "keywords": current_values.get("extracted_keywords", []),
                    "thread_id": self.current_thread_id
                }
            else:
                # Interrupt 데이터가 없다면 이미 완료된 것입니다
                logger.warning("[START MODE] Interrupt 데이터를 찾을 수 없습니다")
                return {
                    "status": "completed",
                    "response": current_values.get("final_response", ""),
                    "thread_id": self.current_thread_id
                }
                
        except Exception as e:
            logger.error(f"[START MODE] 오류: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "thread_id": self.current_thread_id
            }
    
    def continue_with_response(self, user_response: str) -> dict:
        """
        사용자 응답을 받아 워크플로우를 계속 실행합니다.
        
        이 메서드는 사용자가 키워드 확인이나 논문 수 선택에
        응답한 후 호출됩니다. 사용자의 응답을 상태에 반영하고,
        워크플로우를 계속 실행합니다.
        
        두 번째 Interrupt에 도달하면 다시 멈추고,
        모든 Interrupt를 통과하면 최종 답변을 반환합니다.
        
        Args:
            user_response: 사용자의 응답 문자열
            
        Returns:
            상태 정보를 담은 딕셔너리:
            - status: "waiting_for_input", "completed", 또는 "error"
            - interrupt_stage: 현재 단계
            - message: 다음 Interrupt 메시지 (있는 경우)
            - response: 최종 답변 (완료된 경우)
            - papers: 검색된 논문 목록 (완료된 경우)
        """
        
        logger.info(f"[CONTINUE MODE] 사용자 응답 수신 (Stage {self.interrupt_count}): {user_response}")
        
        if not self.current_thread_id:
            logger.error("[CONTINUE MODE] thread_id가 설정되지 않았습니다")
            return {
                "status": "error",
                "message": "먼저 start()를 호출해주세요."
            }
        
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            # 현재 Interrupt 단계에 따라 상태를 업데이트합니다
            if self.interrupt_count == 1:
                # 첫 번째 Interrupt: 키워드 확인 응답 처리
                logger.info("[CONTINUE MODE] Stage 1: 키워드 확인 응답 처리")
                
                # 사용자 응답을 정규화합니다
                normalized_response = user_response.strip().lower()
                if normalized_response in ["다시", "retry", "다시하기", "수정"]:
                    keyword_response = "retry"
                    logger.info("  → '다시' 선택 감지")
                else:
                    keyword_response = "confirmed"
                    logger.info("  → '확인' 선택 감지")
                
                # 상태를 업데이트합니다
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "keyword_confirmation_response": keyword_response,
                        "waiting_for_user": False
                    }
                )
            
            elif self.interrupt_count == 2:
                # 두 번째 Interrupt: 논문 수 응답 처리
                logger.info("[CONTINUE MODE] Stage 2: 논문 수 응답 처리")
                logger.info(f"  → 선택된 논문 수: {user_response}")
                
                # 상태를 업데이트합니다
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "waiting_for_user": False
                    }
                )
            
            # 워크플로우를 계속 실행합니다
            logger.info("[CONTINUE MODE] 워크플로우 계속 실행 중...")
            for event in self.agent.stream(None, config):
                self.current_state = event
                logger.debug(f"  Event received: {list(event.keys())}")
            
            # 최종 상태를 확인합니다
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            logger.info("[CONTINUE MODE] 실행 완료, 상태 확인")
            
            # 워크플로우가 완료되었는지 확인합니다
            if current_values.get("is_complete"):
                logger.info("[CONTINUE MODE] ✓ 워크플로우 완료")
                return {
                    "status": "completed",
                    "interrupt_stage": self.interrupt_count,
                    "response": current_values.get("final_response", ""),
                    "papers": current_values.get("relevant_papers", []),
                    "thread_id": self.current_thread_id
                }
            
            # 다음 Interrupt가 있는지 확인합니다
            if current_values.get("interrupt_data"):
                self.interrupt_count += 1
                interrupt_data = current_values["interrupt_data"]
                logger.info(f"[CONTINUE MODE] → 다음 Interrupt: Stage {self.interrupt_count}")
                
                return {
                    "status": "waiting_for_input",
                    "interrupt_stage": self.interrupt_count,
                    "message": interrupt_data.message,
                    "options": interrupt_data.options,
                    "thread_id": self.current_thread_id
                }
            
            # 예상 외의 상황: 완료도 아니고 Interrupt도 없음
            logger.warning("[CONTINUE MODE] 예상 외의 상태")
            return {
                "status": "unknown",
                "message": "워크플로우 상태를 파악할 수 없습니다.",
                "thread_id": self.current_thread_id
            }
                
        except Exception as e:
            logger.error(f"[CONTINUE MODE] 오류 발생: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "thread_id": self.current_thread_id
            }


# ============================================
# 싱글톤 패턴
# ============================================

_default_assistant = None

def get_assistant() -> ResearchAssistant:
    """
    전역 어시스턴트 인스턴스를 반환합니다.
    
    싱글톤 패턴을 사용하여 앱 전체에서 하나의 어시스턴트 인스턴스만
    사용합니다. 이렇게 하면 메모리를 절약하고, 모델 로딩 시간을
    최소화할 수 있습니다.
    """
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = ResearchAssistant()
    return _default_assistant
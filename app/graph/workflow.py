# -*- coding: utf-8 -*-
"""
완전히 수정된 LangGraph 워크플로우 (노드 기반 라우팅)
====================================================

라우팅을 조건부 함수가 아닌 별도의 노드로 관리합니다.
이렇게 하면 각 단계에서 정확히 무슨 일이 일어나는지 로깅하고 추적할 수 있습니다.

노드 구조:
- 키워드 확인 라우팅: check_keyword_confirmation_status_node
- 논문 수 선택 라우팅: check_paper_count_status_node

이 노드들은 상태를 검사하고, 필요하면 수정하고, 다음 단계로의 경로를 결정합니다.
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
    evaluate_relevance_node,
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
    
    이 노드의 목적:
    1. 현재 상태를 상세히 로깅합니다 (디버깅용)
    2. 사용자의 응답을 검증합니다
    3. 상태를 정리합니다 (waiting_for, waiting_for_user 초기화)
    4. 다음 단계로의 경로를 결정합니다
    
    반환값:
    - "next_node": 다음 노드 이름
    """
    
    # 현재 상태 로깅 (디버깅용)
    waiting_for = state.get("waiting_for")
    waiting_for_user = state.get("waiting_for_user")
    keyword_response = state.get("keyword_confirmation_response")
    
    logger.info("=" * 60)
    logger.info("[CHECK_KEYWORD_CONFIRMATION_STATUS] 상태 검사 시작")
    logger.info(f"  waiting_for: {waiting_for}")
    logger.info(f"  waiting_for_user: {waiting_for_user}")
    logger.info(f"  keyword_confirmation_response: {keyword_response}")
    logger.info("=" * 60)
    
    # 상태 검증
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
        logger.warning(f"  ⚠️ 예상치 못한 응답: {keyword_response}")
        logger.info("  → 기본값으로 처리: request_paper_count로 진행")
        return {
            "next_node": "request_paper_count"
        }


def check_paper_count_status_node(state: AgentState) -> dict:
    """
    논문 수 선택 후 상태를 검사하고 다음 단계를 결정합니다.
    
    이 노드의 목적:
    1. 현재 상태를 상세히 로깅합니다 (디버깅용)
    2. 논문 수가 제대로 설정되었는지 검증합니다
    3. 상태를 정리합니다 (waiting_for, waiting_for_user 초기화)
    4. 검색 단계로 진행합니다
    
    반환값:
    - "next_node": 다음 노드 이름 (항상 "search_papers")
    """
    
    # 현재 상태 로깅 (디버깅용)
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
    
    # 논문 수 검증
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
    노드 기반 라우팅을 사용하는 워크플로우를 구축합니다.
    
    워크플로우 구조:
    
    START
      ↓
    receive_question
      ↓
    analyze_question
      ↓
    request_keyword_confirmation ← [INTERRUPT 1]
      ↓
    process_keyword_confirmation_response
      ↓
    check_keyword_confirmation_status ← [라우팅 노드]
      ↓
    [분기]
    ├─ analyze_question (사용자가 '다시' 선택)
    └─ request_paper_count (사용자가 '확인' 선택) ← [INTERRUPT 2]
      ↓
    process_paper_count_response
      ↓
    check_paper_count_status ← [라우팅 노드]
      ↓
    search_papers
      ↓
    [조건부 분기]
    ├─ generate_response (검색 실패)
    └─ evaluate_relevance (검색 성공)
      ↓
    summarize_papers
      ↓
    generate_response
      ↓
    END
    """
    
    workflow = StateGraph(AgentState)
    
    # ============================================
    # 노드 추가
    # ============================================
    
    # 기존 노드들
    workflow.add_node("receive_question", receive_question_node)
    workflow.add_node("analyze_question", analyze_question_node)
    workflow.add_node("request_keyword_confirmation", request_keyword_confirmation_node)
    workflow.add_node("process_keyword_confirmation_response", process_keyword_confirmation_response_node)
    workflow.add_node("request_paper_count", request_paper_count_node)
    workflow.add_node("process_paper_count_response", process_paper_count_response_node)
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)
    workflow.add_node("summarize_papers", summarize_papers_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # 새로운 라우팅 노드들
    workflow.add_node("check_keyword_confirmation_status", check_keyword_confirmation_status_node)
    workflow.add_node("check_paper_count_status", check_paper_count_status_node)
    
    # ============================================
    # 엣지 추가
    # ============================================
    
    # 시작점
    workflow.set_entry_point("receive_question")
    
    # 기본 흐름
    workflow.add_edge("receive_question", "analyze_question")
    workflow.add_edge("analyze_question", "request_keyword_confirmation")
    workflow.add_edge("request_keyword_confirmation", "process_keyword_confirmation_response")
    
    # 첫 번째 라우팅: 키워드 확인 후
    workflow.add_edge("process_keyword_confirmation_response", "check_keyword_confirmation_status")
    
    # check_keyword_confirmation_status에서 분기
    # 이 노드에서 "next_node" 필드를 반환하므로, 우리는 조건부 엣지를 사용합니다
    def route_after_keyword_check(state: AgentState) -> Literal["analyze_question", "request_paper_count"]:
        """check_keyword_confirmation_status 노드에서 다음 경로를 결정합니다."""
        next_node = state.get("next_node")
        logger.info(f"[ROUTE_AFTER_KEYWORD_CHECK] 경로 결정: {next_node}")
        return next_node or "request_paper_count"
    
    workflow.add_conditional_edges(
        "check_keyword_confirmation_status",
        route_after_keyword_check,
        {
            "analyze_question": "analyze_question",
            "request_paper_count": "request_paper_count"
        }
    )
    
    # 두 번째 Interrupt
    workflow.add_edge("request_paper_count", "process_paper_count_response")
    workflow.add_edge("process_paper_count_response", "check_paper_count_status")
    
    # 두 번째 라우팅: 논문 수 확인 후 (항상 search_papers로 진행)
    workflow.add_edge("check_paper_count_status", "search_papers")
    
    # 검색 결과에 따른 분기
    def check_search_results(state: AgentState) -> Literal["evaluate_relevance", "generate_response"]:
        """검색 결과를 확인하고 다음 경로를 결정합니다."""
        if state.get("error_message"):
            logger.info("[CHECK_SEARCH_RESULTS] 검색 실패 → generate_response")
            return "generate_response"
        logger.info("[CHECK_SEARCH_RESULTS] 검색 성공 → evaluate_relevance")
        return "evaluate_relevance"
    
    workflow.add_conditional_edges(
        "search_papers",
        check_search_results,
        {
            "evaluate_relevance": "evaluate_relevance",
            "generate_response": "generate_response"
        }
    )
    
    # 최종 흐름
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
    
    중요: interrupt_before는 Interrupt 직전의 노드들을 지정합니다.
    따라서:
    - "process_keyword_confirmation_response" 실행 직전에 Interrupt (첫 번째 응답 받기)
    - "process_paper_count_response" 실행 직전에 Interrupt (두 번째 응답 받기)
    """
    
    workflow = build_research_workflow()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "process_keyword_confirmation_response",  # 첫 번째 Interrupt
            "process_paper_count_response"             # 두 번째 Interrupt
        ]
    )
    
    logger.info("✓ 워크플로우 컴파일 완료")
    return compiled


# ============================================
# ResearchAssistant 클래스
# ============================================

class ResearchAssistant:
    """
    두 단계 Human-in-the-Loop을 지원하는 연구 어시스턴트입니다.
    
    노드 기반 라우팅을 사용하므로, 각 단계에서 정확히 무슨 일이
    일어나는지 로깅으로 추적할 수 있습니다.
    """
    
    def __init__(self):
        """어시스턴트를 초기화합니다."""
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
        """
        import uuid
        
        initial_state = create_initial_state(question, session_id)
        initial_state["paper_count"] = paper_count
        initial_state["keyword_confirmation_response"] = "confirmed"
        initial_state["waiting_for_user"] = False
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"[RUN MODE] 자동 실행 시작: {question[:50]}...")
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
        """
        import uuid
        
        logger.info(f"[START MODE] 시작 - 질문: {question[:50]}...")
        
        initial_state = create_initial_state(question, session_id)
        
        self.current_thread_id = str(uuid.uuid4())
        self.interrupt_count = 0
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            # 워크플로우 실행 (첫 번째 Interrupt까지)
            logger.info("[START MODE] 워크플로우 실행 중...")
            for event in self.agent.stream(initial_state, config):
                self.current_state = event
                logger.debug(f"  Event received: {list(event.keys())}")
            
            # 현재 상태 확인
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            self.interrupt_count = 1
            logger.info("[START MODE] 첫 번째 Interrupt 도달")
            
            # 첫 번째 Interrupt 데이터 확인
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
            # 사용자 응답에 따라 상태 업데이트
            if self.interrupt_count == 1:
                logger.info("[CONTINUE MODE] Stage 1: 키워드 확인 응답 처리")
                
                # "확인" 또는 "다시"로 정규화
                normalized_response = user_response.strip().lower()
                if normalized_response in ["다시", "retry", "다시하기", "수정"]:
                    keyword_response = "retry"
                    logger.info("  → '다시' 선택 감지")
                else:
                    keyword_response = "confirmed"
                    logger.info("  → '확인' 선택 감지")
                
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "keyword_confirmation_response": keyword_response,
                        "waiting_for_user": False
                    }
                )
            
            elif self.interrupt_count == 2:
                logger.info("[CONTINUE MODE] Stage 2: 논문 수 응답 처리")
                logger.info(f"  → 선택된 논문 수: {user_response}")
                
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "waiting_for_user": False
                    }
                )
            
            # 워크플로우 계속 실행
            logger.info("[CONTINUE MODE] 워크플로우 계속 실행 중...")
            for event in self.agent.stream(None, config):
                self.current_state = event
                logger.debug(f"  Event received: {list(event.keys())}")
            
            # 최종 상태 확인
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            logger.info("[CONTINUE MODE] 실행 완료, 상태 확인")
            
            # 완료했는지 확인
            if current_values.get("is_complete"):
                logger.info("[CONTINUE MODE] ✓ 워크플로우 완료")
                return {
                    "status": "completed",
                    "interrupt_stage": self.interrupt_count,
                    "response": current_values.get("final_response", ""),
                    "papers": current_values.get("relevant_papers", []),
                    "thread_id": self.current_thread_id
                }
            
            # 다음 Interrupt가 있는지 확인
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
            
            # 예상 외의 상황
            logger.warning("[CONTINUE MODE] 예상 외의 상태: 완료도 아니고 Interrupt도 없음")
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


# 싱글톤 패턴
_default_assistant = None

def get_assistant() -> ResearchAssistant:
    """전역 어시스턴트 인스턴스를 반환합니다."""
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = ResearchAssistant()
    return _default_assistant
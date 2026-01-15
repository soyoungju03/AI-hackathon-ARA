# app/graph/workflow.py
# -*- coding: utf-8 -*-
"""
완전히 수정된 LangGraph 워크플로우 (재분석 모드 지원 - 단순화 버전)
================================================================================

핵심 수정사항:
- "다시" 선택 시 재분석 후 다시 키워드 확인을 거침
- route_after_analyze 조건부 분기 완전히 제거 (단순화)
- analyze_question → request_keyword_confirmation 항상 직행
- 사용자 경험 개선: 새로운 키워드를 항상 확인할 수 있음

파이프라인 흐름:
사용자 질문 → 키워드 추출 → [사용자 확인] →
["다시" 선택 시 → 재분석 → 다시 키워드 확인] →
["확인" 선택 시 → 논문 수 선택] → arXiv 검색 → PDF 처리 → 의미 검색 → 답변 생성
"""

from typing import Literal
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState, create_initial_state, ReActStep
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
# 워크플로우 빌드 (단순화된 버전)
# ============================================

def build_research_workflow() -> StateGraph:
    """
    재분석 모드를 지원하는 워크플로우를 구축합니다.
    
    핵심 변경사항:
    1. 재분석 후에도 항상 키워드 확인을 거침
    2. route_after_analyze 함수 완전히 제거됨 (단순 엣지로 대체)
    3. 사용자는 항상 새로운 키워드를 확인할 수 있음
    
    워크플로우 구조:
    
    START
      ↓
    receive_question (질문 수신)
      ↓
    analyze_question (키워드 추출)
      ↓
    request_keyword_confirmation (키워드 확인 요청) ← 항상 이 단계를 거침
      ↓
    [INTERRUPT 1] 사용자 대기
      ↓
    process_keyword_confirmation_response (사용자 응답 처리)
      ↓
    [조건부 분기]
    ├─ analyze_question (사용자가 '다시' 선택 → 재분석 → 다시 확인)
    └─ request_paper_count (사용자가 '확인' 선택 → 논문 수 선택)
      ↓
    [INTERRUPT 2] 사용자 대기
      ↓
    process_paper_count_response (논문 수 처리)
      ↓
    search_papers (arXiv 검색 → PDF 임베딩 파이프라인)
      ↓
    [조건부 분기]
    ├─ generate_response (오류 발생 시 바로 응답)
    └─ evaluate_relevance (정상: ChromaDB 의미 검색)
      ↓
    summarize_papers (논문 요약)
      ↓
    generate_response (최종 답변 생성)
      ↓
    END
    """
    
    workflow = StateGraph(AgentState)
    
    # 노드 추가
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
    
    # 엣지 정의
    workflow.set_entry_point("receive_question")
    
    # 초기 처리 흐름
    workflow.add_edge("receive_question", "analyze_question")
    
    # 🔑 핵심 단순화: 조건부 분기 제거, 항상 키워드 확인으로 이동
    # 이전: route_after_analyze 함수로 조건부 분기
    # 이후: 단순 엣지로 항상 키워드 확인
    workflow.add_edge("analyze_question", "request_keyword_confirmation")
    
    # 키워드 확인 흐름
    workflow.add_edge("request_keyword_confirmation", "process_keyword_confirmation_response")
    
    # 키워드 확인 후 조건부 분기
    def route_after_keyword_confirmation(state: AgentState) -> Literal["analyze_question", "request_paper_count"]:
        """
        키워드 확인 후 경로를 결정합니다.
        
        사용자가 "다시"를 입력했다면 keyword_confirmation_response가 "retry"로
        설정되어 있을 것이고, 이 경우 analyze_question 노드로 돌아갑니다.
        그러면 새로운 키워드가 추출되고, 다시 request_keyword_confirmation으로
        이동하여 사용자에게 새 키워드를 확인받습니다.
        
        그렇지 않으면 (confirmed인 경우) request_paper_count 노드로 진행하여
        논문 수를 선택받습니다.
        """
        keyword_response = state.get("keyword_confirmation_response")
        
        logger.info("=" * 60)
        logger.info("[ROUTE_AFTER_KEYWORD_CONFIRMATION] 경로 결정")
        logger.info(f"  keyword_confirmation_response: {keyword_response}")
        logger.info("=" * 60)
        
        if keyword_response == "retry":
            logger.info("  → 경로: analyze_question (질문 재분석)")
            return "analyze_question"
        else:
            logger.info("  → 경로: request_paper_count (논문 수 선택)")
            return "request_paper_count"
    
    workflow.add_conditional_edges(
        "process_keyword_confirmation_response",
        route_after_keyword_confirmation,
        {
            "analyze_question": "analyze_question",
            "request_paper_count": "request_paper_count"
        }
    )
    
    # 논문 수 선택 흐름
    workflow.add_edge("request_paper_count", "process_paper_count_response")
    workflow.add_edge("process_paper_count_response", "search_papers")
    
    # 검색 결과에 따른 조건부 분기
    def check_search_results(state: AgentState) -> Literal["evaluate_relevance", "generate_response"]:
        """
        검색 결과를 확인하고 다음 경로를 결정합니다.
        
        error_message가 있으면 검색이 실패한 것이므로 바로 generate_response로
        이동하여 오류 메시지를 사용자에게 전달합니다.
        
        정상적으로 검색이 완료되었다면 evaluate_relevance로 이동하여
        ChromaDB에서 의미 기반 검색을 수행합니다.
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
    
    # 최종 처리 흐름
    workflow.add_edge("evaluate_relevance", "summarize_papers")
    workflow.add_edge("summarize_papers", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow


# ============================================
# 에이전트 생성 및 실행
# ============================================

def create_research_agent(checkpointer=None):
    """
    재분석 모드를 지원하는 연구 어시스턴트 에이전트를 생성합니다.
    
    checkpointer는 워크플로우의 상태를 저장하고 복원하여
    Human-in-the-Loop에서 멈췄다가 나중에 계속 진행할 수 있게 합니다.
    
    Args:
        checkpointer: 상태 저장소 (기본값: MemorySaver)
    
    Returns:
        컴파일된 워크플로우
    """
    
    workflow = build_research_workflow()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # interrupt_before는 워크플로우가 해당 노드를 실행하기 직전에
    # 멈추고 사용자 입력을 기다리도록 합니다.
    # 여기서는 두 곳에서 멈춥니다:
    # 1. process_keyword_confirmation_response 전: 키워드 확인 대기
    # 2. process_paper_count_response 전: 논문 수 선택 대기
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "process_keyword_confirmation_response",
            "process_paper_count_response"
        ]
    )
    
    logger.info("✓ 워크플로우 컴파일 완료 (재분석 모드 지원 - 단순화 버전)")
    return compiled


# ============================================
# ResearchAssistant 클래스
# ============================================

class ResearchAssistant:
    """
    재분석 모드를 지원하는 연구 어시스턴트입니다.
    
    이 클래스는 사용자와의 대화형 워크플로우를 관리합니다.
    
    전체 처리 흐름:
    1. 사용자 질문 수신 (start 메서드)
    2. 키워드 추출 및 사용자 확인 (첫 번째 interrupt)
    3. "다시" 선택 시: 재분석 → 다시 키워드 확인 (첫 번째 interrupt로 복귀)
    4. "확인" 선택 시: 논문 수 선택 (두 번째 interrupt)
    5. arXiv 검색 및 PDF 처리
    6. 의미 기반 청크 검색
    7. 요약 및 답변 생성
    """
    
    def __init__(self):
        """
        어시스턴트를 초기화합니다.
        
        checkpointer를 사용하여 워크플로우 상태를 저장하고,
        interrupt 지점에서 멈췄다가 나중에 계속할 수 있게 합니다.
        """
        self.checkpointer = MemorySaver()
        self.agent = create_research_agent(self.checkpointer)
        self.current_thread_id = None
        self.interrupt_count = 0
    
    def run(
        self,
        question: str,
        paper_count: int = 3,
        session_id: str = "default"
    ) -> str:
        """
        자동 실행 모드: Human-in-the-Loop 없이 전체 워크플로우를 자동으로 실행합니다.
        
        이 모드는 테스트나 자동화된 시나리오에 유용합니다.
        사용자 확인 없이 기본값으로 처리합니다.
        
        Args:
            question: 사용자의 연구 질문
            paper_count: 검색할 논문 수 (기본값: 3)
            session_id: 세션 ID
        
        Returns:
            최종 생성된 답변
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
            
            logger.info("[RUN MODE] ✓ 자동 실행 완료")
            return final_state.get("final_response", "응답을 생성할 수 없습니다.")
        
        except Exception as e:
            logger.error(f"[RUN MODE] 오류: {str(e)}", exc_info=True)
            return f"오류가 발생했습니다: {str(e)}"
    
    def start(self, question: str, session_id: str = "default") -> dict:
        """
        대화형 모드: 첫 번째 Interrupt (키워드 확인)에서 멈춥니다.
        
        이 메서드는 워크플로우를 시작하고, 키워드 확인 단계에서
        멈춰서 사용자의 응답을 기다립니다.
        
        Args:
            question: 사용자의 연구 질문
            session_id: 세션 ID
        
        Returns:
            현재 상태 정보 (키워드, 메시지, 옵션 등)
        """
        
        import uuid
        
        logger.info(f"[START MODE] 시작: {question[:50]}...")
        
        initial_state = create_initial_state(question, session_id)
        
        self.current_thread_id = str(uuid.uuid4())
        self.interrupt_count = 0
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            logger.info("[START MODE] 워크플로우 실행 중...")
            
            # 워크플로우를 stream으로 실행하면서 interrupt까지 진행
            for event in self.agent.stream(initial_state, config):
                pass
            
            # interrupt에서 멈춘 상태 가져오기
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            self.interrupt_count = 1
            logger.info("[START MODE] ✓ 첫 번째 Interrupt 도달")
            
            # interrupt_data가 있으면 사용자에게 보여줄 정보 반환
            if current_values.get("interrupt_data"):
                interrupt_data = current_values["interrupt_data"]
                
                return {
                    "status": "waiting_for_input",
                    "interrupt_stage": 1,
                    "message": interrupt_data.message,
                    "options": interrupt_data.options,
                    "keywords": current_values.get("extracted_keywords", []),
                    "thread_id": self.current_thread_id
                }
            else:
                # interrupt 없이 완료된 경우 (예: 오류 발생)
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
        
        이 메서드는 interrupt에서 멈춰있던 워크플로우를 사용자의 응답과
        함께 재개합니다.
        
        Args:
            user_response: 사용자의 응답 (예: "확인", "다시", "3")
        
        Returns:
            다음 상태 정보 (완료, 다음 interrupt 대기, 또는 오류)
        """
        
        logger.info(f"[CONTINUE MODE] 사용자 응답 (Stage {self.interrupt_count}): {user_response}")
        
        if not self.current_thread_id:
            logger.error("[CONTINUE MODE] thread_id 없음")
            return {
                "status": "error",
                "message": "먼저 start()를 호출해주세요."
            }
        
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            # Stage 1: 키워드 확인 응답 처리
            if self.interrupt_count == 1:
                logger.info("[CONTINUE MODE] Stage 1: 키워드 확인 응답")
                
                # 사용자 응답을 정규화
                normalized_response = user_response.strip().lower()
                keyword_response = "retry" if normalized_response in ["다시", "retry", "수정"] else "confirmed"
                
                logger.info(f"  → 정규화된 응답: {keyword_response}")
                
                # 상태 업데이트
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "keyword_confirmation_response": keyword_response,
                        "waiting_for_user": False
                    }
                )
            
            # Stage 2: 논문 수 선택 응답 처리
            elif self.interrupt_count == 2:
                logger.info("[CONTINUE MODE] Stage 2: 논문 수 선택")
                
                # 상태 업데이트
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "waiting_for_user": False
                    }
                )
            
            logger.info("[CONTINUE MODE] 워크플로우 계속 실행 중...")
            
            # 워크플로우 재개
            for event in self.agent.stream(None, config):
                pass
            
            # 현재 상태 확인
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            logger.info("[CONTINUE MODE] 실행 완료, 상태 확인")
            
            # 워크플로우가 완전히 완료된 경우
            if current_values.get("is_complete"):
                logger.info("[CONTINUE MODE] ✓ 워크플로우 완료")
                return {
                    "status": "completed",
                    "interrupt_stage": self.interrupt_count,
                    "response": current_values.get("final_response", ""),
                    "chunks": current_values.get("relevant_chunks", []),
                    "thread_id": self.current_thread_id
                }
            
            # 다음 interrupt에 도달한 경우
            if current_values.get("interrupt_data"):
                interrupt_data = current_values["interrupt_data"]
                
                # 🔑 핵심: interrupt_type을 확인하여 정확한 stage 설정
                # "다시" 선택 시 재분석 후 다시 키워드 확인(Stage 1)으로 돌아가므로
                # interrupt_count를 단순히 증가시키지 않고 interrupt_type으로 판단
                if interrupt_data.interrupt_type == "confirm_keywords":
                    # 키워드 확인 단계 (재분석 후 돌아온 경우)
                    self.interrupt_count = 1
                    logger.info(f"[CONTINUE MODE] → 키워드 재확인: Stage {self.interrupt_count}")
                elif interrupt_data.interrupt_type == "select_paper_count":
                    # 논문 수 선택 단계
                    self.interrupt_count = 2
                    logger.info(f"[CONTINUE MODE] → 논문 수 선택: Stage {self.interrupt_count}")
                
                return {
                    "status": "waiting_for_input",
                    "interrupt_stage": self.interrupt_count,
                    "message": interrupt_data.message,
                    "options": interrupt_data.options,
                    "keywords": current_values.get("extracted_keywords", []),  # 🔑 새 키워드 전달
                    "thread_id": self.current_thread_id
                }
            
            # 예상치 못한 상태
            logger.warning("[CONTINUE MODE] 예상 외의 상태")
            return {
                "status": "unknown",
                "message": "워크플로우 상태를 파악할 수 없습니다.",
                "thread_id": self.current_thread_id
            }
        
        except Exception as e:
            logger.error(f"[CONTINUE MODE] 오류: {str(e)}", exc_info=True)
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
    
    싱글톤 패턴을 사용하여 애플리케이션 전체에서
    하나의 어시스턴트 인스턴스를 공유합니다.
    
    Returns:
        ResearchAssistant 인스턴스
    """
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = ResearchAssistant()
    return _default_assistant
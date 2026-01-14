# -*- coding: utf-8 -*-
"""
완전히 수정된 AI Research Assistant - 메인 진입점
==================================================

두 단계 Human-in-the-Loop을 완전히 지원합니다:
1. 첫 번째 대화: 질문 입력 → 키워드 확인 (확인/다시)
2. 두 번째 대화: 키워드 확인 응답 → 논문 수 선택 (1-10)
3. 세 번째 대화: 논문 수 선택 → 최종 응답 생성

세션 기반 상태 관리로 사용자의 대화 흐름을 완벽하게 추적합니다.

수정 사항: Gradio 6.3.0 최신 API - type 파라미터 제거, theme을 launch()로 이동
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from typing import Tuple
import uuid

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ============================================
# 환경 변수 검증
# ============================================

def check_api_key():
    """OpenAI API 키가 설정되어 있는지 확인합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return False, """
경고: OPENAI_API_KEY가 설정되지 않았습니다.

Hugging Face Spaces 사용자:
1. Settings 탭으로 이동
2. Repository secrets 섹션 찾기
3. OPENAI_API_KEY를 이름으로, API 키를 값으로 추가
4. Space를 다시 시작

로컬 개발자:
1. 프로젝트 루트에 .env 파일 생성
2. OPENAI_API_KEY=sk-your-key-here 추가
        """
    
    return True, api_key


# ============================================
# 메시지 헬퍼 (Gradio 6.3.0 호환)
# ============================================

def create_user_message(content: str) -> dict:
    """Gradio 6.3.0 호환 사용자 메시지 생성"""
    return {"role": "user", "content": content}


def create_assistant_message(content: str) -> dict:
    """Gradio 6.3.0 호환 어시스턴트 메시지 생성"""
    return {"role": "assistant", "content": content}


# ============================================
# 세션 관리
# ============================================

session_data = {}  # {session_id: {"assistant": ResearchAssistant, "state": {...}}}


def get_or_create_session(session_id: str):
    """세션을 가져오거나 새로 생성합니다."""
    if session_id not in session_data:
        try:
            from app.graph.workflow import ResearchAssistant
            session_data[session_id] = {
                "assistant": ResearchAssistant(),
                "current_interrupt_stage": 0,  # 0=아직 시작 안함, 1=키워드 확인 대기, 2=논문수 선택 대기
                "last_user_message": None
            }
        except ImportError as e:
            logger.error(f"ResearchAssistant import 실패: {str(e)}")
            raise
    return session_data[session_id]


# ============================================
# 핵심 채팅 함수
# ============================================

def process_chat_message(
    message: str,
    history: list,
    state: dict
) -> Tuple[str, list, dict]:
    """
    사용자의 메시지를 처리합니다.
    
    이 함수는 복잡한 상태 관리를 하고 있습니다:
    1. 첫 메시지: 질문 입력 → 워크플로우 시작 → 첫 번째 Interrupt
    2. 두 번째 메시지: 키워드 확인 응답 → 워크플로우 계속 → 두 번째 Interrupt
    3. 세 번째 메시지: 논문 수 응답 → 워크플로우 계속 → 완료
    
    각 단계에서 다른 로직이 실행됩니다.
    """
    
    if not message.strip():
        return "", history, state
    
    # API 키 확인
    has_key, key_or_message = check_api_key()
    if not has_key:
        # Gradio 6.3.0: dict 형식의 메시지 사용
        history.append(create_user_message(message))
        history.append(create_assistant_message(key_or_message))
        return "", history, state
    
    # 세션 ID 관리
    if "session_id" not in state or state["session_id"] is None:
        state["session_id"] = str(uuid.uuid4())
    
    session_id = state["session_id"]
    
    try:
        session = get_or_create_session(session_id)
    except Exception as e:
        error_msg = f"세션 생성 실패: {str(e)}\n\napp/graph/workflow.py 파일이 존재하는지 확인해주세요."
        logger.error(error_msg)
        history.append(create_user_message(message))
        history.append(create_assistant_message(error_msg))
        return "", history, state
    
    assistant = session["assistant"]
    
    # 사용자 메시지를 히스토리에 추가 (Gradio 6.3.0: dict 사용)
    history.append(create_user_message(message))
    
    try:
        current_stage = session.get("current_interrupt_stage", 0)
        
        if current_stage == 0:
            # ===================================
            # 첫 번째 단계: 워크플로우 시작
            # ===================================
            logger.info(f"[STAGE 0] 새 질문 시작: {message}")
            
            result = assistant.start(message, session_id)
            
            if result["status"] == "waiting_for_input":
                # 첫 번째 Interrupt: 키워드 확인
                response = result["message"]
                session["current_interrupt_stage"] = 1
                logger.info("[STAGE 0 → 1] 첫 번째 Interrupt 도달: 키워드 확인 대기")
                
            elif result["status"] == "completed":
                # 드물게 Interrupt 없이 바로 완료된 경우
                response = result["response"]
                session["current_interrupt_stage"] = 0
                logger.info("[STAGE 0] 경고: Interrupt 없이 완료됨 (드문 경우)")
                
            else:
                response = f"오류가 발생했습니다: {result.get('message', '알 수 없음')}"
                session["current_interrupt_stage"] = 0
                logger.error(f"[STAGE 0] 에러: {response}")
        
        elif current_stage == 1:
            # ===================================
            # 두 번째 단계: 키워드 확인 응답 처리
            # ===================================
            logger.info(f"[STAGE 1] 키워드 확인 응답: {message}")
            
            result = assistant.continue_with_response(message)
            
            if result["status"] == "waiting_for_input" and result.get("interrupt_stage") == 2:
                # 두 번째 Interrupt: 논문 수 선택
                response = result["message"]
                session["current_interrupt_stage"] = 2
                logger.info("[STAGE 1 → 2] 두 번째 Interrupt 도달: 논문 수 선택 대기")
                
            elif result["status"] == "waiting_for_input" and result.get("interrupt_stage") == 1:
                # 사용자가 "다시"라고 했으므로 다시 키워드 확인 요청
                response = result["message"]
                session["current_interrupt_stage"] = 1
                logger.info("[STAGE 1] 사용자가 '다시' 선택: 키워드 재확인 요청")
                
            elif result["status"] == "completed":
                response = result["response"]
                session["current_interrupt_stage"] = 0
                logger.info("[STAGE 1] 경고: 예상 외로 완료됨")
                
            else:
                response = f"오류: {result.get('message', '알 수 없음')}"
                session["current_interrupt_stage"] = 0
                logger.error(f"[STAGE 1] 에러: {response}")
        
        elif current_stage == 2:
            # ===================================
            # 세 번째 단계: 논문 수 응답 처리
            # ===================================
            logger.info(f"[STAGE 2] 논문 수 응답: {message}")
            
            result = assistant.continue_with_response(message)
            
            if result["status"] == "completed":
                response = result["response"]
                session["current_interrupt_stage"] = 0
                logger.info("[STAGE 2 → 완료] 최종 응답 생성 완료")
                
            elif result["status"] == "waiting_for_input":
                # 다음 Interrupt가 있으면 (드문 경우)
                response = result["message"]
                session["current_interrupt_stage"] = result.get("interrupt_stage", 0)
                logger.info(f"[STAGE 2] 다음 Interrupt: stage {session['current_interrupt_stage']}")
                
            else:
                response = f"오류: {result.get('message', '알 수 없음')}"
                session["current_interrupt_stage"] = 0
                logger.error(f"[STAGE 2] 에러: {response}")
        
        else:
            # 예상 외의 단계
            response = f"알 수 없는 단계입니다: {current_stage}"
            session["current_interrupt_stage"] = 0
            logger.error(f"[UNKNOWN STAGE] {current_stage}")
        
        # AI 응답을 히스토리에 추가 (Gradio 6.3.0: dict 사용)
        history.append(create_assistant_message(response))
        
    except ImportError as e:
        error_msg = f"모듈 임포트 오류: {str(e)}\nrequirements.txt를 확인해주세요."
        logger.error(f"ImportError: {str(e)}")
        history.append(create_assistant_message(error_msg))
        
    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e)}"
        logger.error(f"Exception: {str(e)}", exc_info=True)
        history.append(create_assistant_message(error_msg))
    
    return "", history, state


def quick_search(question: str, paper_count: int) -> str:
    """
    빠른 검색: Interrupt 없이 자동으로 진행합니다.
    """
    
    if not question.strip():
        return "질문을 입력해주세요."
    
    has_key, key_or_message = check_api_key()
    if not has_key:
        return key_or_message
    
    try:
        from app.graph.workflow import ResearchAssistant
        
        assistant = ResearchAssistant()
        response = assistant.run(question, paper_count=int(paper_count))
        return response
        
    except Exception as e:
        logger.error(f"Quick search error: {str(e)}", exc_info=True)
        return f"오류가 발생했습니다: {str(e)}"


# ============================================
# Gradio 인터페이스 생성
# ============================================
# PDF 파이프라인 미리 초기화 (모델 로드)
logger.info("PDF 임베딩 파이프라인을 미리 로드하는 중...")
try:
    from app.graph.workflow import get_pdf_pipeline
    pipeline = get_pdf_pipeline()
    logger.info("✓ PDF 파이프라인 로드 완료")
except Exception as e:
    logger.warning(f"PDF 파이프라인 로드 경고: {str(e)}")
    logger.info("파이프라인은 첫 검색 시 로드됩니다")

def create_app():
    """
    Gradio 6.3.0에 최적화된 앱을 생성합니다.
    
    주의: theme은 더 이상 Blocks 생성자에서 사용되지 않습니다.
    대신 launch() 메서드에서 사용합니다.
    """
    
    # Gradio 6.3.0: theme 파라미터 제거 (launch()에서 사용)
    with gr.Blocks(title="AI-Research-Assistant") as demo:
        # 헤더
        gr.Markdown("""
        # AI Research Assistant
        ### 학술 논문 기반 지능형 연구 도우미
        
        **기술 스택**: LangGraph + ReAct Pattern + Human-in-the-Loop + arXiv API
        
        질문을 입력하면 AI가 단계별로 당신과 상호작용하며 관련 논문을 검색하고 핵심 내용을 요약해드립니다.
        
        ---
        """)
        
        with gr.Tabs():
            
            # 탭 1: 대화형 검색 (권장)
            with gr.Tab("대화형 검색 (권장)"):
                
                gr.Markdown("""
                **Human-in-the-Loop 워크플로우:**
                
                1. **첫 번째 대화**: 당신의 연구 질문 입력
                   - AI가 키워드를 분석하여 추출합니다
                   
                2. **두 번째 대화**: 키워드 확인
                   - "확인"을 입력하면 진행
                   - "다시"를 입력하면 AI가 다시 분석합니다
                   
                3. **세 번째 대화**: 논문 개수 선택 (1-10)
                   - 검색할 논문의 개수를 선택합니다
                   
                4. **최종 결과**: 논문 검색 및 요약
                   - AI가 논문을 검색하고 종합적인 답변을 제공합니다
                """)
                
                state = gr.State({
                    "session_id": None
                })
                
                # Gradio 6.3.0: type 파라미터 제거
                # Chatbot은 기본적으로 메시지 형식을 지원합니다
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png")
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="당신의 연구 질문을 입력하세요...\n예: 자율주행 자동차의 LiDAR 센서 기술\n또는 이전 단계의 응답을 입력하세요.",
                        lines=2,
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("전송", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("대화 초기화", size="sm")
                
                gr.Examples(
                    examples=[
                        "자율주행 자동차의 LiDAR 센서 데이터 처리 최신 기법",
                        "Transformer 모델의 attention 메커니즘 연구",
                        "딥러닝 기반 의료 영상 분석",
                        "강화학습을 활용한 로봇 제어",
                        "Graph Neural Network 응용 연구",
                    ],
                    inputs=msg_input,
                    label="예시 질문"
                )
                
                # 이벤트 핸들러
                send_btn.click(
                    process_chat_message,
                    inputs=[msg_input, chatbot, state],
                    outputs=[msg_input, chatbot, state]
                )
                
                msg_input.submit(
                    process_chat_message,
                    inputs=[msg_input, chatbot, state],
                    outputs=[msg_input, chatbot, state]
                )
                
                clear_btn.click(
                    lambda: ([], {"session_id": None}),
                    outputs=[chatbot, state]
                )
            
            # 탭 2: 빠른 검색
            with gr.Tab("빠른 검색"):
                
                gr.Markdown("""
                **빠른 검색**: Interrupt 없이 자동으로 모든 단계를 진행합니다.
                
                미리 논문 개수를 선택하고 질문을 입력하면, AI가 자동으로 검색하고 답변을 생성합니다.
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        quick_input = gr.Textbox(
                            label="연구 질문",
                            placeholder="검색하고 싶은 주제를 입력하세요...",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        paper_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="논문 개수"
                        )
                        search_btn = gr.Button("검색", variant="primary", size="lg")
                
                quick_output = gr.Markdown(value="*검색 결과가 여기에 표시됩니다*")
                
                search_btn.click(
                    quick_search,
                    inputs=[quick_input, paper_slider],
                    outputs=quick_output
                )
            
            # 탭 3: 정보
            with gr.Tab("정보"):
                
                has_key, _ = check_api_key()
                status_text = "설정됨 ✓" if has_key else "설정 필요 ✗"
                
                gr.Markdown(f"""
                ## 시스템 정보
                
                ### API 상태
                - **OpenAI API Key**: {status_text}
                
                ### 두 단계 Human-in-the-Loop 구조
                
                이 시스템은 AI와 사용자가 상호작용하는 두 가지 주요 단계를 가집니다:
                
                **첫 번째 단계: 키워드 확인**
                - AI가 당신의 질문을 분석하여 핵심 키워드를 추출합니다
                - 당신은 추출된 키워드가 정확한지 확인합니다
                - "확인"을 입력하면 진행, "다시"를 입력하면 재분석합니다
                
                **두 번째 단계: 논문 개수 선택**
                - 검색할 논문의 개수를 선택합니다 (1-10)
                - AI가 선택한 개수의 논문을 검색합니다
                
                **최종 단계: 응답 생성**
                - AI가 검색된 논문들을 분석하여 종합적인 답변을 제공합니다
                
                ### 기술 스택
                
                - **LangGraph**: AI 워크플로우 오케스트레이션
                - **ReAct 패턴**: Thought-Action-Observation 구조
                - **OpenAI GPT-4o**: 질문 분석 및 응답 생성
                - **arXiv API**: 학술 논문 검색
                - **Gradio 6.3.0**: 웹 인터페이스
                
                ### 주요 특징
                
                - **Human-in-the-Loop**: 사용자가 중간에 개입하여 정확성을 높입니다
                - **ReAct 패턴**: AI의 사고 과정을 투명하게 보여줍니다
                - **상태 관리**: 대화 상태를 정확하게 추적합니다
                - **조건부 라우팅**: 사용자의 응답에 따라 동적으로 워크플로우가 진행됩니다
                
                ---
                
                **버전**: 2.3 (Gradio 6.3.0 완전 호환) | **개발**: AI Hackathon Project
                """)
        
        # 푸터
        gr.Markdown("""
        ---
        Made with LangGraph + Gradio 6.3.0
        """)
    
    return demo


# ============================================
# 앱 실행
# ============================================

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("ARA (AI Research Assistant) 애플리케이션 시작")
    logger.info("Gradio 버전: 6.3.0 완전 호환")
    logger.info("="*60)
    
    try:
        demo = create_app()
        
        # Gradio 6.3.0: theme 파라미터는 launch()에서 사용
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        )
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            theme=theme
        )
    except Exception as e:
        logger.error(f"앱 시작 실패: {str(e)}", exc_info=True)
        raise
"""
AI Research Assistant - 메인 진입점
====================================

이 파일은 Hugging Face Spaces에서 앱을 실행하기 위한 메인 진입점입니다.
"""

import os
import sys
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from gradio import ChatMessage 
from typing import List, Tuple
import uuid

# 환경 변수 로드 (로컬 개발용)
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)


# ============================================
# 환경 변수 검증
# ============================================

def check_api_key():
    """
    OpenAI API 키가 설정되어 있는지 확인합니다.
    
    Hugging Face Spaces에서는 Settings > Repository secrets에서
    OPENAI_API_KEY를 설정해야 합니다.
    """
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
# 전역 상태
# ============================================

session_data = {}


# ============================================
# 채팅 메시지 처리
# ============================================

def process_chat_message(
    message: str,
    history: list,
    state: dict
) -> Tuple[str, list, dict]:
    """
    채팅 메시지를 처리합니다.
    LangGraph 워크플로우와 통합되어 Human-in-the-Loop을 지원합니다.
    
    Args:
        message: 사용자의 메시지
        history: 대화 히스토리
        state: 세션 상태
    
    Returns:
        빈 문자열, 업데이트된 히스토리, 업데이트된 상태
    """
    
    # 빈 메시지 무시
    if not message.strip():
        return "", history, state
    
    # API 키 확인
    has_key, key_or_message = check_api_key()
    
    if not has_key:
        # API 키 없으면 사용자 메시지와 에러 메시지 추가
        history.append(ChatMessage(role="user", content=message))
        history.append(ChatMessage(role="assistant", content=key_or_message))
        return "", history, state
    
    # 세션 ID 설정
    if "session_id" not in state or state["session_id"] is None:
        state["session_id"] = str(uuid.uuid4())
    
    session_id = state["session_id"]
    
    # 먼저 사용자 메시지를 히스토리에 추가
    history.append(ChatMessage(role="user", content=message))
    
    try:
        # LangGraph 워크플로우 임포트 (API 키가 있을 때만)
        from app.graph.workflow import ResearchAssistant
        
        # 세션별 어시스턴트 관리
        if session_id not in session_data:
            session_data[session_id] = {
                "assistant": ResearchAssistant(),
                "waiting": False
            }
        
        session = session_data[session_id]
        assistant = session["assistant"]
        waiting = session.get("waiting", False)
        
        response = ""  # 응답 초기화
        
        if waiting:
            # Interrupt 응답 처리: 사용자가 논문 수를 입력한 상황
            logger.info(f"사용자 응답 처리: {message}")
            result = assistant.continue_with_response(message)
            
            if result["status"] == "completed":
                response = result["response"]
                session["waiting"] = False
                logger.info("검색 완료")
                
            elif result["status"] == "waiting_for_input":
                response = result["message"]
                # waiting 상태 유지
                
            else:
                response = f"오류: {result.get('message', '알 수 없는 오류')}"
                session["waiting"] = False
        
        else:
            # 새 질문 처리
            logger.info(f"새 질문 처리: {message}")
            result = assistant.start(message, session_id)
            
            if result["status"] == "waiting_for_input":
                # Interrupt 발생: 사용자에게 논문 수 선택 요청
                keywords = result.get("keywords", [])
                response = result["message"]
                
                if keywords:
                    response += f"\n\n추출된 키워드: {', '.join(keywords)}"
                response += "\n\n검색할 논문 수를 입력해주세요 (1-10):"
                
                session["waiting"] = True
                logger.info("논문 수 선택 대기 중")
                
            elif result["status"] == "completed":
                # 바로 완료된 경우
                response = result["response"]
                session["waiting"] = False
                logger.info("검색 완료")
                
            else:
                response = f"오류: {result.get('message', '알 수 없는 오류')}"
                session["waiting"] = False
        
        # AI 응답을 히스토리에 추가
        history.append(ChatMessage(role="assistant", content=response))
        
    except ImportError as e:
        # 모듈 임포트 실패
        error_msg = f"모듈 임포트 오류: {str(e)}\nrequirements.txt를 확인해주세요."
        logger.error(f"ImportError: {str(e)}")
        history.append(ChatMessage(role="assistant", content=error_msg))
        
    except Exception as e:
        # 기타 오류
        error_msg = f"오류가 발생했습니다: {str(e)}"
        logger.error(f"Exception: {str(e)}", exc_info=True)
        history.append(ChatMessage(role="assistant", content=error_msg))
    
    # 입력 필드 초기화, 히스토리와 상태 반환
    return "", history, state


def quick_search(question: str, paper_count: int) -> str:
    """빠른 검색 - Human-in-the-Loop 없이 바로 실행"""
    
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
# Gradio 인터페이스
# ============================================

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

def create_app():
    """Gradio 앱을 생성합니다."""
    
    with gr.Blocks(title="AI Research Assistant") as demo:
        # 헤더
        gr.Markdown("""
        # AI Research Assistant
        ### 학술 논문 기반 지능형 연구 도우미
        
        **기술 스택**: LangGraph + ReAct Pattern + Human-in-the-Loop + arXiv API
        
        질문을 입력하면 AI가 관련 논문을 검색하고 핵심 내용을 요약해드립니다.
        
        ---
        """)
        
        with gr.Tabs():
            
            # 탭 1: 대화형 검색
            with gr.Tab("대화형 검색"):
                
                gr.Markdown("""
                **Human-in-the-Loop 워크플로우:**
                1. 연구 질문 입력 → 2. AI 키워드 분석 → 3. **논문 수 선택** → 4. 검색 및 요약
                """)
                
                state = gr.State({
                    "session_id": None,
                    "waiting": False
                })
                
                chatbot = gr.Chatbot(
                    height=450,
                    show_label=False,
                    avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png")
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="연구 질문을 입력하세요... (예: 자율주행 LiDAR 센서 기술)",
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
                    lambda: ([], {"session_id": None, "waiting": False}),
                    outputs=[chatbot, state]
                )
            
            # 탭 2: 빠른 검색
            with gr.Tab("빠른 검색"):
                
                gr.Markdown("""
                **빠른 검색**: 논문 수를 미리 선택하고 바로 검색을 실행합니다.
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
                            label="논문 수"
                        )
                        search_btn = gr.Button("검색", variant="primary", size="lg")
                
                quick_output = gr.Markdown(value="*검색 결과가 여기에 표시됩니다*")
                
                search_btn.click(
                    quick_search,
                    inputs=[quick_input, paper_slider],
                    outputs=quick_output
                )
            
            # 탭 3: 시스템 정보
            with gr.Tab("정보"):
                
                # API 키 상태 확인
                has_key, _ = check_api_key()
                status_emoji = "설정됨" if has_key else "설정 필요"
                
                gr.Markdown(f"""
                ## 시스템 정보
                
                ### API 상태
                - **OpenAI API Key**: {status_emoji}
                
                ### 주요 기능
                
                - 질문 분석: AI가 질문을 분석하여 핵심 키워드 추출
                - Human-in-the-Loop: 사용자가 검색 설정을 확인/조정
                - 논문 검색: arXiv에서 관련 논문 검색
                - 연관성 평가: 검색 결과 품질 필터링
                - 요약 생성: 구조화된 논문 요약
                
                ### 기술 스택
                
                - **LangGraph**: 워크플로우 오케스트레이션
                - **ReAct 패턴**: Thought-Action-Observation 구조
                - **OpenAI GPT-4o**: 질문 분석 및 요약
                - **arXiv API**: 논문 검색
                - **Gradio**: 웹 인터페이스
                
                ---
                
                **버전**: 2.0 | **개발**: AI Hackathon Project
                """)
        
        # 푸터
        gr.Markdown("""
        ---
        Made with LangGraph + Gradio
        """)
    
    return demo


# ============================================
# 앱 실행
# ============================================

if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme
    )
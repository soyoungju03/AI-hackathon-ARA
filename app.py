"""
AI Research Assistant - 메인 진입점
====================================

이 파일은 Hugging Face Spaces에서 앱을 실행하기 위한 메인 진입점입니다.

Hugging Face Spaces 배포 시 주의사항:
-------------------------------------
1. 파일명이 반드시 app.py여야 합니다
2. 환경 변수는 Spaces의 Settings > Repository secrets에서 설정해야 합니다
3. requirements.txt에 모든 의존성이 포함되어 있어야 합니다

현재 발생한 에러 해결:
----------------------
기존 에러: "OPENAI_API_KEY must be set"
해결: os.getenv()로 환경 변수를 읽어오되, 없으면 적절한 에러 메시지 표시
"""

import os
import sys

# 현재 디렉토리를 Python 경로에 추가
# 이렇게 해야 app.graph, app.tools 등의 모듈을 import할 수 있습니다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from typing import List, Tuple
import uuid

# 환경 변수 로드 (로컬 개발용)
from dotenv import load_dotenv
load_dotenv()


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
        ⚠️ **OPENAI_API_KEY가 설정되지 않았습니다.**
        
        **Hugging Face Spaces 사용자:**
        1. Settings 탭으로 이동
        2. Repository secrets 섹션 찾기
        3. `OPENAI_API_KEY`를 이름으로, API 키를 값으로 추가
        4. Space를 다시 시작
        
        **로컬 개발자:**
        1. 프로젝트 루트에 `.env` 파일 생성
        2. `OPENAI_API_KEY=sk-your-key-here` 추가
        """
    
    return True, api_key


# ============================================
# 간단한 검색 함수 (API 키 없이도 UI 표시용)
# ============================================

# 전역 상태
session_data = {}


def process_chat_message(
    message: str,
    history: List[Tuple[str, str]],
    state: dict
) -> Tuple[str, List[Tuple[str, str]], dict]:
    """
    채팅 메시지를 처리합니다.
    LangGraph 워크플로우와 통합되어 Human-in-the-Loop을 지원합니다.
    """
    
    if not message.strip():
        return "", history, state
    
    # API 키 확인
    has_key, key_or_message = check_api_key()
    
    if not has_key:
        history.append((message, key_or_message))
        return "", history, state
    
    # 세션 ID 설정
    if "session_id" not in state or state["session_id"] is None:
        state["session_id"] = str(uuid.uuid4())
    
    session_id = state["session_id"]
    
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
        
        if waiting:
            # Interrupt 응답 처리
            result = assistant.continue_with_response(message)
            
            if result["status"] == "completed":
                response = result["response"]
                session["waiting"] = False
            elif result["status"] == "waiting_for_input":
                response = result["message"]
            else:
                response = f"❌ 오류: {result.get('message', '알 수 없는 오류')}"
                session["waiting"] = False
        
        else:
            # 새 질문 처리
            result = assistant.start(message, session_id)
            
            if result["status"] == "waiting_for_input":
                keywords = result.get("keywords", [])
                response = result["message"]
                
                if keywords:
                    response += f"\n\n**🔑 추출된 키워드**: {', '.join(keywords)}"
                response += "\n\n---\n📊 **검색할 논문 수를 입력해주세요 (1-10):**"
                
                session["waiting"] = True
                
            elif result["status"] == "completed":
                response = result["response"]
            else:
                response = f"❌ 오류: {result.get('message', '알 수 없는 오류')}"
        
        history.append((message, response))
        
    except ImportError as e:
        error_msg = f"""
        ❌ **모듈 임포트 오류**
        
        필요한 패키지가 설치되지 않았습니다: {str(e)}
        
        requirements.txt를 확인해주세요.
        """
        history.append((message, error_msg))
    
    except Exception as e:
        error_msg = f"❌ **오류가 발생했습니다**: {str(e)}"
        history.append((message, error_msg))
    
    return "", history, state


def quick_search(question: str, paper_count: int) -> str:
    """빠른 검색 - Human-in-the-Loop 없이 바로 실행"""
    
    if not question.strip():
        return "❌ 질문을 입력해주세요."
    
    has_key, key_or_message = check_api_key()
    if not has_key:
        return key_or_message
    
    try:
        from app.graph.workflow import ResearchAssistant
        
        assistant = ResearchAssistant()
        response = assistant.run(question, paper_count=int(paper_count))
        return response
        
    except Exception as e:
        return f"❌ 오류가 발생했습니다: {str(e)}"


# ============================================
# Gradio 인터페이스
# ============================================

def create_app():
    """Gradio 앱을 생성합니다."""
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )
    
    with gr.Blocks(
        title="📚 AI Research Assistant",
        theme=theme,
        css="""
        .container { max-width: 1200px; margin: auto; }
        footer { display: none !important; }
        """
    ) as demo:
        
        # 헤더
        gr.Markdown("""
        # 📚 AI Research Assistant
        ### 학술 논문 기반 지능형 연구 도우미
        
        **🔧 기술 스택**: LangGraph + ReAct Pattern + Human-in-the-Loop + arXiv API
        
        질문을 입력하면 AI가 관련 논문을 검색하고 핵심 내용을 요약해드립니다.
        
        ---
        """)
        
        with gr.Tabs():
            
            # 탭 1: 대화형 검색
            with gr.Tab("💬 대화형 검색"):
                
                gr.Markdown("""
                **🔄 Human-in-the-Loop 워크플로우:**
                1. 연구 질문 입력 → 2. AI 키워드 분석 → 3. **논문 수 선택** → 4. 검색 및 요약
                """)
                
                state = gr.State({
                    "session_id": None,
                    "waiting": False
                })
                
                chatbot = gr.Chatbot(
                    height=450,
                    show_label=False,
                    bubble_full_width=False,
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
                    clear_btn = gr.Button("🗑️ 대화 초기화", size="sm")
                
                gr.Examples(
                    examples=[
                        "자율주행 자동차의 LiDAR 센서 데이터 처리 최신 기법",
                        "Transformer 모델의 attention 메커니즘 연구",
                        "딥러닝 기반 의료 영상 분석",
                        "강화학습을 활용한 로봇 제어",
                        "Graph Neural Network 응용 연구",
                    ],
                    inputs=msg_input,
                    label="💡 예시 질문"
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
            with gr.Tab("🔍 빠른 검색"):
                
                gr.Markdown("""
                **빠른 검색**: 논문 수를 미리 선택하고 바로 검색을 실행합니다.
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        quick_input = gr.Textbox(
                            label="📝 연구 질문",
                            placeholder="검색하고 싶은 주제를 입력하세요...",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        paper_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="🔢 논문 수"
                        )
                        search_btn = gr.Button("🔍 검색", variant="primary", size="lg")
                
                quick_output = gr.Markdown(value="*검색 결과가 여기에 표시됩니다*")
                
                search_btn.click(
                    quick_search,
                    inputs=[quick_input, paper_slider],
                    outputs=quick_output
                )
            
            # 탭 3: 시스템 정보
            with gr.Tab("ℹ️ 정보"):
                
                # API 키 상태 확인
                has_key, _ = check_api_key()
                status_emoji = "✅" if has_key else "❌"
                status_text = "설정됨" if has_key else "설정 필요"
                
                gr.Markdown(f"""
                ## 📖 시스템 정보
                
                ### API 상태
                - **OpenAI API Key**: {status_emoji} {status_text}
                
                ### ✨ 주요 기능
                
                | 기능 | 설명 |
                |------|------|
                | 질문 분석 | AI가 질문을 분석하여 핵심 키워드 추출 |
                | Human-in-the-Loop | 사용자가 검색 설정을 확인/조정 |
                | 논문 검색 | arXiv에서 관련 논문 검색 |
                | 연관성 평가 | 검색 결과 품질 필터링 |
                | 요약 생성 | 구조화된 논문 요약 |
                
                ### 🛠️ 기술 스택
                
                - **LangGraph**: 워크플로우 오케스트레이션
                - **ReAct 패턴**: Thought-Action-Observation 구조
                - **OpenAI GPT-4o**: 질문 분석 및 요약
                - **arXiv API**: 논문 검색
                - **Gradio**: 웹 인터페이스
                
                ### 🔄 ReAct 워크플로우
                
                ```
                [Thought] 질문 분석: "자율주행 LiDAR 기술에 대해 알고 싶어함"
                    ↓
                [Action] 키워드 추출: ["autonomous driving", "LiDAR", "sensor"]
                    ↓
                [INTERRUPT] 사용자에게 논문 수 확인 요청
                    ↓
                [Action] arXiv 검색 실행
                    ↓
                [Observation] 5개 논문 발견, 연관성 평가
                    ↓
                [Action] 고연관성 논문 요약 생성
                    ↓
                [Output] 최종 응답 제공
                ```
                
                ---
                
                **버전**: 2.0 | **개발**: AI Hackathon Project
                """)
        
        # 푸터
        gr.Markdown("""
        ---
        <center>
        Made with ❤️ using LangGraph + Gradio | 
        📚 <a href="https://arxiv.org" target="_blank">arXiv</a> | 
        🔗 <a href="https://github.com" target="_blank">GitHub</a>
        </center>
        """)
    
    return demo


# ============================================
# 앱 실행
# ============================================

# Gradio 앱 생성
demo = create_app()

# Hugging Face Spaces에서 실행될 때
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

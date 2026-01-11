"""
Gradio 웹 인터페이스
====================

이 파일은 사용자가 실제로 상호작용하는 웹 인터페이스를 정의합니다.

기존 app.py와의 차이점:
-----------------------
1. LangGraph 워크플로우와 통합
2. Human-in-the-Loop 지원 (채팅 형식)
3. ReAct 과정 시각화
4. 개선된 UI/UX
"""

import gradio as gr
from typing import List, Tuple, Optional
import json
import uuid

# 로컬 모듈 임포트
from app.graph.workflow import ResearchAssistant
from app.config import get_settings

# 설정 로드
settings = get_settings()


# ============================================
# 전역 상태 관리
# ============================================

session_assistants = {}


def get_or_create_assistant(session_id: str) -> ResearchAssistant:
    """세션별 어시스턴트를 가져오거나 생성합니다."""
    if session_id not in session_assistants:
        session_assistants[session_id] = ResearchAssistant()
    return session_assistants[session_id]


# ============================================
# 채팅 처리 함수
# ============================================

def process_message(
    message: str,
    history: List[Tuple[str, str]],
    session_state: dict
) -> Tuple[str, List[Tuple[str, str]], dict]:
    """
    사용자 메시지를 처리하고 응답을 생성합니다.
    Human-in-the-Loop을 지원합니다.
    """
    
    if not message.strip():
        return "", history, session_state
    
    session_id = session_state.get("session_id", str(uuid.uuid4()))
    session_state["session_id"] = session_id
    
    assistant = get_or_create_assistant(session_id)
    
    waiting_for_input = session_state.get("waiting_for_input", False)
    
    if waiting_for_input:
        # Interrupt에 대한 응답 처리
        result = assistant.continue_with_response(message)
        
        if result["status"] == "completed":
            response = result["response"]
            session_state["waiting_for_input"] = False
            history.append((message, response))
            
        elif result["status"] == "waiting_for_input":
            response = result["message"]
            history.append((message, response))
            
        else:
            response = f"❌ 오류가 발생했습니다: {result.get('message', '알 수 없는 오류')}"
            session_state["waiting_for_input"] = False
            history.append((message, response))
    
    else:
        # 새로운 질문 처리
        result = assistant.start(message, session_id)
        
        if result["status"] == "waiting_for_input":
            response = result["message"]
            
            keywords = result.get("keywords", [])
            if keywords:
                response += f"\n\n**🔑 추출된 키워드**: {', '.join(keywords)}"
            
            response += "\n\n---\n**📊 검색할 논문 수를 입력해주세요 (1-10):**"
            
            session_state["waiting_for_input"] = True
            session_state["thread_id"] = result.get("thread_id")
            
            history.append((message, response))
            
        elif result["status"] == "completed":
            response = result["response"]
            history.append((message, response))
            
        else:
            response = f"❌ 오류가 발생했습니다: {result.get('message', '알 수 없는 오류')}"
            history.append((message, response))
    
    return "", history, session_state


def quick_search(question: str, paper_count: int) -> str:
    """
    빠른 검색 기능 - Human-in-the-Loop 없이 바로 검색합니다.
    """
    if not question.strip():
        return "❌ 질문을 입력해주세요."
    
    assistant = ResearchAssistant()
    
    try:
        response = assistant.run(question, paper_count=paper_count)
        return response
    except Exception as e:
        return f"❌ 오류가 발생했습니다: {str(e)}"


# ============================================
# Gradio 인터페이스 생성
# ============================================

def create_gradio_interface() -> gr.Blocks:
    """
    Gradio 웹 인터페이스를 생성합니다.
    """
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    )
    
    with gr.Blocks(
        title="📚 AI Research Assistant",
        theme=theme,
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        """
    ) as demo:
        
        # 헤더
        gr.Markdown("""
        # 📚 AI Research Assistant
        ### 학술 논문 기반 지능형 연구 도우미
        
        질문을 입력하면 AI가 관련 논문을 검색하고 핵심 내용을 요약해드립니다.
        
        **🔧 기술 스택**: LangGraph + ReAct Pattern + Human-in-the-Loop
        
        ---
        """)
        
        # 탭 인터페이스
        with gr.Tabs():
            
            # 탭 1: 대화형 검색
            with gr.Tab("💬 대화형 검색", id="chat"):
                
                gr.Markdown("""
                **사용 방법:**
                1. 연구 질문을 입력하세요
                2. AI가 키워드를 분석하고 확인을 요청합니다
                3. 검색할 논문 수를 선택하세요 (1-10 사이 숫자 입력)
                4. 결과를 확인하세요
                """)
                
                session_state = gr.State({
                    "session_id": None,
                    "waiting_for_input": False,
                    "thread_id": None
                })
                
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                    show_label=False,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="메시지 입력",
                        placeholder="연구 질문을 입력하세요... (예: 자율주행 LiDAR 센서 기술)",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("전송", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "자율주행 자동차의 LiDAR 센서 데이터 처리 최신 기법",
                        "Transformer 아키텍처를 활용한 자연어 처리",
                        "딥러닝 기반 의료 영상 분석 연구 동향",
                        "강화학습의 로봇 제어 응용",
                    ],
                    inputs=chat_input,
                    label="예시 질문"
                )
                
                send_btn.click(
                    fn=process_message,
                    inputs=[chat_input, chatbot, session_state],
                    outputs=[chat_input, chatbot, session_state]
                )
                
                chat_input.submit(
                    fn=process_message,
                    inputs=[chat_input, chatbot, session_state],
                    outputs=[chat_input, chatbot, session_state]
                )
            
            # 탭 2: 빠른 검색
            with gr.Tab("🔍 빠른 검색", id="quick"):
                
                gr.Markdown("""
                **빠른 검색 모드:**
                논문 수를 미리 선택하고 바로 검색을 실행합니다.
                단계별 확인 과정 없이 빠르게 결과를 얻을 수 있습니다.
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
                        search_btn = gr.Button(
                            "🔍 검색",
                            variant="primary",
                            size="lg"
                        )
                
                quick_output = gr.Markdown(
                    label="검색 결과",
                    value="*검색 결과가 여기에 표시됩니다*"
                )
                
                search_btn.click(
                    fn=quick_search,
                    inputs=[quick_input, paper_slider],
                    outputs=quick_output
                )
            
            # 탭 3: 정보
            with gr.Tab("ℹ️ 정보", id="info"):
                
                gr.Markdown("""
                ## 📖 AI Research Assistant 소개
                
                이 도구는 **LangGraph**와 **ReAct 패턴**을 활용한 
                학술 논문 검색 및 요약 어시스턴트입니다.
                
                ### ✨ 주요 기능
                
                1. **질문 분석**: AI가 질문을 분석하여 핵심 키워드를 추출합니다
                2. **Human-in-the-Loop**: 사용자가 검색 설정을 확인하고 조정할 수 있습니다
                3. **논문 검색**: arXiv에서 관련 논문을 검색합니다
                4. **연관성 평가**: 검색 결과의 연관성을 평가하여 필터링합니다
                5. **요약 생성**: 각 논문의 핵심 내용을 구조화된 형식으로 요약합니다
                
                ### 🛠️ 기술 스택
                
                - **LangGraph**: 워크플로우 오케스트레이션
                - **ReAct 패턴**: Thought-Action-Observation 구조
                - **OpenAI GPT-4o**: 질문 분석 및 요약 생성
                - **arXiv API**: 논문 검색
                - **Gradio**: 웹 인터페이스
                
                ### 🔄 워크플로우
                
                ```
                질문 입력 → 질문 분석 → 키워드 추출
                    ↓
                [Human-in-the-Loop: 사용자 확인]
                    ↓
                논문 검색 → 연관성 평가 → 필터링
                    ↓
                요약 생성 → 최종 응답
                ```
                
                ### 📝 ReAct 패턴이란?
                
                ReAct는 **Reasoning + Acting**의 약자로, AI가 다음 과정을 명시적으로 수행합니다:
                
                1. **Thought (생각)**: 현재 상황을 분석하고 다음 행동을 계획
                2. **Action (행동)**: 도구를 사용하거나 작업을 실행
                3. **Observation (관찰)**: 행동의 결과를 관찰하고 기록
                
                이 과정을 반복하여 복잡한 문제를 단계적으로 해결합니다.
                
                ### 🚀 향후 계획
                
                - 다중 논문 소스 지원 (Semantic Scholar, PubMed 등)
                - 웹 검색 통합 (Tavily API)
                - Vector DB 연동 (Weaviate)
                - Long-term Memory 기능
                - 국내 논문 검색 (DBpia, RISS)
                
                ---
                
                **개발자**: AI Hackathon Project  
                **버전**: 2.0
                """)
        
        # 푸터
        gr.Markdown("""
        ---
        <center>
        Made with ❤️ using LangGraph + Gradio | 
        <a href="https://arxiv.org" target="_blank">arXiv</a>
        </center>
        """)
    
    return demo


# ============================================
# 메인 실행
# ============================================

def main():
    """Gradio 앱을 실행합니다."""
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()

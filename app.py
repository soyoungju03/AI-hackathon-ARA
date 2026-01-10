"""
학술 논문 기반 AI 리서치 어시스턴트 - Gradio 웹 인터페이스
사용자가 웹 브라우저를 통해 쉽게 논문을 검색하고 분석할 수 있습니다.
"""

import os
import arxiv
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 사용자의 대화 이력을 저장할 파일명
CONVERSATION_HISTORY_FILE = "conversation_history.json"


def load_conversation_history():
    """
    이전에 저장된 대화 이력을 불러옵니다.
    이 기능을 통해 사용자가 다시 접속했을 때 이전 대화를 기억할 수 있습니다.
    """
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        try:
            with open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_conversation_history(history):
    """
    사용자의 대화 이력을 저장합니다.
    매번 새로운 질문이 들어올 때마다 이것을 호출해서 대화를 기록합니다.
    """
    with open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def extract_keywords(user_question):
    """
    사용자의 질문에서 핵심 키워드를 추출합니다.
    웹 인터페이스에서 요청이 들어올 때마다 이 함수가 호출됩니다.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""다음 질문에서 가장 중요한 기술 키워드 2-3개를 추출해줘.

질문: {user_question}

형식: keyword1, keyword2, keyword3"""
            }
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content


def search_papers_on_arxiv(keywords, max_results=5):
    """
    arXiv에서 키워드에 해당하는 논문들을 검색합니다.
    아무 비용도 들지 않는 무료 데이터베이스입니다.
    """
    try:
        client_arxiv = arxiv.Client()
        
        search = arxiv.Search(
            query=keywords,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client_arxiv.results(search):
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d")
            }
            papers.append(paper_info)
        
        return papers
    except Exception as e:
        raise Exception(f"arXiv 검색 중 오류가 발생했습니다: {str(e)}")


def summarize_paper_with_gpt4(paper_title, paper_abstract):
    """
    논문의 제목과 초록을 읽고 핵심을 요약합니다.
    웹 인터페이스에서 사용자가 결과를 보기까지 약간의 시간이 걸릴 수 있습니다.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""다음 논문의 초록을 읽고 핵심을 정리해줘.

논문 제목: {paper_title}

초록:
{paper_abstract}

다음 형식으로 답변해줘:
1. 주요 문제: [논문이 해결하려는 문제]
2. 제안 방법: [어떻게 해결했는지]
3. 주요 성과: [결과가 어떻게 되었는지]"""
            }
        ],
        temperature=0.5
    )
    
    return response.choices[0].message.content


def process_user_query_for_web(user_question, num_papers=3):
    """
    웹 인터페이스에서 사용자의 질문을 처리하는 메인 함수입니다.
    이 함수는 당신의 기존 main.py의 process_user_query 함수와 비슷하지만,
    웹 인터페이스에 최적화되어 있습니다.
    
    Args:
        user_question: 사용자가 입력한 질문
        num_papers: 검색할 논문의 개수 (기본값: 3)
        
    Returns:
        str: 웹 인터페이스에 표시할 형식의 결과
    """
    
    try:
        # Step 1: 키워드 추출
        keywords = extract_keywords(user_question)
        
        # Step 2: arXiv에서 논문 검색
        papers = search_papers_on_arxiv(keywords, max_results=num_papers)
        
        if not papers:
            return "죄송합니다. 관련된 논문을 찾을 수 없습니다. 다른 키워드로 다시 시도해보세요."
        
        # Step 3: 각 논문을 요약
        results = []
        results.append(f"🔍 **검색 결과**\n\n")
        results.append(f"📌 **추출된 키워드**: {keywords}\n")
        results.append(f"📚 **찾은 논문 수**: {len(papers)}개\n")
        results.append(f"\n{'='*60}\n\n")
        
        # 각 논문에 대해 요약을 생성합니다
        for i, paper in enumerate(papers, 1):
            results.append(f"### 논문 {i}: {paper['title']}\n\n")
            results.append(f"**저자**: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n\n")
            results.append(f"**발표 날짜**: {paper['published']}\n\n")
            results.append(f"**초록**:\n{paper['abstract']}\n\n")
            
            # GPT-4o를 사용해서 논문 요약
            summary = summarize_paper_with_gpt4(paper['title'], paper['abstract'])
            results.append(f"**AI 요약**:\n{summary}\n\n")
            results.append(f"**논문 링크**: [{paper['url']}]({paper['url']})\n\n")
            results.append(f"\n{'='*60}\n\n")
        
        return "".join(results)
    
    except Exception as e:
        return f"❌ 오류가 발생했습니다: {str(e)}\n\n문제가 지속되면 API 키가 올바르게 설정되어 있는지 확인해주세요."


def create_gradio_interface():
    """
    Gradio 웹 인터페이스를 구성합니다.
    이것이 사용자가 실제로 보게 될 웹 페이지입니다.
    """
    
    # 인터페이스의 제목과 설명
    title = "📚 학술 논문 AI 리서치 어시스턴트"
    description = """
이 도구는 당신의 질문을 받아서 자동으로 arXiv에서 관련 논문을 찾고,
각 논문의 핵심을 AI가 요약해주는 어시스턴트입니다.

**사용 방법**:
1. 아래 텍스트 박스에 당신의 질문을 입력하세요
2. "검색" 버튼을 클릭하세요
3. AI가 관련 논문들을 찾아서 요약해줄 것입니다

**팁**:
- 최대한 구체적인 질문을 하면 더 정확한 결과를 얻을 수 있습니다
- 예: "자율주행 자동차의 LiDAR 센서 데이터 처리" (좋음)
- 예: "LiDAR" (덜 구체적)
"""
    
    # 입력 필드 설정
    # Gradio는 이런 "입력 필드" 객체들을 자동으로 웹 인터페이스로 변환합니다
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        # 페이지 상단의 제목과 설명
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        # 구분선
        gr.Markdown("---")
        
        # 사용자 입력 영역
        with gr.Row():
            with gr.Column():
                # 사용자가 질문을 입력하는 텍스트 상자
                user_input = gr.Textbox(
                    label="📝 당신의 질문을 입력하세요",
                    placeholder="예: 자율주행 자동차의 LiDAR 센서 데이터 처리 최신 기법",
                    lines=3  # 텍스트 상자의 높이
                )
                
                # 검색할 논문의 개수를 선택하는 슬라이더
                num_papers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="🔢 검색할 논문의 개수",
                    info="1개에서 10개 사이의 논문을 선택할 수 있습니다"
                )
            
            # 검색 버튼
            search_button = gr.Button(
                "🔍 검색",
                variant="primary",  # 버튼을 파란색으로 강조합니다
                scale=1
            )
        
        # 구분선
        gr.Markdown("---")
        
        # 결과 출력 영역
        with gr.Row():
            # 검색 결과를 보여주는 마크다운 영역
            output = gr.Markdown(
                label="📖 검색 결과",
                value="여기에 검색 결과가 표시됩니다. 위에 질문을 입력하고 검색 버튼을 클릭하세요."
            )
        
        # 구분선
        gr.Markdown("---")
        
        # 사용 설명서
        gr.Markdown("""
## 📖 사용 설명서

### 이 도구는 어떻게 작동하나요?
1. **질문 분석**: 당신의 질문에서 핵심 키워드를 추출합니다
2. **논문 검색**: arXiv 데이터베이스에서 해당 키워드의 논문들을 찾습니다
3. **요약 생성**: GPT-4o가 각 논문의 핵심을 정리해서 보여줍니다

### 좋은 질문의 예시
- "Transformer를 사용한 자연언어처리 최신 기법"
- "딥러닝 기반 의료 영상 분석"
- "양자 컴퓨팅의 응용 분야"

### 주의사항
- 각 논문 요약 생성에 약 5-10초 정도 시간이 걸립니다
- API 호출 비용이 발생할 수 있습니다
- 검색 결과는 최신 arXiv 논문을 기반으로 합니다

---
*이 도구는 AI 해커톤 프로젝트입니다*
""")
    
    # 검색 버튼이 클릭되었을 때 실행될 함수들을 연결합니다
    # 사용자가 "검색" 버튼을 클릭하면, 
    # user_input과 num_papers 값이 process_user_query_for_web 함수에 전달되고,
    # 결과가 output에 표시됩니다
    search_button.click(
        fn=process_user_query_for_web,
        inputs=[user_input, num_papers],  # 함수에 전달할 입력값들
        outputs=output  # 함수의 결과를 표시할 출력값
    )
    
    return demo


def main():
    """
    Gradio 웹 서버를 시작합니다.
    이 함수를 실행하면 웹 인터페이스가 로컬 호스트에서 실행됩니다.
    """
    
    # Gradio 인터페이스 생성
    demo = create_gradio_interface()
    
    # 웹 서버 실행
    # share=True를 설정하면 공개 URL을 생성해서 다른 사람도 접속할 수 있습니다
    # share=False (기본값)를 사용하면 로컬에서만 접속 가능합니다
    demo.launch(
        share=True,  # 공개 링크 생성 (해커톤 심사를 위해 설정)
        server_name="0.0.0.0",  # 모든 인터페이스에서 접속 가능하게 설정
        server_port=7860  # 웹 포트 (기본값)
    )


if __name__ == "__main__":
    main()
else:
    # Hugging Face Spaces에서 직접 앱을 실행할 때 사용됩니다
    # Hugging Face Spaces는 app.py를 찾아서 자동으로 이 객체를 배포합니다
    demo = create_gradio_interface()
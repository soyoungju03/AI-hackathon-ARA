"""
학술 논문 기반 AI 리서치 어시스턴트 - 테스트 프로토타입
사용자의 질문을 받아서 arXiv에서 논문을 검색하고 요약해주는 프로그램입니다.
"""

import os
import arxiv
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

# 환경 변수에서 API 키를 읽습니다
# .env 파일을 사용하면 API 키를 안전하게 보관할 수 있습니다
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트를 초기화합니다
client = OpenAI(api_key=api_key)

# 비용 추적을 위한 변수들
# API 호출을 할 때마다 토큰 사용량을 기록해서, 나중에 비용을 계산할 수 있습니다
api_calls_log = []


def log_api_call(model, input_tokens, output_tokens, cost):
    """
    API 호출 기록을 남기는 함수입니다.
    나중에 총 비용을 계산하기 위해 각 호출의 정보를 저장합니다.
    """
    api_calls_log.append({
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost
    })


def extract_keywords(user_question):
    """
    사용자의 질문에서 핵심 키워드를 추출합니다.
    이것은 간단한 작업이므로 GPT-4o mini를 사용해서 비용을 절감합니다.
    
    Args:
        user_question (str): 사용자가 입력한 질문
        
    Returns:
        str: 추출된 키워드들 (콤마로 구분됨)
    """
    print("\n[STEP 1] 질문 분석 중... (GPT-4o mini 사용)")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 저렴한 미니 모델 사용
        messages=[
            {
                "role": "user",
                "content": f"""다음 질문에서 가장 중요한 기술 키워드 2-3개를 추출해줘.
                
질문: {user_question}

형식: keyword1, keyword2, keyword3"""
            }
        ],
        temperature=0.3  # 일관성 있는 결과를 위해 낮은 온도 사용
    )
    
    # 토큰 사용량과 비용을 기록합니다
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    # GPT-4o mini의 가격 (1M 토큰 기준)
    # 입력: $0.075, 출력: $0.30
    cost = (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
    
    log_api_call("gpt-4o-mini", input_tokens, output_tokens, cost)
    
    keywords = response.choices[0].message.content
    print(f"추출된 키워드: {keywords}")
    print(f"비용: ${cost:.6f}")
    
    return keywords


def search_papers_on_arxiv(keywords, max_results=5):
    """
    arXiv에서 키워드에 해당하는 논문들을 검색합니다.
    arXiv는 무료로 접근 가능한 학술 논문 데이터베이스입니다.
    
    Args:
        keywords (str): 검색할 키워드들
        max_results (int): 최대 검색 결과 수
        
    Returns:
        list: 검색된 논문들의 정보 (제목, 저자, 초록, URL 등)
    """
    print(f"\n[STEP 2] arXiv에서 논문 검색 중... (검색어: {keywords})")
    
    # arXiv 클라이언트를 초기화합니다
    client_arxiv = arxiv.Client()
    
    # 검색 쿼리를 만듭니다
    search = arxiv.Search(
        query=keywords,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 최신순으로 정렬
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
        print(f"  - {result.title} ({result.published.strftime('%Y-%m-%d')})")
    
    print(f"총 {len(papers)}개의 논문을 찾았습니다.")
    return papers


def summarize_paper_with_gpt4(paper_title, paper_abstract):
    """
    논문의 제목과 초록을 읽고 핵심을 요약합니다.
    이것은 복잡한 이해가 필요하므로 GPT-4o를 사용합니다.
    
    Args:
        paper_title (str): 논문의 제목
        paper_abstract (str): 논문의 초록 (abstract)
        
    Returns:
        str: 논문의 요약
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",  # 더 강력한 모델 사용
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
    
    # 토큰 사용량과 비용을 기록합니다
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    # GPT-4o의 가격 (1M 토큰 기준)
    # 입력: $5, 출력: $15
    cost = (input_tokens * 5 + output_tokens * 15) / 1_000_000
    
    log_api_call("gpt-4o", input_tokens, output_tokens, cost)
    
    return response.choices[0].message.content, cost


def process_user_query(user_question):
    """
    사용자의 질문을 처음부터 끝까지 처리하는 메인 함수입니다.
    Step 1: 키워드 추출
    Step 2: arXiv에서 논문 검색
    Step 3: 각 논문의 초록 요약
    
    Args:
        user_question (str): 사용자의 질문
        
    Returns:
        dict: 처리 결과 (키워드, 찾은 논문들, 각 논문의 요약 등)
    """
    
    print("=" * 60)
    print(f"사용자 질문: {user_question}")
    print("=" * 60)
    
    # Step 1: 키워드 추출 (GPT-4o mini 사용)
    keywords = extract_keywords(user_question)
    
    # Step 2: arXiv에서 논문 검색
    papers = search_papers_on_arxiv(keywords, max_results=3)
    
    # Step 3: 각 논문을 요약 (GPT-4o 사용)
    print("\n[STEP 3] 각 논문 요약 중... (GPT-4o 사용)")
    
    summarized_papers = []
    total_summary_cost = 0
    
    for i, paper in enumerate(papers, 1):
        print(f"\n  논문 {i}: {paper['title']}")
        summary, cost = summarize_paper_with_gpt4(paper['title'], paper['abstract'])
        total_summary_cost += cost
        
        summarized_papers.append({
            "title": paper['title'],
            "authors": paper['authors'],
            "published": paper['published'],
            "url": paper['url'],
            "abstract": paper['abstract'],
            "summary": summary,
            "cost": cost
        })
        print(f"  요약 완료 (비용: ${cost:.6f})")
    
    print(f"\n논문 요약 총 비용: ${total_summary_cost:.6f}")
    
    return {
        "question": user_question,
        "keywords": keywords,
        "papers_found": len(papers),
        "summarized_papers": summarized_papers,
        "total_cost": sum([p['cost'] for p in summarized_papers])
    }


def calculate_total_cost():
    """
    지금까지의 모든 API 호출로 인한 총 비용을 계산합니다.
    """
    total_cost = sum([call['cost'] for call in api_calls_log])
    print(f"\n총 API 비용: ${total_cost:.6f}")
    
    # 각 모델별로 비용을 계산해서 보여줍니다
    mini_cost = sum([call['cost'] for call in api_calls_log if call['model'] == 'gpt-4o-mini'])
    gpt4_cost = sum([call['cost'] for call in api_calls_log if call['model'] == 'gpt-4o'])
    
    print(f"  - GPT-4o mini 비용: ${mini_cost:.6f}")
    print(f"  - GPT-4o 비용: ${gpt4_cost:.6f}")
    print(f"  - 절감된 비용: ${mini_cost:.6f} (GPT-4o로만 했다면 약 10배 이상 비용이 들었을 것입니다)")
    
    return total_cost


def print_detailed_log():
    """
    모든 API 호출의 상세 기록을 출력합니다.
    이것을 통해 어느 부분에서 비용이 많이 드는지 분석할 수 있습니다.
    """
    print("\n" + "=" * 60)
    print("API 호출 상세 기록")
    print("=" * 60)
    
    for i, call in enumerate(api_calls_log, 1):
        print(f"\n호출 {i}:")
        print(f"  모델: {call['model']}")
        print(f"  입력 토큰: {call['input_tokens']}")
        print(f"  출력 토큰: {call['output_tokens']}")
        print(f"  비용: ${call['cost']:.6f}")
        print(f"  시간: {call['timestamp']}")


if __name__ == "__main__":
    # 테스트용 사용자 질문
    # 당신이 실제로 관심 있는 주제로 바꿔서 테스트해보세요
    test_question = "자율주행 자동차의 LiDAR 센서 데이터 처리 최신 기법"
    
    # 쿼리 처리
    result = process_user_query(test_question)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("처리 결과 요약")
    print("=" * 60)
    print(f"질문: {result['question']}")
    print(f"추출된 키워드: {result['keywords']}")
    print(f"찾은 논문 수: {result['papers_found']}")
    print(f"이 쿼리의 비용: ${result['total_cost']:.6f}")
    
    # 각 논문의 요약을 출력합니다
    for i, paper in enumerate(result['summarized_papers'], 1):
        print(f"\n--- 논문 {i}: {paper['title']} ---")
        print(f"저자: {', '.join(paper['authors'][:3])}...")
        print(f"발표 날짜: {paper['published']}")
        print(f"\n요약:\n{paper['summary']}")
        print(f"\nURL: {paper['url']}")
        print(f"이 논문 요약 비용: ${paper['cost']:.6f}")
    
    # 전체 비용 계산 및 출력
    calculate_total_cost()
    
    # 상세 기록 출력
    print_detailed_log()
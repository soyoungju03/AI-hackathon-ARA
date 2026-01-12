# -*- coding: utf-8 -*-
"""
LangGraph 노드 정의
===================

이 파일은 LangGraph 워크플로우의 각 노드(Node)를 정의합니다.

노드(Node)란?
-------------
LangGraph에서 노드는 워크플로우의 각 단계를 나타냅니다.
각 노드는 상태(State)를 입력받아 작업을 수행하고,
수정된 상태를 반환합니다.

ReAct 패턴 적용
---------------
각 노드는 ReAct 패턴의 일부를 담당합니다:
- Thought (생각): 현재 상황 분석
- Action (행동): 도구 실행 등의 행동 수행
- Observation (관찰): 행동 결과 관찰 및 기록

Human-in-the-Loop
-----------------
특정 노드는 Interrupt를 발생시켜 사용자 입력을 기다립니다.
이 기능은 LangGraph의 `interrupt_before` 또는 `interrupt_after` 옵션으로 구현됩니다.
"""

import os
from typing import Literal
from datetime import datetime

# LangChain/LangGraph 임포트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 로컬 모듈 임포트
from app.graph.state import (
    AgentState, 
    Paper, 
    ReActStep, 
    InterruptData,
    add_react_step
)
from app.tools.paper_search.arxiv_tool import search_arxiv
from app.config import get_settings

# 설정 로드
settings = get_settings()


def get_llm(model: str = None):
    """
    LLM 인스턴스를 생성합니다.
    
    이 함수는 설정된 API 키와 모델을 사용하여 
    ChatOpenAI 인스턴스를 생성합니다.
    """
    return ChatOpenAI(
        model=model or settings.default_model,
        api_key=settings.openai_api_key,
        temperature=0.3  # 일관된 결과를 위해 낮은 temperature
    )


# ============================================
# 노드 1: 질문 수신 (receive_question)
# ============================================

def receive_question_node(state: AgentState) -> dict:
    """
    사용자 질문을 수신하고 초기 분석을 시작합니다.
    
    이 노드는 워크플로우의 시작점입니다.
    사용자 질문을 받아서 ReAct의 첫 번째 Thought를 기록합니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        dict: 상태 업데이트 딕셔너리
    """
    user_question = state["user_question"]
    
    # ReAct Thought 기록: 질문을 받았음을 인식
    thought_content = f"""
사용자 질문을 수신했습니다: "{user_question}"
이제 질문을 분석하여 핵심 키워드와 의도를 파악해야 합니다.
    """.strip()
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "react_steps": [new_step]
    }


# ============================================
# 노드 2: 질문 분석 (analyze_question)
# ============================================

QUESTION_ANALYSIS_PROMPT = """
당신은 학술 연구 질문을 분석하는 전문가입니다.
사용자의 질문을 분석하여 다음 정보를 추출해주세요.

## 사용자 질문
{question}

## 분석해야 할 항목

1. **핵심 키워드**: 논문 검색에 사용할 핵심 기술 키워드 2-5개
   - 영어로 변환해주세요 (arXiv는 영어 논문이 대부분입니다)
   - 구체적이고 검색 효과가 좋은 키워드를 선택해주세요

2. **질문 의도**: 사용자가 알고 싶어하는 것이 무엇인지
   - "최신 연구 동향" / "특정 기술 설명" / "비교 분석" / "응용 사례" 등

3. **연구 도메인**: 어떤 학문 분야에 해당하는지
   - "computer science" / "physics" / "mathematics" / "biology" 등

## 응답 형식 (반드시 이 형식을 따라주세요)
KEYWORDS: keyword1, keyword2, keyword3
INTENT: 질문 의도 설명
DOMAIN: 연구 도메인
"""


import logging

logger = logging.getLogger(__name__)

def analyze_question_node(state: AgentState) -> dict:
    """
    사용자 질문을 분석하여 키워드, 의도, 도메인을 추출합니다.
    """
    user_question = state["user_question"]
    
    # 로깅: 분석 시작
    logger.info(f"🔍 질문 분석 시작: {user_question}")
    
    # LLM 호출
    llm = get_llm(settings.light_model)
    prompt = QUESTION_ANALYSIS_PROMPT.format(question=user_question)
    
    # 로깅: LLM 호출 전
    logger.info("📡 LLM에 요청 전송 중...")
    
    response = llm.invoke([
        SystemMessage(content="당신은 학술 연구 질문 분석 전문가입니다."),
        HumanMessage(content=prompt)
    ])
    
    # 로깅: LLM 응답 수신
    logger.info(f"✅ LLM 응답 수신: {response.content[:100]}...")
    
    # 응답 파싱
    response_text = response.content
    
    # 키워드 추출
    keywords = []
    intent = ""
    domain = ""
    
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line.startswith("KEYWORDS:"):
            keywords_str = line.replace("KEYWORDS:", "").strip()
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        elif line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip()
        elif line.startswith("DOMAIN:"):
            domain = line.replace("DOMAIN:", "").strip()
    
    # 로깅: 파싱 완료
    logger.info(f"🔑 추출된 키워드: {keywords}")
    logger.info(f"🎯 질문 의도: {intent}")
    logger.info(f"📚 연구 도메인: {domain}")
    
    # ReAct Observation 기록
    observation_content = f"""
질문 분석 완료:
- 추출된 키워드: {keywords}
- 질문 의도: {intent}
- 연구 도메인: {domain}
    """.strip()
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "extracted_keywords": keywords,
        "question_intent": intent,
        "question_domain": domain,
        "react_steps": [new_step]
    }


# ============================================
# 노드 3: 사용자 확인 요청 (request_user_confirmation)
# Human-in-the-Loop Interrupt 발생
# ============================================

def request_user_confirmation_node(state: AgentState) -> dict:
    """
    사용자에게 키워드와 검색 설정을 확인받습니다.
    
    이 노드는 Human-in-the-Loop의 핵심입니다.
    Interrupt를 발생시켜 워크플로우를 일시 중지하고,
    사용자의 입력을 기다립니다.
    
    사용자에게 보여줄 정보:
    1. 추출된 키워드 (수정 가능)
    2. 검색할 논문 수 선택
    3. 검색 소스 선택
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        dict: Interrupt 데이터를 포함한 상태 업데이트
    """
    keywords = state["extracted_keywords"]
    
    # 사용자에게 보여줄 메시지 구성
    message = f"""
## 🔍 검색 설정 확인

분석된 키워드를 확인하고 검색 설정을 선택해주세요.

### 추출된 키워드
{', '.join(keywords)}

### 검색 옵션
- 검색할 논문 수: 1-10개 중 선택
- 검색 소스: arXiv (기본)

수정이 필요하면 알려주세요. 그대로 진행하려면 "확인"을 눌러주세요.
    """.strip()
    
    interrupt_data = InterruptData(
        interrupt_type="confirm_keywords",
        message=message,
        options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default_value="3",
        metadata={
            "keywords": keywords,
            "suggested_sources": ["arxiv"]
        }
    )
    
    # ReAct Thought 기록
    thought_content = """
키워드 추출이 완료되었습니다. 
사용자에게 확인을 요청하고, 검색할 논문 수를 선택받아야 합니다.
워크플로우를 일시 중지하고 사용자 입력을 기다립니다.
    """.strip()
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "interrupt_data": interrupt_data,
        "waiting_for_user": True,
        "react_steps": [new_step]
    }


# ============================================
# 노드 4: 사용자 응답 처리 (process_user_response)
# ============================================

def process_user_response_node(state: AgentState) -> dict:
    """
    사용자의 응답을 처리하고 검색 설정을 업데이트합니다.
    
    이 노드는 사용자가 Interrupt에 응답한 후 실행됩니다.
    사용자의 선택에 따라:
    1. 키워드 수정 (필요한 경우)
    2. 논문 수 설정
    3. 검색 소스 설정
    
    Args:
        state: 현재 워크플로우 상태 (user_response 포함)
    
    Returns:
        dict: 업데이트된 검색 설정
    """
    user_response = state.get("user_response", "3")
    
    # 사용자 응답 파싱
    # 간단한 구현: 숫자만 있으면 논문 수로 해석
    try:
        paper_count = int(user_response)
        paper_count = max(1, min(10, paper_count))  # 1-10 범위로 제한
    except ValueError:
        paper_count = 3  # 파싱 실패 시 기본값
    
    # ReAct Observation 기록
    observation_content = f"""
사용자 응답 처리 완료:
- 선택된 논문 수: {paper_count}
- 검색 소스: arXiv
검색을 시작합니다.
    """.strip()
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "paper_count": paper_count,
        "waiting_for_user": False,
        "interrupt_data": None,
        "react_steps": [new_step]
    }


# ============================================
# 노드 5: 논문 검색 (search_papers)
# ============================================

def search_papers_node(state: AgentState) -> dict:
    """
    설정된 키워드와 옵션으로 논문을 검색합니다.
    
    이 노드는 ReAct 패턴에서 Action을 담당합니다.
    arXiv (그리고 나중에 다른 소스들)에서 논문을 검색합니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        dict: 검색된 논문 목록을 포함한 상태 업데이트
    """
    keywords = state["extracted_keywords"]
    paper_count = state.get("paper_count", 3)
    domain = state.get("question_domain", None)
    
    # ReAct Action 기록
    action_content = f"""
논문 검색을 실행합니다:
- 키워드: {keywords}
- 검색 수: {paper_count}
- 도메인: {domain or '전체'}
- 소스: arXiv
    """.strip()
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    try:
        # arXiv 검색 실행
        papers = search_arxiv(
            keywords=keywords,
            max_results=paper_count,
            domain=domain
        )
        
        # ReAct Observation 기록
        observation_content = f"""
검색 완료: {len(papers)}개의 논문을 찾았습니다.
논문 목록:
"""
        for i, paper in enumerate(papers, 1):
            observation_content += f"\n{i}. {paper.title} (연관성: {paper.relevance_score})"
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content.strip()
        )
        
        return {
            "papers": papers,  # Annotated[List, operator.add]이므로 추가됨
            "react_steps": [action_step, observation_step],
            "error_message": None
        }
        
    except Exception as e:
        # 에러 발생 시
        error_step = ReActStep(
            step_type="observation",
            content=f"검색 중 오류 발생: {str(e)}"
        )
        
        return {
            "react_steps": [action_step, error_step],
            "error_message": str(e)
        }


# ============================================
# 노드 6: 연관성 평가 (evaluate_relevance)
# ============================================

def evaluate_relevance_node(state: AgentState) -> dict:
    """
    검색된 논문들의 연관성을 재평가하고 필터링합니다.
    
    이 노드는 검색 결과 중에서 실제로 사용자 질문과
    관련 있는 논문만 선별합니다.
    
    연관성이 낮은 논문을 걸러내어 품질을 보장합니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        dict: 필터링된 논문 목록
    """
    papers = state.get("papers", [])
    threshold = settings.relevance_threshold
    
    # 연관성 점수가 임계값 이상인 논문만 선택
    relevant_papers = [p for p in papers if p.relevance_score >= threshold]
    
    # 만약 필터링 후 논문이 없으면, 상위 논문이라도 포함
    if not relevant_papers and papers:
        relevant_papers = papers[:min(3, len(papers))]
    
    # ReAct Thought 기록
    thought_content = f"""
연관성 평가 완료:
- 전체 검색 결과: {len(papers)}개
- 임계값({threshold}) 이상: {len(relevant_papers)}개
- 선별된 논문: {[p.title[:30] + '...' for p in relevant_papers]}
    """.strip()
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "relevant_papers": relevant_papers,
        "react_steps": [new_step]
    }


# ============================================
# 노드 7: 논문 요약 생성 (summarize_papers)
# ============================================

SUMMARIZE_PROMPT = """
다음 논문의 초록을 읽고 핵심 내용을 한국어로 요약해주세요.

## 논문 제목
{title}

## 초록
{abstract}

## 요약 형식 (다음 형식을 따라주세요)

### 핵심 아이디어
[논문의 주요 기여점을 2-3문장으로 설명]

### 연구 배경 및 문제점
[해결하고자 하는 문제를 설명]

### 제안 방법론
[문제 해결 접근법을 설명]

### 주요 성과
[실험 결과나 달성한 성과를 설명]
"""


def summarize_papers_node(state: AgentState) -> dict:
    """
    선별된 논문들의 요약을 생성합니다.
    
    각 논문에 대해 LLM을 호출하여 구조화된 요약을 생성합니다.
    이 과정은 시간이 걸릴 수 있습니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        dict: 요약이 추가된 논문 목록
    """
    relevant_papers = state.get("relevant_papers", [])
    
    if not relevant_papers:
        return {
            "react_steps": [ReActStep(
                step_type="observation",
                content="요약할 논문이 없습니다."
            )]
        }
    
    llm = get_llm()  # 기본 모델 사용 (요약은 품질이 중요)
    
    # ReAct Action 기록
    action_step = ReActStep(
        step_type="action",
        content=f"{len(relevant_papers)}개 논문의 요약을 생성합니다."
    )
    
    summarized_papers = []
    
    for paper in relevant_papers:
        try:
            prompt = SUMMARIZE_PROMPT.format(
                title=paper.title,
                abstract=paper.abstract
            )
            
            response = llm.invoke([
                HumanMessage(content=prompt)
            ])
            
            # 요약을 논문 객체에 추가
            paper.summary = response.content
            summarized_papers.append(paper)
            
        except Exception as e:
            # 개별 논문 요약 실패 시에도 계속 진행
            paper.summary = f"요약 생성 실패: {str(e)}"
            summarized_papers.append(paper)
    
    # ReAct Observation 기록
    observation_step = ReActStep(
        step_type="observation",
        content=f"{len(summarized_papers)}개 논문의 요약이 완료되었습니다."
    )
    
    return {
        "relevant_papers": summarized_papers,
        "react_steps": [action_step, observation_step]
    }


# ============================================
# 노드 8: 최종 응답 생성 (generate_response)
# ============================================

FINAL_RESPONSE_PROMPT = """
사용자의 질문에 대해 검색된 논문들을 바탕으로 종합적인 답변을 생성해주세요.

## 사용자 질문
{question}

## 검색된 논문들
{papers_info}

## 답변 형식

친절하고 자세한 답변을 작성해주세요. 다음 요소를 포함해야 합니다:

1. **질문에 대한 직접적인 답변** (2-3문장)
2. **관련 연구 동향 요약** (논문들의 공통 주제 및 트렌드)
3. **각 논문 요약** (이미 생성된 요약을 활용)
4. **추가 탐구 제안** (더 알아볼 만한 주제나 키워드)

답변은 한국어로 작성하되, 기술 용어는 영어 병기해주세요.
"""


def generate_response_node(state: AgentState) -> dict:
    """
    검색 결과를 종합하여 최종 응답을 생성합니다.
    
    이 노드는 ReAct 패턴에서 최종 Decision을 담당합니다.
    모든 분석과 검색 결과를 종합하여 사용자에게
    제공할 최종 답변을 생성합니다.
    
    Args:
        state: 현재 워크플로우 상태
    
    Returns:
        dict: 최종 응답을 포함한 상태 업데이트
    """
    user_question = state["user_question"]
    relevant_papers = state.get("relevant_papers", [])
    error_message = state.get("error_message")
    
    # 에러가 있으면 에러 메시지 반환
    if error_message:
        return {
            "final_response": f"죄송합니다. 검색 중 오류가 발생했습니다: {error_message}",
            "is_complete": True
        }
    
    # 검색 결과가 없으면
    if not relevant_papers:
        return {
            "final_response": """
죄송합니다. 입력하신 질문과 관련된 논문을 찾지 못했습니다.

다음을 시도해보세요:
- 더 구체적인 키워드로 질문해주세요
- 영어 키워드를 사용해보세요
- 다른 관점에서 질문을 다시 작성해보세요
            """.strip(),
            "is_complete": True
        }
    
    # 논문 정보 포맷팅
    papers_info = ""
    for i, paper in enumerate(relevant_papers, 1):
        papers_info += f"""
### 논문 {i}: {paper.title}
- 저자: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
- 출판일: {paper.published_date}
- URL: {paper.url}
- 연관성 점수: {paper.relevance_score}

**요약:**
{paper.summary or '요약 없음'}

---
"""
    
    # LLM으로 최종 응답 생성
    llm = get_llm()
    
    prompt = FINAL_RESPONSE_PROMPT.format(
        question=user_question,
        papers_info=papers_info
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content="당신은 친절하고 전문적인 학술 연구 어시스턴트입니다."),
            HumanMessage(content=prompt)
        ])
        
        final_response = response.content
        
    except Exception as e:
        # LLM 호출 실패 시 기본 응답
        final_response = f"""
## 검색 결과

질문: {user_question}

{len(relevant_papers)}개의 관련 논문을 찾았습니다:

{papers_info}
        """.strip()
    
    # ReAct Decision 기록
    decision_step = ReActStep(
        step_type="thought",
        content="최종 응답 생성이 완료되었습니다. 워크플로우를 종료합니다."
    )
    
    return {
        "final_response": final_response,
        "is_complete": True,
        "react_steps": [decision_step]
    }

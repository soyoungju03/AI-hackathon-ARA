# -*- coding: utf-8 -*-
"""
수정된 LangGraph 노드들 (PDF 임베딩 파이프라인 완전 통합)

주요 기능:
1. 두 단계 Human-in-the-Loop (키워드 확인 + 논문 수 선택)
2. arXiv 검색 + PDF 임베딩 파이프라인 통합
3. ChromaDB를 사용한 의미 기반 검색
4. ReAct 패턴을 따른 투명한 AI 사고 과정

데이터 흐름:
사용자 질문 → 분석 → 키워드 확인 → 논문 수 선택 →
arXiv 검색 → PDF 다운로드 → 텍스트 추출 → 청킹 → 임베딩 →
ChromaDB 저장 → 의미 기반 검색 → 요약 및 답변
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import (
    AgentState, 
    ReActStep, 
    InterruptData
)
from app.tools.paper_search.arxiv_tool import search_arxiv
from app.config import get_settings
from app.tools.pdf_embedding_pipeline_final import PDFEmbeddingPipeline
from app.tools.embeddings import SentenceTransformerEmbedding
from app.tools.vectorstore import ArxivPaperVectorStore

settings = get_settings()
logger = logging.getLogger(__name__)


# ============================================
# PDF 임베딩 파이프라인 초기화 (싱글톤)
# ============================================

_pdf_pipeline = None

def get_pdf_pipeline():
    """
    PDF 처리 파이프라인을 초기화합니다.
    
    싱글톤 패턴을 사용하므로 첫 번째 호출 시에만 초기화되고,
    이후 호출에서는 이미 초기화된 인스턴스를 반환합니다.
    
    임베딩 모델과 벡터 스토어는 메모리에 상주하므로,
    여러 요청에서 재사용할 수 있어 효율적입니다.
    """
    
    global _pdf_pipeline
    
    if _pdf_pipeline is None:
        try:
            logger.info("[INIT] PDF 임베딩 파이프라인 초기화 중...")
            
            # Step 1: 임베딩 모델 초기화
            # distiluse-base-multilingual-cased-v2는 한국어를 포함한 여러 언어를 지원합니다
            # 모델 크기는 약 100MB이고, 처음 로드 시 Hugging Face에서 다운로드됩니다
            embedding_model = SentenceTransformerEmbedding(
                model_name="distiluse-base-multilingual-cased-v2"
            )
            
            # Step 2: ChromaDB 벡터 스토어 초기화
            # 이것은 당신의 vectorstore.py에 정의된 ArxivPaperVectorStore입니다
            vectorstore = ArxivPaperVectorStore(
                persist_directory="./data/arxiv_chunks",
                collection_name="arxiv_chunks"
            )
            
            # Step 3: PDF 처리 파이프라인 생성
            # 이 파이프라인은 PDF 다운로드부터 임베딩까지 모든 과정을 관리합니다
            _pdf_pipeline = PDFEmbeddingPipeline(
                embedding_model=embedding_model,
                vectorstore=vectorstore,
                chunk_chars=1800,      # 각 청크의 대략적인 문자 수 (약 450 토큰)
                overlap_chars=350,     # 청크 간 오버래프 (문맥 손실 방지)
                batch_size=32          # 한 번에 임베딩할 청크 수
            )
            
            logger.info("✓ PDF 임베딩 파이프라인 초기화 완료")
            logger.info(f"  - 임베딩 모델: distiluse-base-multilingual-cased-v2")
            logger.info(f"  - 벡터 저장소: ChromaDB")
            logger.info(f"  - 청크 크기: 1800 문자 (~450 토큰)")
        
        except ImportError as e:
            logger.error(f"PDF 파이프라인 모듈 임포트 실패: {str(e)}")
            logger.error("pdf_embedding_pipeline_final.py가 app/tools/ 디렉토리에 있는지 확인하세요")
            raise
        except Exception as e:
            logger.error(f"PDF 파이프라인 초기화 실패: {str(e)}")
            raise
    
    return _pdf_pipeline


def get_llm(model: str = None):
    """LLM 인스턴스를 생성합니다."""
    return ChatOpenAI(
        model=model or settings.default_model,
        api_key=settings.openai_api_key,
        temperature=0.3
    )


# ============================================
# 노드 1: 질문 수신 (receive_question_node)
# ============================================

def receive_question_node(state: AgentState) -> dict:
    """
    사용자의 질문을 수신하고 처리를 시작합니다.
    
    이것은 워크플로우의 첫 번째 노드로, 단순히 질문을 확인하고
    다음 단계인 질문 분석을 준비합니다.
    """
    user_question = state.get("user_question", "")
    
    logger.info("="*60)
    logger.info("[RECEIVE_QUESTION] 사용자 질문 수신")
    logger.info("="*60)
    logger.info(f"질문: {user_question}")
    
    thought_content = f'사용자 질문을 수신했습니다: "{user_question}"\n이제 질문을 분석하여 핵심 키워드와 의도를 파악해야 합니다.'
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "react_steps": [new_step]
    }


# ============================================
# 노드 2: 질문 분석 (analyze_question_node)
# ============================================

QUESTION_ANALYSIS_PROMPT = """
당신은 학술 연구 질문을 분석하는 전문가입니다.
사용자의 질문을 분석하여 다음 정보를 추출해주세요.

## 사용자 질문
{question}

## 분석해야 할 항목

1. **핵심 키워드**: 논문 검색에 사용할 핵심 기술 키워드 2-5개
   - 영어로 변환해주세요
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


def analyze_question_node(state: AgentState) -> dict:
    """
    사용자 질문을 분석하여 핵심 키워드를 추출합니다.
    
    이 노드는 LLM을 사용하여 사용자의 자연어 질문을 파싱하고,
    arXiv 검색에 사용할 수 있는 구조화된 정보로 변환합니다.
    """
    
    user_question = state.get("user_question", "")
    
    logger.info("="*60)
    logger.info("[ANALYZE_QUESTION] 질문 분석 시작")
    logger.info("="*60)
    logger.info(f"분석 대상: {user_question[:50]}...")
    
    try:
        llm = get_llm(settings.light_model)
        prompt = QUESTION_ANALYSIS_PROMPT.format(question=user_question)
        
        logger.info("LLM에 질문 분석 요청 전송...")
        
        response = llm.invoke([
            SystemMessage(content="당신은 학술 연구 질문 분석 전문가입니다."),
            HumanMessage(content=prompt)
        ])
        
        logger.info("✓ LLM 응답 수신")
        
        # LLM 응답 파싱
        response_text = response.content
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
        
        logger.info(f"✓ 분석 완료")
        logger.info(f"  추출된 키워드: {keywords}")
        logger.info(f"  질문 의도: {intent}")
        logger.info(f"  연구 도메인: {domain}")
        
        observation_content = f"질문 분석 완료:\n- 추출된 키워드: {keywords}\n- 질문 의도: {intent}\n- 연구 도메인: {domain}"
        
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
        
    except Exception as e:
        logger.error(f"질문 분석 중 오류: {str(e)}", exc_info=True)
        return {
            "extracted_keywords": ["research"],  # 기본 키워드
            "question_intent": "general research",
            "question_domain": "computer science",
            "error_message": str(e),
            "react_steps": [ReActStep(step_type="observation", content=f"분석 실패, 기본값 사용: {str(e)}")]
        }


# ============================================
# 노드 3: 키워드 확인 요청 (request_keyword_confirmation_node)
# 첫 번째 Human-in-the-Loop Interrupt 지점
# ============================================

def request_keyword_confirmation_node(state: AgentState) -> dict:
    """
    추출된 키워드가 맞는지 사용자에게 확인받습니다.
    
    첫 번째 Human-in-the-Loop Interrupt 지점입니다.
    워크플로우는 여기서 멈추고 사용자가 응답할 때까지 대기합니다.
    """
    
    keywords = state.get("extracted_keywords", [])
    
    logger.info("[REQUEST_KEYWORD_CONFIRMATION] 사용자 확인 대기 시작")
    
    message = f"""
추출된 키워드를 확인해주세요.

키워드: {', '.join(keywords) if keywords else '없음'}

맞으면 "확인"을, 수정이 필요하면 "다시"라고 입력해주세요.
    """.strip()
    
    interrupt_data = InterruptData(
        interrupt_type="confirm_keywords",
        message=message,
        options=["확인", "다시"],
        default_value="확인",
        metadata={
            "keywords": keywords,
            "stage": 1
        }
    )
    
    thought_content = "키워드 추출이 완료되었습니다. 사용자에게 확인을 요청합니다."
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "interrupt_data": interrupt_data,
        "waiting_for": "keyword_confirmation",
        "interrupt_stage": 1,
        "waiting_for_user": True,
        "react_steps": [new_step]
    }


# ============================================
# 노드 4: 키워드 확인 응답 처리
# ============================================

def process_keyword_confirmation_response_node(state: AgentState) -> dict:
    """
    사용자의 키워드 확인 응답을 처리합니다.
    
    - "다시" → 질문 분석 단계로 돌아감
    - 그 외 ("확인" 등) → 논문 수 선택 단계로 진행
    """
    
    user_response = state.get("user_response", "").strip().lower()
    
    logger.info("[PROCESS_KEYWORD_CONFIRMATION] 사용자 응답 처리")
    logger.info(f"  응답: {user_response}")
    
    # "다시" 응답 확인
    if user_response in ["다시", "retry", "다시하기", "수정", "다시해", "reanalyze"]:
        logger.info("  → '다시' 선택: 질문 분석 단계로 이동")
        
        observation_content = "사용자가 키워드 재분석을 요청했습니다. 질문 분석 단계로 돌아갑니다."
        
        new_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "keyword_confirmation_response": "retry",
            "waiting_for": None,
            "waiting_for_user": False,
            "interrupt_data": None,
            "react_steps": [new_step],
            "user_response": None
        }
    
    # 그 외의 경우 "확인"으로 처리
    logger.info("  → '확인' 선택: 논문 수 선택 단계로 이동")
    
    observation_content = f"사용자가 키워드를 확인했습니다. 키워드: {', '.join(state.get('extracted_keywords', []))}"
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "keyword_confirmation_response": "confirmed",
        "waiting_for": None,
        "waiting_for_user": False,
        "interrupt_data": None,
        "interrupt_stage": 1,
        "react_steps": [new_step],
        "user_response": None
    }


# ============================================
# 노드 5: 논문 수 선택 요청 (request_paper_count_node)
# 두 번째 Human-in-the-Loop Interrupt 지점
# ============================================

def request_paper_count_node(state: AgentState) -> dict:
    """
    몇 개의 논문을 검색할지 사용자에게 선택받습니다.
    
    두 번째 Human-in-the-Loop Interrupt 지점입니다.
    워크플로우는 여기서 멈추고 사용자의 선택을 기다립니다.
    """
    
    logger.info("[REQUEST_PAPER_COUNT] 사용자 선택 대기 시작")
    
    message = """
검색할 논문의 개수를 선택해주세요.

1부터 10 사이의 숫자를 입력해주세요.
(기본값: 3개)

더 많은 논문을 선택할수록 처리 시간이 길어집니다.
    """.strip()
    
    interrupt_data = InterruptData(
        interrupt_type="select_paper_count",
        message=message,
        options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default_value="3",
        metadata={
            "stage": 2
        }
    )
    
    thought_content = "키워드 확인이 완료되었습니다. 이제 검색할 논문 수를 사용자에게 선택받습니다."
    
    new_step = ReActStep(
        step_type="thought",
        content=thought_content
    )
    
    return {
        "interrupt_data": interrupt_data,
        "waiting_for": "paper_count_selection",
        "interrupt_stage": 2,
        "waiting_for_user": True,
        "react_steps": [new_step]
    }


# ============================================
# 노드 6: 논문 수 응답 처리
# ============================================

def process_paper_count_response_node(state: AgentState) -> dict:
    """
    사용자가 선택한 논문 수를 처리합니다.
    
    입력값을 정수로 파싱하고, 1-10 범위로 제한합니다.
    잘못된 입력의 경우 기본값 3을 사용합니다.
    """
    
    user_response = state.get("user_response", "3")
    
    logger.info("[PROCESS_PAPER_COUNT] 사용자 응답 처리")
    logger.info(f"  응답: {user_response}")
    
    try:
        paper_count = int(user_response)
        # 유효한 범위로 제한
        paper_count = max(1, min(10, paper_count))
        logger.info(f"  → 해석됨: {paper_count}개")
    except ValueError:
        logger.warning(f"  → 유효하지 않은 입력, 기본값 3 사용")
        paper_count = 3
    
    observation_content = f"사용자가 논문 수를 선택했습니다: {paper_count}개"
    
    new_step = ReActStep(
        step_type="observation",
        content=observation_content
    )
    
    return {
        "paper_count": paper_count,
        "waiting_for": None,
        "waiting_for_user": False,
        "interrupt_data": None,
        "interrupt_stage": 2,
        "react_steps": [new_step],
        "user_response": None
    }


# ============================================
# 노드 7: 논문 검색 + PDF 처리 (search_papers_node)
# ============================================

def search_papers_node(state: AgentState) -> dict:
    """
    arXiv에서 논문을 검색한 후 PDF 임베딩 파이프라인을 실행합니다.
    
    이 노드가 당신의 시스템에서 가장 복잡하고 시간이 많이 걸리는 부분입니다.
    
    실행 단계:
    1. arXiv API를 사용하여 논문 검색
    2. 각 논문의 PDF를 다운로드
    3. PDF에서 텍스트 추출
    4. 텍스트를 청크로 분할 (약 450 토큰씩)
    5. 각 청크를 Sentence Transformers로 임베딩
    6. 임베딩된 청크를 ChromaDB에 저장
    
    이러한 저장된 청크들은 나중에 evaluate_relevance_node에서
    사용자의 질문과 의미론적으로 유사한 부분을 찾는 데 사용됩니다.
    """
    
    keywords = state.get("extracted_keywords", [])
    paper_count = state.get("paper_count", 3)
    domain = state.get("question_domain", None)
    
    logger.info("="*60)
    logger.info("[SEARCH_PAPERS] 논문 검색 + PDF 처리 시작")
    logger.info("="*60)
    logger.info(f"키워드: {keywords}")
    logger.info(f"검색 개수: {paper_count}개")
    logger.info(f"도메인: {domain or '전체'}")
    
    action_content = f"""논문 검색 및 PDF 처리 파이프라인 시작:
- 키워드: {', '.join(keywords)}
- 검색 개수: {paper_count}개
- 도메인: {domain or '전체'}

처리 단계:
1) arXiv에서 논문 검색
2) 각 논문의 PDF 다운로드
3) PDF에서 텍스트 추출
4) 텍스트를 청크로 분할
5) 각 청크를 임베딩
6) ChromaDB에 저장"""
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    try:
        # Step 1: arXiv에서 논문 검색
        logger.info("\nStep 1: arXiv 검색 실행...")
        
        papers = search_arxiv(
            keywords=keywords,
            max_results=paper_count,
            domain=domain
        )
        
        if not papers:
            logger.warning("검색 결과 없음")
            observation_step = ReActStep(
                step_type="observation",
                content="arXiv에서 관련 논문을 찾지 못했습니다. 다른 키워드로 시도해주세요."
            )
            
            return {
                "papers": [],
                "chunks_saved": 0,
                "react_steps": [action_step, observation_step],
                "error_message": "검색 결과 없음"
            }
        
        logger.info(f"✓ {len(papers)}개 논문 검색 완료")
        
        # Step 2: PDF 처리 파이프라인 초기화
        logger.info("\nStep 2: PDF 처리 파이프라인 초기화...")
        
        pipeline = get_pdf_pipeline()
        
        logger.info(f"✓ 파이프라인 준비 완료")
        
        # Step 3: 논문을 파이프라인 형식으로 변환
        logger.info("\nStep 3: 논문 데이터 변환...")
        
        papers_for_pipeline = []
        for paper in papers:
            # paper 객체의 속성을 딕셔너리로 변환
            arxiv_id = paper.url.split('/')[-1] if hasattr(paper, 'url') and paper.url else 'unknown'
            
            paper_dict = {
                'arxiv_id': arxiv_id,
                'title': getattr(paper, 'title', ''),
                'abstract': getattr(paper, 'abstract', ''),
                'authors': getattr(paper, 'authors', []),
                'published_date': getattr(paper, 'published_date', ''),
                'categories': getattr(paper, 'categories', []),
                'url': getattr(paper, 'url', ''),
            }
            papers_for_pipeline.append(paper_dict)
        
        logger.info(f"✓ {len(papers_for_pipeline)}개 논문 변환 완료")
        
        # Step 4: PDF 처리 파이프라인 실행 (배치 모드)
        logger.info("\nStep 4: PDF 처리 파이프라인 실행 중...")
        logger.info("  (이 단계는 시간이 걸릴 수 있습니다)")
        
        batch_result = pipeline.process_papers_batch(
            papers=papers_for_pipeline,
            max_pages=10  # 각 논문에서 최대 10페이지만 처리
        )
        
        logger.info(f"✓ PDF 처리 완료")
        
        # 처리 결과 분석
        total_chunks_created = sum(r.get('chunks_created', 0) for r in batch_result['results'])
        total_chunks_saved = batch_result['total_chunks']
        
        logger.info(f"\n처리 결과 통계:")
        logger.info(f"  - 처리된 논문: {batch_result['successful']}/{len(papers)}")
        logger.info(f"  - 생성된 청크: {total_chunks_created}개")
        logger.info(f"  - 저장된 청크: {total_chunks_saved}개")
        logger.info(f"  - 처리 시간: {batch_result['time']:.1f}초")
        
        # 처리 결과 요약
        observation_content = f"""검색 및 PDF 처리 완료:

검색 결과:
- 검색된 논문: {len(papers)}개
- 처리 성공: {batch_result['successful']}개
- 처리 실패: {batch_result['failed']}개

청크 처리 통계:
- 생성된 청크: {total_chunks_created}개
- 저장된 청크: {total_chunks_saved}개
- 처리 시간: {batch_result['time']:.1f}초

다음 단계에서 사용자의 질문과 의미론적으로 
가장 유사한 청크들을 검색합니다."""
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "papers": papers,
            "chunks_saved": total_chunks_saved,
            "pdf_processing_result": batch_result,
            "react_steps": [action_step, observation_step],
            "error_message": None
        }
    
    except Exception as e:
        logger.error(f"처리 중 오류: {str(e)}", exc_info=True)
        
        error_observation = ReActStep(
            step_type="observation",
            content=f"""처리 중 오류 발생:
{str(e)}

다음을 확인해주세요:
1. 인터넷 연결 상태
2. arXiv 서버 상태
3. 디스크 공간
4. 메모리 용량"""
        )
        
        return {
            "papers": [],
            "chunks_saved": 0,
            "react_steps": [action_step, error_observation],
            "error_message": str(e)
        }


# ============================================
# 노드 8: 의미 기반 관련성 평가 (evaluate_relevance_node)
# ============================================

def evaluate_relevance_node(state: AgentState) -> dict:
    """
    ChromaDB에 저장된 청크들 중에서 사용자의 질문과
    의미론적으로 가장 유사한 청크들을 검색합니다.
    
    이 노드의 핵심은 코사인 유사도 계산입니다.
    사용자의 질문을 벡터로 변환한 후, 저장된 모든 청크 벡터와 비교합니다.
    가장 높은 유사도를 가진 청크들이 반환됩니다.
    
    이러한 청크 기반 검색은 논문 초록만 보는 것보다 훨씬 정밀합니다.
    """
    
    user_question = state.get("user_question", "")
    paper_count = state.get("paper_count", 3)
    chunks_saved = state.get("chunks_saved", 0)
    
    logger.info("="*60)
    logger.info("[EVALUATE_RELEVANCE] 의미 기반 검색 시작")
    logger.info("="*60)
    logger.info(f"질문: {user_question[:50]}...")
    logger.info(f"검색할 청크 수: {paper_count * 3}개")
    logger.info(f"사용 가능한 청크: {chunks_saved}개")
    
    action_content = f"""의미 기반 청크 검색:
- 질문: "{user_question[:60]}..."
- 검색 방식: 코사인 유사도 (Sentence Transformers)
- 목표 청크 수: {paper_count * 3}개
- 사용 가능한 청크: {chunks_saved}개"""
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    if chunks_saved == 0:
        logger.warning("저장된 청크가 없음")
        observation_step = ReActStep(
            step_type="observation",
            content="""의미 기반 검색 실패:
이전 단계에서 청크가 저장되지 않았습니다.
검색이 실패했거나 모든 PDF 처리가 실패했을 가능성이 있습니다."""
        )
        
        return {
            "relevant_chunks": [],
            "evaluation_result": {
                "success": False,
                "message": "저장된 청크가 없습니다"
            },
            "react_steps": [action_step, observation_step]
        }
    
    try:
        # PDF 파이프라인에서 벡터스토어 접근
        logger.info("\nPDF 파이프라인에서 벡터스토어 접근...")
        
        pipeline = get_pdf_pipeline()
        vectorstore = pipeline.vectorstore
        embedding_model = pipeline.embedding_model
        
        logger.info("✓ 벡터스토어 접근 성공")
        
        # Step 1: 사용자 질문을 임베딩
        logger.info("\nStep 1: 질문 임베딩 생성...")
        
        query_embedding = embedding_model.embed(user_question)
        
        logger.info(f"✓ 임베딩 완료 (차원: {len(query_embedding)})")
        
        # Step 2: ChromaDB에서 검색
        logger.info(f"\nStep 2: ChromaDB에서 검색 (상위 {paper_count * 3}개)...")
        
        search_results = vectorstore.collection.query(
            query_embeddings=[query_embedding],
            n_results=paper_count * 3,
            include=["documents", "metadatas", "distances"]
        )
        
        logger.info(f"✓ 검색 완료: {len(search_results['ids'][0]) if search_results['ids'] else 0}개 청크 발견")
        
        # Step 3: 검색 결과 처리
        logger.info("\nStep 3: 검색 결과 처리...")
        
        relevant_chunks = []
        
        if search_results['ids'] and len(search_results['ids']) > 0:
            for i, chunk_id in enumerate(search_results['ids'][0]):
                distance = search_results['distances'][0][i]
                similarity_score = 1 - distance  # 거리를 유사도로 변환
                
                metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                chunk_content = search_results['documents'][0][i] if search_results['documents'] else ''
                
                chunk_info = {
                    'chunk_id': chunk_id,
                    'content': chunk_content,
                    'similarity_score': float(similarity_score),
                    'arxiv_id': metadata.get('arxiv_id', ''),
                    'title': metadata.get('title', ''),
                    'section': metadata.get('section', ''),
                    'page_number': int(metadata.get('page_number', 1)) if metadata.get('page_number') else 1,
                    'chunk_index': metadata.get('chunk_index', ''),
                    'authors': metadata.get('authors', ''),
                    'metadata': metadata
                }
                
                relevant_chunks.append(chunk_info)
                
                if i < 3:  # 상위 3개만 로깅
                    logger.debug(f"  청크 {i+1}: 유사도 {similarity_score:.4f}")
        
        logger.info(f"✓ {len(relevant_chunks)}개 청크 처리 완료")
        
        # 상위 청크 정보로 관찰 작성
        observation_parts = [f"의미 기반 검색 완료: {len(relevant_chunks)}개 청크 발견"]
        observation_parts.append("\n가장 관련성 높은 상위 3개 청크:")
        
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            observation_parts.append(f"\n{i}. 유사도: {chunk['similarity_score']:.4f}")
            observation_parts.append(f"   논문: {chunk['title'][:40] if chunk['title'] else 'N/A'}...")
            if chunk.get('section'):
                observation_parts.append(f"   섹션: {chunk['section']}")
            observation_parts.append(f"   내용: {chunk['content'][:60]}...")
        
        observation_content = "\n".join(observation_parts)
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        logger.info(f"\n✓ 의미 기반 검색 완료")
        
        return {
            "relevant_chunks": relevant_chunks[:paper_count * 2],
            "evaluation_result": {
                "success": True,
                "message": f"{len(relevant_chunks)}개의 관련 청크를 찾았습니다"
            },
            "react_steps": [action_step, observation_step]
        }
    
    except Exception as e:
        logger.error(f"검색 중 오류: {str(e)}", exc_info=True)
        
        error_observation = ReActStep(
            step_type="observation",
            content=f"""의미 기반 검색 중 오류:
{str(e)}

원인 분석:
- ChromaDB 연결 실패
- 임베딩 모델 로드 실패
- 질문 벡터화 실패"""
        )
        
        return {
            "relevant_chunks": [],
            "evaluation_result": {
                "success": False,
                "message": f"검색 중 오류: {str(e)}"
            },
            "react_steps": [action_step, error_observation]
        }


# ============================================
# 노드 9: 논문 요약 생성 (summarize_papers_node)
# ============================================

SUMMARIZE_PROMPT = """
다음 논문의 초록을 읽고 핵심 내용을 한국어로 요약해주세요.

## 논문 제목
{title}

## 초록
{abstract}

## 요약 형식

### 핵심 아이디어
논문의 주요 기여점을 2-3문장으로 설명해주세요.

### 연구 배경 및 문제점
해결하고자 하는 문제를 설명해주세요.

### 제안 방법론
문제 해결 접근법을 설명해주세요.

### 주요 성과
실험 결과나 달성한 성과를 설명해주세요.
"""


def summarize_papers_node(state: AgentState) -> dict:
    """
    관련 청크들이 속한 논문들의 요약을 생성합니다.
    
    청크 기반 검색에서 반환된 관련 청크들로부터
    논문 정보를 추출하고, 각 논문의 초록을 요약합니다.
    """
    
    relevant_chunks = state.get("relevant_chunks", [])
    papers = state.get("papers", [])
    
    logger.info("="*60)
    logger.info("[SUMMARIZE_PAPERS] 논문 요약 생성")
    logger.info("="*60)
    logger.info(f"관련 청크: {len(relevant_chunks)}개")
    logger.info(f"원본 논문: {len(papers)}개")
    
    if not relevant_chunks and not papers:
        logger.warning("요약할 논문이 없음")
        return {
            "summarized_content": "요약할 논문이 없습니다.",
            "react_steps": [ReActStep(
                step_type="observation",
                content="요약할 논문이 없습니다."
            )]
        }
    
    try:
        llm = get_llm()
        
        action_step = ReActStep(
            step_type="action",
            content=f"관련 논문들의 요약을 생성합니다."
        )
        
        # 관련 청크에서 논문 정보 추출
        seen_papers = {}
        for chunk in relevant_chunks[:10]:  # 최대 10개 청크 처리
            arxiv_id = chunk.get('arxiv_id')
            if arxiv_id and arxiv_id not in seen_papers:
                seen_papers[arxiv_id] = {
                    'title': chunk.get('title', ''),
                    'abstract': chunk.get('content', ''),  # 청크 내용을 요약 대상으로
                    'authors': chunk.get('authors', ''),
                }
        
        logger.info(f"추출된 논문: {len(seen_papers)}개")
        
        # 각 논문 요약 생성
        summary_parts = []
        
        for arxiv_id, paper_info in list(seen_papers.items())[:5]:  # 최대 5개 요약
            try:
                if paper_info['title']:
                    prompt = SUMMARIZE_PROMPT.format(
                        title=paper_info['title'],
                        abstract=paper_info['abstract'][:500]
                    )
                    
                    response = llm.invoke([
                        HumanMessage(content=prompt)
                    ])
                    
                    summary_parts.append(f"\n## {paper_info['title']}\n\n{response.content}")
                    logger.debug(f"✓ {paper_info['title'][:30]}... 요약 완료")
                    
            except Exception as e:
                logger.warning(f"논문 요약 실패: {str(e)}")
                summary_parts.append(f"\n## {paper_info['title']}\n\n요약 생성 실패")
        
        summarized_content = "\n".join(summary_parts) if summary_parts else "요약을 생성할 수 없습니다."
        
        observation_step = ReActStep(
            step_type="observation",
            content=f"{len(summary_parts)}개 논문의 요약이 완료되었습니다."
        )
        
        logger.info(f"✓ {len(summary_parts)}개 요약 생성 완료")
        
        return {
            "summarized_content": summarized_content,
            "react_steps": [action_step, observation_step]
        }
        
    except Exception as e:
        logger.error(f"요약 생성 중 오류: {str(e)}", exc_info=True)
        return {
            "summarized_content": f"요약 생성 중 오류: {str(e)}",
            "react_steps": [ReActStep(step_type="observation", content=f"요약 생성 오류: {str(e)}")]
        }


# ============================================
# 노드 10: 최종 응답 생성 (generate_response_node)
# ============================================

FINAL_RESPONSE_PROMPT = """
사용자의 질문에 대해 검색된 정보를 바탕으로 종합적인 답변을 생성해주세요.

## 사용자 질문
{question}

## 검색된 정보
{papers_info}

## 요약된 논문
{summarized_content}

## 답변 작성 지침
1. 질문에 대한 직접적인 답변으로 시작하세요.
2. 관련 연구 동향을 설명하세요.
3. 검색된 논문들의 주요 내용을 인용하면서 설명하세요.
4. 추가 학습이나 탐구를 위한 제안을 제시하세요.
5. 한국어로 자세하고 전문적인 답변을 작성하세요.

답변의 길이: 500-1500 글자
"""


def generate_response_node(state: AgentState) -> dict:
    """
    검색 결과와 요약을 바탕으로 최종 답변을 생성합니다.
    
    이것이 사용자에게 보여질 최종 결과입니다.
    """
    
    user_question = state.get("user_question", "")
    relevant_chunks = state.get("relevant_chunks", [])
    summarized_content = state.get("summarized_content", "")
    error_message = state.get("error_message")
    
    logger.info("="*60)
    logger.info("[GENERATE_RESPONSE] 최종 응답 생성")
    logger.info("="*60)
    
    if error_message:
        logger.warning(f"이전 단계에서 오류 발생: {error_message}")
        return {
            "final_response": f"""죄송하지만 검색 중 오류가 발생했습니다.

오류 내용: {error_message}

다음을 시도해주세요:
1. 다른 키워드로 검색해보세요
2. 검색할 논문 수를 줄여보세요
3. 나중에 다시 시도해보세요""",
            "is_complete": True
        }
    
    if not relevant_chunks:
        logger.warning("관련 청크가 없음")
        return {
            "final_response": """입력하신 질문과 관련된 논문을 찾지 못했습니다.

다음을 시도해주세요:
1. 더 일반적인 키워드를 사용하세요
2. 검색할 논문 수를 늘려보세요
3. 다른 표현으로 질문을 다시 작성하세요""",
            "is_complete": True
        }
    
    try:
        llm = get_llm()
        
        # 관련 청크 정보 정리
        papers_info = ""
        for i, chunk in enumerate(relevant_chunks[:5], 1):
            papers_info += f"\n\n{i}. {chunk['title']}\n"
            papers_info += f"유사도: {chunk['similarity_score']:.2%}\n"
            papers_info += f"내용: {chunk['content'][:150]}..."
        
        prompt = FINAL_RESPONSE_PROMPT.format(
            question=user_question,
            papers_info=papers_info,
            summarized_content=summarized_content[:500]
        )
        
        logger.info("LLM에 최종 응답 요청...")
        
        response = llm.invoke([
            SystemMessage(content="당신은 친절하고 전문적인 학술 연구 어시스턴트입니다."),
            HumanMessage(content=prompt)
        ])
        
        final_response = response.content
        
        logger.info("✓ 최종 응답 생성 완료")
        
    except Exception as e:
        logger.error(f"최종 응답 생성 중 오류: {str(e)}", exc_info=True)
        final_response = f"""검색된 {len(relevant_chunks)}개의 관련 청크를 찾았습니다.

{summarized_content}

오류로 인해 상세한 분석을 완성하지 못했습니다: {str(e)}"""
    
    decision_step = ReActStep(
        step_type="thought",
        content="모든 처리가 완료되었습니다."
    )
    
    return {
        "final_response": final_response,
        "is_complete": True,
        "react_steps": [decision_step]
    }
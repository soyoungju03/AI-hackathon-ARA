# app/graph/workflow.py
# -*- coding: utf-8 -*-
"""
완전히 수정된 LangGraph 워크플로우 (PDF 임베딩 파이프라인 + 의미 기반 검색)
================================================================================

이 워크플로우는 학술 논문 검색을 위한 복잡한 상호작용을 관리합니다.
핵심 특징은 다음과 같습니다:

1. 노드 기반 라우팅: 조건부 함수 대신 명시적인 라우팅 노드를 사용하여
   각 결정 지점을 명확하게 추적할 수 있습니다.

2. 두 단계 Human-in-the-Loop: 사용자가 키워드 확인과 논문 수 선택 단계에서
   개입하여 검색의 정확도를 높입니다.

3. PDF 임베딩 파이프라인: arXiv에서 논문 PDF를 다운로드하고, 텍스트를 
   추출한 후, 의미 있는 크기로 청킹하고, Sentence Transformers로 임베딩하여
   ChromaDB에 저장합니다.

4. 의미 기반 검색: ChromaDB에 저장된 청크들 중에서 사용자의 질문과 
   의미론적으로 가장 유사한 청크들을 검색합니다.

5. ReAct 패턴: 모든 노드가 Thought-Action-Observation 패턴을 따라
   AI의 사고 과정을 투명하게 보여줍니다.

파이프라인 흐름:
사용자 질문 → 키워드 추출 → arXiv 검색 → 
PDF 다운로드 → 텍스트 추출 → 청킹 → 임베딩 (배치 처리) → ChromaDB 저장 →
의미 기반 검색 (코사인 유사도) → 가장 관련성 높은 청크 선별 →
요약 및 답변 생성
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
    summarize_papers_node,
    generate_response_node
)

logger = logging.getLogger(__name__)


# ============================================
# PDF 임베딩 파이프라인 통합
# ============================================

def get_pdf_pipeline():
    """
    PDF 처리 파이프라인을 초기화하고 반환합니다.
    
    싱글톤 패턴을 사용하므로, 이 함수를 여러 번 호출해도
    하나의 파이프라인 인스턴스만 생성됩니다. 이는 메모리를
    효율적으로 사용하고, 임베딩 모델을 한 번만 로드하기 위함입니다.
    
    임베딩 모델은 처음 로드될 때 (약 100MB 크기의) 가중치를 다운로드합니다.
    이후 호출에서는 캐시된 모델을 사용하므로 빠릅니다.
    """
    
    try:
        from app.tools.pdf_embedding_pipeline_final import PDFEmbeddingPipeline
        from app.tools.embeddings import SentenceTransformerEmbedding
        from app.tools.vectorstore import ArxivPaperVectorStore
        
        # 전역 변수로 파이프라인 저장
        if not hasattr(get_pdf_pipeline, '_pipeline'):
            logger.info("[INIT] PDF 임베딩 파이프라인 초기화 중...")
            
            # Sentence Transformers 모델 초기화
            # distiluse-base-multilingual-cased-v2는 한국어를 포함한 여러 언어를 지원합니다
            embedding_model = SentenceTransformerEmbedding(
                model_name="distiluse-base-multilingual-cased-v2"
            )
            
            # ChromaDB 벡터 스토어 초기화
            vectorstore = ArxivPaperVectorStore(
                persist_directory="./data/arxiv_chunks",
                collection_name="arxiv_chunks"
            )
            
            # PDF 처리 파이프라인 생성
            get_pdf_pipeline._pipeline = PDFEmbeddingPipeline(
                embedding_model=embedding_model,
                vectorstore=vectorstore,
                chunk_chars=1800,  # 각 청크의 대략적인 문자 수 (약 450 토큰)
                overlap_chars=350,  # 청크 간 오버래프 (문맥 손실 방지)
                batch_size=32  # 배치로 임베딩할 청크 수
            )
            
            logger.info("✓ PDF 임베딩 파이프라인 초기화 완료")
        
        return get_pdf_pipeline._pipeline
    
    except ImportError as e:
        logger.error(f"PDF 파이프라인 모듈 임포트 실패: {str(e)}")
        raise


# ============================================
# 수정된 노드: search_papers_node
# ============================================

def search_papers_node(state: AgentState) -> dict:
    """
    arXiv에서 논문을 검색하고 PDF 처리 파이프라인을 실행합니다.
    
    이 노드의 역할:
    1. 추출된 키워드를 사용하여 arXiv API를 호출해 논문 검색
    2. 검색된 논문들의 PDF를 다운로드
    3. PDF에서 텍스트 추출
    4. 텍스트를 의미 있는 청크로 분할
    5. 각 청크를 임베딩 (배치 처리로 효율성 증대)
    6. ChromaDB에 청크와 임베딩 저장
    
    이렇게 저장된 청크들은 evaluate_relevance_node에서 사용되어
    사용자의 질문과 가장 유사한 부분을 찾을 수 있습니다.
    """
    
    keywords = state.get("extracted_keywords", [])
    paper_count = state.get("paper_count", 3)
    domain = state.get("question_domain", None)
    
    # ReAct 패턴: 먼저 어떤 행동을 할 것인지 설명합니다
    action_content = f"""논문 검색 및 PDF 처리 파이프라인 시작:
- 키워드: {', '.join(keywords)}
- 검색 개수: {paper_count}개
- 도메인: {domain or '전체'}
- 처리 단계:
  1) arXiv API에서 논문 검색
  2) 각 논문의 PDF 다운로드
  3) PDF에서 텍스트 추출
  4) 텍스트를 약 450토큰 크기의 청크로 분할
  5) 각 청크를 Sentence Transformers로 임베딩 (384차원)
  6) 청크와 임베딩을 ChromaDB에 저장"""
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    try:
        # arXiv 검색 함수는 app/graph/nodes.py에 정의되어야 합니다
        # 현재는 이 함수를 가정하고 있습니다
        from app.tools.paper_search.arxiv_tool import search_arxiv
        
        logger.info(f"[SEARCH_PAPERS] arXiv 검색 시작")
        logger.info(f"  키워드: {', '.join(keywords)}")
        logger.info(f"  검색 개수: {paper_count}")
        
        # Step 1: arXiv에서 논문 검색
        papers = search_arxiv(
            keywords=keywords,
            max_results=paper_count,
            domain=domain
        )
        
        if not papers:
            logger.warning("[SEARCH_PAPERS] 검색 결과 없음")
            
            observation_content = f"""arXiv 검색 결과:
키워드 '{', '.join(keywords)}'로 검색했지만 관련 논문을 찾지 못했습니다.
다른 키워드로 다시 시도해보세요."""
            
            observation_step = ReActStep(
                step_type="observation",
                content=observation_content
            )
            
            return {
                "papers": [],
                "chunks_saved": 0,
                "react_steps": [action_step, observation_step],
                "error_message": "검색 결과 없음",
                "is_complete": False
            }
        
        logger.info(f"[SEARCH_PAPERS] ✓ {len(papers)}개 논문 검색 완료")
        
        # Step 2: PDF 처리 파이프라인 준비
        logger.info("[SEARCH_PAPERS] PDF 처리 파이프라인 초기화")
        
        pipeline = get_pdf_pipeline()
        
        # Step 3: 논문을 파이프라인이 이해할 수 있는 형식으로 변환
        papers_for_pipeline = []
        
        for paper in papers:
            # Paper 객체 또는 딕셔너리를 처리합니다
            if isinstance(paper, dict):
                arxiv_id = paper.get('arxiv_id', 'unknown')
                paper_dict = paper
            else:
                # Paper 객체인 경우
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
        
        logger.info(f"[SEARCH_PAPERS] {len(papers_for_pipeline)}개 논문을 파이프라인에 전달")
        
        # Step 4: 배치 처리로 PDF 다운로드, 텍스트 추출, 청킹, 임베딩, 저장
        logger.info("[SEARCH_PAPERS] PDF 처리 파이프라인 실행 시작...")
        
        batch_result = pipeline.process_papers_batch(
            papers=papers_for_pipeline,
            max_pages=10  # 각 논문에서 최대 10페이지만 처리 (시간 및 메모리 절약)
        )
        
        logger.info("[SEARCH_PAPERS] ✓ PDF 처리 파이프라인 완료")
        
        # Step 5: 처리 결과 요약
        total_chunks_created = sum(r.get('chunks_created', 0) for r in batch_result['results'])
        total_chunks_embedded = sum(r.get('chunks_embedded', 0) for r in batch_result['results'])
        total_chunks_saved = batch_result['total_chunks']
        
        observation_content = f"""검색 및 PDF 처리 완료:
- 검색된 논문: {len(papers)}개
- 성공: {batch_result['successful']}개
- 실패: {batch_result['failed']}개

청크 처리 통계:
- 생성된 청크: {total_chunks_created}개
- 임베딩된 청크: {total_chunks_embedded}개
- 저장된 청크: {total_chunks_saved}개

처리 시간: {batch_result['time']:.1f}초

이제 사용자의 질문과 의미론적으로 가장 유사한 청크들을
ChromaDB에서 검색할 수 있습니다."""
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        logger.info(f"[SEARCH_PAPERS] 최종 결과:")
        logger.info(f"  검색된 논문: {len(papers)}")
        logger.info(f"  생성된 청크: {total_chunks_created}")
        logger.info(f"  저장된 청크: {total_chunks_saved}")
        
        return {
            "papers": papers,
            "chunks_saved": total_chunks_saved,
            "pdf_processing_result": batch_result,
            "react_steps": [action_step, observation_step],
            "error_message": None,
            "is_complete": False
        }
    
    except Exception as e:
        logger.error(f"[SEARCH_PAPERS] 처리 중 오류: {str(e)}", exc_info=True)
        
        error_observation = f"""처리 중 오류 발생:
{str(e)}

다음 단계를 시도해보세요:
1. 키워드를 다시 확인하세요
2. 인터넷 연결을 확인하세요
3. 다시 시도해보세요"""
        
        observation_step = ReActStep(
            step_type="observation",
            content=error_observation
        )
        
        return {
            "papers": [],
            "chunks_saved": 0,
            "react_steps": [action_step, observation_step],
            "error_message": str(e),
            "is_complete": False
        }


# ============================================
# 수정된 노드: evaluate_relevance_node
# ============================================

def evaluate_relevance_node(state: AgentState) -> dict:
    """
    ChromaDB에 저장된 청크들 중에서 사용자의 질문과
    의미론적으로 가장 유사한 청크들을 검색합니다.
    
    이 노드의 역할:
    1. 사용자의 원래 질문을 Sentence Transformers로 임베딩
    2. 임베딩된 질문 벡터를 사용하여 ChromaDB에서 코사인 유사도 기반 검색
    3. 유사도 점수가 높은 상위 청크들을 선별
    4. 청크의 메타데이터(논문 ID, 제목, 섹션 등)를 포함하여 반환
    
    코사인 유사도는 두 벡터가 같은 방향을 얼마나 가리키는지를 나타내는
    척도입니다. 1에 가까울수록 더 유사한 의미를 가지고 있다는 뜻입니다.
    """
    
    original_question = state.get("user_question", "")
    paper_count = state.get("paper_count", 3)
    chunks_saved = state.get("chunks_saved", 0)
    
    action_content = f"""의미 기반 청크 검색 시작:
- 질문: {original_question[:60]}...
- 원하는 청크 개수: {paper_count * 3}개 (각 논문당 약 3개)
- 검색 방식: 코사인 유사도 (Sentence Transformers)
- 벡터 차원: 384차원
- 저장된 총 청크: {chunks_saved}개"""
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    if chunks_saved == 0:
        logger.warning("[EVALUATE_RELEVANCE] 저장된 청크가 없음")
        
        observation_content = f"""의미 기반 검색 실패:
저장된 청크가 없어서 검색할 수 없습니다.
이전 단계에서 오류가 발생했을 가능성이 있습니다."""
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "relevant_papers": [],
            "relevant_chunks": [],
            "evaluation_result": {
                "success": False,
                "message": "저장된 청크가 없습니다"
            },
            "react_steps": [action_step, observation_step],
            "is_complete": False
        }
    
    try:
        logger.info("[EVALUATE_RELEVANCE] 의미 기반 검색 시작")
        logger.info(f"  질문: {original_question[:50]}...")
        logger.info(f"  검색할 청크 수: {paper_count * 3}개")
        
        # PDF 파이프라인에서 파이프라인의 벡터스토어에 접근합니다
        pipeline = get_pdf_pipeline()
        vectorstore = pipeline.vectorstore
        embedding_model = pipeline.embedding_model
        
        # Step 1: 사용자의 질문을 임베딩합니다
        logger.info("[EVALUATE_RELEVANCE] 질문 임베딩 생성 중...")
        
        query_embedding = embedding_model.embed(original_question)
        
        logger.info(f"[EVALUATE_RELEVANCE] ✓ 질문 임베딩 완료 (차원: {len(query_embedding)})")
        
        # Step 2: ChromaDB에서 코사인 유사도 기반 검색
        # ChromaDB는 자동으로 거리를 계산하고 결과를 정렬합니다
        logger.info(f"[EVALUATE_RELEVANCE] ChromaDB에서 검색 중...")
        
        search_results = vectorstore.collection.query(
            query_embeddings=[query_embedding],
            n_results=paper_count * 3,  # 더 많은 결과를 먼저 가져온 후 필터링
            include=["documents", "metadatas", "distances"]
        )
        
        logger.info(f"[EVALUATE_RELEVANCE] ✓ 검색 완료: {len(search_results['ids'][0]) if search_results['ids'] else 0}개 결과")
        
        # Step 3: 검색 결과를 처리하고 유사도 점수를 계산합니다
        relevant_chunks = []
        
        if search_results['ids'] and len(search_results['ids']) > 0:
            for i, chunk_id in enumerate(search_results['ids'][0]):
                # ChromaDB는 거리 메트릭을 사용하므로, 1에서 거리를 빼서 유사도로 변환합니다
                # (거리가 작을수록 유사함을 의미하므로, 유사도 = 1 - 거리)
                distance = search_results['distances'][0][i]
                similarity_score = 1 - distance
                
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
                
                logger.debug(f"  청크 {i+1}: {chunk_id[:20]}... (유사도: {similarity_score:.4f})")
        
        # Step 4: 결과를 요약합니다
        logger.info(f"[EVALUATE_RELEVANCE] ✓ 의미 기반 검색 완료: {len(relevant_chunks)}개 청크 발견")
        
        # 상위 청크들의 정보를 관찰 단계에 포함합니다
        observation_parts = [f"의미 기반 검색 완료: {len(relevant_chunks)}개 관련 청크 발견"]
        observation_parts.append("\n가장 관련성 높은 상위 3개 청크:")
        
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            observation_parts.append(f"\n{i}. 유사도: {chunk['similarity_score']:.4f}")
            observation_parts.append(f"   논문: {chunk['title'][:40]}...")
            if chunk.get('section'):
                observation_parts.append(f"   섹션: {chunk['section']}")
            observation_parts.append(f"   내용: {chunk['content'][:60]}...")
        
        observation_content = "\n".join(observation_parts)
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        return {
            "relevant_papers": [],  # 청크 기반 검색이므로 논문 리스트는 비웁니다
            "relevant_chunks": relevant_chunks[:paper_count * 2],  # 상위 청크들만 반환
            "evaluation_result": {
                "success": True,
                "message": f"{len(relevant_chunks)}개의 관련 청크를 찾았습니다",
                "details": {
                    "evaluation_method": "semantic_similarity_chunks",
                    "total_chunks_searched": chunks_saved,
                    "chunks_found": len(relevant_chunks)
                }
            },
            "react_steps": [action_step, observation_step],
            "is_complete": False
        }
    
    except Exception as e:
        logger.error(f"[EVALUATE_RELEVANCE] 검색 중 오류: {str(e)}", exc_info=True)
        
        error_observation = f"""의미 기반 검색 중 오류 발생:
{str(e)}

원인 분석:
- ChromaDB에 저장된 청크가 손상되었을 수 있습니다
- 임베딩 모델이 로드되지 않았을 수 있습니다
- 질문 텍스트가 유효하지 않았을 수 있습니다"""
        
        observation_step = ReActStep(
            step_type="observation",
            content=error_observation
        )
        
        return {
            "relevant_papers": [],
            "relevant_chunks": [],
            "evaluation_result": {
                "success": False,
                "message": f"검색 중 오류: {str(e)}",
                "details": {}
            },
            "react_steps": [action_step, observation_step],
            "is_complete": False
        }


# ============================================
# 라우팅 노드들
# ============================================

def check_keyword_confirmation_status_node(state: AgentState) -> dict:
    """
    키워드 확인 후 상태를 검사하고 다음 단계를 결정합니다.
    
    사용자가 "확인"을 선택하면 논문 수 선택 단계로 진행하고,
    "다시"를 선택하면 질문 분석 단계로 돌아갑니다.
    """
    
    keyword_response = state.get("keyword_confirmation_response")
    
    logger.info("=" * 60)
    logger.info("[CHECK_KEYWORD_CONFIRMATION_STATUS] 상태 검사 시작")
    logger.info(f"  응답: {keyword_response}")
    logger.info("=" * 60)
    
    if keyword_response == "retry":
        logger.info("  → '다시' 선택: analyze_question으로 이동")
        return {"next_node": "analyze_question"}
    else:
        logger.info("  → '확인' 선택: request_paper_count로 이동")
        return {"next_node": "request_paper_count"}


def check_paper_count_status_node(state: AgentState) -> dict:
    """
    논문 수 선택 후 상태를 검사하고 다음 단계를 결정합니다.
    
    논문 수가 유효한 범위(1-10)에 있는지 확인하고,
    검색 단계로 진행할 준비를 합니다.
    """
    
    paper_count = state.get("paper_count")
    
    logger.info("=" * 60)
    logger.info("[CHECK_PAPER_COUNT_STATUS] 상태 검사 시작")
    logger.info(f"  논문 수: {paper_count}")
    logger.info("=" * 60)
    
    if paper_count is None or paper_count < 1 or paper_count > 10:
        logger.warning(f"  ⚠️ 유효하지 않은 논문 수: {paper_count}")
        logger.info("  → 기본값 3으로 설정")
        return {"paper_count": 3, "next_node": "search_papers"}
    
    logger.info(f"  ✓ 유효한 논문 수: {paper_count}")
    logger.info("  → search_papers로 이동")
    
    return {"next_node": "search_papers"}


# ============================================
# 워크플로우 빌드
# ============================================

def build_research_workflow() -> StateGraph:
    """
    PDF 임베딩 파이프라인을 포함하는 연구 어시스턴트 워크플로우를 구축합니다.
    
    워크플로우 구조:
    
    START
      ↓
    receive_question (질문 수신)
      ↓
    analyze_question (질문 분석 및 키워드 추출)
      ↓
    request_keyword_confirmation ← [INTERRUPT 1]
      ↓
    process_keyword_confirmation_response
      ↓
    check_keyword_confirmation_status
      ↓
    [분기]
    ├─ analyze_question (다시)
    └─ request_paper_count ← [INTERRUPT 2]
      ↓
    process_paper_count_response
      ↓
    check_paper_count_status
      ↓
    search_papers (→ PDF 임베딩 파이프라인 실행)
      ↓
    [조건부 분기]
    ├─ generate_response (오류 발생 시)
    └─ evaluate_relevance (→ ChromaDB 의미 검색)
      ↓
    summarize_papers
      ↓
    generate_response
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
    
    # PDF 임베딩 파이프라인이 포함된 노드들
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)
    
    workflow.add_node("summarize_papers", summarize_papers_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # 라우팅 노드
    workflow.add_node("check_keyword_confirmation_status", check_keyword_confirmation_status_node)
    workflow.add_node("check_paper_count_status", check_paper_count_status_node)
    
    # 엣지 정의
    workflow.set_entry_point("receive_question")
    
    # 초기 처리 흐름
    workflow.add_edge("receive_question", "analyze_question")
    workflow.add_edge("analyze_question", "request_keyword_confirmation")
    workflow.add_edge("request_keyword_confirmation", "process_keyword_confirmation_response")
    workflow.add_edge("process_keyword_confirmation_response", "check_keyword_confirmation_status")
    
    # 첫 번째 조건부 분기
    def route_after_keyword_check(state: AgentState) -> Literal["analyze_question", "request_paper_count"]:
        next_node = state.get("next_node", "request_paper_count")
        return next_node
    
    workflow.add_conditional_edges(
        "check_keyword_confirmation_status",
        route_after_keyword_check,
        {
            "analyze_question": "analyze_question",
            "request_paper_count": "request_paper_count"
        }
    )
    
    # 논문 수 선택 흐름
    workflow.add_edge("request_paper_count", "process_paper_count_response")
    workflow.add_edge("process_paper_count_response", "check_paper_count_status")
    workflow.add_edge("check_paper_count_status", "search_papers")
    
    # 두 번째 조건부 분기: 검색 결과 확인
    def check_search_results(state: AgentState) -> Literal["evaluate_relevance", "generate_response"]:
        """검색이 성공했는지 확인합니다."""
        if state.get("error_message"):
            logger.info("[CHECK_SEARCH_RESULTS] 검색 실패 → 답변 생성으로 이동")
            return "generate_response"
        logger.info("[CHECK_SEARCH_RESULTS] 검색 성공 → 의미 기반 검색으로 이동")
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
    PDF 임베딩 파이프라인이 통합된 연구 어시스턴트 에이전트를 생성합니다.
    
    checkpointer는 워크플로우의 상태를 저장하고 복원하여
    Human-in-the-Loop에서 멈췄다가 나중에 계속 진행할 수 있게 합니다.
    """
    
    workflow = build_research_workflow()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "process_keyword_confirmation_response",
            "process_paper_count_response"
        ]
    )
    
    logger.info("✓ 워크플로우 컴파일 완료 (PDF 임베딩 파이프라인 + 의미 검색)")
    return compiled


# ============================================
# ResearchAssistant 클래스
# ============================================

class ResearchAssistant:
    """
    PDF 임베딩 파이프라인이 통합된 연구 어시스턴트입니다.
    
    이 클래스는 사용하기 쉬운 인터페이스를 제공합니다.
    
    전체 처리 흐름:
    1. 사용자 질문 수신
    2. 키워드 추출 및 사용자 확인
    3. 논문 수 선택
    4. arXiv 검색
    5. PDF 다운로드 및 처리 (새로 추가됨)
    6. 의미 기반 청크 검색
    7. 요약 및 답변 생성
    """
    
    def __init__(self):
        """어시스턴트를 초기화합니다."""
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
        
        Args:
            question: 사용자의 연구 질문
            paper_count: 검색할 논문 개수 (1-10)
            session_id: 세션 식별자
            
        Returns:
            최종 답변 문자열
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
        
        Args:
            question: 사용자의 연구 질문
            session_id: 세션 식별자
            
        Returns:
            상태 정보 딕셔너리
        """
        
        import uuid
        
        logger.info(f"[START MODE] 시작: {question[:50]}...")
        
        initial_state = create_initial_state(question, session_id)
        
        self.current_thread_id = str(uuid.uuid4())
        self.interrupt_count = 0
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        try:
            logger.info("[START MODE] 워크플로우 실행 중...")
            
            for event in self.agent.stream(initial_state, config):
                pass
            
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            self.interrupt_count = 1
            logger.info("[START MODE] ✓ 첫 번째 Interrupt 도달")
            
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
        
        Args:
            user_response: 사용자의 응답
            
        Returns:
            상태 정보 딕셔너리
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
            if self.interrupt_count == 1:
                # 첫 번째 Interrupt: 키워드 확인
                logger.info("[CONTINUE MODE] Stage 1: 키워드 확인 응답")
                
                normalized_response = user_response.strip().lower()
                keyword_response = "retry" if normalized_response in ["다시", "retry", "수정"] else "confirmed"
                
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "keyword_confirmation_response": keyword_response,
                        "waiting_for_user": False
                    }
                )
            
            elif self.interrupt_count == 2:
                # 두 번째 Interrupt: 논문 수 선택
                logger.info("[CONTINUE MODE] Stage 2: 논문 수 선택")
                
                self.agent.update_state(
                    config,
                    {
                        "user_response": user_response,
                        "waiting_for_user": False
                    }
                )
            
            logger.info("[CONTINUE MODE] 워크플로우 계속 실행 중...")
            
            for event in self.agent.stream(None, config):
                pass
            
            state_snapshot = self.agent.get_state(config)
            current_values = state_snapshot.values
            
            logger.info("[CONTINUE MODE] 실행 완료, 상태 확인")
            
            if current_values.get("is_complete"):
                logger.info("[CONTINUE MODE] ✓ 워크플로우 완료")
                return {
                    "status": "completed",
                    "interrupt_stage": self.interrupt_count,
                    "response": current_values.get("final_response", ""),
                    "chunks": current_values.get("relevant_chunks", []),
                    "thread_id": self.current_thread_id
                }
            
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
    """전역 어시스턴트 인스턴스를 반환합니다."""
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = ResearchAssistant()
    return _default_assistant
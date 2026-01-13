# -*- coding: utf-8 -*-
"""
Weaviate 통합 search_papers 노드
================================

이 모듈은 기존의 search_papers 노드를 Weaviate 벡터 검색과 통합합니다.

흐름:
1. 먼저 Weaviate에서 벡터 검색 시도
2. 충분한 결과가 없으면 arXiv API에서 추가 검색
3. 새로운 논문들을 Weaviate에 저장 (캐싱)
4. 모든 결과를 상태에 저장

이렇게 하면 반복된 검색은 매우 빠르고, 새로운 논문은 arXiv에서 가져옵니다.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def search_papers_with_weaviate(
    keywords: List[str],
    paper_count: int,
    weaviate_client,
    arxiv_client = None
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Weaviate와 arXiv를 조합하여 논문을 검색합니다.
    
    Args:
        keywords: 검색 키워드 리스트
        paper_count: 원하는 논문 개수
        weaviate_client: Weaviate 클라이언트
        arxiv_client: arXiv 클라이언트 (선택사항)
    
    Returns:
        (논문 리스트, 에러 메시지)
        성공 시 에러 메시지는 None
    """
    
    logger.info("=" * 70)
    logger.info(f"🔍 논문 검색 시작: {', '.join(keywords)}")
    logger.info(f"   목표 개수: {paper_count}개")
    logger.info("=" * 70)
    
    papers = []
    error_message = None
    
    try:
        # 단계 1: Weaviate에서 벡터 검색
        logger.info("📚 [단계 1] Weaviate 벡터 검색 시작...")
        
        if not weaviate_client.health_check():
            logger.warning("⚠️  Weaviate 서버 응답 없음, arXiv로 진행")
        else:
            query_string = " ".join(keywords)
            
            weaviate_papers = weaviate_client.search_papers(
                query=query_string,
                limit=paper_count,
                keywords=keywords
            )
            
            logger.info(f"✓ Weaviate에서 {len(weaviate_papers)}개 논문 검색됨")
            
            # Weaviate 결과를 표준 형식으로 변환
            for paper in weaviate_papers:
                papers.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', []),
                    'abstract': paper.get('abstract', ''),
                    'summary': paper.get('summary', ''),
                    'arxiv_id': paper.get('arxiv_id', ''),
                    'url': paper.get('url', ''),
                    'published_date': paper.get('published_date', ''),
                    'source': 'weaviate',
                    'relevance_score': paper.get('relevance_score', 0.0)
                })
        
        # 단계 2: 부족하면 arXiv에서 추가 검색
        if len(papers) < paper_count and arxiv_client:
            logger.info(f"📡 [단계 2] arXiv 추가 검색 (필요: {paper_count - len(papers)}개)")
            
            try:
                arxiv_papers = search_arxiv_papers(
                    keywords=keywords,
                    max_results=paper_count - len(papers),
                    arxiv_client=arxiv_client
                )
                
                logger.info(f"✓ arXiv에서 {len(arxiv_papers)}개 논문 검색됨")
                
                # 중복 제거 (Weaviate에 이미 있는 논문)
                existing_arxiv_ids = {p['arxiv_id'] for p in papers}
                
                new_papers = [
                    p for p in arxiv_papers 
                    if p['arxiv_id'] not in existing_arxiv_ids
                ]
                
                logger.info(f"✓ 중복 제거 후 {len(new_papers)}개 새 논문")
                
                papers.extend(new_papers)
                
                # 단계 3: 새로운 논문들을 Weaviate에 저장 (캐싱)
                logger.info("💾 [단계 3] 새 논문들을 Weaviate에 저장 중...")
                
                from app.services.weaviate_client import Paper
                
                papers_to_save = []
                for paper in new_papers:
                    try:
                        paper_obj = Paper(
                            title=paper['title'],
                            authors=paper['authors'],
                            abstract=paper['abstract'],
                            arxiv_id=paper['arxiv_id'],
                            url=paper['url'],
                            published_date=paper['published_date'],
                            summary=paper.get('summary'),
                            keywords=keywords
                        )
                        papers_to_save.append(paper_obj)
                    except Exception as e:
                        logger.warning(f"⚠️  논문 변환 실패: {str(e)}")
                
                if papers_to_save:
                    saved_count = weaviate_client.add_papers_batch(papers_to_save)
                    logger.info(f"✓ {saved_count}개 논문 Weaviate에 저장됨")
            
            except Exception as e:
                logger.error(f"⚠️  arXiv 검색 중 오류: {str(e)}")
                # arXiv 오류는 치명적이지 않음 (Weaviate 결과는 이미 있음)
        
        # 최종 결과 확인
        logger.info("=" * 70)
        logger.info(f"✓ 검색 완료: {len(papers)}개 논문")
        logger.info("=" * 70)
        
        if len(papers) == 0:
            error_message = f"'{', '.join(keywords)}'에 대한 논문을 찾을 수 없습니다."
            logger.warning(f"⚠️  {error_message}")
        
        return papers, error_message
    
    except Exception as e:
        error_message = f"논문 검색 중 오류가 발생했습니다: {str(e)}"
        logger.error(f"✗ {error_message}")
        return [], error_message


def search_arxiv_papers(
    keywords: List[str],
    max_results: int = 5,
    arxiv_client = None
) -> List[Dict[str, Any]]:
    """
    arXiv API를 사용하여 논문을 검색합니다.
    
    Args:
        keywords: 검색 키워드 리스트
        max_results: 최대 결과 개수
        arxiv_client: arXiv 클라이언트
    
    Returns:
        논문 정보 딕셔너리 리스트
    """
    
    try:
        import arxiv
        
        client = arxiv_client or arxiv.Client()
        
        # 검색 쿼리 구성
        query_string = " OR ".join([f"title:{kw}" for kw in keywords])
        
        logger.debug(f"arXiv 쿼리: {query_string}")
        
        # arXiv 검색
        search = arxiv.Search(
            query=query_string,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        
        for entry in client.results(search):
            paper = {
                'title': entry.title,
                'authors': [author.name for author in entry.authors],
                'abstract': entry.summary,
                'summary': None,  # arXiv에서는 없음
                'arxiv_id': entry.entry_id.split('/abs/')[-1],
                'url': entry.pdf_url,
                'published_date': entry.published.isoformat(),
                'source': 'arxiv',
                'relevance_score': 1.0  # arXiv는 모두 같은 점수
            }
            
            papers.append(paper)
            
            if len(papers) >= max_results:
                break
        
        logger.debug(f"arXiv에서 {len(papers)}개 논문 검색됨")
        
        return papers
    
    except ImportError:
        logger.error("arxiv 패키지가 설치되지 않았습니다: pip install arxiv")
        return []
    except Exception as e:
        logger.error(f"arXiv 검색 오류: {str(e)}")
        return []


# LangGraph 노드 함수
def search_papers_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph의 search_papers 노드
    
    Weaviate와 arXiv를 조합하여 논문을 검색합니다.
    """
    
    from app.services.weaviate_client import get_weaviate_client
    
    logger.info("[search_papers_node] 실행 시작")
    
    try:
        # 상태에서 필요한 정보 추출
        keywords = state.get('extracted_keywords', [])
        paper_count = state.get('paper_count', 5)
        
        if not keywords:
            logger.warning("⚠️  추출된 키워드가 없습니다")
            return {
                'papers': [],
                'error_message': '키워드를 추출할 수 없습니다.',
                'react_steps': [
                    {
                        'step_type': 'observation',
                        'content': '키워드 추출 실패'
                    }
                ]
            }
        
        # Weaviate 클라이언트 초기화
        weaviate_client = get_weaviate_client(use_embedded=True)
        
        # 검색 실행
        papers, error_message = search_papers_with_weaviate(
            keywords=keywords,
            paper_count=paper_count,
            weaviate_client=weaviate_client
        )
        
        # ReAct 스텝 기록
        observation = f"'{', '.join(keywords)}'로 검색하여 {len(papers)}개 논문 발견"
        
        react_step = {
            'step_type': 'observation',
            'content': observation
        }
        
        return {
            'papers': papers,
            'error_message': error_message,
            'react_steps': [react_step]
        }
    
    except Exception as e:
        logger.error(f"[search_papers_node] 오류: {str(e)}", exc_info=True)
        
        return {
            'papers': [],
            'error_message': f"논문 검색 중 오류: {str(e)}",
            'react_steps': [
                {
                    'step_type': 'observation',
                    'content': f'오류 발생: {str(e)}'
                }
            ]
        }
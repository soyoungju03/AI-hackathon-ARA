"""
통합 검색 모듈: arXiv + RISS 멀티소스 검색
============================================

당신의 workflow의 search_papers_node를 수정하여
arXiv와 RISS에서 동시에 논문을 검색합니다.

핵심 아이디어:
1. 두 소스에 동시에 검색 요청을 보냅니다
2. 결과를 동일한 Paper 형식으로 통일합니다
3. 출처(source)로 어느 데이터베이스에서 왔는지 표시합니다
4. 결과를 결합하여 반환합니다

이렇게 하면 사용자 입장에서는 통합된 하나의 검색 결과를 받게 됩니다.
"""

import asyncio
import logging
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from app.graph.state import Paper, AgentState, ReActStep
from app.config import get_settings

# arXiv 검색 import
try:
    from app.tools.paper_search.arxiv_tool import search_arxiv
except ImportError:
    search_arxiv = None

# RISS 검색 import (아직 없으면 None)
try:
    from app.tools.paper_search.riss_client import RissClient
    riss_available = True
except ImportError:
    riss_available = False
    RissClient = None

logger = logging.getLogger(__name__)
settings = get_settings()


class MultiSourcePaperSearcher:
    """
    arXiv와 RISS를 통합하여 검색하는 클래스입니다.
    
    이 클래스는 두 소스에 병렬로 검색 요청을 보내고,
    결과를 동일한 Paper 형식으로 통일합니다.
    """
    
    def __init__(self):
        """멀티소스 검색기를 초기화합니다."""
        self.riss_client = None
        
        # RISS 클라이언트 초기화 (선택사항)
        if riss_available:
            try:
                self.riss_client = RissClient(max_results_per_query=100, delay=2)
                logger.info("✓ RISS 클라이언트 초기화 완료")
            except Exception as e:
                logger.warning(f"RISS 클라이언트 초기화 실패: {str(e)}")
                self.riss_client = None
        else:
            logger.warning("RISS 클라이언트를 사용할 수 없습니다 (beautifulsoup4 미설치)")
    
    def search(
        self,
        keywords: List[str],
        max_results: int = 5,
        domain: Optional[str] = None,
        sources: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        멀티소스 검색을 수행합니다.
        
        Args:
            keywords: 검색 키워드 리스트
            max_results: 소스당 최대 결과 수
            domain: 검색 도메인 (arXiv용)
            sources: 검색할 소스 리스트
                    ["arxiv", "riss"]
                    기본값: ["arxiv"] (빠른 검색)
                    원하면 ["arxiv", "riss"]로 설정
        
        Returns:
            통합된 Paper 객체 리스트
        """
        
        if sources is None:
            # 기본값: arXiv만 검색 (빠름)
            # RISS는 느릴 수 있으므로, 필요시에만 추가
            sources = ["arxiv"]
        
        logger.info(f"멀티소스 검색 시작")
        logger.info(f"  - 키워드: {keywords}")
        logger.info(f"  - 소스: {sources}")
        logger.info(f"  - 결과 수: {max_results}개/소스")
        
        all_papers = []
        
        # 스레드 풀을 사용하여 병렬 검색 수행
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = {}
            
            # arXiv 검색
            if "arxiv" in sources and search_arxiv:
                logger.info("arXiv 검색 중...")
                future = executor.submit(
                    self._search_arxiv,
                    keywords=keywords,
                    max_results=max_results,
                    domain=domain
                )
                futures[future] = "arxiv"
            
            # RISS 검색
            if "riss" in sources and self.riss_client:
                logger.info("RISS 검색 중...")
                future = executor.submit(
                    self._search_riss,
                    keywords=keywords,
                    max_results=max_results
                )
                futures[future] = "riss"
            
            # 결과 수집
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    papers = future.result()
                    logger.info(f"✓ {source_name.upper()} 검색 완료: {len(papers)}개")
                    all_papers.extend(papers)
                except Exception as e:
                    logger.error(f"❌ {source_name.upper()} 검색 실패: {str(e)}")
        
        logger.info(f"✓ 멀티소스 검색 완료: 총 {len(all_papers)}개 논문")
        
        return all_papers
    
    def _search_arxiv(
        self,
        keywords: List[str],
        max_results: int,
        domain: Optional[str] = None
    ) -> List[Paper]:
        """
        arXiv에서 검색하여 Paper 객체 리스트로 반환합니다.
        
        당신의 기존 search_arxiv 함수와 호환됩니다.
        """
        
        try:
            # 당신의 기존 search_arxiv 함수 호출
            arxiv_papers = search_arxiv(
                keywords=keywords,
                max_results=max_results,
                domain=domain
            )
            
            # 이미 Paper 객체일 가능성이 높지만, 혹시 모르니 확인
            if arxiv_papers and isinstance(arxiv_papers[0], dict):
                # 딕셔너리를 Paper 객체로 변환
                papers = [
                    Paper(
                        title=p.get('title', ''),
                        authors=p.get('authors', []),
                        abstract=p.get('abstract', ''),
                        url=p.get('url', ''),
                        published_date=p.get('published_date', ''),
                        source='arXiv',
                        relevance_score=p.get('relevance_score', 0.0)
                    )
                    for p in arxiv_papers
                ]
            else:
                # 이미 Paper 객체
                papers = arxiv_papers
                # source 필드를 명시적으로 설정
                for paper in papers:
                    paper.source = 'arXiv'
            
            return papers
        
        except Exception as e:
            logger.error(f"arXiv 검색 실패: {str(e)}")
            return []
    
    def _search_riss(
        self,
        keywords: List[str],
        max_results: int
    ) -> List[Paper]:
        """
        RISS에서 검색하여 Paper 객체 리스트로 반환합니다.
        
        RISS 결과를 Paper 형식으로 통일합니다.
        """
        
        if not self.riss_client:
            logger.warning("RISS 클라이언트를 사용할 수 없습니다")
            return []
        
        try:
            # RISS에서 한국 논문 검색 (우선적으로 한국어 논문 검색)
            riss_papers, total = self.riss_client.search_by_keyword(
                keywords=keywords,
                max_results=max_results,
                search_type="all"  # 국문과 영문 모두
            )
            
            # RISS 결과를 Paper 형식으로 변환
            papers = []
            for riss_paper in riss_papers:
                paper = Paper(
                    title=riss_paper.get('title', ''),
                    authors=riss_paper.get('authors', []),
                    abstract=riss_paper.get('abstract', ''),
                    url=riss_paper.get('url', ''),
                    published_date=riss_paper.get('published_date', ''),
                    source='RISS',  # 출처 표시
                    relevance_score=0.0  # RISS에서는 점수를 직접 제공하지 않음
                )
                papers.append(paper)
            
            return papers
        
        except Exception as e:
            logger.error(f"RISS 검색 실패: {str(e)}")
            return []


# 전역 검색기 인스턴스 (싱글톤)
_searcher = None

def get_multi_source_searcher() -> MultiSourcePaperSearcher:
    """멀티소스 검색기 인스턴스를 가져옵니다."""
    global _searcher
    if _searcher is None:
        _searcher = MultiSourcePaperSearcher()
    return _searcher


# ============================================
# 수정된 search_papers_node
# ============================================

def search_papers_node(state: AgentState) -> dict:
    """
    수정된 search_papers_node: arXiv + RISS 통합 검색
    
    이 노드는 당신의 기존 search_papers_node를 대체합니다.
    arXiv와 RISS에서 동시에 논문을 검색합니다.
    
    Args:
        state: AgentState 객체
    
    Returns:
        상태 업데이트 딕셔너리
    """
    
    keywords = state.get("extracted_keywords", [])
    paper_count = state.get("paper_count", 3)
    domain = state.get("question_domain", None)
    
    # 기본값: arXiv만 검색 (빠름)
    # 필요시 ["arxiv", "riss"]로 변경 가능
    sources_to_search = ["arxiv"]
    
    # 동작 설명
    action_content = f"""논문 검색을 실행합니다:
- 키워드: {', '.join(keywords)}
- 검색 개수: {paper_count}개
- 도메인: {domain or '전체'}
- 검색 소스: {', '.join(sources_to_search)}"""
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    try:
        # 멀티소스 검색 수행
        searcher = get_multi_source_searcher()
        papers = searcher.search(
            keywords=keywords,
            max_results=paper_count,
            domain=domain,
            sources=sources_to_search
        )
        
        # 검색 결과 로깅
        observation_content = f"검색 완료: {len(papers)}개의 논문을 찾았습니다.\n"
        
        for i, paper in enumerate(papers, 1):
            source_badge = f"[{paper.source}]"
            observation_content += f"\n{i}. {source_badge} {paper.title[:50]}..."
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        logger.info(f"✓ 검색 완료: {len(papers)}개 논문")
        
        return {
            "papers": papers,
            "react_steps": [action_step, observation_step],
            "error_message": None
        }
        
    except Exception as e:
        logger.error(f"논문 검색 중 오류: {str(e)}", exc_info=True)
        
        error_step = ReActStep(
            step_type="observation",
            content=f"검색 중 오류 발생: {str(e)}"
        )
        
        return {
            "papers": [],
            "react_steps": [action_step, error_step],
            "error_message": str(e)
        }


# ============================================
# RISS 검색을 포함하는 버전
# ============================================

def search_papers_node_with_riss(state: AgentState) -> dict:
    """
    RISS를 포함하는 검색 노드
    
    이것을 사용하면 arXiv와 RISS 모두에서 검색합니다.
    더 많은 논문을 찾을 수 있지만, 검색이 약간 더 오래 걸립니다.
    
    config.py에서 설정으로 전환할 수 있습니다.
    """
    
    keywords = state.get("extracted_keywords", [])
    paper_count = state.get("paper_count", 3)
    domain = state.get("question_domain", None)
    
    # RISS 포함
    sources_to_search = ["arxiv", "riss"]
    
    action_content = f"""논문 검색을 실행합니다 (멀티소스):
- 키워드: {', '.join(keywords)}
- 검색 개수: {paper_count}개/소스
- 도메인: {domain or '전체'}
- 검색 소스: {', '.join(sources_to_search)}
  (arXiv: 해외 논문, RISS: 한국 논문)"""
    
    action_step = ReActStep(
        step_type="action",
        content=action_content
    )
    
    try:
        searcher = get_multi_source_searcher()
        papers = searcher.search(
            keywords=keywords,
            max_results=paper_count,
            domain=domain,
            sources=sources_to_search
        )
        
        # 출처별 분류
        arxiv_count = sum(1 for p in papers if p.source == 'arXiv')
        riss_count = sum(1 for p in papers if p.source == 'RISS')
        
        observation_content = f"""검색 완료: {len(papers)}개의 논문을 찾았습니다.
- arXiv (해외): {arxiv_count}개
- RISS (한국): {riss_count}개
"""
        
        for i, paper in enumerate(papers, 1):
            source_badge = f"[{paper.source}]"
            observation_content += f"\n{i}. {source_badge} {paper.title[:50]}..."
        
        observation_step = ReActStep(
            step_type="observation",
            content=observation_content
        )
        
        logger.info(f"✓ 멀티소스 검색 완료: arXiv {arxiv_count}개 + RISS {riss_count}개")
        
        return {
            "papers": papers,
            "react_steps": [action_step, observation_step],
            "error_message": None
        }
        
    except Exception as e:
        logger.error(f"논문 검색 중 오류: {str(e)}")
        
        error_step = ReActStep(
            step_type="observation",
            content=f"검색 중 오류 발생: {str(e)}"
        )
        
        return {
            "papers": [],
            "react_steps": [action_step, error_step],
            "error_message": str(e)
        }


# ============================================
# 사용 예시
# ============================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🔍 멀티소스 검색 예시")
    print("="*60 + "\n")
    
    # 테스트 상태 객체
    test_state = {
        "extracted_keywords": ["attention mechanism", "transformer"],
        "paper_count": 3,
        "question_domain": "computer science"
    }
    
    # 1. arXiv만 검색
    print("1️⃣ arXiv만 검색 (빠름)")
    print("-" * 60)
    result = search_papers_node(test_state)
    print(f"검색 완료: {len(result['papers'])}개 논문")
    for paper in result['papers']:
        print(f"  - {paper.title[:50]}... ({paper.source})")
    
    # 2. arXiv + RISS 검색 (RISS가 설치되어 있으면)
    print("\n2️⃣ arXiv + RISS 통합 검색")
    print("-" * 60)
    
    searcher = get_multi_source_searcher()
    if searcher.riss_client:
        papers = searcher.search(
            keywords=test_state["extracted_keywords"],
            max_results=3,
            sources=["arxiv", "riss"]
        )
        print(f"검색 완료: {len(papers)}개 논문")
        for paper in papers:
            print(f"  - {paper.title[:50]}... ({paper.source})")
    else:
        print("RISS 클라이언트를 사용할 수 없습니다")
        print("설치: pip install beautifulsoup4 lxml")
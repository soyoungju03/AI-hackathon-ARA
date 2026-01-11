"""
arXiv 논문 검색 도구
====================

arXiv는 물리학, 수학, 컴퓨터 과학 등의 분야에서 
연구자들이 논문을 무료로 공개하는 오픈 액세스 사이트입니다.

이 파일은 arXiv API를 사용하여 논문을 검색하는 기능을 제공합니다.
기존 코드에서 개선된 점:
1. 더 정교한 검색 쿼리 생성
2. 연관성 점수 계산 추가
3. 에러 처리 강화
4. LangGraph Tool 형식으로 구조화
"""

import arxiv
from typing import List, Optional
from datetime import datetime
import re

# 상위 모듈에서 타입 정의를 가져옵니다
import sys
sys.path.append('..')
from app.graph.state import Paper


class ArxivSearchTool:
    """
    arXiv 논문 검색 도구입니다.
    
    이 클래스는 arXiv API를 래핑하여 더 쉽게 사용할 수 있도록 합니다.
    싱글톤 패턴 대신 인스턴스를 생성하여 사용합니다.
    
    Example:
        >>> tool = ArxivSearchTool()
        >>> papers = tool.search(
        ...     keywords=["autonomous driving", "LiDAR"],
        ...     max_results=5
        ... )
        >>> for paper in papers:
        ...     print(paper.title)
    """
    
    def __init__(self):
        """arXiv 클라이언트를 초기화합니다."""
        self.client = arxiv.Client(
            page_size=100,       # 한 번에 가져올 결과 수
            delay_seconds=3.0,   # 요청 사이 대기 시간 (서버 부하 방지)
            num_retries=3        # 실패 시 재시도 횟수
        )
    
    def _build_query(
        self, 
        keywords: List[str],
        domain: Optional[str] = None
    ) -> str:
        """
        arXiv 검색 쿼리를 생성합니다.
        
        arXiv API는 특별한 검색 문법을 사용합니다:
        - ti:keyword → 제목에서 검색
        - abs:keyword → 초록에서 검색
        - all:keyword → 전체에서 검색
        - AND, OR → 논리 연산자
        - cat:cs.AI → 특정 카테고리에서 검색
        
        Args:
            keywords: 검색 키워드 목록
            domain: 검색 도메인/카테고리 (예: "cs.AI", "physics")
        
        Returns:
            str: 완성된 검색 쿼리 문자열
        
        Example:
            >>> tool = ArxivSearchTool()
            >>> query = tool._build_query(
            ...     keywords=["transformer", "attention"],
            ...     domain="cs.AI"
            ... )
            >>> print(query)
            '(ti:transformer OR abs:transformer) AND (ti:attention OR abs:attention) AND cat:cs.AI'
        """
        
        if not keywords:
            return ""
        
        # 각 키워드에 대해 제목과 초록에서 검색하는 쿼리 생성
        # 예: ["LiDAR", "detection"] → 
        #     "(ti:LiDAR OR abs:LiDAR) AND (ti:detection OR abs:detection)"
        keyword_queries = []
        for keyword in keywords:
            # 키워드 정리 (특수문자 제거)
            clean_keyword = re.sub(r'[^\w\s-]', '', keyword).strip()
            if clean_keyword:
                # 띄어쓰기가 있는 키워드는 따옴표로 감싸기
                if ' ' in clean_keyword:
                    clean_keyword = f'"{clean_keyword}"'
                keyword_queries.append(f"(ti:{clean_keyword} OR abs:{clean_keyword})")
        
        # AND로 연결
        query = " AND ".join(keyword_queries)
        
        # 도메인/카테고리 필터 추가
        if domain:
            category_map = {
                "computer science": "cs.*",
                "physics": "physics.*",
                "mathematics": "math.*",
                "biology": "q-bio.*",
                "finance": "q-fin.*",
                "statistics": "stat.*",
                "electrical engineering": "eess.*",
                # 세부 카테고리
                "machine learning": "cs.LG",
                "artificial intelligence": "cs.AI",
                "computer vision": "cs.CV",
                "natural language processing": "cs.CL",
                "robotics": "cs.RO",
            }
            
            category = category_map.get(domain.lower(), None)
            if category:
                query = f"({query}) AND cat:{category}"
        
        return query
    
    def _calculate_relevance(
        self,
        paper_title: str,
        paper_abstract: str,
        keywords: List[str]
    ) -> float:
        """
        논문과 검색 키워드 간의 연관성 점수를 계산합니다.
        
        현재는 간단한 키워드 매칭 방식을 사용합니다.
        나중에 임베딩 기반 유사도로 개선할 수 있습니다.
        
        점수 계산 방식:
        1. 키워드가 제목에 포함되면 높은 점수 (+0.3)
        2. 키워드가 초록에 포함되면 낮은 점수 (+0.1)
        3. 최대 점수는 1.0
        
        Args:
            paper_title: 논문 제목
            paper_abstract: 논문 초록
            keywords: 검색에 사용된 키워드
        
        Returns:
            float: 0.0 ~ 1.0 사이의 연관성 점수
        """
        if not keywords:
            return 0.5  # 키워드가 없으면 중간 점수
        
        score = 0.0
        title_lower = paper_title.lower()
        abstract_lower = paper_abstract.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 제목에서 키워드 발견 (높은 가중치)
            if keyword_lower in title_lower:
                score += 0.3
            
            # 초록에서 키워드 발견 (낮은 가중치)
            if keyword_lower in abstract_lower:
                score += 0.1
        
        # 점수를 0-1 범위로 정규화
        # 모든 키워드가 제목과 초록에 있으면 최대 0.4 * len(keywords)
        max_possible = 0.4 * len(keywords)
        if max_possible > 0:
            score = min(score / max_possible, 1.0)
        
        return round(score, 2)
    
    def search(
        self,
        keywords: List[str],
        max_results: int = 5,
        domain: Optional[str] = None,
        sort_by: str = "relevance"  # "relevance" 또는 "date"
    ) -> List[Paper]:
        """
        arXiv에서 논문을 검색합니다.
        
        이 메서드는 다음 단계로 동작합니다:
        1. 키워드로 검색 쿼리 생성
        2. arXiv API 호출
        3. 결과를 Paper 객체로 변환
        4. 연관성 점수 계산 및 정렬
        
        Args:
            keywords: 검색 키워드 목록
            max_results: 최대 결과 수 (기본값: 5)
            domain: 검색 도메인 (예: "computer science", "physics")
            sort_by: 정렬 기준 ("relevance" 또는 "date")
        
        Returns:
            List[Paper]: 검색된 논문 목록 (연관성 점수 포함)
        
        Raises:
            Exception: arXiv API 호출 실패 시
        
        Example:
            >>> tool = ArxivSearchTool()
            >>> papers = tool.search(
            ...     keywords=["transformer", "vision"],
            ...     max_results=3,
            ...     domain="computer science"
            ... )
            >>> print(f"Found {len(papers)} papers")
            Found 3 papers
        """
        
        # 검색 쿼리 생성
        query = self._build_query(keywords, domain)
        
        if not query:
            return []
        
        # 정렬 기준 설정
        if sort_by == "date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        else:
            sort_criterion = arxiv.SortCriterion.Relevance
        
        # arXiv 검색 객체 생성
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # 필터링을 위해 여유있게 검색
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending
        )
        
        try:
            # API 호출 및 결과 수집
            papers = []
            for result in self.client.results(search):
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors[:5]],  # 최대 5명
                    abstract=result.summary,
                    url=result.entry_id,
                    published_date=result.published.strftime("%Y-%m-%d"),
                    source="arXiv",
                    relevance_score=self._calculate_relevance(
                        result.title,
                        result.summary,
                        keywords
                    )
                )
                papers.append(paper)
            
            # 연관성 점수로 정렬 (높은 순)
            papers.sort(key=lambda p: p.relevance_score, reverse=True)
            
            # 요청한 수만큼만 반환
            return papers[:max_results]
            
        except Exception as e:
            # 에러 발생 시 빈 리스트 반환하지 않고 에러를 다시 발생시킵니다
            # 이렇게 하면 호출하는 쪽에서 에러를 적절히 처리할 수 있습니다
            raise Exception(f"arXiv 검색 중 오류 발생: {str(e)}")
    
    def search_by_query(
        self,
        raw_query: str,
        max_results: int = 5
    ) -> List[Paper]:
        """
        직접 쿼리 문자열로 검색합니다.
        
        고급 사용자가 arXiv 검색 문법을 직접 사용하고 싶을 때 유용합니다.
        
        Args:
            raw_query: arXiv 검색 쿼리 문자열
            max_results: 최대 결과 수
        
        Returns:
            List[Paper]: 검색된 논문 목록
        """
        search = arxiv.Search(
            query=raw_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            paper = Paper(
                title=result.title,
                authors=[author.name for author in result.authors[:5]],
                abstract=result.summary,
                url=result.entry_id,
                published_date=result.published.strftime("%Y-%m-%d"),
                source="arXiv",
                relevance_score=0.5  # 직접 쿼리의 경우 기본 점수
            )
            papers.append(paper)
        
        return papers


# 모듈 레벨에서 기본 인스턴스 생성 (편의를 위해)
# 대부분의 경우 이 인스턴스를 사용하면 됩니다
default_arxiv_tool = ArxivSearchTool()


def search_arxiv(
    keywords: List[str],
    max_results: int = 5,
    domain: Optional[str] = None
) -> List[Paper]:
    """
    arXiv 검색을 위한 편의 함수입니다.
    
    이 함수는 ArxivSearchTool의 기본 인스턴스를 사용합니다.
    클래스를 직접 인스턴스화하지 않고 바로 검색할 수 있습니다.
    
    Example:
        >>> from app.tools.paper_search.arxiv_tool import search_arxiv
        >>> papers = search_arxiv(
        ...     keywords=["GPT", "language model"],
        ...     max_results=3
        ... )
    """
    return default_arxiv_tool.search(keywords, max_results, domain)

"""
arXiv API 클라이언트
arXiv에서 논문을 검색하고 메타데이터를 수집합니다.

arXiv API 문서: https://arxiv.org/help/api/user-manual
"""

import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
from urllib.parse import quote
import xml.etree.ElementTree as ET


class ArxivClient:
    """arXiv API를 사용하여 논문 정보를 검색하는 클래스"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # arXiv 카테고리 매핑
    CATEGORIES = {
        'cs.AI': '인공지능',
        'cs.LG': '기계학습',
        'cs.CL': '자연어 처리',
        'cs.CV': '컴퓨터 비전',
        'cs.NE': '신경망',
        'stat.ML': '통계 머신러닝',
        'physics.data-an': '데이터 분석'
    }
    
    def __init__(self, max_results_per_query: int = 100, delay: float = 3):
        """
        ArxivClient 초기화
        
        Args:
            max_results_per_query: 한 번의 쿼리로 가져올 최대 논문 수
            delay: API 요청 사이의 지연시간 (초) - arXiv API 정책 준수
        """
        self.max_results_per_query = max_results_per_query
        self.delay = delay
        self.session = requests.Session()
        self.last_request_time = 0
        
        print("✓ arXiv 클라이언트 초기화 완료")
        print(f"  - 최대 결과 수: {max_results_per_query}")
        print(f"  - 요청 지연: {delay}초")
    
    def _rate_limit(self):
        """API 속도 제한을 지키기 위해 대기"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def search_by_keyword(
        self,
        keywords: List[str],
        max_results: int = 50,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        categories: Optional[List[str]] = None
    ) -> Tuple[List[Dict], int]:
        """
        키워드로 arXiv 논문 검색
        
        Args:
            keywords: 검색할 키워드 리스트
            max_results: 최대 검색 결과 수
            sort_by: 정렬 기준 ("submittedDate", "lastUpdatedDate", "relevance")
            sort_order: 정렬 순서 ("ascending", "descending")
            categories: 제한할 카테고리 리스트 (예: ['cs.LG', 'cs.AI'])
        
        Returns:
            (검색된 논문 리스트, 총 결과 수)
        """
        
        # 검색 쿼리 구성
        query_parts = []
        
        for keyword in keywords:
            # 각 키워드를 제목, 요약, 저자에서 검색
            query_parts.append(f"(ti:\"{keyword}\" OR abs:\"{keyword}\")")
        
        query = " AND ".join(query_parts) if query_parts else "*"
        
        # 카테고리 필터 추가
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query += f" AND ({cat_query})"
        
        return self._execute_query(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
    
    def search_by_category(
        self,
        category: str,
        max_results: int = 50,
        days_back: int = 7,
        sort_by: str = "submittedDate"
    ) -> Tuple[List[Dict], int]:
        """
        카테고리별로 최신 논문 검색
        
        Args:
            category: arXiv 카테고리 (예: 'cs.LG', 'cs.AI')
            max_results: 최대 검색 결과 수
            days_back: 몇 일 전까지의 논문을 검색할지
            sort_by: 정렬 기준
        
        Returns:
            (검색된 논문 리스트, 총 결과 수)
        """
        
        # 최근 N일 논문만 검색
        start_date = datetime.utcnow() - timedelta(days=days_back)
        date_str = start_date.strftime("%Y%m%d%H%M%S")
        
        query = f"cat:{category} AND submittedDate:[{date_str} TO 9999999999]"
        
        return self._execute_query(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order="descending"
        )
    
    def search_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        특정 arXiv ID로 논문 검색
        
        Args:
            arxiv_id: arXiv 논문 ID (예: "2401.00001")
        
        Returns:
            논문 정보 딕셔너리, 또는 찾지 못한 경우 None
        """
        
        query = f"arxivID:{arxiv_id}"
        papers, _ = self._execute_query(query, max_results=1)
        
        return papers[0] if papers else None
    
    def _execute_query(
        self,
        query: str,
        max_results: int,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> Tuple[List[Dict], int]:
        """
        arXiv API에 쿼리 실행
        
        Args:
            query: arXiv 검색 쿼리
            max_results: 최대 결과 수
            sort_by: 정렬 기준
            sort_order: 정렬 순서
        
        Returns:
            (논문 정보 리스트, 총 결과 수)
        """
        
        self._rate_limit()
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': min(max_results, self.max_results_per_query),
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        try:
            print(f"\n🔍 arXiv 검색 중...")
            print(f"   쿼리: {query[:100]}..." if len(query) > 100 else f"   쿼리: {query}")
            
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            papers, total_results = self._parse_response(response.text)
            
            print(f"✓ 검색 완료: {len(papers)}개 논문 발견 (총 {total_results}개 중)")
            
            return papers, total_results
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {str(e)}")
            return [], 0
    
    def _parse_response(self, xml_content: str) -> Tuple[List[Dict], int]:
        """
        arXiv API의 XML 응답을 파싱
        
        Args:
            xml_content: API 응답 XML
        
        Returns:
            (논문 정보 리스트, 총 결과 수)
        """
        
        papers = []
        total_results = 0
        
        try:
            root = ET.fromstring(xml_content)
            
            # XML 네임스페이스 정의
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # 전체 결과 수 파싱
            total_elem = root.find('atom:totalResults', namespaces)
            if total_elem is not None:
                total_results = int(total_elem.text)
            
            # 각 논문 파싱
            for entry in root.findall('atom:entry', namespaces):
                try:
                    paper_info = self._extract_paper_info(entry, namespaces)
                    if paper_info:
                        papers.append(paper_info)
                except Exception as e:
                    print(f"⚠️ 논문 파싱 오류: {str(e)}")
                    continue
            
            return papers, total_results
        
        except ET.ParseError as e:
            print(f"❌ XML 파싱 오류: {str(e)}")
            return [], 0
    
    def _extract_paper_info(self, entry, namespaces: Dict) -> Optional[Dict]:
        """
        XML 엔트리에서 논문 정보 추출
        
        Returns:
            논문 정보 딕셔너리
        """
        
        atom_ns = namespaces['atom']
        arxiv_ns = namespaces['arxiv']
        
        # 기본 정보
        paper_id = entry.find(f'{{{arxiv_ns}}}id')
        title = entry.find(f'{{{atom_ns}}}title')
        summary = entry.find(f'{{{atom_ns}}}summary')
        published = entry.find(f'{{{atom_ns}}}published')
        updated = entry.find(f'{{{atom_ns}}}updated')
        
        if not all([paper_id, title, summary]):
            return None
        
        # arXiv ID 정제 (버전 번호 제거)
        arxiv_id = paper_id.text.split('/abs/')[-1]
        
        # 저자 추출
        authors = []
        for author in entry.findall(f'{{{atom_ns}}}author'):
            name_elem = author.find(f'{{{atom_ns}}}name')
            if name_elem is not None:
                authors.append(name_elem.text)
        
        # 카테고리 추출
        categories = []
        for category in entry.findall(f'{{{arxiv_ns}}}primary_category'):
            term = category.get('term')
            if term:
                categories.append(term)
        
        for category in entry.findall(f'{{{atom_ns}}}category'):
            term = category.get('term')
            if term:
                categories.append(term)
        
        categories = list(set(categories))  # 중복 제거
        
        # 링크 추출
        pdf_url = ""
        html_url = ""
        for link in entry.findall(f'{{{atom_ns}}}link'):
            rel = link.get('rel')
            href = link.get('href')
            
            if rel == 'alternate':
                html_url = href
            elif link.get('type') == 'application/pdf':
                pdf_url = href + '.pdf'  # PDF 링크 완성
        
        # 텍스트 정제
        def clean_text(text):
            if text is None:
                return ""
            return ' '.join(text.split())
        
        title_text = clean_text(title.text)
        summary_text = clean_text(summary.text)
        
        # 문서 생성 (벡터스토어에 추가할 형식)
        paper_info = {
            'id': f"arxiv_{arxiv_id.replace('.', '_').replace('/', '_')}",
            'content': f"{title_text}\n\n{summary_text}",
            'metadata': {
                'arxiv_id': arxiv_id,
                'title': title_text,
                'authors': authors,
                'published_date': published.text if published is not None else "",
                'updated_date': updated.text if updated is not None else "",
                'summary': summary_text,
                'categories': categories,
                'primary_category': categories[0] if categories else "unknown",
                'pdf_url': pdf_url,
                'html_url': html_url or f"https://arxiv.org/abs/{arxiv_id}"
            }
        }
        
        return paper_info
    
    def get_trending_papers(
        self,
        category: str = "cs.LG",
        days_back: int = 7,
        max_results: int = 20
    ) -> List[Dict]:
        """
        특정 카테고리의 최신 논문 조회
        
        Args:
            category: arXiv 카테고리
            days_back: 최근 N일의 논문
            max_results: 최대 결과 수
        
        Returns:
            논문 정보 리스트
        """
        
        papers, _ = self.search_by_category(
            category=category,
            max_results=max_results,
            days_back=days_back
        )
        
        return papers
    
    def search_multiple_queries(
        self,
        queries: List[Dict],
        consolidate: bool = True
    ) -> List[Dict]:
        """
        여러 검색 쿼리를 순차적으로 실행
        
        Args:
            queries: 검색 쿼리 리스트. 각 쿼리는:
                {
                    'type': 'keyword' | 'category',
                    'keywords': [...],  # type='keyword'일 때
                    'category': 'cs.LG',  # type='category'일 때
                    'max_results': 20
                }
            consolidate: 중복된 논문 제거 여부
        
        Returns:
            모든 검색 결과를 통합한 논문 리스트
        """
        
        all_papers = []
        seen_arxiv_ids = set()
        
        for query in queries:
            query_type = query.get('type', 'keyword')
            max_results = query.get('max_results', 50)
            
            if query_type == 'keyword':
                papers, _ = self.search_by_keyword(
                    keywords=query.get('keywords', []),
                    max_results=max_results
                )
            elif query_type == 'category':
                papers, _ = self.search_by_category(
                    category=query.get('category', 'cs.LG'),
                    max_results=max_results
                )
            else:
                continue
            
            # 중복 제거
            for paper in papers:
                arxiv_id = paper['metadata']['arxiv_id']
                
                if consolidate and arxiv_id in seen_arxiv_ids:
                    continue
                
                seen_arxiv_ids.add(arxiv_id)
                all_papers.append(paper)
        
        print(f"\n📊 통합 결과: {len(all_papers)}개 논문 (중복 제거됨)")
        
        return all_papers


# 사용 예시 함수
def example_usage():
    """arXiv 클라이언트 사용 예시"""
    
    print("\n" + "="*60)
    print("🔬 arXiv API 클라이언트 예시")
    print("="*60 + "\n")
    
    # 1. 클라이언트 초기화
    client = ArxivClient(max_results_per_query=100, delay=2)
    
    # 2. 키워드 검색
    print("\n" + "-"*60)
    print("1️⃣ 키워드 검색: 'attention mechanism' AND 'transformers'")
    print("-"*60)
    
    papers, total = client.search_by_keyword(
        keywords=['attention mechanism', 'transformers'],
        max_results=5
    )
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['metadata']['title']}")
        print(f"   arXiv ID: {paper['metadata']['arxiv_id']}")
        print(f"   저자: {', '.join(paper['metadata']['authors'][:2])}")
        print(f"   게시일: {paper['metadata']['published_date'][:10]}")
        print(f"   카테고리: {', '.join(paper['metadata']['categories'])}")
    
    # 3. 카테고리별 최신 논문
    print("\n" + "-"*60)
    print("2️⃣ 카테고리별 최신 논문: cs.LG (최근 7일)")
    print("-"*60)
    
    trending_papers = client.get_trending_papers(
        category='cs.LG',
        days_back=7,
        max_results=3
    )
    
    for i, paper in enumerate(trending_papers, 1):
        print(f"\n{i}. {paper['metadata']['title']}")
        print(f"   arXiv ID: {paper['metadata']['arxiv_id']}")
        print(f"   요약: {paper['metadata']['summary'][:200]}...")
    
    # 4. 특정 ID로 검색
    print("\n" + "-"*60)
    print("3️⃣ 특정 논문 검색")
    print("-"*60)
    
    specific_paper = client.search_by_arxiv_id("2401.00001")
    if specific_paper:
        print(f"✓ 논문 발견:")
        print(f"  제목: {specific_paper['metadata']['title']}")
        print(f"  저자: {', '.join(specific_paper['metadata']['authors'])}")
    else:
        print("논문을 찾지 못했습니다")
    
    # 5. 여러 쿼리 검색
    print("\n" + "-"*60)
    print("4️⃣ 복수 쿼리 검색")
    print("-"*60)
    
    queries = [
        {
            'type': 'keyword',
            'keywords': ['vision transformer'],
            'max_results': 5
        },
        {
            'type': 'category',
            'category': 'cs.CV',
            'max_results': 5
        }
    ]
    
    consolidated_papers = client.search_multiple_queries(queries, consolidate=True)
    
    print(f"\n최종 결과: {len(consolidated_papers)}개 논문")
    print("\n상위 3개 논문:")
    for i, paper in enumerate(consolidated_papers[:3], 1):
        print(f"{i}. {paper['metadata']['title']}")


if __name__ == "__main__":
    example_usage()
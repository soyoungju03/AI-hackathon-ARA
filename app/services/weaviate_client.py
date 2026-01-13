# -*- coding: utf-8 -*-
"""
Weaviate 클라이언트
==================

이 모듈은 Weaviate 벡터 데이터베이스와 상호작용합니다.

주요 기능:
1. Weaviate 연결 및 초기화
2. 논문 데이터 저장 (upsert)
3. 의미 기반 논문 검색
4. 벡터 임베딩 생성 및 관리

사용법:
    from app.services.weaviate_client import WeaviateClient
    
    client = WeaviateClient()
    
    # 논문 저장
    client.add_paper({
        'title': '...',
        'abstract': '...',
        ...
    })
    
    # 논문 검색
    papers = client.search_papers('machine learning', limit=5)
"""

import logging
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """논문 데이터 클래스"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    url: str
    published_date: str
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Weaviate 저장용 딕셔너리로 변환"""
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'arxiv_id': self.arxiv_id,
            'url': self.url,
            'published_date': self.published_date,
            'summary': self.summary or '',
            'keywords': self.keywords or [],
            'relevance_score': self.relevance_score or 0.0
        }


class WeaviateClient:
    """
    Weaviate 벡터 데이터베이스 클라이언트
    
    이 클래스는 Weaviate와의 모든 상호작용을 관리합니다.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        use_embedded: bool = False
    ):
        """
        Weaviate 클라이언트 초기화
        
        Args:
            url: Weaviate 서버 URL (기본값: 로컬 호스트)
            api_key: Weaviate API 키 (필요시)
            use_embedded: 임베디드 Weaviate 사용 여부
        """
        
        self.url = url
        self.api_key = api_key
        self.use_embedded = use_embedded
        self.client = None
        
        self._initialize_client()
        self._ensure_schema()
        
        logger.info("✓ Weaviate 클라이언트 초기화 완료")
    
    def _initialize_client(self):
        """Weaviate 클라이언트 초기화"""
        
        try:
            import weaviate
            from weaviate.embedded import EmbeddedOptions
            
            if self.use_embedded:
                logger.info("📦 임베디드 Weaviate 사용 중...")
                try:
                    self.client = weaviate.Client(
                        embedded_options=EmbeddedOptions(version="1.0")
                    )
                    logger.info("✓ 임베디드 Weaviate 초기화 성공")
                except Exception as e:
                    logger.warning(f"⚠️  임베디드 초기화 실패: {str(e)}")
                    logger.info("📡 네트워크 연결 시도...")
                    self.client = weaviate.Client(
                        url=self.url,
                        auth_client_secret=weaviate.AuthApiKey(
                            api_key=self.api_key
                        ) if self.api_key else None
                    )
            else:
                logger.info(f"📡 Weaviate 서버 연결 중... ({self.url})")
                self.client = weaviate.Client(
                    url=self.url,
                    auth_client_secret=weaviate.AuthApiKey(
                        api_key=self.api_key
                    ) if self.api_key else None
                )
            
            # 연결 확인
            if self.client.is_ready():
                logger.info("✓ Weaviate 연결 성공")
            else:
                logger.error("✗ Weaviate 준비 상태 확인 실패")
                
        except ImportError:
            logger.error("✗ weaviate-client 패키지가 설치되지 않았습니다")
            logger.info("   설치: pip install weaviate-client")
            raise
        except Exception as e:
            logger.error(f"✗ Weaviate 초기화 실패: {str(e)}")
            raise
    
    def _ensure_schema(self):
        """스키마 생성 (존재하지 않으면)"""
        
        try:
            # 기존 스키마 확인
            schema = self.client.schema.get()
            
            # Paper 클래스가 이미 존재하는지 확인
            class_names = [cls['class'] for cls in schema.get('classes', [])]
            
            if 'Paper' in class_names:
                logger.info("✓ Paper 스키마 이미 존재")
                return
            
            # 스키마 생성
            paper_schema = {
                'class': 'Paper',
                'description': 'A research paper from arXiv or other sources',
                'properties': [
                    {
                        'name': 'title',
                        'description': 'Title of the paper',
                        'dataType': ['text']
                    },
                    {
                        'name': 'authors',
                        'description': 'Authors of the paper',
                        'dataType': ['text[]']
                    },
                    {
                        'name': 'abstract',
                        'description': 'Abstract of the paper',
                        'dataType': ['text']
                    },
                    {
                        'name': 'summary',
                        'description': 'AI-generated summary',
                        'dataType': ['text']
                    },
                    {
                        'name': 'arxiv_id',
                        'description': 'arXiv paper ID',
                        'dataType': ['text'],
                        'indexInverted': True
                    },
                    {
                        'name': 'url',
                        'description': 'URL to the paper',
                        'dataType': ['text']
                    },
                    {
                        'name': 'published_date',
                        'description': 'Publication date',
                        'dataType': ['date']
                    },
                    {
                        'name': 'keywords',
                        'description': 'Search keywords related to paper',
                        'dataType': ['text[]']
                    },
                    {
                        'name': 'relevance_score',
                        'description': 'Relevance score for current query',
                        'dataType': ['number']
                    }
                ],
                'vectorizer': 'none',  # 수동으로 벡터화 (비용 절감)
                'vectorIndexConfig': {
                    'distance': 'cosine',
                    'hnsw': {
                        'efConstruction': 128,
                        'maxConnections': 64
                    }
                }
            }
            
            self.client.schema.create_class(paper_schema)
            logger.info("✓ Paper 스키마 생성 완료")
            
        except Exception as e:
            logger.error(f"⚠️  스키마 생성 중 오류: {str(e)}")
            # 이미 존재하는 경우 무시
            pass
    
    def add_paper(self, paper: Paper) -> bool:
        """
        논문을 Weaviate에 추가
        
        Args:
            paper: Paper 객체
        
        Returns:
            성공 여부
        """
        
        try:
            # 벡터 생성 (abstract와 summary를 함께 임베딩)
            embedding_text = f"{paper.title}. {paper.abstract}"
            if paper.summary:
                embedding_text += f". {paper.summary}"
            
            vector = self._generate_embedding(embedding_text)
            
            # Weaviate에 추가
            paper_data = paper.to_dict()
            
            uuid = self.client.data_object.create(
                data_object=paper_data,
                class_name='Paper',
                vector=vector
            )
            
            logger.debug(f"✓ 논문 추가: {paper.arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ 논문 추가 실패 ({paper.arxiv_id}): {str(e)}")
            return False
    
    def add_papers_batch(self, papers: List[Paper]) -> int:
        """
        여러 논문을 배치로 추가 (더 빠름)
        
        Args:
            papers: Paper 객체 리스트
        
        Returns:
            성공한 논문 개수
        """
        
        logger.info(f"📝 {len(papers)}개 논문 배치 추가 시작...")
        
        success_count = 0
        
        try:
            # Weaviate 배치 작업 시작
            with self.client.batch as batch:
                batch.batch_size = 100  # 배치 크기
                
                for paper in papers:
                    try:
                        embedding_text = f"{paper.title}. {paper.abstract}"
                        if paper.summary:
                            embedding_text += f". {paper.summary}"
                        
                        vector = self._generate_embedding(embedding_text)
                        paper_data = paper.to_dict()
                        
                        batch.add_data_object(
                            data_object=paper_data,
                            class_name='Paper',
                            vector=vector
                        )
                        
                        success_count += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠️  논문 추가 실패: {paper.arxiv_id} - {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"✗ 배치 작업 실패: {str(e)}")
        
        logger.info(f"✓ {success_count}/{len(papers)}개 논문 추가 완료")
        return success_count
    
    def search_papers(
        self,
        query: str,
        limit: int = 5,
        keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        의미 기반으로 논문 검색
        
        Args:
            query: 검색 쿼리 (자연어)
            limit: 반환할 결과 개수
            keywords: 추가 필터링 키워드
        
        Returns:
            검색된 논문 리스트
        """
        
        try:
            logger.info(f"🔍 논문 검색: '{query}' (limit={limit})")
            
            # 쿼리 벡터 생성
            query_vector = self._generate_embedding(query)
            
            # Weaviate에서 벡터 유사성으로 검색
            where_filter = None
            
            if keywords:
                # 키워드 필터 적용 (선택사항)
                keyword_filters = [
                    {
                        'path': ['keywords'],
                        'operator': 'ContainsAny',
                        'valueText': keyword
                    }
                    for keyword in keywords
                ]
                
                if keyword_filters:
                    where_filter = {
                        'operator': 'Or',
                        'operands': keyword_filters
                    }
            
            # 검색 실행
            results = self.client.query.get(
                'Paper',
                ['title', 'abstract', 'summary', 'authors', 'url', 'arxiv_id', 'published_date', 'relevance_score']
            ).with_near_vector(
                {
                    'vector': query_vector
                }
            ).with_limit(
                limit
            ).with_where(
                where_filter
            ) if where_filter else self.client.query.get(
                'Paper',
                ['title', 'abstract', 'summary', 'authors', 'url', 'arxiv_id', 'published_date', 'relevance_score']
            ).with_near_vector(
                {
                    'vector': query_vector
                }
            ).with_limit(
                limit
            )
            
            results = results.do()
            
            papers = []
            if 'data' in results and 'Get' in results['data']:
                for paper_obj in results['data']['Get'].get('Paper', []):
                    papers.append(paper_obj)
            
            logger.info(f"✓ {len(papers)}개 논문 검색됨")
            return papers
            
        except Exception as e:
            logger.error(f"✗ 논문 검색 실패: {str(e)}")
            return []
    
    def search_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        arXiv ID로 논문 검색
        
        Args:
            arxiv_id: arXiv 논문 ID
        
        Returns:
            논문 데이터 또는 None
        """
        
        try:
            result = self.client.query.get(
                'Paper',
                ['title', 'abstract', 'summary', 'authors', 'url', 'published_date']
            ).with_where(
                {
                    'path': ['arxiv_id'],
                    'operator': 'Equal',
                    'valueText': arxiv_id
                }
            ).do()
            
            if 'data' in result and 'Get' in result['data']:
                papers = result['data']['Get'].get('Paper', [])
                if papers:
                    return papers[0]
            
            return None
            
        except Exception as e:
            logger.error(f"✗ arXiv ID 검색 실패: {str(e)}")
            return None
    
    def delete_paper(self, arxiv_id: str) -> bool:
        """
        arXiv ID로 논문 삭제
        
        Args:
            arxiv_id: arXiv 논문 ID
        
        Returns:
            성공 여부
        """
        
        try:
            # 먼저 ID로 논문 찾기
            result = self.client.query.get(
                'Paper',
                []
            ).with_where(
                {
                    'path': ['arxiv_id'],
                    'operator': 'Equal',
                    'valueText': arxiv_id
                }
            ).with_additional(['id']).do()
            
            if 'data' in result and 'Get' in result['data']:
                papers = result['data']['Get'].get('Paper', [])
                if papers and '_additional' in papers[0]:
                    object_id = papers[0]['_additional']['id']
                    
                    # 객체 삭제
                    self.client.data_object.delete(
                        uuid=object_id,
                        class_name='Paper'
                    )
                    
                    logger.info(f"✓ 논문 삭제: {arxiv_id}")
                    return True
            
            logger.warning(f"⚠️  논문을 찾을 수 없음: {arxiv_id}")
            return False
            
        except Exception as e:
            logger.error(f"✗ 논문 삭제 실패: {str(e)}")
            return False
    
    def get_paper_count(self) -> int:
        """저장된 논문의 총 개수"""
        
        try:
            result = self.client.query.aggregate(
                'Paper'
            ).with_meta_count().do()
            
            if 'data' in result and 'Aggregate' in result['data']:
                count_list = result['data']['Aggregate'].get('Paper', [])
                if count_list:
                    return count_list[0]['meta']['count']
            
            return 0
            
        except Exception as e:
            logger.error(f"⚠️  논문 개수 조회 실패: {str(e)}")
            return 0
    
    def clear_all(self) -> bool:
        """모든 논문 삭제 (주의!)"""
        
        try:
            logger.warning("🗑️  모든 논문 삭제 중...")
            
            self.client.schema.delete_class('Paper')
            self._ensure_schema()
            
            logger.info("✓ 모든 논문 삭제 완료, 스키마 재생성됨")
            return True
            
        except Exception as e:
            logger.error(f"✗ 삭제 실패: {str(e)}")
            return False
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        텍스트를 벡터로 변환
        
        OpenAI나 로컬 모델을 사용합니다.
        """
        
        try:
            # 방법 1: OpenAI 임베딩 (권장)
            try:
                from openai import OpenAI
                import os
                
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    client = OpenAI(api_key=api_key)
                    response = client.embeddings.create(
                        input=text[:8191],  # 토큰 제한
                        model="text-embedding-3-small"
                    )
                    return response.data[0].embedding
            except Exception as e:
                logger.debug(f"OpenAI 임베딩 실패: {str(e)}")
            
            # 방법 2: 로컬 모델 사용 (대체)
            try:
                from sentence_transformers import SentenceTransformer
                
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(text, convert_to_tensor=False)
                
                return embedding.tolist()
            except ImportError:
                logger.error("sentence-transformers 패키지 필요: pip install sentence-transformers")
                # 더미 벡터 반환 (테스트용)
                return [0.0] * 384
            
        except Exception as e:
            logger.error(f"⚠️  임베딩 생성 실패: {str(e)}")
            return [0.0] * 384  # 더미 벡터
    
    def health_check(self) -> bool:
        """Weaviate 연결 상태 확인"""
        
        try:
            return self.client.is_ready()
        except Exception as e:
            logger.error(f"⚠️  Weaviate 헬스 체크 실패: {str(e)}")
            return False


# 싱글톤 패턴으로 전역 클라이언트
_weaviate_client: Optional[WeaviateClient] = None


def get_weaviate_client(
    url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    use_embedded: bool = False
) -> WeaviateClient:
    """전역 Weaviate 클라이언트 인스턴스 반환"""
    
    global _weaviate_client
    
    if _weaviate_client is None:
        _weaviate_client = WeaviateClient(
            url=url,
            api_key=api_key,
            use_embedded=use_embedded
        )
    
    return _weaviate_client
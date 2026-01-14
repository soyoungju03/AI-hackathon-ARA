"""
ARA 프로젝트를 위한 ChromaDB 벡터 스토어 통합 모듈
당신의 workflow.py와 embeddings.py와 완벽하게 호환됩니다.

구조:
1. ArxivPaperVectorStore: 논문 데이터 관리
2. SemanticSearchEngine: 의미 기반 검색 엔진
3. WorkflowIntegration: workflow와의 통합 레이어
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# 당신의 embeddings 모듈에서 필요한 함수 import
try:
    from tools.embeddings import embed_text, calculate_semantic_similarity
except ImportError:
    # 개발 환경에서의 폴백
    embed_text = None
    calculate_semantic_similarity = None

logger = logging.getLogger(__name__)


class ArxivPaperVectorStore:
    """
    arXiv 논문을 위한 ChromaDB 벡터 스토어
    
    이 클래스는 논문의 메타데이터와 임베딩을 저장하고 관리합니다.
    당신의 embeddings.py의 embed_text 함수를 사용합니다.
    
    저장되는 정보:
    - 논문 ID와 제목
    - 초록 (abstract)
    - 저자 정보
    - 카테고리
    - 발표 날짜
    - arXiv 링크 등
    
    모든 메타데이터는 ChromaDB의 필터링 기능을 활용하여 검색할 수 있습니다.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/arxiv_vectorstore",
        collection_name: str = "arxiv_papers"
    ):
        """
        VectorStore 초기화
        
        Args:
            persist_directory: 데이터 저장 디렉토리
            collection_name: 컬렉션 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 디렉토리 생성
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # ChromaDB 초기화
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
        
        # 컬렉션 생성 또는 연결
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 로그 디렉토리
        self.log_dir = Path(persist_directory) / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        logger.info(f"✓ ArxivPaperVectorStore 초기화 완료")
        logger.info(f"  - 디렉토리: {persist_directory}")
        logger.info(f"  - 컬렉션: {collection_name}")
    
    def add_papers_from_arxiv_search(self, arxiv_papers: List[Dict]) -> Dict:
        """
        arXiv API에서 검색한 논문들을 VectorStore에 추가합니다.
        
        이 메서드는 workflow의 search_papers_node에서 반환된
        논문 목록을 받아 VectorStore에 저장합니다.
        
        Args:
            arxiv_papers: arXiv API에서 반환한 논문 리스트
                구조:
                {
                    'arxiv_id': '2401.00001',
                    'title': 'Paper Title',
                    'abstract': 'Paper abstract...',
                    'authors': ['Author1', 'Author2'],
                    'categories': ['cs.LG', 'cs.AI'],
                    'published_date': '2024-01-01',
                    'pdf_url': 'https://...',
                    'html_url': 'https://...'
                }
        
        Returns:
            {
                'success': bool,
                'added_count': int,
                'message': str
            }
        """
        
        if not arxiv_papers:
            return {
                "success": False,
                "added_count": 0,
                "message": "추가할 논문이 없습니다"
            }
        
        try:
            logger.info(f"논문 추가 시작: {len(arxiv_papers)}개")
            
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for i, paper in enumerate(arxiv_papers):
                arxiv_id = paper.get('arxiv_id')
                
                if not arxiv_id:
                    logger.warning(f"논문 {i}: arxiv_id 없음, 건너뜀")
                    continue
                
                # 고유 ID 생성
                doc_id = f"arxiv_{arxiv_id.replace('.', '_')}"
                
                # 문서 내용: 제목 + 초록
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                content = f"{title}\n\n{abstract}".strip()
                
                if not content:
                    logger.warning(f"논문 {arxiv_id}: 제목/초록 없음, 건너뜀")
                    continue
                
                # 당신의 embed_text 함수를 사용하여 임베딩 생성
                try:
                    if embed_text is not None:
                        embedding = embed_text(content)
                        # numpy 배열을 리스트로 변환
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                    else:
                        logger.warning("embed_text를 로드할 수 없습니다")
                        embedding = None
                except Exception as e:
                    logger.error(f"논문 {arxiv_id} 임베딩 실패: {str(e)}")
                    embedding = None
                
                # 메타데이터 준비
                metadata = {
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'authors': ', '.join(paper.get('authors', [])),
                    'categories': ', '.join(paper.get('categories', [])),
                    'published_date': paper.get('published_date', ''),
                    'pdf_url': paper.get('pdf_url', ''),
                    'html_url': paper.get('html_url', '')
                }
                
                # 데이터 수집
                ids.append(doc_id)
                documents.append(content)
                metadatas.append(metadata)
                if embedding:
                    embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  {i + 1}/{len(arxiv_papers)} 임베딩 완료")
            
            if not ids:
                return {
                    "success": False,
                    "added_count": 0,
                    "message": "유효한 논문이 없습니다"
                }
            
            # ChromaDB에 추가
            if embeddings and len(embeddings) == len(ids):
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # 일부 임베딩이 누락된 경우
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"✓ {len(ids)}개 논문 추가 완료")
            
            # 로그 저장
            self._save_operation_log(ids, 'add_papers', len(ids))
            
            return {
                "success": True,
                "added_count": len(ids),
                "message": f"{len(ids)}개 논문이 저장되었습니다"
            }
        
        except Exception as e:
            logger.error(f"논문 추가 실패: {str(e)}")
            return {
                "success": False,
                "added_count": 0,
                "message": f"오류: {str(e)}"
            }
    
    def get_collection_count(self) -> int:
        """저장된 논문의 총 개수"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"컬렉션 크기 조회 실패: {str(e)}")
            return 0
    
    def _save_operation_log(self, doc_ids: List[str], operation: str, count: int):
        """작업 로그 저장"""
        try:
            log_file = self.log_dir / "operations.jsonl"
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "count": count,
                "doc_ids_sample": doc_ids[:5]
            }
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"로그 저장 실패: {str(e)}")


class SemanticSearchEngine:
    """
    의미 기반 검색 엔진
    
    이 클래스는 당신의 embeddings.py의 calculate_semantic_similarity를
    사용하여 논문들을 의미론적으로 평가합니다.
    
    workflow의 evaluate_relevance_node에서 이 엔진을 사용합니다.
    """
    
    def __init__(self, vectorstore: ArxivPaperVectorStore):
        """
        검색 엔진 초기화
        
        Args:
            vectorstore: ArxivPaperVectorStore 인스턴스
        """
        self.vectorstore = vectorstore
        self.min_similarity_threshold = 0.3  # 최소 유사도 임계값
    
    def evaluate_papers_semantic_relevance(
        self,
        papers: List[Dict],
        query: str,
        top_k: Optional[int] = None,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        검색된 논문들의 의미적 관련성을 평가합니다.
        
        이것이 workflow의 evaluate_relevance_node에서 사용되는 핵심 메서드입니다.
        
        Args:
            papers: 평가할 논문 리스트
            query: 사용자의 원래 질문 (의미 유사도 계산 기준)
            top_k: 반환할 상위 논문 개수 (None이면 모두 반환)
            threshold: 유사도 임계값 (이 이상인 논문만 반환)
        
        Returns:
            유사도 점수가 추가된 논문 리스트 (유사도 순 내림차순)
            
            구조:
            {
                'arxiv_id': '2401.00001',
                'title': 'Paper Title',
                'abstract': 'Paper abstract...',
                'authors': [...],
                'categories': [...],
                'semantic_score': 0.75,  # 추가된 필드
                'content_for_summary': '제목\n\n초록...'
            }
        """
        
        if not papers:
            logger.warning("평가할 논문이 없습니다")
            return []
        
        logger.info(f"의미 기반 평가 시작: {len(papers)}개 논문, 쿼리: {query[:50]}...")
        
        try:
            evaluated_papers = []
            
            for paper in papers:
                # 논문의 제목과 초록을 결합하여 평가 대상 텍스트 생성
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                paper_content = f"{title}\n\n{abstract}".strip()
                
                if not paper_content:
                    logger.warning(f"논문 {paper.get('arxiv_id')}: 평가 텍스트 없음")
                    continue
                
                # 당신의 calculate_semantic_similarity 함수 사용
                # 이 함수는 0~1 범위의 정규화된 유사도를 반환합니다
                try:
                    if calculate_semantic_similarity is not None:
                        semantic_score = calculate_semantic_similarity(query, paper_content)
                    else:
                        logger.warning("calculate_semantic_similarity를 로드할 수 없습니다")
                        semantic_score = 0.0
                except Exception as e:
                    logger.error(f"유사도 계산 실패 ({paper.get('arxiv_id')}): {str(e)}")
                    semantic_score = 0.0
                
                # 임계값 체크
                if semantic_score >= threshold:
                    # 평가된 논문을 리스트에 추가
                    evaluated_paper = paper.copy()
                    evaluated_paper['semantic_score'] = semantic_score
                    evaluated_paper['content_for_summary'] = paper_content
                    evaluated_papers.append(evaluated_paper)
                    
                    logger.debug(f"  {paper.get('arxiv_id')}: {semantic_score:.4f}")
            
            # 유사도 점수 기준으로 내림차순 정렬
            evaluated_papers.sort(
                key=lambda x: x.get('semantic_score', 0),
                reverse=True
            )
            
            # top_k 적용
            if top_k and len(evaluated_papers) > top_k:
                evaluated_papers = evaluated_papers[:top_k]
            
            logger.info(f"✓ 평가 완료: {len(evaluated_papers)}개 논문 선별 (임계값: {threshold})")
            
            return evaluated_papers
        
        except Exception as e:
            logger.error(f"의미 기반 평가 실패: {str(e)}")
            return []
    
    def vector_search(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], List[float]]:
        """
        VectorStore에서 벡터 검색을 수행합니다.
        
        이것은 ChromaDB의 벡터 검색 기능을 활용합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
        
        Returns:
            (검색 결과 논문 리스트, 유사도 점수 리스트)
        """
        
        try:
            logger.info(f"벡터 검색: {query[:50]}... (top_k={top_k})")
            
            # 쿼리 임베딩
            if embed_text is None:
                logger.error("embed_text를 로드할 수 없습니다")
                return [], []
            
            query_embedding = embed_text(query)
            
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # ChromaDB 검색
            results = self.vectorstore.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 처리
            papers = []
            similarities = []
            
            if results['ids'] and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1 - distance  # 거리를 유사도로 변환
                    
                    paper = {
                        'arxiv_id': metadata.get('arxiv_id', ''),
                        'title': metadata.get('title', ''),
                        'abstract': results['documents'][0][i] if results['documents'] else '',
                        'authors': metadata.get('authors', '').split(', '),
                        'categories': metadata.get('categories', '').split(', '),
                        'published_date': metadata.get('published_date', ''),
                        'pdf_url': metadata.get('pdf_url', ''),
                        'html_url': metadata.get('html_url', ''),
                        'vector_similarity': similarity
                    }
                    
                    papers.append(paper)
                    similarities.append(similarity)
            
            logger.info(f"✓ 벡터 검색 완료: {len(papers)}개 결과")
            
            return papers, similarities
        
        except Exception as e:
            logger.error(f"벡터 검색 실패: {str(e)}")
            return [], []


class WorkflowIntegration:
    """
    workflow와 VectorStore의 통합 레이어
    
    이 클래스는 workflow의 각 노드에서 사용할 수 있는
    편의 메서드들을 제공합니다.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/arxiv_vectorstore"
    ):
        """
        통합 레이어 초기화
        
        Args:
            persist_directory: 데이터 저장 디렉토리
        """
        self.vectorstore = ArxivPaperVectorStore(persist_directory)
        self.search_engine = SemanticSearchEngine(self.vectorstore)
        
        logger.info("✓ WorkflowIntegration 초기화 완료")
    
    def process_search_results_for_evaluation(
        self,
        arxiv_papers: List[Dict],
        original_query: str,
        num_papers_to_return: int = 3,
        similarity_threshold: float = 0.3
    ) -> Dict:
        """
        workflow의 evaluate_relevance_node에서 호출할 메인 메서드
        
        이 메서드는 다음을 수행합니다:
        1. arXiv API에서 검색된 논문들을 VectorStore에 저장
        2. 각 논문의 의미적 관련성 평가
        3. 관련성 높은 논문들만 선별하여 반환
        
        Args:
            arxiv_papers: arXiv API의 검색 결과
            original_query: 사용자의 원래 질문
            num_papers_to_return: 반환할 논문 개수
            similarity_threshold: 유사도 임계값
        
        Returns:
            {
                'success': bool,
                'relevant_papers': List[Dict],  # 의미 기반으로 선별된 논문
                'evaluation_details': Dict,      # 평가 결과 상세정보
                'message': str
            }
        """
        
        try:
            logger.info(f"[WORKFLOW_INTEGRATION] 평가 시작")
            logger.info(f"  - 검색 결과: {len(arxiv_papers)}개")
            logger.info(f"  - 원래 쿼리: {original_query[:50]}...")
            logger.info(f"  - 임계값: {similarity_threshold}")
            
            # 1단계: 논문들을 VectorStore에 추가
            add_result = self.vectorstore.add_papers_from_arxiv_search(arxiv_papers)
            
            if not add_result['success']:
                logger.warning(f"VectorStore 추가 실패: {add_result['message']}")
                # 그래도 계속 진행 (이미 저장된 논문들이 있을 수 있음)
            
            # 2단계: 의미 기반 평가 수행
            relevant_papers = self.search_engine.evaluate_papers_semantic_relevance(
                papers=arxiv_papers,
                query=original_query,
                top_k=num_papers_to_return,
                threshold=similarity_threshold
            )
            
            # 3단계: 결과 정리
            logger.info(f"✓ 평가 완료: {len(relevant_papers)}개 논문 선별")
            
            return {
                "success": True,
                "relevant_papers": relevant_papers,
                "evaluation_details": {
                    "total_papers_evaluated": len(arxiv_papers),
                    "papers_passed_threshold": len(relevant_papers),
                    "threshold_used": similarity_threshold,
                    "num_papers_returned": min(len(relevant_papers), num_papers_to_return)
                },
                "message": f"{len(relevant_papers)}개의 관련 논문을 찾았습니다"
            }
        
        except Exception as e:
            logger.error(f"[WORKFLOW_INTEGRATION] 평가 중 오류: {str(e)}")
            return {
                "success": False,
                "relevant_papers": [],
                "evaluation_details": {},
                "message": f"오류: {str(e)}"
            }
    
    def get_statistics(self) -> Dict:
        """VectorStore의 통계 정보"""
        return {
            "total_papers": self.vectorstore.get_collection_count(),
            "collection_name": self.vectorstore.collection_name,
            "persist_directory": self.vectorstore.persist_directory
        }


# 사용 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🔬 VectorStore와 Workflow 통합 예시")
    print("="*60 + "\n")
    
    # 1. 통합 레이어 초기화
    integration = WorkflowIntegration(
        persist_directory="./data/arxiv_vectorstore"
    )
    
    # 2. 샘플 arXiv 논문 (실제로는 search_papers_node에서 옴)
    sample_papers = [
        {
            'arxiv_id': '2401.00001',
            'title': 'Attention Mechanisms in Neural Networks',
            'abstract': 'This paper explores attention mechanisms and their role in modern deep learning.',
            'authors': ['John Smith', 'Jane Doe'],
            'categories': ['cs.LG', 'cs.AI'],
            'published_date': '2024-01-15',
            'pdf_url': 'https://arxiv.org/pdf/2401.00001',
            'html_url': 'https://arxiv.org/abs/2401.00001'
        },
        {
            'arxiv_id': '2401.00002',
            'title': 'Efficient Transformers',
            'abstract': 'We propose an efficient transformer architecture for real-world applications.',
            'authors': ['Alice Chen'],
            'categories': ['cs.LG'],
            'published_date': '2024-01-18',
            'pdf_url': 'https://arxiv.org/pdf/2401.00002',
            'html_url': 'https://arxiv.org/abs/2401.00002'
        }
    ]
    
    # 3. 의미 기반 평가 수행
    user_query = "attention mechanisms and efficiency in transformers"
    
    result = integration.process_search_results_for_evaluation(
        arxiv_papers=sample_papers,
        original_query=user_query,
        num_papers_to_return=2,
        similarity_threshold=0.3
    )
    
    print(f"평가 결과: {result['message']}")
    print(f"\n선별된 논문:")
    for i, paper in enumerate(result['relevant_papers'], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   유사도: {paper['semantic_score']:.4f}")
        print(f"   저자: {', '.join(paper['authors'])}")
    
    # 4. 통계
    stats = integration.get_statistics()
    print(f"\n📊 VectorStore 통계: {stats}")
"""
최종 PDF 처리 및 임베딩 파이프라인
당신의 embeddings.py와 vectorstore.py와 100% 호환

구조:
1. PDF 다운로드 및 텍스트 추출
2. 텍스트 청킹
3. 청크 임베딩 (배치 처리)
4. ChromaDB 저장

사용 방법:
from app.tools.embeddings import SentenceTransformerEmbedding
from app.tools.vectorstore import ArxivPaperVectorStore
from app.tools.pdf_embedding_pipeline import PDFEmbeddingPipeline

embedding_model = SentenceTransformerEmbedding()
vectorstore = ArxivPaperVectorStore()
pipeline = PDFEmbeddingPipeline(embedding_model, vectorstore)

# 논문 처리
result = pipeline.process_paper('2401.00001', paper_metadata)
"""

import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import os

logger = logging.getLogger(__name__)


class PDFDownloadAndExtract:
    """arXiv에서 PDF를 다운로드하고 텍스트를 추출합니다."""
    
    def __init__(self, cache_dir: str = "./data/arxiv_pdfs"):
        """
        초기화
        
        Args:
            cache_dir: PDF를 저장할 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF 추출 라이브러리 확인
        self.use_pdfplumber = self._check_pdfplumber()
        self.use_pypdf = self._check_pypdf()
        
        if not self.use_pdfplumber and not self.use_pypdf:
            logger.warning("⚠️ PDF 라이브러리 없음. 설치: pip install pdfplumber")
    
    def _check_pdfplumber(self) -> bool:
        """pdfplumber 사용 가능 여부 확인"""
        try:
            import pdfplumber
            logger.info("✓ pdfplumber 사용 가능")
            return True
        except ImportError:
            return False
    
    def _check_pypdf(self) -> bool:
        """PyPDF 사용 가능 여부 확인"""
        try:
            import pypdf
            logger.info("✓ PyPDF 사용 가능")
            return True
        except ImportError:
            return False
    
    def download_pdf(self, arxiv_id: str) -> Optional[str]:
        """
        arXiv에서 PDF를 다운로드합니다.
        
        Args:
            arxiv_id: arXiv 논문 ID (예: 2401.00001)
        
        Returns:
            PDF 파일 경로 또는 None
        """
        
        arxiv_id = arxiv_id.strip().replace('/', '')
        cache_file = self.cache_dir / f"{arxiv_id}.pdf"
        
        # 이미 다운로드되었으면 바로 반환
        if cache_file.exists():
            logger.debug(f"캐시된 PDF 사용: {arxiv_id}")
            return str(cache_file)
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            logger.info(f"PDF 다운로드: {arxiv_id}")
            
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            file_size = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ 다운로드 완료: {file_size:.2f}MB")
            
            return str(cache_file)
        
        except Exception as e:
            logger.error(f"다운로드 실패: {str(e)}")
            return None
    
    def extract_text(self, pdf_path: str, max_pages: Optional[int] = None) -> Optional[str]:
        """
        PDF에서 텍스트를 추출합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            max_pages: 최대 추출 페이지 (None이면 전체)
        
        Returns:
            추출된 텍스트 또는 None
        """
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일 없음: {pdf_path}")
            return None
        
        try:
            if self.use_pdfplumber:
                return self._extract_pdfplumber(pdf_path, max_pages)
            elif self.use_pypdf:
                return self._extract_pypdf(pdf_path, max_pages)
            else:
                logger.error("사용 가능한 PDF 라이브러리 없음")
                return None
        
        except Exception as e:
            logger.error(f"텍스트 추출 실패: {str(e)}")
            return None
    
    def _extract_pdfplumber(self, pdf_path: str, max_pages: Optional[int]) -> Optional[str]:
        """pdfplumber를 사용한 추출"""
        try:
            import pdfplumber
            
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_extract = total_pages if max_pages is None else min(max_pages, total_pages)
                
                logger.info(f"텍스트 추출 중: {pages_to_extract}/{total_pages} 페이지")
                
                for i, page in enumerate(pdf.pages[:pages_to_extract]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"페이지 {i+1} 추출 실패: {str(e)}")
            
            text = "\n\n".join(text_parts)
            logger.info(f"✓ 추출 완료: {len(text)} 글자")
            
            return text if text.strip() else None
        
        except Exception as e:
            logger.error(f"pdfplumber 실패: {str(e)}")
            return None
    
    def _extract_pypdf(self, pdf_path: str, max_pages: Optional[int]) -> Optional[str]:
        """PyPDF를 사용한 추출"""
        try:
            from pypdf import PdfReader
            
            text_parts = []
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            pages_to_extract = total_pages if max_pages is None else min(max_pages, total_pages)
            
            logger.info(f"텍스트 추출 중: {pages_to_extract}/{total_pages} 페이지")
            
            for i in range(pages_to_extract):
                try:
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"페이지 {i+1} 추출 실패: {str(e)}")
            
            text = "\n\n".join(text_parts)
            logger.info(f"✓ 추출 완료: {len(text)} 글자")
            
            return text if text.strip() else None
        
        except Exception as e:
            logger.error(f"PyPDF 실패: {str(e)}")
            return None


class SimpleTextChunker:
    """
    간단한 텍스트 청킹
    
    논문 텍스트를 의미 있는 크기의 청크로 나눕니다.
    """
    
    def __init__(self, chunk_chars: int = 1800, overlap_chars: int = 350):
        """
        초기화
        
        Args:
            chunk_chars: 청크의 목표 문자 수
                        (Sentence Transformers는 약 512 토큰을 처리할 수 있으며,
                         영어 기준 1 토큰 ≈ 4 문자이므로 chunk_chars ≈ 2000)
            overlap_chars: 청크 간 오버래프 문자 수
        """
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        logger.info(f"✓ TextChunker 초기화: {chunk_chars} 문자, {overlap_chars} 오버래프")
    
    def chunk(self, text: str, arxiv_id: str = "") -> List[Dict]:
        """
        텍스트를 청크로 나눕니다.
        
        Args:
            text: 분할할 텍스트
            arxiv_id: 논문 ID (메타데이터용)
        
        Returns:
            청크 리스트:
            {
                'chunk_id': str,
                'content': str,
                'chunk_index': int
            }
        """
        
        if not text or not text.strip():
            return []
        
        # 문장으로 분리 (간단한 방식)
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if not sentence:
                continue
            
            # 현재 청크에 문장을 추가했을 때의 길이 계산
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # 청크 크기 초과 시 저장
            if len(test_chunk) > self.chunk_chars and current_chunk:
                chunks.append({
                    'chunk_id': f"{arxiv_id}_chunk_{chunk_index}" if arxiv_id else f"chunk_{chunk_index}",
                    'content': current_chunk.strip(),
                    'chunk_index': chunk_index
                })
                
                chunk_index += 1
                
                # 오버래프를 위해 이전 내용의 일부 유지
                sentences_in_chunk = current_chunk.split('. ')
                if len(sentences_in_chunk) > 1:
                    overlap_text = '. '.join(sentences_in_chunk[-2:])
                else:
                    overlap_text = sentences_in_chunk[0]
                
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = test_chunk
        
        # 마지막 청크 저장
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{arxiv_id}_chunk_{chunk_index}" if arxiv_id else f"chunk_{chunk_index}",
                'content': current_chunk.strip(),
                'chunk_index': chunk_index
            })
        
        logger.info(f"✓ 청킹 완료: {len(chunks)}개 청크")
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """간단한 문장 분리"""
        import re
        
        # 마침표, 느낌표, 물음표로 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return sentences


class PDFEmbeddingPipeline:
    """
    완전한 PDF 임베딩 파이프라인
    
    당신의 SentenceTransformerEmbedding과 ArxivPaperVectorStore와 호환됩니다.
    """
    
    def __init__(
        self,
        embedding_model,  # SentenceTransformerEmbedding 인스턴스
        vectorstore,  # ArxivPaperVectorStore 인스턴스
        chunk_chars: int = 1800,
        overlap_chars: int = 350,
        batch_size: int = 32
    ):
        """
        초기화
        
        Args:
            embedding_model: SentenceTransformerEmbedding 인스턴스
            vectorstore: ArxivPaperVectorStore 인스턴스
            chunk_chars: 청크 문자 수
            overlap_chars: 오버래프 문자 수
            batch_size: 배치 처리 크기
        """
        
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.batch_size = batch_size
        
        self.pdf_processor = PDFDownloadAndExtract()
        self.chunker = SimpleTextChunker(
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars
        )
        
        logger.info("✓ PDFEmbeddingPipeline 초기화 완료")
    
    def process_paper(
        self,
        arxiv_id: str,
        paper_metadata: Dict,
        max_pages: int = 10
    ) -> Dict:
        """
        단일 논문을 처리합니다.
        
        Args:
            arxiv_id: arXiv ID
            paper_metadata: 논문 메타데이터
            max_pages: 최대 처리 페이지
        
        Returns:
            처리 결과:
            {
                'success': bool,
                'chunks_created': int,
                'chunks_embedded': int,
                'chunks_saved': int,
                'message': str,
                'time': float
            }
        """
        
        start_time = time.time()
        
        logger.info("="*60)
        logger.info(f"[처리 시작] {arxiv_id}")
        logger.info("="*60)
        
        try:
            # 1단계: PDF 다운로드
            logger.info("1단계: PDF 다운로드 및 텍스트 추출...")
            
            pdf_path = self.pdf_processor.download_pdf(arxiv_id)
            
            if not pdf_path:
                return {
                    "success": False,
                    "arxiv_id": arxiv_id,
                    "chunks_created": 0,
                    "chunks_embedded": 0,
                    "chunks_saved": 0,
                    "message": "PDF 다운로드 실패",
                    "time": time.time() - start_time
                }
            
            # 2단계: 텍스트 추출
            text = self.pdf_processor.extract_text(pdf_path, max_pages=max_pages)
            
            if not text:
                return {
                    "success": False,
                    "arxiv_id": arxiv_id,
                    "chunks_created": 0,
                    "chunks_embedded": 0,
                    "chunks_saved": 0,
                    "message": "텍스트 추출 실패",
                    "time": time.time() - start_time
                }
            
            logger.info(f"텍스트 추출 완료: {len(text)} 글자")
            
            # 3단계: 청킹
            logger.info("2단계: 텍스트 청킹...")
            
            chunks = self.chunker.chunk(text, arxiv_id=arxiv_id)
            
            if not chunks:
                return {
                    "success": False,
                    "arxiv_id": arxiv_id,
                    "chunks_created": 0,
                    "chunks_embedded": 0,
                    "chunks_saved": 0,
                    "message": "청킹 실패",
                    "time": time.time() - start_time
                }
            
            # 4단계: 임베딩 (배치 처리)
            logger.info("3단계: 청크 임베딩 (배치 처리)...")
            
            chunk_texts = [chunk['content'] for chunk in chunks]
            
            try:
                # 당신의 embedding_model의 embed_batch 사용
                embeddings = self.embedding_model.embed_batch(chunk_texts)
                logger.info(f"✓ {len(embeddings)}개 청크 임베딩 완료")
            except Exception as e:
                logger.error(f"임베딩 실패: {str(e)}")
                return {
                    "success": False,
                    "arxiv_id": arxiv_id,
                    "chunks_created": len(chunks),
                    "chunks_embedded": 0,
                    "chunks_saved": 0,
                    "message": f"임베딩 실패: {str(e)}",
                    "time": time.time() - start_time
                }
            
            # 5단계: ChromaDB 저장
            logger.info("4단계: ChromaDB에 저장...")
            
            saved_count = self._save_to_vectorstore(
                chunks,
                embeddings,
                arxiv_id,
                paper_metadata
            )
            
            elapsed = time.time() - start_time
            
            logger.info(f"✓ 처리 완료: {saved_count}개 청크 저장 ({elapsed:.2f}초)")
            
            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "chunks_created": len(chunks),
                "chunks_embedded": len(embeddings),
                "chunks_saved": saved_count,
                "message": f"{saved_count}개 청크 저장 완료",
                "time": elapsed
            }
        
        except Exception as e:
            logger.error(f"처리 중 오류: {str(e)}", exc_info=True)
            
            return {
                "success": False,
                "arxiv_id": arxiv_id,
                "chunks_created": 0,
                "chunks_embedded": 0,
                "chunks_saved": 0,
                "message": f"오류: {str(e)}",
                "time": time.time() - start_time
            }
    
    def _save_to_vectorstore(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        arxiv_id: str,
        paper_metadata: Dict
    ) -> int:
        """
        청크와 임베딩을 ChromaDB에 저장합니다.
        당신의 vectorstore.collection을 직접 사용합니다.
        """
        
        try:
            ids = [chunk['chunk_id'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            
            # 메타데이터 준비
            metadatas = []
            for chunk in chunks:
                metadata = {
                    'arxiv_id': arxiv_id,
                    'chunk_index': str(chunk['chunk_index']),
                    'title': paper_metadata.get('title', ''),
                    'authors': ', '.join(paper_metadata.get('authors', [])),
                }
                metadatas.append(metadata)
            
            # ChromaDB 컬렉션에 직접 저장
            self.vectorstore.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"✓ {len(ids)}개 청크 저장 완료")
            
            return len(ids)
        
        except Exception as e:
            logger.error(f"저장 실패: {str(e)}")
            return 0
    
    def process_papers_batch(
        self,
        papers: List[Dict],
        max_pages: int = 10
    ) -> Dict:
        """
        여러 논문을 처리합니다.
        
        Args:
            papers: 논문 리스트
                각 항목:
                {
                    'arxiv_id': str,
                    'title': str,
                    'authors': List[str],
                    ...
                }
            max_pages: 최대 처리 페이지
        
        Returns:
            배치 처리 결과
        """
        
        logger.info("="*60)
        logger.info(f"[배치 처리 시작] {len(papers)}개 논문")
        logger.info("="*60)
        
        results = []
        successful = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, paper in enumerate(papers):
            arxiv_id = paper.get('arxiv_id')
            
            logger.info(f"\n[{i+1}/{len(papers)}] {arxiv_id} 처리 중...")
            
            result = self.process_paper(
                arxiv_id=arxiv_id,
                paper_metadata=paper,
                max_pages=max_pages
            )
            
            results.append(result)
            
            if result['success']:
                successful += 1
                total_chunks += result['chunks_saved']
            
            # API 요청 사이에 잠깐 대기 (arXiv 서버 부하 고려)
            if i < len(papers) - 1:
                time.sleep(2)
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("[배치 처리 완료]")
        logger.info("="*60)
        logger.info(f"성공: {successful}/{len(papers)}")
        logger.info(f"총 저장된 청크: {total_chunks}")
        logger.info(f"총 소요 시간: {total_time:.2f}초")
        
        return {
            "total": len(papers),
            "successful": successful,
            "failed": len(papers) - successful,
            "total_chunks": total_chunks,
            "time": total_time,
            "results": results,
            "message": f"{successful}개 논문 처리 완료, {total_chunks}개 청크 저장"
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🚀 PDF 임베딩 파이프라인 테스트")
    print("="*60 + "\n")
    
    try:
        from embeddings import SentenceTransformerEmbedding
        from vectorstore import ArxivPaperVectorStore
        
        # 초기화
        embedding_model = SentenceTransformerEmbedding(
            model_name="distiluse-base-multilingual-cased-v2"
        )
        
        vectorstore = ArxivPaperVectorStore(
            persist_directory="./data/arxiv_chunks",
            collection_name="arxiv_chunks"
        )
        
        pipeline = PDFEmbeddingPipeline(
            embedding_model=embedding_model,
            vectorstore=vectorstore
        )
        
        # 테스트
        test_papers = [
            {
                'arxiv_id': '2401.01111',
                'title': 'Test Paper',
                'authors': ['Author1']
            }
        ]
        
        result = pipeline.process_papers_batch(test_papers, max_pages=2)
        print(f"\n결과: {result['message']}")
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        print("embeddings.py와 vectorstore.py가 필요합니다")
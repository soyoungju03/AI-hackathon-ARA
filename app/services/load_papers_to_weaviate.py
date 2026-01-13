# -*- coding: utf-8 -*-
"""
Weaviate 초기 데이터 로드 스크립트
=================================

이 스크립트는 arXiv에서 논문들을 검색하여 Weaviate 데이터베이스에 저장합니다.

사용법:
    python load_papers_to_weaviate.py --keywords "machine learning" "deep learning" --count 50

또는 기본값으로 실행:
    python load_papers_to_weaviate.py
"""

import logging
import argparse
import sys
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_papers_from_arxiv(
    keywords: List[str],
    count_per_keyword: int = 50
) -> int:
    """
    arXiv에서 논문을 검색하여 Weaviate에 저장합니다.
    
    Args:
        keywords: 검색할 키워드 리스트
        count_per_keyword: 각 키워드당 검색할 논문 개수
    
    Returns:
        저장된 논문의 총 개수
    """
    
    try:
        import arxiv
        from app.services.weaviate_client import WeaviateClient, Paper
        
        logger.info("=" * 70)
        logger.info("🚀 Weaviate 초기 데이터 로드 시작")
        logger.info("=" * 70)
        
        # Weaviate 클라이언트 초기화
        try:
            weaviate_client = WeaviateClient(use_embedded=True)
            logger.info("✓ Weaviate 연결 성공")
        except Exception as e:
            logger.error(f"✗ Weaviate 연결 실패: {str(e)}")
            return 0
        
        total_saved = 0
        
        # 각 키워드별로 검색
        for keyword in keywords:
            logger.info(f"\n📝 '{keyword}'로 검색 중...")
            
            try:
                client = arxiv.Client()
                
                # arXiv 검색 쿼리 작성
                search = arxiv.Search(
                    query=f"cat:cs.AI OR cat:cs.LG OR title:{keyword}",
                    max_results=count_per_keyword,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                papers_to_save = []
                
                for entry in client.results(search):
                    try:
                        # 논문 정보 추출
                        arxiv_id = entry.entry_id.split('/abs/')[-1]
                        
                        paper = Paper(
                            title=entry.title,
                            authors=[author.name for author in entry.authors],
                            abstract=entry.summary,
                            arxiv_id=arxiv_id,
                            url=entry.pdf_url,
                            published_date=entry.published.isoformat(),
                            keywords=[keyword]
                        )
                        
                        papers_to_save.append(paper)
                        
                    except Exception as e:
                        logger.debug(f"논문 처리 실패: {str(e)}")
                        continue
                
                # 배치로 저장
                if papers_to_save:
                    saved_count = weaviate_client.add_papers_batch(papers_to_save)
                    total_saved += saved_count
                    logger.info(f"✓ '{keyword}': {saved_count}/{len(papers_to_save)}개 저장됨")
                
            except Exception as e:
                logger.error(f"✗ '{keyword}' 검색 중 오류: {str(e)}")
                continue
        
        # 최종 통계
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ 로드 완료!")
        logger.info(f"  총 {total_saved}개 논문이 Weaviate에 저장되었습니다.")
        logger.info(f"  현재 데이터베이스 크기: {weaviate_client.get_paper_count()}개 논문")
        logger.info("=" * 70)
        
        return total_saved
    
    except ImportError as e:
        logger.error(f"필수 패키지 누락: {str(e)}")
        logger.info("설치: pip install arxiv weaviate-client")
        return 0
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return 0


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(
        description="Weaviate에 arXiv 논문 로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본값으로 실행 (기본 키워드, 50개 논문)
  python load_papers_to_weaviate.py

  # 커스텀 키워드와 개수
  python load_papers_to_weaviate.py \\
    --keywords "machine learning" "deep learning" "neural networks" \\
    --count 100

  # 빠른 테스트 (적은 개수)
  python load_papers_to_weaviate.py --count 5
        """
    )
    
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=['machine learning', 'deep learning', 'neural networks', 'natural language processing'],
        help='검색할 키워드들 (기본값: machine learning, deep learning, etc.)'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help='각 키워드당 검색할 논문 개수 (기본값: 50)'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='시작 전에 Weaviate의 기존 데이터 모두 삭제'
    )
    
    args = parser.parse_args()
    
    # 기존 데이터 삭제 옵션
    if args.clear:
        logger.warning("⚠️  기존 Weaviate 데이터를 삭제합니다...")
        
        try:
            from app.services.weaviate_client import WeaviateClient
            client = WeaviateClient(use_embedded=True)
            
            if client.clear_all():
                logger.info("✓ 기존 데이터 삭제 완료")
            else:
                logger.error("✗ 삭제 실패")
                return 1
        
        except Exception as e:
            logger.error(f"✗ 삭제 중 오류: {str(e)}")
            return 1
    
    # 논문 로드
    result = load_papers_from_arxiv(
        keywords=args.keywords,
        count_per_keyword=args.count
    )
    
    return 0 if result > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
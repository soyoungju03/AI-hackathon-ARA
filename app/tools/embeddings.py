# app/tools/embeddings.py
"""
임베딩 기반 의미 유사도 계산 모듈

Sentence Transformers를 사용하여 텍스트를 벡터로 변환하고,
코사인 유사도를 계산합니다.
"""

import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 전역 모델 인스턴스 (싱글톤 패턴)
_embedding_model = None


def get_embedding_model():
    """
    Sentence Transformer 모델을 로드합니다.
    
    all-MiniLM-L6-v2 모델을 사용합니다:
    - 빠른 속도 (CPU에서도 잘 작동)
    - 384차원 벡터 생성
    - 영어에 최적화되어 있지만 다국어도 어느 정도 지원
    """
    global _embedding_model
    
    if _embedding_model is None:
        logger.info("Sentence Transformer 모델 로딩 중...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("모델 로딩 완료")
    
    return _embedding_model


def embed_text(text: str) -> np.ndarray:
    """
    텍스트를 384차원 벡터로 변환합니다.
    
    Args:
        text: 임베딩할 텍스트
        
    Returns:
        384차원의 numpy 배열
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    두 벡터 사이의 코사인 유사도를 계산합니다.
    
    코사인 유사도는 -1에서 1 사이의 값을 가집니다:
    - 1에 가까울수록 매우 유사함
    - 0에 가까우면 관련성이 적음
    - -1에 가까우면 반대 의미
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
        
    Returns:
        -1.0에서 1.0 사이의 유사도 점수
    """
    # 벡터의 내적을 벡터 크기의 곱으로 나눔
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 0으로 나누는 것을 방지
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def calculate_semantic_similarity(query: str, document: str) -> float:
    """
    질문과 문서 사이의 의미적 유사도를 계산합니다.
    
    이것은 embed_text와 cosine_similarity를 결합한 편의 함수입니다.
    
    Args:
        query: 사용자 질문
        document: 논문 제목 또는 초록
        
    Returns:
        0.0에서 1.0 사이의 유사도 점수
    """
    try:
        query_embedding = embed_text(query)
        doc_embedding = embed_text(document)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        
        # -1~1 범위를 0~1 범위로 정규화
        normalized_similarity = (similarity + 1) / 2
        return normalized_similarity
        
    except Exception as e:
        logger.error(f"유사도 계산 중 오류: {str(e)}")
        return 0.0
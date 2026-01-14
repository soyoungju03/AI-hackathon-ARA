"""
임베딩 모듈
텍스트를 벡터로 변환하여 의미론적 검색을 가능하게 합니다.
"""

from typing import List, Union
import numpy as np
from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """임베딩 모델의 추상 기본 클래스"""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩으로 변환"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩 변환"""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformers를 사용한 임베딩"""
    
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v2"):
        """
        SentenceTransformer 모델 초기화
        
        Args:
            model_name: 사용할 모델 이름
                - "all-MiniLM-L6-v2": 빠름, 한국어 지원 제한적
                - "distiluse-base-multilingual-cased-v2": 다국어 지원 (한국어 포함)
                - "paraphrase-multilingual-MiniLM-L12-v2": 의미론적으로 유사한 문장 인식
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"🔄 '{model_name}' 모델 로딩 중...")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            print(f"✓ 임베딩 모델 로드 완료")
            print(f"  - 모델: {model_name}")
            print(f"  - 차원: {self.model.get_sentence_embedding_dimension()}")
        
        except ImportError:
            raise ImportError(
                "sentence-transformers 패키지가 필요합니다. "
                "설치: pip install sentence-transformers"
            )
    
    def embed(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩으로 변환"""
        if not text or not text.strip():
            # 빈 문자열의 경우 영벡터 반환
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩 변환 (더 효율적)"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 반환"""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI의 임베딩 API를 사용"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        OpenAI 임베딩 초기화
        
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델
                - "text-embedding-3-small": 1536차원, 저렴
                - "text-embedding-3-large": 3072차원, 고성능
        """
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=api_key)
            self.model = model
            print(f"✓ OpenAI 임베딩 초기화 완료 (모델: {model})")
        
        except ImportError:
            raise ImportError(
                "openai 패키지가 필요합니다. "
                "설치: pip install openai"
            )
    
    def embed(self, text: str) -> List[float]:
        """OpenAI API를 이용한 단일 임베딩"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """OpenAI API를 이용한 배치 임베딩"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        # 응답 순서가 입력 순서와 같음을 보장
        return [item.embedding for item in response.data]


class CachedEmbedding(EmbeddingModel):
    """임베딩을 캐싱하여 성능을 개선하는 래퍼"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        캐싱 래퍼 초기화
        
        Args:
            embedding_model: 실제 임베딩을 수행할 모델
        """
        self.model = embedding_model
        self.cache = {}
        print("✓ 캐싱 임베딩 활성화")
    
    def embed(self, text: str) -> List[float]:
        """캐시를 이용한 임베딩 (동일 텍스트는 재계산 안함)"""
        # 텍스트를 정규화하여 캐시 키로 사용
        cache_key = text.strip().lower()
        
        if cache_key not in self.cache:
            self.cache[cache_key] = self.model.embed(text)
        
        return self.cache[cache_key]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 (캐시 활용)"""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # 캐시되지 않은 텍스트만 필터링
        for idx, text in enumerate(texts):
            cache_key = text.strip().lower()
            
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(idx)
                embeddings.append(None)
        
        # 캐시되지 않은 텍스트들을 배치로 임베딩
        if texts_to_embed:
            new_embeddings = self.model.embed_batch(texts_to_embed)
            
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
                cache_key = texts[idx].strip().lower()
                self.cache[cache_key] = embedding
        
        return embeddings
    
    def get_cache_stats(self) -> dict:
        """캐시 통계"""
        return {
            "cached_items": len(self.cache),
            "model": str(self.model)
        }


# 사용 예시 함수
def example_usage():
    """임베딩 모듈 사용 예시"""
    
    # 1. Sentence Transformers 사용 (권장)
    print("="*50)
    print("1. Sentence Transformers 임베딩")
    print("="*50)
    
    embedding_model = SentenceTransformerEmbedding(
        model_name="distiluse-base-multilingual-cased-v2"
    )
    
    # 2. 단일 텍스트 임베딩
    text = "This is a sample text about machine learning"
    embedding = embedding_model.embed(text)
    print(f"\n📝 텍스트: {text}")
    print(f"📊 임베딩 차원: {len(embedding)}")
    print(f"📊 임베딩 (처음 5개): {embedding[:5]}")
    
    # 3. 배치 임베딩 (여러 문서)
    print("\n" + "="*50)
    print("2. 배치 임베딩")
    print("="*50)
    
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing helps computers understand text",
        "Deep learning uses neural networks with multiple layers"
    ]
    
    embeddings = embedding_model.embed_batch(texts)
    print(f"\n✓ {len(texts)}개 텍스트 임베딩 완료")
    print(f"📊 각 임베딩 차원: {len(embeddings[0])}")
    
    # 4. 캐싱 활용 (반복되는 텍스트가 있을 때 효율적)
    print("\n" + "="*50)
    print("3. 캐싱 임베딩 (성능 개선)")
    print("="*50)
    
    cached_embedding = CachedEmbedding(embedding_model)
    
    # 같은 텍스트를 여러 번 임베딩
    test_text = "Artificial intelligence and machine learning"
    
    import time
    
    start = time.time()
    emb1 = cached_embedding.embed(test_text)
    time1 = time.time() - start
    
    start = time.time()
    emb2 = cached_embedding.embed(test_text)  # 캐시에서 가져옴
    time2 = time.time() - start
    
    print(f"\n첫 번째 임베딩: {time1*1000:.2f}ms")
    print(f"두 번째 임베딩 (캐시): {time2*1000:.2f}ms")
    print(f"속도 향상: {time1/time2:.1f}배")
    
    # 캐시 통계
    stats = cached_embedding.get_cache_stats()
    print(f"\n📊 캐시 통계: {stats}")


if __name__ == "__main__":
    example_usage()
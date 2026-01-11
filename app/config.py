# AI Research Assistant - 학술 논문 기반 지능형 연구 도우미
# 환경 설정 파일

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    애플리케이션 설정을 관리하는 클래스입니다.
    환경 변수에서 값을 자동으로 읽어옵니다.
    
    Pydantic의 BaseSettings를 사용하면 다음과 같은 이점이 있습니다:
    1. 환경 변수 자동 로드
    2. 타입 검증
    3. 기본값 설정
    """
    
    # === API Keys ===
    # OpenAI API 키 - GPT 모델 사용을 위해 필요합니다
    openai_api_key: str = ""
    
    # Anthropic API 키 (선택) - Claude 모델 사용 시 필요합니다
    anthropic_api_key: Optional[str] = None
    
    # Tavily API 키 (선택) - 웹 검색 기능에 사용됩니다
    tavily_api_key: Optional[str] = None
    
    # === Model Settings ===
    # 기본 LLM 모델 - 질문 분석 및 요약에 사용됩니다
    default_model: str = "gpt-4o"
    
    # 경량 모델 - 간단한 작업(키워드 추출 등)에 사용됩니다
    light_model: str = "gpt-4o-mini"
    
    # 임베딩 모델 - 문서 벡터화에 사용됩니다
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # === Vector Database ===
    # Weaviate 연결 URL
    weaviate_url: str = "http://localhost:8080"
    
    # === Server Settings ===
    # 서버 호스트
    host: str = "0.0.0.0"
    
    # 서버 포트
    port: int = 7860
    
    # 디버그 모드
    debug: bool = False
    
    # === Search Settings ===
    # 기본 검색 논문 수
    default_paper_count: int = 3
    
    # 최대 검색 논문 수
    max_paper_count: int = 10
    
    # 연관성 점수 임계값 (0-1 사이, 이 값 이상만 결과에 포함)
    relevance_threshold: float = 0.5
    
    class Config:
        """Pydantic 설정"""
        # .env 파일에서 환경 변수를 읽습니다
        env_file = ".env"
        # 대소문자를 구분하지 않습니다 (OPENAI_API_KEY == openai_api_key)
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    설정 객체를 반환합니다.
    
    @lru_cache() 데코레이터를 사용하면 이 함수가 처음 호출될 때만 
    Settings 객체가 생성되고, 이후에는 캐시된 객체를 반환합니다.
    이렇게 하면 매번 .env 파일을 읽지 않아도 됩니다.
    
    Returns:
        Settings: 애플리케이션 설정 객체
    """
    return Settings()

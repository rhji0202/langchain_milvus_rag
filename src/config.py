"""설정 관리 모듈"""
import os
import sys

# Windows에서 한글 출력을 위한 인코딩 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class Config:
    """애플리케이션 설정 클래스"""
    
    # API Keys
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-477e1de110644a7ca8273bca60639449")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    
    JINAAI_API_KEY: str = os.getenv(
        "JINAAI_API_KEY", 
        "jina_46a8336db03c480e830c45cca221a9ccMSIy8vbCNvnvQTyfIozEG0kLHj6S"
    )
    
    # Milvus 설정
    # 참조: https://milvus.io/docs/ko/build_RAG_with_milvus_and_deepseek.md
    # URI 설정 옵션:
    # - 로컬 파일: "./milvus_demo.db" (Milvus Lite 자동 사용)
    # - 서버: "http://localhost:19530"
    # - Zilliz Cloud: Public Endpoint URL
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    COLLECTION_NAME: str = "my_rag_collection"  # 문서 패턴: "my_rag_collection"
    
    # 문서 경로
    DOCS_PATHS: list[str] = [
        "milvus_docs/en/faq/*.md",
        "milvus_docs/en/embeddings/*.md",
    ]
    
    # 텍스트 스플리터 설정
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # RAG 설정
    SEARCH_K: int = 3  # 검색 결과 개수
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.0
    
    # Jina Embedding 설정
    JINA_MODEL_NAME: str = "jina-embeddings-v4"
    JINA_LATE_CHUNKING: bool = False
    
    @classmethod
    def validate(cls) -> bool:
        """필수 설정값 검증"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 환경 변수가 설정되지 않았습니다.")
        if not cls.JINAAI_API_KEY:
            raise ValueError("JINAAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return True


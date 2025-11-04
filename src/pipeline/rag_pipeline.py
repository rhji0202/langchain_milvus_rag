"""RAG 파이프라인 통합 모듈"""
from typing import List, Optional
from langchain_deepseek  import ChatDeepSeek

from ..config import Config
from ..loaders import MarkdownDocumentLoader
from ..embeddings import JinaEmbeddings
from ..vector_store import MilvusVectorStore
from ..rag import RAGChain


class RAGPipeline:
    """전체 RAG 파이프라인 클래스"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        rebuild_index: bool = False,
    ):
        """
        Args:
            config: 설정 객체
            rebuild_index: 인덱스 재구성 여부
        """
        self.config = config or Config()
        self.config.validate()
        
        # 임베딩 모델 초기화
        self.embeddings = JinaEmbeddings(
            model_name=self.config.JINA_MODEL_NAME,
            api_key=self.config.JINAAI_API_KEY,
            late_chunking=self.config.JINA_LATE_CHUNKING,
        )
        
        # 벡터 스토어 초기화
        self.vector_store = MilvusVectorStore(
            uri=self.config.MILVUS_URI,
            collection_name=self.config.COLLECTION_NAME,
            embedding=self.embeddings,
            drop_old=rebuild_index,
        )
        
        # DeepSeek LLM 클라이언트 초기화 (OpenAI 호환 API 사용)
        self.llm_client = ChatDeepSeek(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
        )
        
        # 문서 로더 초기화
        self.document_loader = MarkdownDocumentLoader(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )
        
        self.rag_chain: Optional[RAGChain] = None
    
    def build_index(self, file_patterns: Optional[List[str]] = None):
        """
        벡터 인덱스 구축
        
        Args:
            file_patterns: 문서 파일 패턴 리스트
        """
        if file_patterns is None:
            file_patterns = self.config.DOCS_PATHS
        
        print("문서 로딩 중...")
        documents = self.document_loader.load_documents(file_patterns)
        print(f"로드된 문서 수: {len(documents)}")
        
        print("벡터 스토어 생성 중...")
        self.vector_store.create_from_documents(documents)
        print("벡터 스토어 생성 완료!")
    
    def initialize_rag_chain(self, search_k: Optional[int] = None):
        """
        RAG 체인 초기화
        
        Args:
            search_k: 검색 결과 개수
        """
        if search_k is None:
            search_k = self.config.SEARCH_K
        
        print(f"[디버깅] RAG 체인 초기화 시작 (search_k={search_k})...")
        self.rag_chain = RAGChain(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            model_name=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            search_kwargs={"k": search_k},
        )
        print("[디버깅] RAG 체인 초기화 완료")
    
    def query(self, question: str) -> str:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
        
        Returns:
            생성된 답변
        """
        if self.rag_chain is None:
            self.initialize_rag_chain()
        
        print(f"[디버깅] 질문 처리 시작: {question}")
        print("[디버깅] 검색 단계 실행 중...")
        answer = self.rag_chain.invoke(question)
        print("[디버깅] 답변 생성 완료")
        return answer


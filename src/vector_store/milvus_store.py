"""Milvus 벡터 스토어 관리 모듈"""
from typing import List, Optional
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient


class MilvusVectorStore:
    """Milvus 벡터 스토어 래퍼 클래스"""
    
    def __init__(
        self,
        uri: str,
        collection_name: str,
        embedding: Embeddings,
        drop_old: bool = False,
        metric_type: str = "IP",  # Inner product distance
        consistency_level: str = "Bounded",  # 문서 패턴: Bounded 사용 (Strong, Session, Bounded, Eventually)
    ):
        """
        Args:
            uri: Milvus URI (로컬 파일 경로 또는 서버 URI)
            collection_name: 컬렉션 이름
            embedding: 임베딩 모델
            drop_old: 기존 컬렉션 삭제 여부
            metric_type: 거리 메트릭 타입 (IP: Inner Product, L2: Euclidean)
            consistency_level: 일관성 레벨 (Strong, Session, Bounded, Eventually)
                - 문서 패턴: Bounded 사용 (https://milvus.io/docs/ko/build_RAG_with_milvus_and_deepseek.md)
        """
        self.uri = uri
        self.collection_name = collection_name
        self.embedding = embedding
        self.drop_old = drop_old
        self.metric_type = metric_type
        self.consistency_level = consistency_level
        self.vectorstore: Optional[Milvus] = None
        self.milvus_client: Optional[MilvusClient] = None
    
    def create_from_documents(self, documents: List[Document]) -> Milvus:
        """
        문서로부터 벡터 스토어 생성
        문서 패턴 참조: https://milvus.io/docs/ko/build_RAG_with_milvus_and_deepseek.md
        
        Args:
            documents: Document 리스트
        
        Returns:
            Milvus 벡터 스토어 인스턴스
        """
        # 문서 패턴: MilvusClient를 사용하여 컬렉션 존재 여부 확인 및 처리
        # 참조: https://milvus.io/docs/ko/build_RAG_with_milvus_and_deepseek.md
        self.milvus_client = MilvusClient(uri=self.uri)
        
        # 문서 패턴: 컬렉션이 존재하면 drop_old에 따라 처리
        if self.milvus_client.has_collection(self.collection_name):
            if self.drop_old:
                self.milvus_client.drop_collection(self.collection_name)
                print(f"기존 컬렉션 '{self.collection_name}' 삭제됨")
            else:
                print(f"기존 컬렉션 '{self.collection_name}' 사용 중")
        
        # LangChain Milvus를 사용하여 문서 로드
        # LangChain이 내부적으로 컬렉션을 생성하고 데이터를 삽입
        # 문서 패턴: dimension, metric_type="IP", consistency_level="Bounded" 사용
        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embedding,
            connection_args={"uri": self.uri},
            collection_name=self.collection_name,
            drop_old=self.drop_old,
        )
        return self.vectorstore
    
    def load_existing(self) -> Milvus:
        """
        기존 벡터 스토어 로드
        문서 패턴 참조: https://milvus.io/docs/ko/build_RAG_with_milvus_and_deepseek.md
        
        Returns:
            Milvus 벡터 스토어 인스턴스
        
        Raises:
            ValueError: 컬렉션이 존재하지 않는 경우
        """
        print(f"[디버깅] load_existing 시작 (uri={self.uri}, collection={self.collection_name})")
        # 문서 패턴: MilvusClient를 사용하여 컬렉션 존재 여부 확인
        print("[디버깅] MilvusClient 생성 중...")
        self.milvus_client = MilvusClient(uri=self.uri)
        print("[디버깅] 컬렉션 존재 여부 확인 중...")
        
        if not self.milvus_client.has_collection(self.collection_name):
            raise ValueError(
                f"컬렉션 '{self.collection_name}'이 존재하지 않습니다. "
                "먼저 build_index()를 실행하여 인덱스를 구축하세요."
            )
        
        print("[디버깅] 컬렉션 존재 확인됨. Milvus.from_documents 호출 중...")
        # 기존 컬렉션을 로드할 때는 from_documents를 빈 리스트로 사용
        # embed_documents()에서 빈 리스트 처리를 이미 했으므로 안전함
        # drop_old=False로 설정하여 기존 컬렉션을 유지
        self.vectorstore = Milvus.from_documents(
            documents=[],  # 빈 문서 리스트로 기존 컬렉션에 연결
            embedding=self.embedding,
            connection_args={"uri": self.uri},
            collection_name=self.collection_name,
            drop_old=False,  # 기존 컬렉션 유지
        )
        print("[디버깅] Milvus.from_documents 완료")
        return self.vectorstore
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        검색기(Retriever) 가져오기
        
        Args:
            search_kwargs: 검색 옵션
        
        Returns:
            Retriever 인스턴스
        """
        print("[디버깅] get_retriever 호출됨")
        if self.vectorstore is None:
            print("[디버깅] vectorstore가 None이므로 load_existing() 호출...")
            self.load_existing()
            print("[디버깅] load_existing() 완료")
        
        print("[디버깅] as_retriever 호출 중...")
        if search_kwargs:
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        else:
            retriever = self.vectorstore.as_retriever()
        print("[디버깅] as_retriever 완료")
        return retriever
    
    def similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            **kwargs: 추가 검색 옵션
        
        Returns:
            검색된 Document 리스트
        """
        if self.vectorstore is None:
            self.load_existing()
        
        return self.vectorstore.similarity_search(query, k=k, **kwargs)


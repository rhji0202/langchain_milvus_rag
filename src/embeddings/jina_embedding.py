"""Jina Embedding을 LangChain과 호환되도록 래핑"""
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from pymilvus.model.dense import JinaEmbeddingFunction


class JinaEmbeddings(Embeddings):
    """Jina Embedding을 LangChain Embeddings 인터페이스로 래핑"""
    
    def __init__(
        self,
        model_name: str = "jina-embeddings-v4",
        api_key: Optional[str] = None,
        late_chunking: bool = False,
    ):
        """
        Args:
            model_name: Jina 모델 이름
            api_key: Jina API 키
            late_chunking: 지연 청킹 여부
        """
        self.jina_ef = JinaEmbeddingFunction(
            model_name=model_name,
            api_key=api_key,
            late_chunking=late_chunking,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들을 임베딩으로 변환"""
        # 빈 리스트인 경우 빈 리스트 반환 (Jina API가 빈 리스트를 거부함)
        if not texts:
            return []
        embeddings = self.jina_ef.encode_documents(texts)
        return [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트를 임베딩으로 변환"""
        embeddings = self.jina_ef.encode_queries([text])
        result = embeddings[0]
        return result.tolist() if hasattr(result, 'tolist') else result
    
    @property
    def embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        test_embedding = self.embed_query("test")
        return len(test_embedding)


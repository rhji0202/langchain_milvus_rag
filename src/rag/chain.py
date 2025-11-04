"""RAG 체인 구성 모듈"""
from typing import Optional
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek  import ChatDeepSeek
from openai import OpenAI

from ..vector_store.milvus_store import MilvusVectorStore

class RAGChain:
    """RAG 체인 클래스"""
    
    DEFAULT_PROMPT_TEMPLATE = """Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.

Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>

Assistant:"""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        llm_client: OpenAI,
        model_name: str = "deepseek-chat",
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        search_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            vector_store: Milvus 벡터 스토어
            llm_client: DeepSeek OpenAI 호환 클라이언트 (base_url이 DeepSeek API로 설정됨)
            model_name: DeepSeek 모델 이름 (기본값: "deepseek-chat")
            temperature: LLM temperature
            prompt_template: 프롬프트 템플릿
            search_kwargs: 검색 옵션
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.search_kwargs = search_kwargs or {"k": 3}
        
        # Retriever 생성
        print("[디버깅] Retriever 생성 중...")
        self.retriever = vector_store.get_retriever(search_kwargs=self.search_kwargs)
        print("[디버깅] Retriever 생성 완료")
        
        # 프롬프트 생성
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"],
        )
        
        # DeepSeek LLM 초기화 (OpenAI 호환 API 사용)
        self.llm = ChatDeepSeek(
            model=model_name,
            temperature=temperature,
        )
        
        # RAG 체인 구성
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """RAG 체인 구성"""
        def format_docs(docs):
            """검색된 문서 포맷팅"""
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def invoke(self, question: str) -> str:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
        
        Returns:
            생성된 답변
        """
        print("[디버깅] RAG 체인 invoke 시작...")
        try:
            result = self.chain.invoke(question)
            print("[디버깅] RAG 체인 invoke 완료")
            return result
        except Exception as e:
            print(f"[디버깅] RAG 체인 invoke 중 에러 발생: {type(e).__name__}: {e}")
            raise
    
    def get_chain(self):
        """RAG 체인 반환"""
        return self.chain


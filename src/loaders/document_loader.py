"""문서 로더 모듈"""
from glob import glob
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MarkdownDocumentLoader:
    """마크다운 문서를 로드하고 분할하는 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹치는 부분
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "# ", " ", ""],
        )
    
    def load_documents(self, file_patterns: List[str]) -> List[Document]:
        """
        파일 패턴에 매칭되는 모든 파일을 로드하고 분할
        
        Args:
            file_patterns: 파일 경로 패턴 리스트 (glob 패턴)
        
        Returns:
            분할된 Document 리스트
        """
        documents = []
        
        for pattern in file_patterns:
            file_paths = glob(pattern, recursive=True)
            for file_path in file_paths:
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    
                    # 메타데이터 추가
                    metadata = {
                        "source": str(Path(file_path)),
                        "file_path": file_path,
                    }
                    
                    # 문서를 "# "로 분할하여 더 작은 청크로 나눔
                    sections = content.split("# ")
                    for i, section in enumerate(sections):
                        if section.strip():
                            # 섹션 메타데이터 추가
                            section_metadata = metadata.copy()
                            section_metadata["section_index"] = i
                            
                            # 텍스트 스플리터로 분할
                            split_docs = self.text_splitter.create_documents(
                                [section],
                                metadatas=[section_metadata],
                            )
                            documents.extend(split_docs)
                
                except Exception as e:
                    print(f"파일 로드 중 오류 발생: {file_path}, 오류: {e}")
                    continue
        
        return documents
    
    def load_and_split(self, file_patterns: List[str]) -> List[Document]:
        """load_documents의 별칭"""
        return self.load_documents(file_patterns)


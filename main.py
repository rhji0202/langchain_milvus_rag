"""메인 실행 파일 - 벡터 인덱스 구축"""
import sys
from src.config import Config
from src.pipeline import RAGPipeline

# Windows에서 한글 출력을 위한 인코딩 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def main():
    """메인 함수"""
    config = Config()
    config.validate()
    
    # RAG 파이프라인 초기화 및 인덱스 구축
    pipeline = RAGPipeline(config=config, rebuild_index=True)
    pipeline.build_index()
    
    print("\n벡터 인덱스 구축이 완료되었습니다!")
    print(f"컬렉션 이름: {config.COLLECTION_NAME}")
    print(f"Milvus URI: {config.MILVUS_URI}")


if __name__ == "__main__":
    main()

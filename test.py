"""테스트 실행 파일 - RAG 질문 답변"""
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
    
    # RAG 파이프라인 초기화
    pipeline = RAGPipeline(config=config, rebuild_index=False)
    
    # 질문 예시
    question = "Voyage 사용법을 알려줘"
    
    print(f"질문: {question}\n")
    print("답변 생성 중...\n")
    
    # 답변 생성
    answer = pipeline.query(question)
    
    print("=" * 50)
    print("답변:")
    print("=" * 50)
    print(answer)


if __name__ == "__main__":
    main()

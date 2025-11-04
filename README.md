# Milvus RAG System with LangChain

LangChain을 사용하여 구조화된 Milvus 기반 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 프로젝트 구조

```
milvus/
├── src/
│   ├── __init__.py
│   ├── config.py              # 설정 관리
│   ├── embeddings/            # 임베딩 모듈
│   │   ├── __init__.py
│   │   └── jina_embedding.py
│   ├── loaders/               # 문서 로더 모듈
│   │   ├── __init__.py
│   │   └── document_loader.py
│   ├── vector_store/          # 벡터 스토어 모듈
│   │   ├── __init__.py
│   │   └── milvus_store.py
│   ├── rag/                   # RAG 체인 모듈
│   │   ├── __init__.py
│   │   └── chain.py
│   └── pipeline/              # 통합 파이프라인
│       ├── __init__.py
│       └── rag_pipeline.py
├── main.py                    # 벡터 인덱스 구축
├── test.py                    # RAG 질문 답변 테스트
├── docker-compose.yml         # Docker Compose 설정
├── embedEtcd.yaml             # etcd 설정 파일
├── user.yaml                  # Milvus 사용자 설정 파일
├── pyproject.toml
└── README.md
```

## 설치

### 의존성 설치

```bash
uv sync
```

또는

```bash
pip install -e .
```

### Milvus 실행

Docker Compose를 사용하여 Milvus를 실행합니다:

```bash
# Milvus 시작
docker-compose up -d

# Milvus 상태 확인
docker-compose ps

# Milvus 로그 확인
docker-compose logs -f milvus-standalone

# Milvus 중지
docker-compose stop

# Milvus 중지 및 컨테이너 삭제
docker-compose down

# Milvus 중지 및 볼륨까지 삭제 (데이터 삭제)
docker-compose down -v
```

> **참고**: Milvus가 정상적으로 시작되려면 약 90초 정도 소요됩니다. Health check가 완료될 때까지 기다려주세요.

### 환경 변수 설정

`.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
JINAAI_API_KEY=your_jina_api_key
MILVUS_URI=http://localhost:19530
```

## 사용 방법

### 0. Milvus 실행 확인

먼저 Milvus가 정상적으로 실행 중인지 확인하세요:

```bash
docker-compose ps
```

`milvus-standalone` 컨테이너의 상태가 `healthy`로 표시되어야 합니다.

### 1. 벡터 인덱스 구축

```bash
python main.py
```

이 명령은 `milvus_docs` 디렉토리의 문서를 로드하고 벡터 인덱스를 구축합니다.

### 2. RAG 질문 답변

```bash
python test.py
```

이 명령은 저장된 벡터를 검색하고 LLM을 사용하여 답변을 생성합니다.

## 주요 기능

### 1. 문서 로딩 및 분할
- `MarkdownDocumentLoader`: 마크다운 문서를 로드하고 청크로 분할
- `RecursiveCharacterTextSplitter`를 사용한 스마트 텍스트 분할

### 2. 임베딩
- `JinaEmbeddings`: Jina Embedding을 LangChain 인터페이스로 래핑
- LangChain의 `Embeddings` 인터페이스와 호환

### 3. 벡터 스토어
- `MilvusVectorStore`: Milvus 벡터 스토어 관리
- LangChain의 `Milvus` 통합 사용

### 4. RAG 체인
- `RAGChain`: LangChain LCEL을 사용한 RAG 체인 구성
- 검색기(Retriever) + LLM을 통한 답변 생성

### 5. 통합 파이프라인
- `RAGPipeline`: 전체 RAG 파이프라인 통합 관리
- 인덱스 구축부터 질문 답변까지 원스톱 솔루션

## 설정 커스터마이징

### 애플리케이션 설정

`src/config.py`에서 다음 설정을 변경할 수 있습니다:

- `CHUNK_SIZE`: 텍스트 청크 크기
- `CHUNK_OVERLAP`: 청크 간 겹치는 부분
- `SEARCH_K`: 검색 결과 개수
- `LLM_MODEL`: LLM 모델 이름
- `DOCS_PATHS`: 문서 경로 패턴

### Milvus 설정

Milvus의 기본 설정을 변경하려면 `user.yaml` 파일을 수정한 후 다음 명령으로 재시작하세요:

```bash
docker-compose restart
```

`embedEtcd.yaml` 파일은 내장 etcd의 설정을 관리합니다. 일반적으로 수정할 필요가 없습니다.

## 기술 스택

- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **Milvus**: 벡터 데이터베이스
- **Jina AI**: 임베딩 모델
- **DeepSeek**: LLM 모델 (OpenAI 호환 API 사용)

## 라이선스

MIT

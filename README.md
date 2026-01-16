# Agent Development Kit (ADK) + Vertex AI RAG Engine

구글의 [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) 과 [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview)으로 제작된 검색 증강 생성(RAG) 엔진. 이 프로젝트는 Google Cloud Storage (GCS) 버킷, RAG corpora 및 문서 검색을 관리하기 위한 모듈식 프레임워크를 제공합니다.

![RAG Query Interface](.Images/RAG-Single-Query-Search-Web.gif)

![GCS File Upload Interface](.Images/GCS-File-Upload-Web.gif)

## Vertex AI RAG Engine

Vertex AI RAG Engine은 Retrieval-Augmented Generation (RAG, 검색증강생성)을 촉진하는 Vertex AI 플랫폼의 구성 요소로, context 증강 large language model (LLM) 애플리케이션을 개발하기 위한 데이터 프레임워크 역할을 합니다. 이를 통해 조직의 사적인 지식을 기반으로 LLM context를 풍부하게 하고, 할루시네이션(환각)을 줄이고 답변 정확도를 증가시킬 수 있습니다.

### RAG 프로세스 개념

이 개념들은 retrieval-augmented generation(RAG, 검색 증강 생성) 프로세스의 순서를 나열했습니다.:

1. **Data ingestion**(데이터 수집): 다양한 데이터 소스에서 데이터를 가져옵니다. 예를 들어, 로컬 파일, GCS(구글 클라우드 스토리지), Google 드라이브 등.

2. **Data transformation**(데이터 변환): 분류를 위해 데이터를 변환합니다(전처리). 예를 들어, 데이터를 청크 단위로 나눕니다.

3. **Embedding**(임베딩): 단어들 텍스트 조각들의 수치화. 이러한 수치들은 텍스트의 의미와 맥락을 포착합니다. 비슷하거나 관련있는 단어 및 텍스트는 같은 임베딩을 갖는 경향이 있습니다. 이는 고차원 벡터상에서 가깝게 붙어있다는 이야기입니다.

4. **Data indexing**(데이터 분류): Vertex AI RAG Engine은 corpus라고 불리는 색인을 만듭니다. 이것은 지식 기반을 구조화하여 검색에 최적화합니다. 예를 들자면, 방대한 참고 서적의 상세한 목차와 같습니다.

5. **Retrieval**(검색): 유저가 질문을 하거나 프롬포트를 제공하면, Vertex AI RAG Engine의 검색 구성 요소는 자체 지식 기반에서 쿼리와 관련된 정보를 검색합니다.

6. **Generation**(생성): 검색된 정보는 사용자의 쿼리에 포함되어 가이드로서 생성형 AI 모델이 사실에 근거하고 관련있는 응답을 생성하는데 도움을 줍니다.

## Agent Development Kit (ADK)

[Agent Development Kit (ADK)](https://google.github.io/adk-docs/) 는 AI 에이전트를 개발하고 배포하기 위한 유연하고 모듈식 프레임워크입니다. 주요 기능은 다음과 같습니다.:

- **Model-Agnostic**(모델에 구애받지 않음): ADK는 Gemini 및 Google 생태계에 최적화되어 있지만 모든 모델에서 작동합니다.
- **Flexible Orchestration**(유연한 통합): 워크플로 에이전트(Sequential, Parallel, Loop)를 사용하여 워크플로를 정의하거나, LLM 기반 동적 라우팅을 활용하여 적응형 동작을 구현할 수 있습니다..
- **Multi-Agent Architecture**(다중 에이전트 구조): 계층 구조로 여러 개의 특화된 에이전트를 조합하여 모듈형 애플리케이션을 구축합니다.
- **Rich Tool Ecosystem**(풍부한 도구 생태계): 사전 구축된 도구, 사용자 지정 기능, 타사 통합 또는 다른 에이전트를 도구로 활용하는 등 다양한 기능을 에이전트에게 제공합니다.
- **Deployment Ready**(배포 가능성): 에이전트를 어디든지 배포할 수 있습니다. – 로컬 혹은 Vertex AI Agent Engine 또는 Cloud Run/Docker에도.
- **Built-in Evaluation**(내장된 평가 기능): 응답 품질과 실행 과정을 평가하여 에이전트 성능을 향상할 수 있습니다.

ADK는 에이전트 개발을 소프트웨어 개발처럼 만들어주어 간단한 작업부터 복잡한 워크플로우에 이르기까지 다양한 에이전트를 더 쉽게 생성, 배포 및 통합할 수 있도록 합니다.

## Table of Contents

- [Vertex AI RAG Engine](#vertex-ai-rag-engine)
  - [RAG Process Concepts](#rag-process-concepts)
- [Agent Development Kit (ADK)](#agent-development-kit-adk)
- [Features](#features)
- [Pre-created RAG Corpora](#pre-created-rag-corpora)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Agent](#running-the-agent)
  - [Example Commands](#example-commands)
- [Configuration](#configuration)
- [Supported File Types](#supported-file-types)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Example Workflow](#example-workflow)
  - [Create GCS Buckets](#1-create-gcs-buckets)
  - [Upload PDF Files to GCS Buckets](#2-upload-pdf-files-to-gcs-buckets)
  - [Create RAG Corpora and Import Files](#3-create-rag-corpora-and-import-files) 
  - [Query Across All Corpora](#4-query-across-all-corpora)
- [Author](#author)

## Features

- 🗂️ **GCS Bucket Management**: Create, list, and manage GCS buckets for file storage.
- 📚 **RAG Corpus Management**: Create, update, list, and delete RAG corpora in Vertex AI.
- 📄 **Document Management**: Import documents from GCS into RAG corpora for vector search.
- 🔎 **Semantic Search**: Query RAG corpora for relevant information with citations.
- 🤖 **Agent-based Interface**: Interact with all functionalities through a natural language interface.
- ⚙️ **Configurable & Extensible**: Centralized configuration, emoji-enhanced responses, and schema-compliant tools.

## Pre-created RAG Corpora

The project includes several pre-created RAG corpora covering major AI topics:

- **Foundation Models & Prompt Engineering**: Resources on large language models and effective prompt design
- **Embeddings & Vector Stores**: Details on text embeddings and vector databases
- **Generative AI Agents**: Information on agent design, implementation, and usage
- **Domain-Specific LLMs**: Techniques for applying LLMs to solve domain-specific problems
- **MLOps for Generative AI**: Deployment and production considerations for GenAI systems

Each corpus contains relevant PDF documents imported from Google and Kaggle's Gen AI Intensive course:

- [Day 1: Foundational Models & Prompt Engineering](https://lnkd.in/d-_w3gXj)
- [Day 2: Embeddings & Vector Stores / Databases](https://lnkd.in/dkmfDUcp)
- [Day 3: Generative AI Agents](https://lnkd.in/dd3Zd2-F)
- [Day 4: Domain-Specific LLMs](https://lnkd.in/d6Z39yqt)
- [Day 5: MLOps for Generative AI](https://lnkd.in/dcXCTPVF)

These documents are from Google and Kaggle's Gen AI Intensive course, which broke the GUINNESS WORLD RECORDS™ title for the Largest Attendance at a Virtual AI Conference in One Week with more than 280,000 signups in just 20 days. The materials provide a comprehensive overview of Vertex AI capabilities and best practices for working with generative AI.

## Architecture

The project follows a modular architecture based on the ADK framework:

![ADK Vertex AI RAG Architecture](.Images/ADK-VertexAI-RAG-Architecture.png)

The architecture consists of several key components:

1. **User Interface**: Interact with the system through ADK Web or CLI
2. **Agent Development Kit (ADK)**: The core orchestration layer that manages tools and user interactions
3. **Function Tools**: Modular components divided into:
   - **Storage Tools**: For GCS bucket and file management
   - **RAG Corpus Tools**: For corpus management and semantic search
4. **Google Cloud Services**:
   - **Google Cloud Storage**: Stores document files
   - **Vertex AI RAG Engine**: Provides embedding, indexing and retrieval capabilities
   - **Gemini 2.0 LLM Model**: Generates responses grounded in retrieved contexts

File structure:
```
adk-vertex-ai-rag-engine/
├── rag/                          # Main project package
│   ├── __init__.py               # Package initialization
│   ├── agent.py                  # The main RAG corpus manager agent
│   ├── config/                   # Configuration directory
│   │   └── __init__.py           # Centralized configuration settings
│   └── tools/                    # ADK function tools
│       ├── __init__.py           # Tools package initialization
│       ├── corpus_tools.py       # RAG corpus management tools
│       └── storage_tools.py      # GCS bucket management tools
├── .Images/                      # Demo images and GIFs
└── README.md                     # Project documentation
```

## Prerequisites

- 파이썬 3.11+
- Vertex AI API가 활성화된 Google Cloud project
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Vertex AI and Cloud Storage의 접근

## 설치

```bash
# repository 복사
git clone https://github.com/arjunprabhulal/adk-vertex-ai-rag-engine.git
cd adk-vertex-ai-rag-engine

# (선택사항) 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 본인의 Google Cloud project으로 설정
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Google Cloud services 활성화하기 (필요)
gcloud services enable aiplatform.googleapis.com --project=${GOOGLE_CLOUD_PROJECT}
gcloud services enable storage.googleapis.com --project=${GOOGLE_CLOUD_PROJECT}

# IAM 권한 설정
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="user:YOUR_EMAIL@domain.com" \
    --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} \
    --member="user:YOUR_EMAIL@domain.com" \
    --role="roles/storage.objectAdmin"

# Gemini API key 설정
# API key를 Google AI Studio에서 받으세요: https://ai.google.dev/
export GOOGLE_API_KEY=your_gemini_api_key_here

# 인증 자격 증명 설정
# 선택 1: Use gcloud application-default credentials (recommended for development)
gcloud auth application-default login

# 선택 2: Use a service account key (for production or CI/CD environments)
# Download your service account key from GCP Console and set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

## 사용법

### Agent 실행
Agent를 실행하는 방법은 2가지 입니다.

```bash
# 선택사항 1: Use ADK web interface (recommended for interactive usage)
adk web 

# 선택사항 2: Run the agent directly in the terminal
adk run rag
```

The web interface provides a chat-like experience for interacting with the agent, while the direct run option is suitable for scripting and automated workflows.

### 예시 명령어

```
# 모든 GCS buckets 리스트화 하기
[user]: GCS 버킷 리스트해서 보여줘

# LLM을 위한 GCS bucket 만들기
[user]: "adk-embedding-vector-stores"라는 이름의 GCS 버킷 만들어줘

# 문서 업로드
[user]: (파일을 업로드 한 뒤) 이 PDF 파일을 GCS 버킷 gs://adk-embedding-vector-stores/에 업로드하고 같은 이름으로 유지해줘.

# RAG corpus 생성
[user]: "adk-embedding-vector-stores"라는 이름의 rag 만들어주고 설명은 "adk-embedding-vector-stores"로 해줘

# 문서를 RAG corpus에 학습시키기
[user]: gs://adk-embedding-vector-stores/emebddings-vector-stores.pdf를 RAG에 학습시켜줘

# RAG corpus한테 질문하기
[user]: Chain of Thought (CoT)가 뭐야?

```

## Configuration

`rag/config/__init__.py`을 수정해서 세팅을 커스터마이즈 하세요:

- `PROJECT_ID`: Your Google Cloud project ID
- `LOCATION`: Default location for Vertex AI and GCS resources
- `GCS_DEFAULT_*`: Defaults for GCS operations
- `RAG_DEFAULT_*`: Defaults for RAG operations
- `AGENT_*`: Settings for the agent

## 가능한 파일 타입

The engine supports various document types, including:
- PDF
- TXT
- DOC/DOCX
- XLS/XLSX
- PPT/PPTX
- CSV
- JSON
- HTML
- Markdown

## Troubleshooting

### Common Issues

- **403 Errors**: Make sure you've authenticated with `gcloud auth application-default login`
- **Resource Exhausted**: Check your quota limits in the GCP Console
- **Upload Issues**: Ensure your file format is supported and file size is within limits

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## References

- [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview)
- [Google Cloud Storage](https://cloud.google.com/storage)

## Example Workflow

Below is a complete example workflow showing how to set up the entire RAG environment with the Google Gen AI Intensive course materials:

### 1. Create GCS Buckets

![GCS Bucket Creation CLI](.Images/GCS-Bucket-creation-cli.gif)

```
Create the following 7 Google Cloud Storage buckets for my project, using the default settings (location: US, storage class: STANDARD) for all of them. Do not ask for confirmation for each bucket.

1. adk-foundation-llm
2. adk-prompt-engineering
3. adk-embedding-vector-stores
4. adk-agents-llm
5. adk-agents-companion
6. adk-solving-domain-problem-using-llms
7. adk-operationalizing-genai-vertex-ai
```

### 2. Upload PDF Files to GCS Buckets

![GCS File Upload Web](.Images/GCS-File-Upload-Web.gif)

![GCS Multiple Uploads](.Images/GCS-Multiple-Uploads.png)

```
Upload the file "promptengineering.pdf" to the GCS bucket gs://adk-prompt-engineering/ and use "promptengineering.pdf" as the destination blob name. Do not ask for confirmation.

Upload the file "foundational-large-language-models-text-generation.pdf" to the GCS bucket gs://adk-foundation-llm/ and use "foundational-large-language-models-text-generation.pdf" as the destination blob name. Do not ask for confirmation.

Upload the file "agents.pdf" to the GCS bucket gs://adk-agents-llm/ and use "agents.pdf" as the destination blob name. Do not ask for confirmation.

Upload the file "agents-companion.pdf" to the GCS bucket gs://adk-agents-companion/ and use "agents-companion.pdf" as the destination blob name. Do not ask for confirmation.

Upload the file "emebddings-vector-stores.pdf" to the GCS bucket gs://adk-embedding-vector-stores/ and use "emebddings-vector-stores.pdf" as the destination blob name. Do not ask for confirmation.

Upload the file "operationalizing-generative-ai-on-vertex-ai.pdf" to the GCS bucket gs://adk-operationalizing-genai-vertex-ai/ and use "operationalizing-generative-ai-on-vertex-ai.pdf" as the destination blob name. Do not ask for confirmation.

Upload the file "solving-domain-specific-problems-using-llms.pdf" to the GCS bucket gs://adk-solving-domain-problem-using-llms/ and use "solving-domain-specific-problems-using-llms.pdf" as the destination blob name. Do not ask for confirmation.
```

### 3. Create RAG Corpora and Import Files

![RAG Create Import Web](.Images/RAG-Create-Import-Web.gif)

![RAG Create Multiple Upload CLI](.Images/RAG-Create-Mutliple-Upload-CLI.gif)

```
Create a RAG corpus named "adk-agents-companion" with description of rag as "adk-agents-companion" and import the gs://adk-agents-companion/agents-companion.pdf into RAG

Create a RAG corpus named "adk-agents-llm" with description "adk-agents-llm" and import the file gs://adk-agents-llm/agents.pdf into the RAG corpus.

Create a RAG corpus named "adk-embedding-vector-stores" with description "adk-embedding-vector-stores" and import the file gs://adk-embedding-vector-stores/emebddings-vector-stores.pdf into the RAG corpus.

Create a RAG corpus named "adk-foundation-llm" with description "adk-foundation-llm" and import the file gs://adk-foundation-llm/foundational-large-language-models-text-generation.pdf into the RAG corpus.

Create a RAG corpus named "adk-operationalizing-genai-vertex-ai" with description "adk-operationalizing-genai-vertex-ai" and import the file gs://adk-operationalizing-genai-vertex-ai/operationalizing-generative-ai-on-vertex-ai.pdf into the RAG corpus.

Create a RAG corpus named "adk-solving-domain-problem-using-llms" with description "adk-solving-domain-problem-using-llms" and import the file gs://adk-solving-domain-problem-using-llms/solving-domain-specific-problems-using-llms.pdf into the RAG corpus.
```

### 4. Query Across All Corpora

![RAG Multiple Query Search CLI](.Images/RAG-Multiple-Query-Search-CLI.gif)

![RAG Multiple Search Corpus Web](.Images/RAG-Multiple-Search-Corpus-Web.gif)

![RAG Single Query Search Web](.Images/RAG-Single-Query-Search-Web.gif)

```
# Questions about Prompt Engineering
What is Chain of Thought (CoT)?
What is Tree of Thoughts (ToT)?
What is ReAct (reason & act)?

# Questions about Embeddings & Vector Stores
What are Types of embeddings?
What is Vector search?
What is Vector databases?

# Questions about Agents
What is Agent Lifecycle?

# Questions about MLOps & Operationalization
How do multiple teams collaborate to operationalize GenAI models?
How multiple teams collaborate to operationalize both models and GenAI applications?
```

## Author

For more articles on AI/ML and Generative AI, follow me on Medium: https://medium.com/@arjun-prabhulal

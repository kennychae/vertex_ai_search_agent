"""
Configuration settings for the Vertex AI RAG engine.
"""
import os

# Google Cloud Project Settings
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Replace with your project ID
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")  # Default location for Vertex AI and GCS resources
APP_LOCATION = os.environ.get("GOOGLE_APP_LOCATION", "global") # Default location for Vertex AI Search.

# GCS Storage Settings
GCS_DEFAULT_STORAGE_CLASS = "STANDARD"
GCS_DEFAULT_LOCATION = "US"
GCS_LIST_BUCKETS_MAX_RESULTS = 50
GCS_LIST_BLOBS_MAX_RESULTS = 100
GCS_DEFAULT_CONTENT_TYPE = "application/pdf"  # Default content type for uploaded files

# RAG Corpus Settings
RAG_DEFAULT_EMBEDDING_MODEL = "text-multilingual-embedding-002"
RAG_DEFAULT_TOP_K = 10  # Default number of results for single corpus query
RAG_DEFAULT_SEARCH_TOP_K = 5  # Default number of results per corpus for search_all
RAG_DEFAULT_VECTOR_DISTANCE_THRESHOLD = 0.5
RAG_DEFAULT_PAGE_SIZE = 50  # Default page size for listing files

# Vertex AI Search (Discovery Engine) Settings
SEARCH_DEFAULT_SERVING_CONFIG_ID = "default_config"

SEARCH_DEFAULT_PAGE_SIZE = 10

# Vertex AI Search Summary
SEARCH_DEFAULT_SUMMARY_RESULT_COUNT = 3
SEARCH_DEFAULT_INCLUDE_CITATIONS = True
SEARCH_DEFAULT_USE_SEMANTIC_CHUNKS = True
SEARCH_DEFAULT_SUMMARY_MODEL_VERSION = "stable"

SEARCH_DEFAULT_SUMMARY_PREAMBLE = (
    "너는 검색 결과에 대한 정보를 사용자에게 잘 전달하기 위해 요약하는 비서다.\n"
    "- 한국어로 작성한다.\n"
    "- 요약본에 중요한 내용은 무조건 포함할 것\n"
    "- 불필요한 기호를 남발하지 않는다.\n"
)

# Vertex Search Snippet / Extractive settings
SEARCH_DEFAULT_MAX_SNIPPET_COUNT = 5
SEARCH_DEFAULT_MAX_EXTRACTIVE_ANSWER_COUNT = 3
SEARCH_DEFAULT_MAX_EXTRACTIVE_SEGMENT_COUNT = 3

# Vertex Search Query expansion / spell correction
SEARCH_DEFAULT_QUERY_EXPANSION = "AUTO"      # AUTO / DISABLED
SEARCH_DEFAULT_SPELL_CORRECTION = "AUTO"     # AUTO / DISABLED

# Agent Settings
AGENT_NAME = "rag_corpus_manager"
AGENT_MODEL = "gemini-2.5-flash"
AGENT_OUTPUT_KEY = "last_response"

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

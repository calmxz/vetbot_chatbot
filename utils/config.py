"""Configuration constants for the chatbot application."""

class Config:
    # RAG settings (~300 tokens, within BGE's 512 token limit)
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150
    SIMILARITY_SEARCH_K = 3
    CHROMA_DB_PATH = "./chroma_db"
    DOCUMENTS_FOLDER = "./documents"

    # Embeddings (768-dim, 512 max tokens)
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DEVICE = "cpu"

    # LLM settings
    DEFAULT_MODEL = "gemini-2.5-flash"
    TEMPERATURE_NORMAL = 0.4
    TEMPERATURE_PROFESSIONAL = 0.3

    # TTS settings
    TTS_LANG_CODE = "a"  # 'a' for American English
    TTS_DEFAULT_VOICE = "af_heart"
    TTS_SAMPLE_RATE = 24000

    # Conversation context
    MAX_CONTEXT_MESSAGES = 6

    # API retry settings
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0

    # RAG relevance filtering (ChromaDB returns distance, lower = better)
    SIMILARITY_DISTANCE_THRESHOLD = 0.5

    # Incremental indexing
    INDEXING_HASH_ALGORITHM = "sha256"

    # Cross-encoder re-ranking
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATE_K = 10
    RERANK_TOP_N = 4
    RERANK_ENABLED = True

    # Context assembly
    MAX_CONTEXT_TOKENS = 3000
    CHARS_PER_TOKEN_ESTIMATE = 4

    # Query rewriting
    QUERY_REWRITE_ENABLED = True
    QUERY_REWRITE_MODEL = "gemini-2.5-flash"
    QUERY_REWRITE_TEMPERATURE = 0.0

    # Input sanitization
    MAX_INPUT_LENGTH = 4000

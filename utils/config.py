"""Configuration constants for the chatbot application."""

class Config:
    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    SIMILARITY_SEARCH_K = 3
    CHROMA_DB_PATH = "./chroma_db"
    DOCUMENTS_FOLDER = "./documents"

    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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

# Utils package for shared chatbot functionality
from .config import Config
from .tts import get_kokoro_pipeline, text_to_speech, cleanup_audio_files
from .rag import (
    load_and_index_documents,
    get_embeddings,
    retrieve_relevant_documents,
    build_rag_context,
    build_prompt_with_context,
    detect_metadata_filters,
)
from .audio_player import render_audio_button
from .query import rewrite_query

__all__ = [
    'Config',
    'get_kokoro_pipeline',
    'text_to_speech',
    'cleanup_audio_files',
    'load_and_index_documents',
    'get_embeddings',
    'retrieve_relevant_documents',
    'build_rag_context',
    'build_prompt_with_context',
    'detect_metadata_filters',
    'render_audio_button',
    'rewrite_query',
]

"""Text-to-Speech utilities using Kokoro TTS."""

import os
import re
import tempfile
import streamlit as st
import soundfile as sf
import numpy as np
from kokoro import KPipeline

from .config import Config


# Track temporary audio files for cleanup
_temp_audio_files = []


@st.cache_resource
def get_kokoro_pipeline():
    """Initialize and cache Kokoro TTS pipeline."""
    return KPipeline(lang_code=Config.TTS_LANG_CODE)


def _clean_text_for_tts(text: str) -> str:
    """Clean up text for better TTS pronunciation."""
    cleaned = text
    # Remove markdown bold formatting (e.g., **text** -> text)
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    # Remove markdown italic formatting (e.g., *text* -> text)
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    # Remove any remaining asterisks
    cleaned = cleaned.replace('*', '')
    # Replace colons with periods for more natural speech
    cleaned = cleaned.replace(':', '.')
    return cleaned


def text_to_speech(text: str, voice: str = None) -> str | None:
    """
    Convert text to speech using Kokoro TTS and return audio file path.

    Args:
        text: The text to convert to speech
        voice: The voice to use (default: Config.TTS_DEFAULT_VOICE)

    Returns:
        Path to the generated audio file, or None if generation failed
    """
    if voice is None:
        voice = Config.TTS_DEFAULT_VOICE

    try:
        pipeline = get_kokoro_pipeline()
        cleaned_text = _clean_text_for_tts(text)

        # Generate audio using Kokoro
        generator = pipeline(cleaned_text, voice=voice)

        # Kokoro yields (graphemes, phonemes, audio) tuples
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            st.error("No audio generated")
            return None

        # Concatenate all audio chunks
        audio_data = np.concatenate(audio_chunks)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            sf.write(fp.name, audio_data, Config.TTS_SAMPLE_RATE)
            _temp_audio_files.append(fp.name)
            return fp.name

    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None


def cleanup_audio_files(audio_cache: dict = None):
    """
    Clean up temporary audio files.

    Args:
        audio_cache: Optional dict of cached audio file paths to also clean
    """
    global _temp_audio_files

    # Clean files from the cache
    if audio_cache:
        for file_path in audio_cache.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        audio_cache.clear()

    # Clean tracked temp files
    for file_path in _temp_audio_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass

    _temp_audio_files.clear()

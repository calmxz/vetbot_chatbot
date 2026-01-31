"""Audio player UI component for Streamlit."""

import base64
import streamlit as st

from .tts import text_to_speech


def render_audio_button(
    message_content: str,
    message_idx: int,
    audio_cache: dict,
    playing_audio_key: str,
    session_state_prefix: str = ""
):
    """
    Render an audio play/pause button for a message.

    Args:
        message_content: The text content of the message
        message_idx: Index of the message in the messages list
        audio_cache: Dict to cache generated audio files
        playing_audio_key: Session state key for tracking which audio is playing
        session_state_prefix: Prefix for session state keys (e.g., "vet_")
    """
    generating_key = f"{session_state_prefix}generating_audio_for" if session_state_prefix else "generating_audio_for"
    playing_state_key = f"{session_state_prefix}playing_audio" if session_state_prefix else "playing_audio"

    # Check if audio is being generated for this message
    is_generating = st.session_state.get(generating_key) == message_idx

    # Get current playing state
    is_playing = st.session_state.get(playing_state_key) == message_idx
    button_icon = "\u23f8\ufe0f" if is_playing else "\U0001f50a"

    if st.button(
        button_icon,
        key=f"tts_{session_state_prefix}{message_idx}",
        help="Generating audio..." if is_generating else "Play/Stop audio",
        disabled=is_generating
    ):
        if is_playing:
            # Stop audio
            st.session_state[playing_state_key] = None
            st.rerun()
        else:
            # Start playing audio
            st.session_state[playing_state_key] = message_idx

            # Generate audio on-demand if not cached
            if message_idx not in audio_cache:
                audio_file = text_to_speech(message_content)
                if audio_file:
                    audio_cache[message_idx] = audio_file

            st.rerun()

    # Play audio if this message is playing
    if is_playing and message_idx in audio_cache:
        _play_audio_hidden(audio_cache[message_idx])


def _play_audio_hidden(audio_file_path: str):
    """Play audio without showing controls using hidden HTML audio element."""
    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()

        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = (
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">'
            f'</audio>'
        )
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
        pass  # Silently fail if audio file is missing

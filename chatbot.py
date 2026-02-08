# Standard library imports
import os
import time
import logging
import random
from pathlib import Path

# Third-party imports
import streamlit as st
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Script directory for relative paths
SCRIPT_DIR = Path(__file__).parent

# Local imports
from utils import (
    Config,
    text_to_speech,
    cleanup_audio_files,
    load_and_index_documents,
    render_audio_button,
    retrieve_relevant_documents,
    build_rag_context,
    build_prompt_with_context,
    detect_metadata_filters,
    rewrite_query,
)

# Load environment variables
load_dotenv()


def load_system_prompt(filepath=None):
    """Load system prompt from file, or return default if not found."""
    if filepath is None:
        filepath = SCRIPT_DIR / "system_prompt.txt"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.warning("system_prompt.txt not found. Using default fallback prompt.")
        return (
            "You are a helpful and friendly veterinary assistant chatbot. "
            "Always remind users to consult a veterinarian for professional medical advice."
        )
    except UnicodeDecodeError:
        logger.error(f"Invalid UTF-8 encoding in {filepath}")
        st.warning("System prompt file has encoding issues. Using default.")
        return (
            "You are a helpful and friendly veterinary assistant chatbot. "
            "Always remind users to consult a veterinarian for professional medical advice."
        )


@st.cache_resource
def initialize_client():
    """Initialize Gemini API client using API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        st.stop()

    client = genai.Client(api_key=api_key)

    # Validate API key by making a lightweight request
    try:
        list(client.models.list())
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        st.error("Invalid GEMINI_API_KEY. Please check your API key.")
        st.stop()

    return client


def get_response_stream(client, contents, system_prompt=None, model=None):
    """Stream response from Gemini API with retry logic.

    Retry only happens before any data is yielded. Once streaming starts,
    failures are propagated to avoid duplicate content.
    """
    if model is None:
        model = Config.DEFAULT_MODEL

    config = types.GenerateContentConfig(
        temperature=Config.TEMPERATURE_NORMAL,
        system_instruction=system_prompt,
    )

    last_exception = None
    for attempt in range(Config.MAX_RETRIES):
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            # Track if we've started yielding data
            started_streaming = False

            for chunk in stream:
                if chunk.text:
                    started_streaming = True
                    yield chunk.text
            return

        except (google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.DeadlineExceeded) as e:
            # Only retry if we haven't started streaming yet
            if started_streaming:
                logger.error(f"Stream failed mid-response: {e}")
                raise

            last_exception = e
            if attempt < Config.MAX_RETRIES - 1:
                # Exponential backoff with jitter
                delay = Config.RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{Config.MAX_RETRIES})")
                time.sleep(delay)

        except Exception as e:
            logger.error(f"API error: {e}")
            raise

    if last_exception:
        raise last_exception


def build_conversation_contents(messages, current_input, rag_context=None,
                                has_vectorstore=True):
    """Build multi-turn Content objects for Gemini API."""
    contents = []

    # Add history (excluding current message)
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )

    # Build current prompt with RAG context
    prompt = build_prompt_with_context(
        current_input, rag_context or "", mode="normal",
        has_vectorstore=has_vectorstore
    )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    )

    return contents


def sanitize_input(text: str) -> str | None:
    """Sanitize user input with length limit and validation.

    Returns None if input is empty/whitespace only.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    if len(text) > Config.MAX_INPUT_LENGTH:
        text = text[:Config.MAX_INPUT_LENGTH]
        logger.info(f"Input truncated to {Config.MAX_INPUT_LENGTH} chars")

    return text


def main():
    """Main Streamlit app: sets up UI and handles chat interactions with RAG."""
    st.set_page_config(
        page_title="Veterinary Chatbot",
        page_icon="\U0001f43e",
        layout="wide"
    )

    st.title("\U0001f43e Veterinary Information Chatbot")
    st.markdown("*Ask questions about pet health and care*")

    # Hide anchor links on markdown headings
    st.markdown("""
    <style>
    .stMarkdown a[href^="#"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    if "playing_audio" not in st.session_state:
        st.session_state.playing_audio = None
    if "generating_audio_for" not in st.session_state:
        st.session_state.generating_audio_for = None

    # Load resources
    vectorstore = load_and_index_documents(Config.DOCUMENTS_FOLDER)
    client = initialize_client()
    system_prompt = load_system_prompt()

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(message["content"])
            with col2:
                if message["role"] == "assistant":
                    render_audio_button(
                        message_content=message["content"],
                        message_idx=idx,
                        audio_cache=st.session_state.audio_cache,
                        playing_audio_key="playing_audio",
                    )

    # Handle user input
    if raw_input := st.chat_input("Ask about your pet's health..."):
        user_input = sanitize_input(raw_input)
        if user_input is None:
            st.warning("Please enter a valid question.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response (shown via spinner, not inline)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build conversation context
                recent_messages = st.session_state.messages[-Config.MAX_CONTEXT_MESSAGES:]

                # Rewrite query for better retrieval
                search_query = rewrite_query(client, user_input, recent_messages)

                # Build RAG context with relevance filtering
                rag_context = ""
                if vectorstore is not None:
                    where_filter = detect_metadata_filters(user_input)
                    docs_with_scores = retrieve_relevant_documents(
                        vectorstore, search_query, where_filter=where_filter
                    )
                    rag_context = build_rag_context(docs_with_scores)

                # Build multi-turn contents
                contents = build_conversation_contents(
                    recent_messages, user_input, rag_context,
                    has_vectorstore=(vectorstore is not None)
                )

                # Collect streamed response without displaying
                try:
                    response = ""
                    for chunk in get_response_stream(client, contents, system_prompt=system_prompt):
                        response += chunk
                except Exception as e:
                    response = "I apologize, but I encountered an error. Please try again."
                    logger.error(f"Response generation error: {e}")

        # Save message to session state and queue audio generation
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.generating_audio_for = len(st.session_state.messages) - 1
        st.rerun()  # Rerun to show response from history with disabled button

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot provides veterinary information using:
        - **RAG** (Retrieval-Augmented Generation)
        - **Gemini Flash** LLM
        - **LangChain** for document processing

         Documents are loaded from `./documents/` folder

        **Conversation Style:** Step-by-step guidance with natural follow-ups
        """)

        st.markdown("---")
        if st.button("Clear Chat History"):
            cleanup_audio_files(st.session_state.audio_cache)
            st.session_state.messages = []
            st.session_state.playing_audio = None
            st.session_state.generating_audio_for = None
            st.rerun()

        st.markdown("---")
        st.caption("For informational purposes only. Always consult a veterinarian for medical advice.")

    # Generate audio for pending message (runs after UI is rendered)
    if st.session_state.generating_audio_for is not None:
        idx = st.session_state.generating_audio_for
        if idx < len(st.session_state.messages) and idx not in st.session_state.audio_cache:
            msg = st.session_state.messages[idx]
            audio_file = text_to_speech(msg["content"])
            if audio_file:
                st.session_state.audio_cache[idx] = audio_file
        st.session_state.generating_audio_for = None
        st.rerun()


if __name__ == "__main__":
    main()

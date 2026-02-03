# Standard library imports
import os

# Third-party imports
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Local imports
from utils import (
    Config,
    text_to_speech,
    cleanup_audio_files,
    load_and_index_documents,
    render_audio_button,
)

# Load environment variables
load_dotenv()


def load_system_prompt(filepath="system_prompt_vet.txt"):
    """Load system prompt from file, or return default if not found."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.warning("system_prompt_vet.txt not found. Using default fallback prompt.")
        return (
            "You are an expert veterinary clinical assistant. Provide concise, accurate, "
            "and up-to-date information for licensed veterinarians and veterinary staff. "
            "Always cite sources if possible. Do not give owner-facing advice unless explicitly requested."
        )


@st.cache_resource
def initialize_client():
    """Initialize Gemini API client using API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        st.stop()
    return genai.Client(api_key=api_key)


def get_response(client, prompt, system_prompt=None, model=None):
    """Get streaming response from Gemini API."""
    if model is None:
        model = Config.DEFAULT_MODEL

    if system_prompt:
        full_prompt = f"""[Instruction — for professional reference use]
{system_prompt}

[User question]
{prompt}"""
    else:
        full_prompt = prompt

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=Config.TEMPERATURE_PROFESSIONAL,
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            response_text += chunk.text

    return response_text


def main():
    """Main Streamlit app for veterinary professional reference."""
    st.set_page_config(
        page_title="Veterinary Professional Reference Chatbot",
        layout="wide"
    )

    st.title("\U0001fa7a Veterinary Clinical Reference Chatbot")
    st.markdown("Ask clinical, diagnostic, or treatment questions — for veterinary professionals only")

    # Hide anchor links on markdown headings
    st.markdown("""
    <style>
    .stMarkdown a[href^="#"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "vet_messages" not in st.session_state:
        st.session_state.vet_messages = []
    if "vet_audio_cache" not in st.session_state:
        st.session_state.vet_audio_cache = {}
    if "vet_playing_audio" not in st.session_state:
        st.session_state.vet_playing_audio = None
    if "vet_generating_audio_for" not in st.session_state:
        st.session_state.vet_generating_audio_for = None

    # Load resources
    vectorstore = load_and_index_documents(Config.DOCUMENTS_FOLDER)
    client = initialize_client()
    system_prompt = load_system_prompt()

    # Display chat history
    for idx, message in enumerate(st.session_state.vet_messages):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(message["content"])
            with col2:
                if message["role"] == "assistant":
                    render_audio_button(
                        message_content=message["content"],
                        message_idx=idx,
                        audio_cache=st.session_state.vet_audio_cache,
                        playing_audio_key="vet_playing_audio",
                        session_state_prefix="vet_",
                    )

    # Handle user input
    if user_input := st.chat_input("Veterinary professional question (diagnosis, clinical signs, drug doses, etc)..."):
        st.session_state.vet_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response and audio (shown via spinner, not inline)
        with st.chat_message("assistant"):
            with st.spinner("Consulting knowledge base..."):
                # Build conversation context
                conversation_context = ""
                if len(st.session_state.vet_messages) > 1:
                    recent_messages = st.session_state.vet_messages[-Config.MAX_CONTEXT_MESSAGES:]
                    conversation_context = "\n".join([
                        f"{msg['role'].title()}: {msg['content']}"
                        for msg in recent_messages
                    ])

                # Build prompt with RAG context if available
                if vectorstore is not None:
                    relevant_docs = vectorstore.similarity_search(
                        user_input, k=Config.SIMILARITY_SEARCH_K
                    )

                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        source_text = "Knowledge base documents"

                        prompt = f"""[PROFESSIONAL MODE]
Context from clinical knowledge base:
{context}

Previous conversation:
{conversation_context}

User question: {user_input}

Source: {source_text}"""
                    else:
                        prompt = f"""[PROFESSIONAL MODE]
Previous conversation:
{conversation_context}

User question: {user_input}"""
                else:
                    prompt = f"""[PROFESSIONAL MODE]
Previous conversation:
{conversation_context}

User question: {user_input}"""

                response = get_response(client, prompt, system_prompt=system_prompt)

        # Save message to session state and queue audio generation
        st.session_state.vet_messages.append({"role": "assistant", "content": response})
        st.session_state.vet_generating_audio_for = len(st.session_state.vet_messages) - 1
        st.rerun()  # Rerun to show response from history with disabled button

    # Sidebar
    with st.sidebar:
        st.header("About (Veterinarian Mode)")
        st.markdown("""
This chatbot is designed for use by veterinarians and licensed techs only.

- **Professional reference** based on Merck Veterinary Manual, clinical guides, RAG (Retrieval-Augmented Generation), and Gemini LLM
- Optimized for rapid clinical Q&A
- Cites sources when possible
- **Data source:** ./documents/ folder

**Never use the output for layperson advice or client handouts directly.**

If unsure, always consult the latest literature or a board-certified specialist for advanced cases.
""")

        st.markdown("---")
        if st.button("Clear Chat History"):
            cleanup_audio_files(st.session_state.vet_audio_cache)
            st.session_state.vet_messages = []
            st.session_state.vet_playing_audio = None
            st.session_state.vet_generating_audio_for = None
            st.rerun()

        st.markdown("---")
        st.caption("For licensed professional use only.")

    # Generate audio for pending message (runs after UI is rendered)
    if st.session_state.vet_generating_audio_for is not None:
        idx = st.session_state.vet_generating_audio_for
        if idx < len(st.session_state.vet_messages) and idx not in st.session_state.vet_audio_cache:
            msg = st.session_state.vet_messages[idx]
            audio_file = text_to_speech(msg["content"])
            if audio_file:
                st.session_state.vet_audio_cache[idx] = audio_file
        st.session_state.vet_generating_audio_for = None
        st.rerun()


if __name__ == "__main__":
    main()

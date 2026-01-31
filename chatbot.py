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


def load_system_prompt(filepath="system_prompt.txt"):
    """Load system prompt from file, or return default if not found."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.warning("system_prompt.txt not found. Using default fallback prompt.")
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
    return genai.Client(api_key=api_key)


def get_response(client, prompt, system_prompt=None, model=None):
    """Get streaming response from Gemini API."""
    if model is None:
        model = Config.DEFAULT_MODEL

    if system_prompt:
        full_prompt = f"[Instruction — read before replying]\n{system_prompt}\n\n[User question]\n{prompt}"
    else:
        full_prompt = prompt

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=Config.TEMPERATURE_NORMAL,
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
    if user_input := st.chat_input("Ask about your pet's health..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response and audio (shown via spinner, not inline)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build conversation context
                conversation_context = ""
                if len(st.session_state.messages) > 1:
                    recent_messages = st.session_state.messages[-Config.MAX_CONTEXT_MESSAGES:]
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

                        # Check if it's a how-to question
                        is_howto = any(
                            phrase in user_input.lower()
                            for phrase in ['how to', 'how do i', 'how can i', 'steps to', 'way to']
                        )

                        if is_howto:
                            prompt = f"""Context from knowledge base:
{context}

Previous conversation:
{conversation_context}

User question: {user_input}

IMPORTANT: This is a "how-to" question. Provide ALL steps in numbered format from start to finish. Do not stop halfway through the procedure. Give the complete process in one response.

Source: {source_text}"""
                        else:
                            prompt = f"""Context from knowledge base:
{context}

Previous conversation:
{conversation_context}

User question: {user_input}

Source: {source_text}"""
                    else:
                        prompt = f"""Previous conversation:
{conversation_context}

User question: {user_input}"""
                else:
                    prompt = f"""Previous conversation:
{conversation_context}

User question: {user_input}"""

                response = get_response(client, prompt, system_prompt=system_prompt)

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

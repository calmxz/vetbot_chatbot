# Standard library imports
import os  # For environment variables and file operations
import tempfile  # For temporary file creation for audio

# Third-party imports
import streamlit as st  # Web application framework for creating the chatbot UI
from google import genai  # Google's Gemini AI model client
from google.genai import types  # Gemini API types for message formatting
from dotenv import load_dotenv  # Load environment variables from .env file
from kokoro import KPipeline  # Kokoro Text-to-Speech for audio generation
import soundfile as sf  # For saving audio files
import numpy as np  # For audio processing

# LangChain imports for document processing and RAG functionality
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader  # Document loading utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text chunking for embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding model for vector search
from langchain_chroma import Chroma  # Vector database for similarity search  # pyright: ignore[reportMissingImports]

# Load environment variables (API keys, etc.) from .env file
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
def get_kokoro_pipeline():
    """Initialize and cache Kokoro TTS pipeline."""
    return KPipeline(lang_code='a')  # 'a' for American English

def text_to_speech(text, voice='af_heart'):
    """Convert text to speech using Kokoro TTS and return audio file path."""
    try:
        pipeline = get_kokoro_pipeline()

        # Clean up text for better TTS pronunciation (doesn't affect displayed text)
        import re
        cleaned_text = text

        # Remove markdown bold formatting (e.g., **Diagnosis Title** -> Diagnosis Title)
        cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_text)
        # Remove markdown italic formatting (e.g., *text* -> text)
        cleaned_text = re.sub(r'\*([^*]+)\*', r'\1', cleaned_text)
        # Remove any remaining asterisks
        cleaned_text = cleaned_text.replace('*', '')

        # Replace colons with periods for more natural speech
        cleaned_text = cleaned_text.replace(':', '.')

        # Generate audio using Kokoro with cleaned text
        generator = pipeline(cleaned_text, voice=voice)

        # Kokoro yields (graphemes, phonemes, audio) tuples
        # Concatenate all audio chunks together
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            st.error("No audio generated")
            return None

        # Concatenate all audio chunks into one array
        audio_data = np.concatenate(audio_chunks)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            sf.write(fp.name, audio_data, 24000)
            return fp.name
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

@st.cache_resource
def load_and_index_documents(folder_path):
    """Load documents and create vector store for RAG, or load existing if available."""
    chroma_db_path = "./chroma_db"
    if os.path.exists(chroma_db_path) and os.path.exists(os.path.join(chroma_db_path, "chroma.sqlite3")):
        with st.spinner("Loading existing vector store..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings
            )
            st.success("Vector store loaded!")
            return vectorstore
    
    if not os.path.exists(folder_path):
        st.warning(f"Folder '{folder_path}' not found!")
        return None
    
    with st.spinner("Loading documents..."):
        documents = []
        
        try:
            txt_loader = DirectoryLoader(
                folder_path, 
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            st.error(f"Error loading TXT files: {e}")
        
        try:
            pdf_loader = DirectoryLoader(
                folder_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents.extend(pdf_loader.load())
        except Exception as e:
            st.error(f"Error loading PDF files: {e}")
        
        if not documents:
            st.warning("No documents found!")
            return None
        
        st.success(f"Loaded {len(documents)} documents.")
    
    with st.spinner("Splitting documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"Created {len(chunks)} chunks.")
    
    with st.spinner("Creating embeddings and vector store..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
        st.success("Vector store created and ready!")
    
    return vectorstore

@st.cache_resource
def initialize_client():
    """Initialize Gemini API client using API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        st.stop()
    
    return genai.Client(api_key=api_key)

def get_response(client, prompt, system_prompt=None, model="gemini-2.5-flash"):
    """Get streaming response from Gemini API."""
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
        temperature=0.4,
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

def build_direct_answer_prompt(user_input, conversation_context, context=None, source_text=None):
    """Build prompt template for direct answers with emergency detection."""
    if context:
        return f"""You're a friendly veterinary assistant helping a pet owner. Here's our conversation so far:

{conversation_context}

Knowledge base information:
{context}

Current message: {user_input}

Your approach: If this is a general question about pet care, grooming, nutrition, or health information, answer it directly in a helpful, conversational way. Only ask clarifying questions if the situation involves concerning symptoms or potential emergencies where you need to assess urgency.

Write naturally in conversational sentences without bullet points or lists. If it's clearly an emergency (heavy bleeding, can't breathe, seizures, poisoning), give immediate first-aid steps and urge them to get to a vet right away.

This information comes from: {source_text}"""
    else:
        return f"""You're a friendly veterinary assistant helping a pet owner. Here's our conversation:

{conversation_context}

Current message: {user_input}

Your approach: If this is a general question about pet care, answer it directly in a helpful way. Only ask questions if you need to assess the severity of symptoms or an emergency situation.

Write naturally in conversational sentences without bullet points or lists. If it's an emergency, give immediate advice and tell them to see a vet. Always remind them to consult their vet for professional diagnosis."""

def main():
    """Main Streamlit app: sets up UI and handles chat interactions with RAG."""
    st.set_page_config(
        page_title="Veterinary Chatbot",
        page_icon="🐾",
        layout="wide"
    )
    
    st.title("🐾 Veterinary Information Chatbot")
    st.markdown("*Ask questions about pet health and care*")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    if "playing_audio" not in st.session_state:
        st.session_state.playing_audio = None
    
    vectorstore = load_and_index_documents('./documents')
    client = initialize_client()
    system_prompt = load_system_prompt()
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(message["content"])
            with col2:
                if message["role"] == "assistant":
                    # Determine button icon based on playing state
                    is_playing = st.session_state.playing_audio == idx
                    button_icon = "⏸️" if is_playing else "🔊"

                    if st.button(button_icon, key=f"tts_{idx}", help="Play/Stop audio"):
                        if is_playing:
                            # Stop audio by clearing the playing state
                            st.session_state.playing_audio = None
                            st.rerun()
                        else:
                            # Start playing audio
                            st.session_state.playing_audio = idx
                            # Check if audio is already cached
                            if idx in st.session_state.audio_cache:
                                audio_file = st.session_state.audio_cache[idx]
                            else:
                                # Generate if not cached (for old messages)
                                audio_file = text_to_speech(message["content"])
                                if audio_file:
                                    st.session_state.audio_cache[idx] = audio_file
                            st.rerun()

                    # Play audio without showing controls
                    if is_playing and idx in st.session_state.audio_cache:
                        audio_file = st.session_state.audio_cache[idx]
                        # Use HTML audio element with autoplay and hidden controls
                        import base64
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode()
                        audio_html = f'<audio autoplay style="display:none"><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
                        st.markdown(audio_html, unsafe_allow_html=True)
    
    if user_input := st.chat_input("Ask about your pet's health..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            sources_for_display = None
            with st.spinner("Thinking..."):
                conversation_context = ""
                if len(st.session_state.messages) > 1:
                    recent_messages = st.session_state.messages[-6:]
                    conversation_context = "\n".join([
                        f"{msg['role'].title()}: {msg['content']}" 
                        for msg in recent_messages
                    ])
                
                if vectorstore is not None:
                    relevant_docs = vectorstore.similarity_search(user_input, k=3)
                    
                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        sources = set([doc.metadata.get('source', 'documents') for doc in relevant_docs])
                        sources_for_display = sources  # Store for display outside spinner
                        
                        # Create source attribution for prompt
                        source_text = "Knowledge base documents"
                        
                        is_howto = any(phrase in user_input.lower() for phrase in ['how to', 'how do i', 'how can i', 'steps to', 'way to'])
                        
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

                st.markdown(response)

            # Display source information if available (outside spinner so it appears after thinking)
            if sources_for_display:
                # Just show that information comes from Merck Veterinary Manual if applicable
                has_web_cache = any('web_cache' in str(s) for s in sources_for_display)
                if has_web_cache:
                    st.caption("📚 Information sources: Merck Veterinary Manual")

        # Save message to session state BEFORE audio generation to prevent loss on rerun
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Pre-generate audio for assistant response in background
        current_idx = len(st.session_state.messages) - 1  # Adjust index since we already appended
        with st.spinner("Generating audio..."):
            audio_file = text_to_speech(response)
            if audio_file:
                st.session_state.audio_cache[current_idx] = audio_file

        # Rerun to display the TTS button now that audio is ready
        st.rerun()
    
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
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("For informational purposes only. Always consult a veterinarian for medical advice.")

if __name__ == "__main__":
    main()
import os
import tempfile
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_system_prompt(filepath="system_prompt_vet.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.warning("system_prompt_vet.txt not found. Using default fallback prompt.")
        return (
            "You are an expert veterinary clinical assistant. Provide concise, accurate, and up-to-date information for licensed veterinarians and veterinary staff. Always cite sources if possible. Do not give owner-facing advice unless explicitly requested."
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        st.stop()
    return genai.Client(api_key=api_key)

def get_response(client, prompt, system_prompt=None, model="gemini-2.5-flash"):
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
        temperature=0.3,
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
    st.set_page_config(
        page_title="Veterinary Professional Reference Chatbot",
        layout="wide"
    )
    st.title("🩺 Veterinary Clinical Reference Chatbot")
    st.markdown("Ask clinical, diagnostic, or treatment questions — for veterinary professionals only")

    if "vet_messages" not in st.session_state:
        st.session_state.vet_messages = []
    if "vet_audio_cache" not in st.session_state:
        st.session_state.vet_audio_cache = {}
    if "vet_playing_audio" not in st.session_state:
        st.session_state.vet_playing_audio = None
    vectorstore = load_and_index_documents('./documents')
    client = initialize_client()
    system_prompt = load_system_prompt()
    for idx, message in enumerate(st.session_state.vet_messages):
        with st.chat_message(message["role"]):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(message["content"])
            with col2:
                if message["role"] == "assistant":
                    # Determine button icon based on playing state
                    is_playing = st.session_state.vet_playing_audio == idx
                    button_icon = "⏸️" if is_playing else "🔊"

                    if st.button(button_icon, key=f"tts_{idx}", help="Play/Stop audio"):
                        if is_playing:
                            # Stop audio by clearing the playing state
                            st.session_state.vet_playing_audio = None
                            st.rerun()
                        else:
                            # Start playing audio
                            st.session_state.vet_playing_audio = idx
                            # Check if audio is already cached
                            if idx in st.session_state.vet_audio_cache:
                                audio_file = st.session_state.vet_audio_cache[idx]
                            else:
                                # Generate if not cached (for old messages)
                                audio_file = text_to_speech(message["content"])
                                if audio_file:
                                    st.session_state.vet_audio_cache[idx] = audio_file
                            st.rerun()

                    # Play audio without showing controls
                    if is_playing and idx in st.session_state.vet_audio_cache:
                        audio_file = st.session_state.vet_audio_cache[idx]
                        # Use HTML audio element with autoplay and hidden controls
                        import base64
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode()
                        audio_html = f'<audio autoplay style="display:none"><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
                        st.markdown(audio_html, unsafe_allow_html=True)
    if user_input := st.chat_input("Veterinary professional question (diagnosis, clinical signs, drug doses, etc)..."):
        st.session_state.vet_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            sources_for_display = None
            with st.spinner("Consulting knowledge base..."):
                conversation_context = ""
                if len(st.session_state.vet_messages) > 1:
                    recent_messages = st.session_state.vet_messages[-6:]
                    conversation_context = "\n".join([
                        f"{msg['role'].title()}: {msg['content']}" 
                        for msg in recent_messages
                    ])
                if vectorstore is not None:
                    relevant_docs = vectorstore.similarity_search(user_input, k=3)
                    if relevant_docs:
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        sources = set([doc.metadata.get('source', 'documents') for doc in relevant_docs])
                        sources_for_display = sources
                        source_text = "Knowledge base documents"
                        prompt = f"[PROFESSIONAL MODE]\nContext from clinical knowledge base:\n{context}\n\nPrevious conversation:\n{conversation_context}\n\nUser question: {user_input}\n\nSource: {source_text}"
                    else:
                        prompt = f"[PROFESSIONAL MODE]\nPrevious conversation:\n{conversation_context}\n\nUser question: {user_input}"
                else:
                    prompt = f"[PROFESSIONAL MODE]\nPrevious conversation:\n{conversation_context}\n\nUser question: {user_input}"
                response = get_response(client, prompt, system_prompt=system_prompt)

                st.markdown(response)

            if sources_for_display:
                has_web_cache = any('web_cache' in str(s) for s in sources_for_display)
                if has_web_cache:
                    st.caption("Information sources: Merck Veterinary Manual and other verified materials")

        # Save message to session state BEFORE audio generation to prevent loss on rerun
        st.session_state.vet_messages.append({"role": "assistant", "content": response})

        # Pre-generate audio for assistant response in background
        current_idx = len(st.session_state.vet_messages) - 1  # Adjust index since we already appended
        with st.spinner("Generating audio..."):
            audio_file = text_to_speech(response)
            if audio_file:
                st.session_state.vet_audio_cache[current_idx] = audio_file

        # Rerun to display the TTS button now that audio is ready
        st.rerun()
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
            st.session_state.vet_messages = []
            st.rerun()
        st.markdown("---")
        st.caption("For licensed professional use only.")

if __name__ == "__main__":
    main()

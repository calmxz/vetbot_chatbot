# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import os
import streamlit as st  # Web framework for creating the chatbot UI
from google import genai  # Google's Gemini AI API
from google.genai import types  # Types for structuring API requests
from dotenv import load_dotenv  # Load environment variables (API keys)
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader  # Load documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split text into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Generate embeddings for semantic search
from langchain_community.vectorstores import Chroma  # Vector database for storing embeddings

# Load environment variables from .env file (contains GEMINI_API_KEY)
load_dotenv()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_system_prompt(filepath="system_prompt.txt"):
    """
    Load the system prompt from a text file.
    The system prompt defines the bot's personality and behavior guidelines.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()  # Read and return the file content
    except FileNotFoundError:
        # If file doesn't exist, use a default prompt
        st.warning("⚠️ system_prompt.txt not found. Using default fallback prompt.")
        return (
            "You are a helpful and friendly veterinary assistant chatbot. "
            "Always remind users to consult a veterinarian for professional medical advice."
        )

@st.cache_resource
def load_and_index_documents(folder_path):
    """
    RAG SETUP: Load documents and create a searchable vector database.
    
    This function:
    1. Loads all PDF and TXT files from the specified folder
    2. Splits them into small chunks (for better retrieval)
    3. Converts chunks into embeddings (vector representations)
    4. Stores them in a Chroma vector database for fast similarity search
    
    @st.cache_resource caches the result so it only runs once per session.
    """
    if not os.path.exists(folder_path):
        st.warning(f"Folder '{folder_path}' not found!")
        return None
    
    # STEP 1: Load documents from the folder
    with st.spinner("Loading documents..."):
        documents = []
        try:
            # Load all .txt files recursively from subdirectories
            txt_loader = DirectoryLoader(
                folder_path, 
                glob="**/*.txt",  # ** means search in all subdirectories
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            st.error(f"Error loading TXT files: {e}")
        
        try:
            # Load all .pdf files recursively from subdirectories
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
    
    # STEP 2: Split documents into smaller chunks
    # This is important because:
    # - LLMs have token limits
    # - Smaller chunks provide more focused information
    # - We can retrieve only the most relevant pieces
    with st.spinner("Splitting documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Each chunk is ~500 characters
            chunk_overlap=50,    # 50 characters overlap between chunks (keeps context)
            separators=["\n\n", "\n", " ", ""]  # Try to split at these boundaries first
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"Created {len(chunks)} chunks.")
    
    # STEP 3: Create embeddings and vector store
    # Embeddings convert text into numerical vectors that capture meaning
    # Similar meanings = similar vectors = can be found via similarity search
    with st.spinner("Creating embeddings and vector store..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Pre-trained model for generating embeddings
            model_kwargs={'device': 'cpu'}  # Use CPU (change to 'cuda' for GPU)
        )
        
        # Store embeddings in Chroma database for fast retrieval
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"  # Save to disk so we don't rebuild each time
        )
        st.success("Vector store ready!")
    
    return vectorstore

@st.cache_resource
def initialize_client():
    """
    Create and return a Gemini API client.
    Uses the API key from environment variables.
    @st.cache_resource ensures we only create one client per session.
    """
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def get_response(client, prompt, system_prompt=None, model="gemini-2.5-flash"):
    """
    Send a prompt to Gemini and return the response.
    
    Args:
        client: Gemini API client
        prompt: The user's question or conversation context
        system_prompt: Instructions for how the AI should behave
        model: Which Gemini model to use (flash is faster/cheaper)
    
    Returns:
        The AI's text response
    """
    # Combine system prompt and user prompt
    if system_prompt:
        full_prompt = f"[Instruction — read before replying]\n{system_prompt}\n\n[User question]\n{prompt}"
    else:
        full_prompt = prompt

    # Structure the API request
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)],
        )
    ]

    # Stream the response (process it as it comes in)
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
    ):
        if chunk.text:
            response_text += chunk.text
    return response_text

def build_direct_answer_prompt(user_input, conversation_context, context=None, source_text=None):
    """
    Build a detailed prompt for the AI with conversation history and knowledge base context.
    
    This function isn't currently used in the main flow, but could be used for
    more advanced prompting strategies.
    """
    
    if context:
        # With RAG context - the AI has retrieved information from the knowledge base
        return f"""You're a friendly veterinary assistant helping a pet owner. Here's our conversation so far:

{conversation_context}

Knowledge base information:
{context}

Current message: {user_input}

Your approach: If this is a general question about pet care, grooming, nutrition, or health information, answer it directly in a helpful, conversational way. Only ask clarifying questions if the situation involves concerning symptoms or potential emergencies where you need to assess urgency.

Write naturally in conversational sentences without bullet points or lists. If it's clearly an emergency (heavy bleeding, can't breathe, seizures, poisoning), give immediate first-aid steps and urge them to get to a vet right away.

This information comes from: {source_text}"""
    else:
        # Without RAG context - the AI only has the conversation history
        return f"""You're a friendly veterinary assistant helping a pet owner. Here's our conversation:

{conversation_context}

Current message: {user_input}

Your approach: If this is a general question about pet care, answer it directly in a helpful way. Only ask questions if you need to assess the severity of symptoms or an emergency situation.

Write naturally in conversational sentences without bullet points or lists. If it's an emergency, give immediate advice and tell them to see a vet. Always remind them to consult their vet for professional diagnosis."""

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application - creates the chatbot UI and handles interactions"""
    
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Veterinary Chatbot",
        page_icon="🐾",
        layout="wide"
    )
    st.title("🐾 Veterinary Information Chatbot")
    st.markdown("*Ask questions about pet health and care*")
    
    # Initialize the conversation history (persists across user inputs)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load resources (these are cached, so they only run once)
    vectorstore = load_and_index_documents('./documents')  # RAG knowledge base
    client = initialize_client()  # Gemini API client
    system_prompt = load_system_prompt()  # AI behavior instructions
    
    # Display previous messages in the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle new user input
    if user_input := st.chat_input("Ask about your pet's health..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # STEP 1: Build conversation context from recent messages
                # This helps the AI remember what was said earlier
                conversation_context = ""
                if len(st.session_state.messages) > 1:
                    recent_messages = st.session_state.messages[-6:]  # Last 6 messages
                    conversation_context = "\n".join([
                        f"{msg['role'].title()}: {msg['content']}" 
                        for msg in recent_messages
                    ])
                
                # STEP 2: Retrieve relevant documents from knowledge base (RAG)
                if vectorstore is not None:
                    # Search for the 3 most relevant document chunks
                    relevant_docs = vectorstore.similarity_search(user_input, k=3)
                    
                    if relevant_docs:
                        # Extract the text content from retrieved documents
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        # Get the source filenames for attribution
                        sources = set([doc.metadata.get('source', 'documents') for doc in relevant_docs])
                        source_text = ", ".join(sources)
                        
                        # Build prompt with retrieved context
                        prompt = f"""Context from knowledge base:
{context}

Previous conversation:
{conversation_context}

User question: {user_input}

Source: {source_text}"""
                    else:
                        # No relevant docs found - use conversation history only
                        prompt = f"""Previous conversation:
{conversation_context}

User question: {user_input}"""
                else:
                    # Vector store failed to load - use conversation history only
                    prompt = f"""Previous conversation:
{conversation_context}

User question: {user_input}"""
                
                # STEP 3: Send prompt to Gemini and get response
                response = get_response(client, prompt, system_prompt=system_prompt)
                st.markdown(response)
        
        # Add AI response to conversation history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with information and controls
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot provides veterinary information using:
        - **RAG** (Retrieval-Augmented Generation)
        - **Gemini Flash** LLM
        - **LangChain** for document processing
        
        📁 Documents are loaded from `./documents/` folder
        
        **Conversation Style:** Step-by-step guidance with natural follow-ups
        """)
        
        # Button to clear conversation history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()  # Refresh the page to clear the UI
        
        st.markdown("---")
        st.caption("⚠️ For informational purposes only. Always consult a veterinarian for medical advice.")

# Run the application
if __name__ == "__main__":
    main()
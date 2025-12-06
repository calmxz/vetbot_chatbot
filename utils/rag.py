"""RAG (Retrieval-Augmented Generation) utilities for document loading and indexing."""

import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]

from .config import Config


@st.cache_resource
def get_embeddings():
    """Initialize and cache the embeddings model (singleton)."""
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': Config.EMBEDDING_DEVICE}
    )


@st.cache_resource
def load_and_index_documents(folder_path: str = None):
    """
    Load documents and create vector store for RAG, or load existing if available.

    Args:
        folder_path: Path to folder containing documents (default: Config.DOCUMENTS_FOLDER)

    Returns:
        Chroma vectorstore instance, or None if loading failed
    """
    if folder_path is None:
        folder_path = Config.DOCUMENTS_FOLDER

    chroma_db_path = Config.CHROMA_DB_PATH
    embeddings = get_embeddings()

    # Try to load existing vector store
    if os.path.exists(chroma_db_path) and os.path.exists(os.path.join(chroma_db_path, "chroma.sqlite3")):
        with st.spinner("Loading existing vector store..."):
            vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings
            )
            st.success("Vector store loaded!")
            return vectorstore

    # Check if documents folder exists
    if not os.path.exists(folder_path):
        st.warning(f"Folder '{folder_path}' not found!")
        return None

    # Load documents
    documents = _load_documents_from_folder(folder_path)
    if not documents:
        return None

    # Split documents into chunks
    chunks = _split_documents(documents)

    # Create and return vector store
    return _create_vectorstore(chunks, embeddings, chroma_db_path)


def _load_documents_from_folder(folder_path: str) -> list:
    """Load TXT and PDF documents from a folder."""
    documents = []

    with st.spinner("Loading documents..."):
        # Load TXT files
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

        # Load PDF files
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
            return []

        st.success(f"Loaded {len(documents)} documents.")

    return documents


def _split_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    with st.spinner("Splitting documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"Created {len(chunks)} chunks.")

    return chunks


def _create_vectorstore(chunks: list, embeddings, persist_directory: str):
    """Create a Chroma vector store from document chunks."""
    with st.spinner("Creating embeddings and vector store..."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        st.success("Vector store created and ready!")

    return vectorstore

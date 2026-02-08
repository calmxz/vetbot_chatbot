"""RAG (Retrieval-Augmented Generation) utilities for document loading and indexing."""

import glob as glob_module
import hashlib
import logging
import os
import re
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]

from .config import Config

logger = logging.getLogger(__name__)

# Known pet types and categories from folder structure
_PET_TYPES = {"cats", "dogs", "general"}
_CATEGORIES = {"care", "clinical", "diseases", "first-aid"}


@st.cache_resource
def get_embeddings():
    """Initialize and cache the embeddings model (singleton)."""
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': Config.EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )


_SENTINEL_ID = "__vectorstore_metadata__"


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's content."""
    h = hashlib.new(Config.INDEXING_HASH_ALGORITHM)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_stored_embedding_model(vectorstore) -> str | None:
    """Read the embedding model name from the sentinel document."""
    try:
        result = vectorstore.get(ids=[_SENTINEL_ID], include=["metadatas"])
        if result and result["metadatas"]:
            return result["metadatas"][0].get("embedding_model")
    except Exception:
        pass
    return None


def _store_vectorstore_metadata(vectorstore):
    """Upsert a sentinel document with vectorstore metadata."""
    vectorstore._collection.upsert(
        ids=[_SENTINEL_ID],
        documents=["Vectorstore metadata sentinel"],
        metadatas=[{"embedding_model": Config.EMBEDDING_MODEL, "is_sentinel": "true"}],
    )


def _get_stored_file_hashes(vectorstore) -> dict:
    """Get content hashes for all indexed files from vectorstore metadata.

    Returns:
        Dict mapping source file path to content_hash
    """
    try:
        result = vectorstore.get(
            where={"content_hash": {"$ne": ""}},
            include=["metadatas"]
        )
        hashes = {}
        if result and result["metadatas"]:
            for meta in result["metadatas"]:
                source = meta.get("source", "")
                content_hash = meta.get("content_hash", "")
                if source and content_hash and meta.get("is_sentinel") != "true":
                    hashes[source] = content_hash
        return hashes
    except Exception as e:
        logger.warning(f"Failed to read stored file hashes: {e}")
        return {}


def _get_current_file_hashes(folder_path: str) -> dict:
    """Compute content hashes for all indexable files in folder.

    Returns:
        Dict mapping file path to content_hash
    """
    hashes = {}
    for ext in ("**/*.txt", "**/*.pdf"):
        for file_path in glob_module.glob(os.path.join(folder_path, ext), recursive=True):
            norm_path = os.path.normpath(file_path)
            hashes[norm_path] = _compute_file_hash(norm_path)
    return hashes


def _delete_chunks_by_source(vectorstore, source_path: str):
    """Delete all chunks from vectorstore that came from a given source file."""
    try:
        norm = os.path.normpath(source_path)
        result = vectorstore.get(
            where={"source": norm},
            include=[]
        )
        if result and result["ids"]:
            vectorstore.delete(ids=result["ids"])
            logger.info(f"Deleted {len(result['ids'])} chunks from {norm}")
    except Exception as e:
        logger.warning(f"Failed to delete chunks for {source_path}: {e}")


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

    # Check if documents folder exists
    if not os.path.exists(folder_path):
        st.warning(f"Folder '{folder_path}' not found!")
        return None

    # Try to load existing vector store
    if os.path.exists(chroma_db_path) and os.path.exists(os.path.join(chroma_db_path, "chroma.sqlite3")):
        with st.spinner("Loading existing vector store..."):
            vectorstore = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings
            )

        # Check if embedding model matches
        stored_model = _get_stored_embedding_model(vectorstore)
        if stored_model and stored_model != Config.EMBEDDING_MODEL:
            logger.info(
                f"Embedding model changed ({stored_model} -> {Config.EMBEDDING_MODEL}), "
                "full re-index required"
            )
            st.info("Embedding model changed. Re-indexing all documents...")
            # Fall through to full re-index below
        else:
            # Incremental indexing: check for new/modified/deleted files
            stored_hashes = _get_stored_file_hashes(vectorstore)
            current_hashes = _get_current_file_hashes(folder_path)

            new_files = [f for f in current_hashes if f not in stored_hashes]
            modified_files = [
                f for f in current_hashes
                if f in stored_hashes and current_hashes[f] != stored_hashes[f]
            ]
            deleted_files = [f for f in stored_hashes if f not in current_hashes]

            if not new_files and not modified_files and not deleted_files:
                st.success("Vector store loaded! (all documents up to date)")
                return vectorstore

            logger.info(
                f"Incremental update: {len(new_files)} new, "
                f"{len(modified_files)} modified, {len(deleted_files)} deleted"
            )

            with st.spinner("Updating vector store..."):
                # Delete chunks from modified and deleted files
                for f in modified_files + deleted_files:
                    _delete_chunks_by_source(vectorstore, f)

                # Load and index new/modified files
                files_to_index = new_files + modified_files
                if files_to_index:
                    documents = _load_specific_files(files_to_index)
                    if documents:
                        documents = _enrich_metadata(documents, folder_path)
                        # Add content hashes to metadata
                        for doc in documents:
                            source = os.path.normpath(doc.metadata.get("source", ""))
                            doc.metadata["content_hash"] = current_hashes.get(source, "")
                        chunks = _split_documents(documents)
                        vectorstore.add_documents(chunks)

                _store_vectorstore_metadata(vectorstore)
                st.success(f"Vector store updated! ({len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted)")

            return vectorstore

    # Full index: load all documents
    documents = _load_documents_from_folder(folder_path)
    if not documents:
        return None

    # Enrich metadata from directory structure
    documents = _enrich_metadata(documents, folder_path)

    # Add content hashes to metadata
    current_hashes = _get_current_file_hashes(folder_path)
    for doc in documents:
        source = os.path.normpath(doc.metadata.get("source", ""))
        doc.metadata["content_hash"] = current_hashes.get(source, "")

    # Split documents into chunks
    chunks = _split_documents(documents)

    # Create vector store
    vectorstore = _create_vectorstore(chunks, embeddings, chroma_db_path)

    # Store sentinel metadata
    _store_vectorstore_metadata(vectorstore)

    return vectorstore


def _load_specific_files(file_paths: list) -> list:
    """Load documents from specific file paths (for incremental indexing)."""
    from langchain_community.document_loaders import TextLoader as TL, PyPDFLoader as PL
    documents = []
    for file_path in file_paths:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PL(file_path)
            else:
                loader = TL(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    logger.info(f"Loaded {len(documents)} documents from {len(file_paths)} files")
    return documents


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
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""]
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


def _enrich_metadata(documents: list, base_folder: str) -> list:
    """Parse document source paths to extract pet_type and category metadata.

    Expects paths like: documents/cats/diseases/fip.txt
    Extracts pet_type='cats', category='diseases' from the path components.
    """
    base_folder = os.path.normpath(base_folder)
    for doc in documents:
        source = doc.metadata.get("source", "")
        source_norm = os.path.normpath(source)

        # Get relative path from base folder
        try:
            rel_path = os.path.relpath(source_norm, base_folder)
        except ValueError:
            rel_path = source_norm

        parts = rel_path.replace("\\", "/").split("/")

        pet_type = "general"
        category = "general"

        for part in parts:
            part_lower = part.lower()
            if part_lower in _PET_TYPES:
                pet_type = part_lower
            if part_lower in _CATEGORIES:
                category = part_lower

        doc.metadata["pet_type"] = pet_type
        doc.metadata["category"] = category

    return documents


def detect_metadata_filters(query: str) -> dict | None:
    """Detect species keywords in query and return ChromaDB where filter.

    Returns a filter that includes the detected species AND general documents,
    or None if no species is detected.
    """
    query_lower = query.lower()

    cat_keywords = r'\b(cat|cats|kitten|kittens|feline|felines)\b'
    dog_keywords = r'\b(dog|dogs|puppy|puppies|canine|canines)\b'

    has_cat = bool(re.search(cat_keywords, query_lower))
    has_dog = bool(re.search(dog_keywords, query_lower))

    if has_cat and has_dog:
        # Both species mentioned, no filtering needed
        return None
    elif has_cat:
        return {"pet_type": {"$in": ["cats", "general"]}}
    elif has_dog:
        return {"pet_type": {"$in": ["dogs", "general"]}}

    return None


@st.cache_resource
def _get_cross_encoder():
    """Initialize and cache the cross-encoder model for re-ranking."""
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading cross-encoder model: {Config.RERANK_MODEL}")
    return CrossEncoder(Config.RERANK_MODEL)


def rerank_documents(query: str, docs_with_scores: list,
                     top_n: int = None) -> list:
    """Re-rank documents using cross-encoder for better precision.

    Args:
        query: Search query string
        docs_with_scores: List of (Document, embedding_score) tuples
        top_n: Number of top results to return (default: Config.RERANK_TOP_N)

    Returns:
        List of (Document, cross_encoder_score) tuples sorted by score descending
    """
    if top_n is None:
        top_n = Config.RERANK_TOP_N

    if not docs_with_scores:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
    scores = cross_encoder.predict(pairs)

    # Pair documents with cross-encoder scores and sort descending
    reranked = sorted(
        zip([doc for doc, _ in docs_with_scores], scores),
        key=lambda x: x[1],
        reverse=True
    )

    logger.info(
        f"Re-ranked {len(reranked)} documents, "
        f"returning top {min(top_n, len(reranked))}"
    )

    return [(doc, float(score)) for doc, score in reranked[:top_n]]


def retrieve_relevant_documents(vectorstore, query: str,
                                k: int = None,
                                distance_threshold: float = None,
                                where_filter: dict = None,
                                rerank: bool = None) -> list:
    """Retrieve documents with similarity scores, filtered by distance threshold.

    When re-ranking is enabled, retrieves more candidates initially, filters
    by distance threshold, then re-ranks to the top N.

    Args:
        vectorstore: ChromaDB vectorstore instance
        query: Search query string
        k: Number of candidates to retrieve (default: Config.SIMILARITY_SEARCH_K
           or Config.RERANK_CANDIDATE_K when re-ranking)
        distance_threshold: Max distance for relevance (default: Config.SIMILARITY_DISTANCE_THRESHOLD)
        where_filter: ChromaDB metadata filter dict
        rerank: Whether to apply cross-encoder re-ranking (default: Config.RERANK_ENABLED)

    Returns:
        List of (Document, score) tuples
    """
    if rerank is None:
        rerank = Config.RERANK_ENABLED
    if k is None:
        k = Config.RERANK_CANDIDATE_K if rerank else Config.SIMILARITY_SEARCH_K
    if distance_threshold is None:
        distance_threshold = Config.SIMILARITY_DISTANCE_THRESHOLD

    search_kwargs = {"k": k}
    if where_filter:
        search_kwargs["filter"] = where_filter
        logger.info(f"Applying metadata filter: {where_filter}")

    docs_with_scores = vectorstore.similarity_search_with_score(
        query, **search_kwargs
    )

    # Filter by distance threshold
    relevant = [
        (doc, score) for doc, score in docs_with_scores
        if score < distance_threshold
    ]

    logger.info(
        f"Retrieval for query '{query[:80]}': "
        f"{len(docs_with_scores)} candidates, "
        f"{len(relevant)} passed threshold ({distance_threshold})"
    )

    # Apply cross-encoder re-ranking if enabled and we have candidates
    if rerank and relevant:
        relevant = rerank_documents(query, relevant)

    return relevant


def build_rag_context(docs_with_scores: list) -> str:
    """Build labeled RAG context string with token budget enforcement.

    Each chunk is numbered and labeled with source file and relevance score.
    Chunks are ordered by relevance (most relevant first).
    A token budget prevents context from exceeding model limits.

    Args:
        docs_with_scores: List of (Document, score) tuples

    Returns:
        Labeled, budget-constrained context string
    """
    if not docs_with_scores:
        return ""

    max_chars = Config.MAX_CONTEXT_TOKENS * Config.CHARS_PER_TOKEN_ESTIMATE
    context_parts = []
    total_chars = 0
    dropped = 0

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "unknown")
        # Use forward slashes and get just the relative part
        source_short = source.replace("\\", "/").split("documents/")[-1] if "documents" in source.replace("\\", "/") else os.path.basename(source)

        header = f"[Source {i}: {source_short} | Score: {score:.3f}]"
        chunk_text = f"{header}\n{doc.page_content}"

        if total_chars + len(chunk_text) > max_chars:
            dropped += 1
            continue

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    if dropped:
        logger.info(f"Context budget: included {len(context_parts)} chunks, dropped {dropped} due to token limit")

    logger.info(f"RAG context assembled: {len(context_parts)} chunks, ~{total_chars // Config.CHARS_PER_TOKEN_ESTIMATE} tokens")

    return "\n\n".join(context_parts)


def build_prompt_with_context(user_input: str, rag_context: str,
                              mode: str = "normal",
                              has_vectorstore: bool = True) -> str:
    """Assemble the user prompt with RAG context for Gemini API.

    Args:
        user_input: The user's question
        rag_context: RAG context string (may be empty)
        mode: "normal" for pet owner, "professional" for vet mode
        has_vectorstore: Whether a vectorstore is available

    Returns:
        Assembled prompt string
    """
    prefix = "[PROFESSIONAL MODE]\n" if mode == "professional" else ""

    if rag_context:
        prompt = (
            f"{prefix}Context from knowledge base:\n{rag_context}\n\n"
            f"User question: {user_input}"
        )
    elif has_vectorstore:
        prompt = (
            f"{prefix}[No relevant knowledge base documents were found for this query.]\n\n"
            f"User question: {user_input}"
        )
    else:
        prompt = (
            f"{prefix}[Knowledge base is unavailable.]\n\n"
            f"User question: {user_input}"
        )

    return prompt

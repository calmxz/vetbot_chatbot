# Veterinary Assistant Chatbot

A Streamlit-based veterinary chatbot with RAG (Retrieval-Augmented Generation) capabilities and text-to-speech output.

## Features

- **Two Modes**: Pet Owner (friendly) and Veterinary Professional (clinical)
- **RAG-Powered**: Retrieves relevant information from a curated knowledge base
- **Text-to-Speech**: Audio responses via Kokoro TTS
- **Conversation Memory**: Maintains context across the chat session

## Tech Stack

- **LLM**: Google Gemini API (`gemini-2.5-flash`)
- **RAG**: LangChain + ChromaDB with HuggingFace embeddings (`all-MiniLM-L6-v2`)
- **TTS**: Kokoro TTS
- **UI**: Streamlit

## Setup

1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

**Pet Owner Mode** (friendly, empathetic responses):
```bash
streamlit run chatbot.py
```

**Veterinary Professional Mode** (clinical reference):
```bash
streamlit run chatbot_vet.py
```

## Document Structure

Knowledge base organized by pet type in `./documents/`:

```
documents/
├── cats/           # Cat-specific content
├── dogs/           # Dog-specific content
└── general/        # Content for both
```

Each folder contains subfolders: `care/`, `clinical/`, `diseases/`, `first-aid/`

To update the knowledge base:
1. Add documents to the appropriate folder
2. Delete `./chroma_db/` to force reindexing
3. Restart the application

## Configuration

Tunable parameters in `utils/config.py`:
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - Document splitting
- `SIMILARITY_SEARCH_K` - Number of retrieved documents
- `TEMPERATURE_NORMAL`, `TEMPERATURE_PROFESSIONAL` - Response creativity
- `MAX_CONTEXT_MESSAGES` - Conversation history length

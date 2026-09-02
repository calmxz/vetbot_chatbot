# Veterinary Assistant Chatbot

A Streamlit-based veterinary chatbot powered by Google Gemini with a Retrieval-Augmented Generation (RAG) pipeline and text-to-speech output. Answers are grounded in a curated veterinary knowledge base rather than the model's memory alone.

## Features

- **Two Modes**: Pet Owner (friendly, empathetic) and Veterinary Professional (clinical, technical)
- **RAG-Powered**: Retrieves relevant chunks from the knowledge base, re-ranks them with a cross-encoder, and cites sources in the prompt
- **Species-Aware Retrieval**: Detects cat/dog intent in the query and filters results by species metadata
- **Query Rewriting**: Rewrites conversational follow-ups ("what about kittens?") into standalone search queries
- **Incremental Indexing**: Only re-embeds documents that changed (SHA-256 file hashing); tracks which embedding model built the index
- **Streaming Responses**: Token-by-token output with exponential backoff retry on API errors
- **Text-to-Speech**: Audio playback of responses via Kokoro TTS
- **Conversation Memory**: Maintains recent context across the chat session
- **Input Validation**: Length limits and basic sanitization on user input

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Google Gemini API (`gemini-2.5-flash`) via `google-genai` |
| UI | Streamlit |
| RAG orchestration | LangChain |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) via HuggingFace / sentence-transformers |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Document parsing | PyPDF (PDF), plain text (TXT) |
| TTS | Kokoro TTS (voice `af_heart`) + soundfile |
| Scraping | requests + BeautifulSoup4 (ASPCA articles) |

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

First run downloads the embedding and re-ranking models and indexes `./documents/` into `./chroma_db/`. This takes a few minutes; later runs reuse the index.

## Usage

**Pet Owner Mode** (friendly, empathetic responses):
```bash
streamlit run chatbot.py
```

**Veterinary Professional Mode** (clinical reference):
```bash
streamlit run chatbot_vet.py
```

Both apps share the same knowledge base and RAG pipeline. They differ in system prompt, temperature, and tone.

## How a Query Is Answered

1. **Sanitize** user input (length cap, basic cleaning)
2. **Rewrite** the query into a standalone form using recent conversation history (lightweight Gemini call, temperature 0)
3. **Detect species filter** (cats / dogs / general) from the query wording
4. **Retrieve** top 10 candidates from ChromaDB, drop any beyond the distance threshold
5. **Re-rank** survivors with the cross-encoder and keep the top 4
6. **Assemble context**: labeled chunks with source path and score, trimmed to a token budget
7. **Generate** a streamed Gemini response using the system prompt, conversation history, and assembled context
8. **Optional TTS**: convert the reply to audio on request

## Project Structure

```
chatbot_test/
├── chatbot.py                 # Pet Owner Mode - Streamlit app
├── chatbot_vet.py             # Vet Professional Mode - Streamlit app
├── scraper.py                 # ASPCA article scraper for the knowledge base
├── system_prompt.txt          # System prompt (pet owner mode)
├── system_prompt_vet.txt      # System prompt (professional mode)
├── requirements.txt
├── .env                       # GEMINI_API_KEY (not committed)
│
├── utils/
│   ├── config.py              # All tunable constants
│   ├── rag.py                 # Indexing, retrieval, re-ranking, context assembly
│   ├── query.py               # Query rewriting
│   ├── tts.py                 # Kokoro TTS pipeline
│   └── audio_player.py        # Streamlit audio button component
│
├── documents/                 # Knowledge base
│   ├── cats/{care,clinical,diseases,first-aid}/
│   ├── dogs/{care,clinical,diseases,first-aid}/
│   └── general/
│
└── chroma_db/                 # Vector store (auto-generated, gitignored)
```

## Knowledge Base

Documents live in `./documents/`, organized by species. The top-level folder (`cats`, `dogs`, `general`) becomes the `species` metadata field used for filtering at retrieval time. PDF and TXT files are supported.

**Adding or updating documents:**
1. Place the file in the appropriate folder
2. Restart the application. Indexing is incremental: new and changed files are embedded, removed files are pruned, unchanged files are skipped.

**Full rebuild** is required after changing `EMBEDDING_MODEL`, `CHUNK_SIZE`, or `CHUNK_OVERLAP`:
```bash
rm -rf chroma_db   # or delete the folder manually
```

**Scraping ASPCA articles:** `scraper.py` fetches a fixed list of ASPCA pet-care pages into `documents/`. Edit the URL list in the file to add sources.

## Configuration

All tunable parameters are in `utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1200 / 150 | Chunk size in characters and overlap between chunks |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `SIMILARITY_DISTANCE_THRESHOLD` | 0.5 | Max ChromaDB distance to keep a candidate (lower is more similar) |
| `RERANK_ENABLED` | `True` | Toggle cross-encoder re-ranking |
| `RERANK_CANDIDATE_K` / `RERANK_TOP_N` | 10 / 4 | Candidates fetched vs. chunks kept after re-ranking |
| `MAX_CONTEXT_TOKENS` | 3000 | Token budget for assembled RAG context |
| `QUERY_REWRITE_ENABLED` | `True` | Toggle follow-up query rewriting |
| `DEFAULT_MODEL` | `gemini-2.5-flash` | Gemini model for responses |
| `TEMPERATURE_NORMAL` / `TEMPERATURE_PROFESSIONAL` | 0.4 / 0.3 | Sampling temperature per mode |
| `MAX_CONTEXT_MESSAGES` | 6 | Conversation turns sent to the model |
| `MAX_INPUT_LENGTH` | 4000 | Max user input characters |
| `MAX_RETRIES` / `RETRY_BASE_DELAY` | 3 / 1.0s | API retry policy |
| `TTS_DEFAULT_VOICE` | `af_heart` | Kokoro voice |

## Development

```bash
black .     # format
flake8      # lint
pytest      # tests
```

Modify tone or behavior by editing `system_prompt.txt` (pet owner) or `system_prompt_vet.txt` (professional). No code change needed.

## Disclaimer

Pet Owner Mode is for informational purposes only. Always consult a licensed veterinarian for medical advice. Veterinary Professional Mode is intended for licensed professional use.

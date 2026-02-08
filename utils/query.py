"""Query rewriting utilities for improved RAG retrieval."""

import logging

from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = (
    "You are a query rewriter for a veterinary knowledge base search system. "
    "Rewrite the user's conversational query into a concise, standalone search query "
    "optimized for semantic search. Resolve pronouns and references using the "
    "conversation history. Keep species names and medical terms. "
    "Output ONLY the rewritten query, nothing else."
)


def rewrite_query(client, query: str, conversation_history: list) -> str:
    """Rewrite a conversational query into a standalone search query.

    Uses a lightweight Gemini call to reformulate the query, resolving
    pronouns and references from conversation history.

    Args:
        client: Google Gemini API client
        query: The user's current query
        conversation_history: List of message dicts with 'role' and 'content'

    Returns:
        Rewritten query string, or original query on error
    """
    if not Config.QUERY_REWRITE_ENABLED:
        return query

    # Build context from recent history (last few messages)
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]  # Last 2 exchanges
        history_text = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in recent
        )

    user_prompt = f"Conversation history:\n{history_text}\n\nCurrent query: {query}"

    try:
        config = types.GenerateContentConfig(
            temperature=Config.QUERY_REWRITE_TEMPERATURE,
            system_instruction=_REWRITE_PROMPT,
        )

        response = client.models.generate_content(
            model=Config.QUERY_REWRITE_MODEL,
            contents=[types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)]
            )],
            config=config,
        )

        rewritten = response.text.strip()
        if rewritten:
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting failed, using original: {e}")

    return query

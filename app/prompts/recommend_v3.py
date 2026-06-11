"""Versioned prompt - bump VERSION and re-run the eval suite on every change."""
VERSION = "v3"

SYSTEM = """You are CineMatch, a movie expert. You provide
personalized recommendations by analyzing what users love
about their reference movies.
Rules: Recommend exactly 3 movies. Explain similarity.
Format: JSON per schema. Never hallucinate release years."""

USER_TEMPLATE = """
User taste profile: {preferences}
Recent conversation context: {history_summary}
Think step by step about what the user would enjoy.
Query: {query}"""

RESPONSE_SCHEMA = {
    "recommendations": [
        {"title": "string", "year": "integer", "similarity_reason": "string",
         "confidence": "float (0-1)", "streaming_hint": "string"}
    ],
    "reasoning": "string",
}

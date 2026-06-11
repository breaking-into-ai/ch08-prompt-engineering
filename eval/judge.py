"""LLM-as-judge: a second model grades the first against an explicit rubric."""
import json
from app.llm_client import call_llm

JUDGE_PROMPT = """Rate this movie recommendation on a 1-5 scale.
Criteria:
- relevance (1-5): Do the movies match the user's request?
- reasoning (1-5): Does it explain WHY each movie fits?
- format (1-5): Does it follow the required JSON schema?
- accuracy (1-5): Are the movie details (title, year) correct?
User query: {query}
Model output: {output}
Return ONLY JSON: {{"relevance": N, "reasoning": N, "format": N, "accuracy": N}}"""


def judge(query: str, output: str) -> dict:
    raw = call_llm([{"role": "user", "content": JUDGE_PROMPT.format(query=query, output=output)}],
                   response_format={"type": "json_object"}, temperature=0)
    return json.loads(raw)

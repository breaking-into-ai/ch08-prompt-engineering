"""Production LLM client: retries, exponential backoff, timeouts, streaming."""
import time
from openai import OpenAI, RateLimitError, APITimeoutError

client = OpenAI()  # reads OPENAI_API_KEY from environment
MODEL = "gpt-4o-mini"  # pin an exact dated version in production


def call_llm(messages, model=MODEL, max_retries=3, backoff=2, **kwargs):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, timeout=30, **kwargs)
            return response.choices[0].message.content
        except RateLimitError:
            wait = backoff ** attempt
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APITimeoutError:
            print(f"Timeout on attempt {attempt + 1}")
    raise RuntimeError("LLM call failed after retries")


async def stream_llm(messages, model=MODEL, **kwargs):
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Chapter 8 — Prompt Engineering and Building with LLM APIs

Companion code for **Breaking Into AI**, Chapter 8: the complete LLM-powered FlickSage Q&A application — versioned prompts, a retrying LLM client, a token-budgeted context pipeline, a streaming FastAPI endpoint, and an LLM-as-judge evaluation suite.

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload      # streaming /recommend endpoint
python -m eval.run_eval            # evaluation-driven development loop
```

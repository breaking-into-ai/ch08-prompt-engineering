from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .context import ContextPipeline
from .llm_client import stream_llm
from .models import RecommendRequest
from .prompts.recommend_v3 import SYSTEM

app = FastAPI(title="CineMatch Q&A")
pipeline = ContextPipeline(system_prompt=SYSTEM)


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    messages = pipeline.build_messages(request.query)

    async def generate():
        async for chunk in stream_llm(messages):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

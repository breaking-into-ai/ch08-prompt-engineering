from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    title: str
    year: int
    similarity_reason: str
    confidence: float = Field(ge=0, le=1)
    streaming_hint: str = ""


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    reasoning: str


class RecommendRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    history: list[dict] = []

# api/models.py

from typing import List
from pydantic import BaseModel

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[str]
    model: str = "Hybrid Recommendation Model v1.0"

class HealthResponse(BaseModel):
    status: str
    message: str
    uptime_seconds: float
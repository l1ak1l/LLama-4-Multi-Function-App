from pydantic import BaseModel
from typing import Dict, List

class ChatResponse(BaseModel):
    query: str
    response: str

class OCRResponse(BaseModel):
    filename: str
    text: str

class RAGResponse(BaseModel):
    query: str
    answer: str
    context: List[str]
    evaluation: Dict[str, float]

class AgentResponse(BaseModel):
    task: str
    agents_used: List[str]
    results: Dict[str, str]
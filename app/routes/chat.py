from fastapi import APIRouter, Form
from app.services.chat import ChatService
from app.models.schemas import ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])
service = ChatService()

@router.post("/", response_model=ChatResponse)
async def chat_message(query: str = Form(...)):
    response = await service.process_message(query)
    return {"query": query, "response": response}
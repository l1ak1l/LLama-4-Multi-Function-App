"""
RAG API Routes
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.rag import RAGService
from app.models.schemas import RAGResponse

router = APIRouter(prefix="/rag", tags=["RAG"])
service = RAGService()

@router.post("/", response_model=RAGResponse)
async def rag_endpoint(
    file: UploadFile = File(...),
    query: str = Form(...),
    evaluate: bool = Form(True)
):
    """Main RAG endpoint"""
    try:
        if not file.filename:
            raise HTTPException(400, "Filename is required")
            
        document = await file.read()
        result = service.process_rag(
            document=document,
            filename=file.filename,
            query=query,
            evaluate=evaluate
        )
        
        if "error" in result:
            raise HTTPException(500, result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
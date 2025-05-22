"""
RAG API Routes - Simplified with only /upload and /rag_query
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from app.services.rag import RAGService
from app.models.schemas import RAGResponse, DocumentResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

# Create service instance lazily
def get_rag_service():
    return RAGService()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: RAGService = Depends(get_rag_service)
):
    """
    Upload and process a document for later querying
    
    This endpoint handles:
    - Document upload
    - Text extraction and chunking
    - Embedding generation using Segmind API
    - Vector database creation and storage
    
    Args:
        file: Document to process (PDF, TXT, CSV)
    
    Returns:
        DocumentResponse with document_id for future queries
    """
    try:
        if not file.filename:
            raise HTTPException(400, "Filename is required")
            
        # Read the uploaded file
        document = await file.read()
        
        # Process the document (embeddings + vector DB)
        result = service.process_document(
            document=document,
            filename=file.filename
        )
        
        if result.get("status") == "error":
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(500, f"Document processing failed: {error_msg}")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@router.post("/rag_query", response_model=RAGResponse)
async def rag_query(
    document_id: str = Form(...),
    query: str = Form(...),
    top_k: Optional[int] = Form(3),
    service: RAGService = Depends(get_rag_service)
):
    """
    Query a previously processed document with Gemini evaluation
    
    This endpoint handles:
    - Retrieving relevant context from vector database
    - Generating answer using Groq + Llama 4 Scout
    - Evaluating response quality using Google Gemini
    
    Args:
        document_id: ID of the previously uploaded document
        query: Question to answer based on the document
        top_k: Number of relevant contexts to retrieve (default: 3)
    
    Returns:
        RAGResponse with answer, contexts, and Gemini evaluation scores
    """
    try:
        if not document_id.strip():
            raise HTTPException(400, "document_id is required")
            
        if not query.strip():
            raise HTTPException(400, "query is required")
            
        # Query the document with automatic Gemini evaluation
        result = service.query(
            document_id=document_id,
            query=query,
            evaluate=True,  # Always evaluate with Gemini
            top_k=top_k
        )
        
        if result.get("status") == "error":
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(500, f"Query failed: {error_msg}")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"RAG query failed: {str(e)}")
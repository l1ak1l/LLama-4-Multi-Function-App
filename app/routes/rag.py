"""
RAG API Routes
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from typing import Optional, List
from app.services.rag import RAGService
from app.models.schemas import RAGResponse, DocumentResponse, DocumentListResponse

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
    
    - **file**: Document to process (PDF, TXT, CSV)
    
    Returns the document_id to use for querying
    """
    try:
        if not file.filename:
            raise HTTPException(400, "Filename is required")
            
        document = await file.read()
        result = service.process_document(
            document=document,
            filename=file.filename
        )
        
        if result.get("status") == "error":
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(500, error_msg)
            
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    service: RAGService = Depends(get_rag_service)
):
    """
    List all processed documents
    
    Returns a list of document IDs and metadata
    """
    try:
        documents = service.document_cache.list_documents()
        return {
            "documents": [
                {
                    "document_id": doc_id,
                    "filename": info["filename"],
                    "timestamp": info["timestamp"],
                    **info.get("metadata", {})
                }
                for doc_id, info in documents.items()
            ],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/query", response_model=RAGResponse)
async def query_document(
    document_id: str = Form(...),
    query: str = Form(...),
    evaluate: bool = Form(True),
    top_k: Optional[int] = Form(3),
    service: RAGService = Depends(get_rag_service)
):
    """
    Query a previously processed document
    
    - **document_id**: ID of the document to query
    - **query**: Question to answer based on the document
    - **evaluate**: Whether to evaluate the answer
    - **top_k**: Number of contexts to retrieve
    """
    try:
        result = service.query(
            document_id=document_id,
            query=query,
            evaluate=evaluate,
            top_k=top_k
        )
        
        if result.get("status") == "error":
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(500, error_msg)
            
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/", response_model=RAGResponse)
async def rag_endpoint(
    query: str = Form(...),
    evaluate: bool = Form(True),
    top_k: Optional[int] = Form(3),
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Form(None),
    service: RAGService = Depends(get_rag_service)
):
    """
    Main RAG endpoint - Combines upload and query
    
    - **query**: Question to answer based on the document
    - **evaluate**: Whether to evaluate the answer
    - **top_k**: Number of contexts to retrieve
    - **file**: Document to process (optional if document_id is provided)
    - **document_id**: ID of a previously processed document (optional if file is provided)
    """
    try:
        # Check if we have either a file or a document_id
        if not file and not document_id:
            raise HTTPException(400, "Either file or document_id must be provided")
            
        # If we have a file, process it first
        if file and file.filename:
            document = await file.read()
            result = service.process_rag(
                document=document,
                filename=file.filename,
                query=query,
                evaluate=evaluate,
                top_k=top_k
            )
        else:
            # Otherwise use the provided document_id
            result = service.query(
                document_id=document_id,
                query=query,
                evaluate=evaluate,
                top_k=top_k
            )
        
        if result.get("status") == "error":
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(500, error_msg)
            
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
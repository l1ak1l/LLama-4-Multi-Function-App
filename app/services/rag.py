"""
RAG Service with Segmind Embeddings and Gemini Evaluation
Optimized for /upload and /rag_query workflow
"""
import os
import tempfile
import logging
from typing import Dict, List, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from dotenv import load_dotenv

# Import our custom services
from app.services.rag_evaluation import GeminiEvaluator
from app.services.Document_cache import DocumentCache

load_dotenv()
logger = logging.getLogger(__name__)

class SegmindEmbedder:
    """Segmind API integration for text embeddings"""
    def __init__(self):
        self.api_key = os.getenv("SEGMIND_API_KEY")
        self.base_url = "https://api.segmind.com/v1/text-embedding-3-large"
        
        if not self.api_key:
            raise ValueError("SEGMIND_API_KEY environment variable is required")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents"""
        import requests
        
        try:
            headers = {"x-api-key": self.api_key}
            response = requests.post(
                self.base_url,
                json={"text": texts},
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if "embeddings" not in result:
                raise ValueError("Invalid response format from Segmind API")
                
            return result["embeddings"]
            
        except Exception as e:
            logger.error(f"Segmind API error: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

class RAGService:
    """Main RAG processing service optimized for upload/query workflow"""
    
    def __init__(self):
        self.embedder = SegmindEmbedder()
        self.evaluator = GeminiEvaluator()
        self.document_cache = DocumentCache()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_document(self, document: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a document for the /upload endpoint
        
        This method handles:
        1. Document loading and text extraction
        2. Text chunking
        3. Embedding generation via Segmind
        4. Vector store creation
        5. Caching for future queries
        
        Args:
            document: Raw document bytes
            filename: Original filename with extension
            
        Returns:
            Dictionary with document_id and processing metadata
        """
        try:
            logger.info(f"Starting document processing for: {filename}")
            
            # Validate file format
            supported_formats = ['.pdf', '.txt', '.csv']
            file_ext = os.path.splitext(filename.lower())[1]
            if file_ext not in supported_formats:
                return {
                    "error": f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}",
                    "status": "error"
                }
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(document)
                tmp.flush()
                tmp_path = tmp.name
            
            try:
                # Load document content
                loader = self._get_loader(tmp_path, filename)
                docs = loader.load()
                
                if not docs:
                    return {
                        "error": "No content could be extracted from the document",
                        "status": "error"
                    }
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(docs)
                texts = [doc.page_content.strip() for doc in chunks if doc.page_content.strip()]
                
                if not texts:
                    return {
                        "error": "No meaningful content found after processing",
                        "status": "error" 
                    }
                
                logger.info(f"Created {len(texts)} chunks from document")
                
                # Generate embeddings using Segmind
                logger.info("Generating embeddings via Segmind API...")
                embeddings = self.embedder.embed_documents(texts)
                
                if len(embeddings) != len(texts):
                    return {
                        "error": "Embedding generation failed - count mismatch",
                        "status": "error"
                    }
                
                # Create FAISS vector store
                vector_store = FAISS.from_texts(
                    texts, 
                    embedding=self.embedder,
                    metadatas=[{
                        "source": filename,
                        "chunk_id": i,
                        "chunk_text": text[:100] + "..." if len(text) > 100 else text
                    } for i, text in enumerate(texts)]
                )
                
                # Store in cache for future queries
                doc_id = self.document_cache.store(
                    filename=filename,
                    vector_store=vector_store,
                    metadata={
                        "chunk_count": len(texts),
                        "file_size": len(document),
                        "file_extension": file_ext
                    }
                )
                
                logger.info(f"Document processed successfully. ID: {doc_id}")
                
                return {
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_count": len(texts),
                    "file_size": len(document),
                    "status": "success"
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
    
    def query(self, document_id: str, query: str, evaluate: bool = True, top_k: int = 3) -> Dict[str, Any]:
        """
        Query a processed document for the /rag_query endpoint
        
        This method handles:
        1. Vector similarity search
        2. Context retrieval
        3. Answer generation via Groq + Llama 4 Scout
        4. Response evaluation via Google Gemini
        
        Args:
            document_id: Document ID from process_document
            query: User's question
            evaluate: Whether to evaluate with Gemini (default True)
            top_k: Number of chunks to retrieve
            
        Returns:
            Complete RAG response with answer and evaluation
        """
        try:
            logger.info(f"Processing query for document {document_id}: {query[:100]}...")
            
            # Retrieve vector store from cache
            vector_store = self.document_cache.retrieve(document_id)
            document_info = self.document_cache.get_info(document_id)
            
            # Perform similarity search
            results = vector_store.similarity_search(query, k=top_k)
            contexts = [doc.page_content for doc in results]
            
            if not contexts:
                return {
                    "error": "No relevant context found for the query",
                    "status": "error"
                }
            
            logger.info(f"Retrieved {len(contexts)} relevant contexts")
            
            # Generate answer using Groq + Llama 4 Scout
            answer = self._generate_answer(query, contexts)
            
            # Evaluate response using Gemini
            evaluation = {}
            if evaluate:
                logger.info("Evaluating response with Gemini...")
                evaluation = self.evaluator.evaluate_rag(
                    question=query,
                    answer=answer,
                    contexts=contexts
                )
            
            return {
                "query": query,
                "answer": answer,
                "contexts": contexts,
                "document_id": document_id,
                "filename": document_info["filename"],
                "evaluation": evaluation,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }

    def _get_loader(self, file_path: str, filename: str):
        """Get appropriate document loader based on file extension"""
        file_ext = filename.lower()
        
        if file_ext.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_ext.endswith('.txt'):
            return TextLoader(file_path, encoding='utf-8')
        elif file_ext.endswith('.csv'):
            return CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer using Groq Cloud with Llama 4 Scout
        
        Args:
            query: User's question
            contexts: Retrieved relevant contexts
            
        Returns:
            Generated answer string
        """
        import requests
        
        # Prepare context for the model
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

Guidelines:
- Use ONLY the information in the provided context to answer the question
- If the context doesn't contain enough information, clearly state this
- Be concise, accurate, and directly address the question
- Cite specific parts of the context when relevant
- If multiple contexts are relevant, synthesize the information appropriately"""
        
        user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set")
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_api_key}"
            }
            
            payload = {
                "model": "llama4-scout",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
                "top_p": 0.9
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            logger.info("Answer generated successfully with Llama 4 Scout")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Groq: {str(e)}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
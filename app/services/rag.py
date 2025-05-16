"""
RAG Service with Segmind Embeddings and Gemini Evaluation
"""
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from dotenv import load_dotenv

# Import our custom services
from app.services.rag_evaluation import GeminiEvaluator
from app.services.Document_cache import DocumentCache

load_dotenv()

class SegmindEmbedder:
    """Segmind API integration for text embeddings"""
    def __init__(self):
        self.api_key = os.getenv("SEGMIND_API_KEY")
        self.base_url = "https://api.segmind.com/v1/text-embedding-3-large"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents"""
        import requests
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            headers = {"x-api-key": self.api_key}
            response = requests.post(
                self.base_url,
                json={"text": texts},
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"Segmind API error: {str(e)}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]

class RAGService:
    """Main RAG processing service"""
    
    def __init__(self):
        self.embedder = SegmindEmbedder()
        self.evaluator = GeminiEvaluator()
        self.document_cache = DocumentCache()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_document(
        self,
        document: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Process a document and store it in the cache
        
        Args:
            document: Raw document bytes
            filename: Original filename
            
        Returns:
            Dictionary with document_id and metadata
        """
        try:
            # Save and load document
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
                tmp.write(document)
                tmp.flush()
                tmp_path = tmp.name
            
            try:
                loader = self._get_loader(tmp_path, filename)
                docs = loader.load()
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            # Process chunks
            chunks = self.text_splitter.split_documents(docs)
            texts = [doc.page_content for doc in chunks]
            
            if not texts:
                return {
                    "error": "No content extracted from document",
                    "status": "error" 
                }
            
            # Generate embeddings
            embeddings = self.embedder.embed_documents(texts)
            
            # Create vector store
            vector_store = FAISS.from_texts(
                texts, 
                embedding=self.embedder,
                metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
            )
            
            # Store in cache
            doc_id = self.document_cache.store(
                filename=filename,
                vector_store=vector_store,
                metadata={"chunk_count": len(texts)}
            )
            
            return {
                "document_id": doc_id,
                "filename": filename,
                "chunk_count": len(texts),
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
    
    def query(
        self,
        document_id: str,
        query: str,
        evaluate: bool = True,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Query a processed document
        
        Args:
            document_id: Document ID from process_document
            query: Query string
            evaluate: Whether to evaluate the results
            top_k: Number of chunks to retrieve
            
        Returns:
            Query results
        """
        try:
            # Retrieve vector store from cache
            vector_store = self.document_cache.retrieve(document_id)
            document_info = self.document_cache.get_info(document_id)
            
            # Search for similar chunks
            results = vector_store.similarity_search(query, k=top_k)
            context = [doc.page_content for doc in results]
            
            # Generate answer
            answer = self._generate_answer(query, context)
            
            # Evaluation
            evaluation = {}
            if evaluate:
                evaluation = self.evaluator.evaluate_rag(
                    question=query,
                    answer=answer,
                    contexts=context
                )
            
            return {
                "query": query,
                "answer": answer,
                "context": context,
                "document_id": document_id,
                "filename": document_info["filename"],
                "evaluation": evaluation,
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }
    
    def process_rag(
        self,
        document: Optional[bytes] = None,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
        query: Optional[str] = None,
        evaluate: bool = True,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline execution - either process new document or query existing one
        
        This is a convenience method that combines process_document and query
        """
        try:
            # Case 1: Process document only
            if document and filename and not query:
                return self.process_document(document, filename)
                
            # Case 2: Query existing document
            elif document_id and query:
                return self.query(document_id, query, evaluate, top_k)
                
            # Case 3: Upload document and query in one go
            elif document and filename and query:
                doc_result = self.process_document(document, filename)
                if doc_result.get("status") == "error":
                    return doc_result
                    
                doc_id = doc_result["document_id"]
                return self.query(doc_id, query, evaluate, top_k)
                
            else:
                return {
                    "error": "Invalid combination of parameters. Either provide document+filename, document_id+query, or document+filename+query",
                    "status": "error"
                }
                
        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "error"
            }

    def _get_loader(self, file_path: str, filename: str):
        """Get appropriate document loader"""
        if filename.lower().endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif filename.lower().endswith('.txt'):
            return TextLoader(file_path)
        elif filename.lower().endswith('.csv'):
            return CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def _generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using context and Groq Cloud with Llama 4 Scout"""
        import requests
        
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Use ONLY the information in the context to answer the question. If the context doesn't 
contain the answer, say "I don't have enough information to answer this question."
Be concise, accurate, and helpful."""
        
        user_prompt = f"""Context:
{context_text}

Question: {query}"""
        
        try:
            # Use Groq with Llama 4 Scout
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
                "max_tokens": 1024
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            
            return answer
            
        except Exception as e:
            import logging
            logging.error(f"Error generating answer with Groq: {str(e)}")
            return "An error occurred while generating the answer with Llama 4 Scout."
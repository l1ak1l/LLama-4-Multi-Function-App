"""
Document Cache for RAG Persistence
"""
import os
import time
import uuid
import pickle
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class DocumentCache:
    """
    Cache for storing processed documents and their vector stores
    to enable querying without re-uploading documents
    """
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        """
        Initialize the document cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time to live for cache entries in hours
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self.cache_info = {}  # In-memory record of cache entries
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache info
        self._load_cache_info()
        
        # Clean expired cache entries
        self._clean_expired()
        
    def _load_cache_info(self):
        """Load cache info from disk"""
        cache_info_path = os.path.join(self.cache_dir, "cache_info.pkl")
        if os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, "rb") as f:
                    self.cache_info = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache info: {str(e)}")
                self.cache_info = {}
    
    def _save_cache_info(self):
        """Save cache info to disk"""
        cache_info_path = os.path.join(self.cache_dir, "cache_info.pkl")
        try:
            with open(cache_info_path, "wb") as f:
                pickle.dump(self.cache_info, f)
        except Exception as e:
            logger.error(f"Error saving cache info: {str(e)}")
    
    def _clean_expired(self):
        """Remove expired cache entries"""
        now = time.time()
        expired_ids = []
        
        for doc_id, info in self.cache_info.items():
            if now - info["timestamp"] > self.ttl_seconds:
                expired_ids.append(doc_id)
                # Delete the cache file
                cache_path = os.path.join(self.cache_dir, f"{doc_id}.pkl")
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except Exception as e:
                        logger.error(f"Error removing cache file {doc_id}: {str(e)}")
        
        # Remove expired entries from cache info
        for doc_id in expired_ids:
            del self.cache_info[doc_id]
            
        # Save updated cache info
        if expired_ids:
            self._save_cache_info()
    
    def store(self, filename: str, vector_store: Any, metadata: Optional[Dict] = None) -> str:
        """
        Store a processed document and its vector store
        
        Args:
            filename: Original document filename
            vector_store: Vector store object
            metadata: Additional metadata
            
        Returns:
            Document ID for later retrieval
        """
        # Generate a unique ID
        doc_id = str(uuid.uuid4())
        
        # Store cache info
        self.cache_info[doc_id] = {
            "filename": filename,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Store the vector store
        cache_path = os.path.join(self.cache_dir, f"{doc_id}.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(vector_store, f)
        except Exception as e:
            logger.error(f"Error storing cache for {doc_id}: {str(e)}")
            raise HTTPException(500, f"Error storing document cache: {str(e)}")
        
        # Save cache info
        self._save_cache_info()
        
        return doc_id
    
    def retrieve(self, doc_id: str) -> Any:
        """
        Retrieve a vector store by document ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Vector store object
        """
        # Check if document exists
        if doc_id not in self.cache_info:
            raise HTTPException(404, f"Document with ID {doc_id} not found")
        
        # Check if document has expired
        now = time.time()
        if now - self.cache_info[doc_id]["timestamp"] > self.ttl_seconds:
            del self.cache_info[doc_id]
            self._save_cache_info()
            raise HTTPException(404, f"Document with ID {doc_id} has expired")
        
        # Retrieve the vector store
        cache_path = os.path.join(self.cache_dir, f"{doc_id}.pkl")
        try:
            with open(cache_path, "rb") as f:
                vector_store = pickle.load(f)
            
            # Update timestamp
            self.cache_info[doc_id]["timestamp"] = now
            self._save_cache_info()
            
            return vector_store
        except Exception as e:
            logger.error(f"Error retrieving cache for {doc_id}: {str(e)}")
            raise HTTPException(500, f"Error retrieving document cache: {str(e)}")
    
    def get_info(self, doc_id: str) -> Dict:
        """Get information about a cached document"""
        if doc_id not in self.cache_info:
            raise HTTPException(404, f"Document with ID {doc_id} not found")
        
        return self.cache_info[doc_id]
    
    def list_documents(self) -> Dict[str, Dict]:
        """List all documents in the cache"""
        # Clean expired documents first
        self._clean_expired()
        return self.cache_info
# """
# RAG Service with Segmind Embeddings
# """
# import os
# import tempfile
# from typing import Dict, List
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
# from .rag_evaluation import SegmindEmbedder, RAGEvaluator

# load_dotenv()

# class RAGService:
#     """Main RAG processing service"""
    
#     def __init__(self):
#         self.embedder = SegmindEmbedder()
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )

#     def process_rag(
#         self,
#         document: bytes,
#         filename: str,
#         query: str,
#         evaluate: bool = True
#     ) -> Dict:
#         """Full RAG pipeline execution"""
#         try:
#             # Save and load document
#             with tempfile.NamedTemporaryFile(suffix=filename) as tmp:
#                 tmp.write(document)
#                 tmp.flush()
#                 loader = self._get_loader(tmp.name, filename)
#                 docs = loader.load()

#             # Process chunks
#             chunks = self.text_splitter.split_documents(docs)
#             texts = [doc.page_content for doc in chunks]
            
#             # Generate embeddings
#             embeddings = self.embedder.embed_documents(texts)
            
#             # Create vector store
#             vector_store = FAISS.from_embeddings(
#                 list(zip(texts, embeddings)),
#                 self.embedder
#             )
            
#             # Search and generate answer
#             results = vector_store.similarity_search(query, k=3)
#             context = [doc.page_content for doc in results]
#             answer = self._generate_answer(query, context)
            
#             # Evaluation
#             evaluation = {}
#             if evaluate:
#                 evaluation = RAGEvaluator.evaluate_with_ragas(
#                     question=query,
#                     answer=answer,
#                     contexts=context
#                 )
            
#             return {
#                 "query": query,
#                 "answer": answer,
#                 "context": context,
#                 "evaluation": evaluation
#             }
            
#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "status": "error"
#             }

#     def _get_loader(self, file_path: str, filename: str):
#         """Get appropriate document loader"""
#         if filename.endswith('.pdf'):
#             return PyPDFLoader(file_path)
#         if filename.endswith('.txt'):
#             return TextLoader(file_path)
#         if filename.endswith('.csv'):
#             return CSVLoader(file_path)
#         raise ValueError("Unsupported file format")

#     def _generate_answer(self, query: str, context: List[str]) -> str:
#         """Generate answer using context"""
#         context_text = "\n\n".join(context)
#         prompt = f"""Answer based on context:
#         {context_text}
        
#         Question: {query}
#         """
#         # Implement actual LLM call here
#         return "Sample answer based on context"  # Replace with actual LLM call
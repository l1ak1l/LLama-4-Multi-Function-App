# """
# RAG Evaluation Service with Segmind Embeddings
# """
# import os
# import logging
# import requests
# import numpy as np
# from typing import Dict, List, Optional
# from datasets import Dataset
# from dotenv import load_dotenv
# from ragas import evaluate
# from ragas.metrics import AnswerRelevancy, Faithfulness, ContextPrecision

# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SegmindEmbedder:
#     """Segmind API integration for text embeddings"""
#     def __init__(self):
#         self.api_key = os.getenv("SEGMIND_API_KEY")
#         self.base_url = "https://api.segmind.com/v1/text-3-large"
        
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Batch embed documents"""
#         try:
#             headers = {"x-api-key": self.api_key}
#             response = requests.post(
#                 self.base_url,
#                 json={"text": texts},
#                 headers=headers,
#                 timeout=60
#             )
#             response.raise_for_status()
#             return response.json()["embeddings"]
#         except Exception as e:
#             logger.error(f"Segmind API error: {str(e)}")
#             return []

#     def embed_query(self, text: str) -> List[float]:
#         """Embed single query"""
#         return self.embed_documents([text])[0]

# # Initialize RAGAS metrics with Segmind embeddings
# segmind_embeddings = SegmindEmbedder()
# answer_relevancy = AnswerRelevancy(embeddings=segmind_embeddings)
# faithfulness = Faithfulness(embeddings=segmind_embeddings)
# context_precision = ContextPrecision(embeddings=segmind_embeddings)

# class RAGEvaluator:
#     """Handles RAG evaluation workflows"""
    
#     @staticmethod
#     def evaluate_with_ragas(
#         question: str,
#         answer: str, 
#         contexts: List[str],
#         ground_truth: Optional[str] = None
#     ) -> Dict[str, float]:
#         """Run RAGAS evaluation with Segmind embeddings"""
#         try:
#             metrics = [answer_relevancy, faithfulness]
#             if ground_truth:
#                 metrics.append(context_precision)
            
#             dataset = Dataset.from_dict({
#                 "question": [question],
#                 "answer": [answer],
#                 "contexts": [contexts],
#                 **({"ground_truth": [ground_truth]} if ground_truth else {})
#             })
            
#             results = evaluate(dataset, metrics=metrics)
#             return {metric: float(results[metric]) for metric in results.columns}
#         except Exception as e:
#             logger.error(f"RAGAS evaluation failed: {str(e)}")
#             return {"answer_relevancy": 0.0, "faithfulness": 0.0}
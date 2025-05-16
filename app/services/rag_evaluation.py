"""
RAG Evaluation Service with Gemini
"""
import os
import logging
import requests
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEvaluator:
    """Gemini API integration for RAG evaluation"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        
    def evaluate_rag(
        self, 
        question: str,
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, float]:
        """Evaluate RAG results using Gemini for faithfulness and confidence"""
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(question, answer, contexts)
            
            # Call Gemini API
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            url = f"{self.api_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract evaluation metrics
            return self._parse_evaluation_response(response_text)
            
        except Exception as e:
            logger.error(f"Gemini evaluation error: {str(e)}")
            return {"faithfulness": 0.0, "confidence": 0.0}
    
    def _create_evaluation_prompt(self, question: str, answer: str, contexts: List[str]) -> str:
        """Create prompt for Gemini evaluation"""
        contexts_text = "\n\n".join([f"Context {i+1}:\n{context}" for i, context in enumerate(contexts)])
        
        prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. Your task is to evaluate the faithfulness and confidence of the provided answer based on the retrieved contexts.

QUESTION:
{question}

RETRIEVED CONTEXTS:
{contexts_text}

GENERATED ANSWER:
{answer}

EVALUATION TASK:
1. Faithfulness: Evaluate on a scale of 0.0 to 1.0 how faithful the answer is to the provided contexts. 
   - 1.0 means the answer is completely supported by the contexts
   - 0.0 means the answer contains information not present in any of the contexts or contradicts the contexts
   
2. Confidence: Evaluate on a scale of 0.0 to 1.0 how confident we should be in this answer.
   - 1.0 means high confidence (the contexts are highly relevant and the answer is comprehensive)
   - 0.0 means no confidence (irrelevant contexts or incomplete/incorrect answer)

RESPONSE FORMAT:
Provide a brief analysis followed by the scores in this exact format:
FAITHFULNESS: <score between 0.0 and 1.0>
CONFIDENCE: <score between 0.0 and 1.0>
"""
        return prompt
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, float]:
        """Parse Gemini's response to extract evaluation metrics"""
        try:
            # Extract scores using string manipulation
            faithfulness_line = [line for line in response_text.split('\n') if 'FAITHFULNESS:' in line]
            confidence_line = [line for line in response_text.split('\n') if 'CONFIDENCE:' in line]
            
            faithfulness = 0.0
            confidence = 0.0
            
            if faithfulness_line:
                faithfulness_str = faithfulness_line[0].split('FAITHFULNESS:')[1].strip()
                faithfulness = float(faithfulness_str)
                
            if confidence_line:
                confidence_str = confidence_line[0].split('CONFIDENCE:')[1].strip()
                confidence = float(confidence_str)
            
            return {
                "faithfulness": faithfulness,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {str(e)}")
            return {"faithfulness": 0.0, "confidence": 0.0}
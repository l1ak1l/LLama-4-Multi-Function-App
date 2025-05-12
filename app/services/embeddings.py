import requests
import numpy as np
import os
from pathlib import Path
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

class SegmindEmbedder:
    def __init__(self):
        self.api_key = os.getenv("SEGMIND_API_KEY")
        self.temp_dir = Path("temp/embeddings")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def generate_embeddings(self, texts: list):
        try:
            headers = {"x-api-key": self.api_key}
            response = requests.post(
                "https://api.segmind.com/v1/text-embedding-3-large",
                json={"text": texts},
                headers=headers
            )
            response.raise_for_status()
            
            embeddings = response.json().get("embeddings", [])
            file_id = f"emb_{uuid.uuid4().hex}_{int(time.time())}"
            np.save(self.temp_dir / file_id, np.array(embeddings))
            return {"file_id": file_id, "status": "success"}
            
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def load_embeddings(self, file_id: str):
        try:
            file_path = self.temp_dir / f"{file_id}.npy"
            return np.load(file_path)
        except Exception as e:
            return None
from groq import Groq
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

class OCRService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    async def process_image(self, image_data: bytes, prompt: str, model_name: str):
        try:
            image = Image.open(io.BytesIO(image_data))
            img_format = image.format or "PNG"
            buffered = io.BytesIO()
            image.save(buffered, format=img_format)
            base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{img_format.lower()};base64,{base64_str}"
                        }}
                    ]
                }],
                temperature=1,
                max_tokens=1024,
                top_p=1,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"OCR processing error: {str(e)}"
import base64
import io
import os
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class OCRService:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("Groq API key not set in environment.")
        self.client = Groq(api_key=GROQ_API_KEY)

    def encode_image_to_base64(self, image: Image.Image):
        """Convert PIL Image to base64 string"""
        img_format = image.format if hasattr(image, 'format') and image.format else "PNG"
        buffered = io.BytesIO()
        image.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str, img_format.lower()

    def process_with_groq(self, model_name: str, image: Image.Image, prompt: str):
        """Process the image with Groq model (e.g., LLaMA-4 Scout)"""
        try:
            base64_string, img_format = self.encode_image_to_base64(image)
            media_type = "image/jpeg" if img_format in ['jpg', 'jpeg'] else "image/png"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_string}"
                                },
                            },
                        ],
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OCR processing error: {str(e)}"

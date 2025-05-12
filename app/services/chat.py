from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class ChatService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    async def process_message(self, message: str, chat_history: list = None):
        try:
            messages = [{"role": "user", "content": message}]
            if chat_history:
                messages = chat_history + messages
                
            response = self.client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing chat: {str(e)}"
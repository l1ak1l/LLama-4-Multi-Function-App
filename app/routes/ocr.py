from fastapi import APIRouter, UploadFile, File
from app.services.ocr import OCRService
import io
from PIL import Image
from app.models.schemas import OCRResponse

router = APIRouter(prefix="/ocr", tags=["OCR"])
service = OCRService()

@router.post("/", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        prompt = "Extract all the visible text from this image."
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        text = service.process_with_groq(model_name, image, prompt)
        return {"filename": file.filename, "text": text}
    except Exception as e:
        return {"filename": file.filename, "text": f"OCR endpoint error: {str(e)}"}
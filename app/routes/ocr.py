from fastapi import APIRouter, UploadFile, File
from app.services.ocr import OCRService
from app.models.schemas import OCRResponse

router = APIRouter(prefix="/ocr", tags=["OCR"])
service = OCRService()

@router.post("/", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    text = service.process_image(image_data)
    return {"filename": file.filename, "text": text}
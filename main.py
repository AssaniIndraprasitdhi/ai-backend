from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import tempfile, os

app = FastAPI()
ocr = PaddleOCR(lang='latin', use_angle_cls=False, use_gpu=False)

@app.post("/ocr")
async def ocr_from_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(content={"error": "Only PDF supported"}, status_code=400)

        pdf_bytes = await file.read()
        images = convert_from_bytes(pdf_bytes)

        results = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image_path = tmp.name
                image.save(image_path, "JPEG")

            ocr_result = ocr.ocr(image_path, cls=False)
            os.remove(image_path)

            text_blocks = []
            for line in ocr_result[0]:
                text_blocks.append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "box": line[0]
                })

            results.append({
                "page": i + 1,
                "text_blocks": text_blocks
            })

        return {"pages": results}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

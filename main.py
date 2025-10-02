import io, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# импортируй свой пайплайн
from GCAI_Pipeline import GCAI_Classifier

# создаём FastAPI-приложение
app = FastAPI()

# создаём классификатор (твой класс из GCAI_Pipeline)
classifier = GCAI_Classifier()

@app.on_event("startup")
async def startup_event():
    # запуск воркеров в фоне
    await classifier.start(num_workers=1)

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # запускаем инференс → получаем результат
        result = await classifier.classify(raw, return_annotated=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    detections = result.get("detections", [])
    img_bytes = result.get("annotated_image_jpg", None)

    if not img_bytes:
        img_bytes = raw  # fallback: если не сработало

    # готовим StreamingResponse с JPEG
    buf = io.BytesIO(img_bytes)
    resp = StreamingResponse(buf, media_type="image/jpeg")

    # прикладываем JSON с детекциями в заголовок
    resp.headers["X-Detections"] = json.dumps(detections, ensure_ascii=False)
    return resp

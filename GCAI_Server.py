# gcai_api.py
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
from typing import Any, Dict
import base64

# ЕСЛИ GStreamer реально нужен — оставь, иначе убери эти 3 строки:
# import gi
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst  # type: ignore

from GCAI_Pipeline import GCAI_Classifier  # твой класс из предыдущего шага

# gcai_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import io, json

from GCAI_Pipeline import GCAI_Classifier

class GCAI_APIHandler:
    def __init__(self, app: FastAPI, classifier: GCAI_Classifier | None = None) -> None:
        self.app = app
        self.classifier = classifier or GCAI_Classifier()
        self.register_routes()

        @app.on_event("startup")
        async def _startup():
            await self.classifier.start(num_workers=1)

    def register_routes(self) -> None:

        @self.app.post("/classify-image")
        async def classify_image(file: UploadFile = File(...)):
            if not file.content_type or not file.content_type.startswith(("image/", "application/octet-stream")):
                raise HTTPException(status_code=400, detail="file must be an image")

            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="empty file")

            # инференс (просим вернуть аннотированную картинку в bytes)
            try:
                result = await self.classifier.classify(raw, return_annotated=True)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"inference error: {e}")

            detections = result.get("detections", [])
            img_bytes = result.get("annotated_image_jpg", None)
            if not img_bytes:
                # запасной вариант: вернём исходник как JPEG
                img_bytes = raw

            # оборачиваем bytes в StreamingResponse как JPEG
            buf = io.BytesIO(img_bytes)
            resp = StreamingResponse(buf, media_type="image/jpeg")
            # положим JSON c детекциями в заголовок (короткий и удобный для клиента)
            resp.headers["X-Detections"] = json.dumps(detections, ensure_ascii=False)
            return resp

















# class GCAI_APIHandler:
#     """
#     HTTP-слой для общения с мобильным приложением.
#     """
#     def __init__(self, app: FastAPI, classifier: GCAI_Classifier | None = None) -> None:
#         self.app = app
#         self.classifier = classifier or GCAI_Classifier()
#         self.register_routes()

#         @app.on_event("startup")
#         async def _startup():
#             # запускаем воркеры (1 достаточно, увеличишь при необходимости)
#             await self.classifier.start(num_workers=1)

#     def _ensure_b64(self, result: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Унифицируем ответ: если classifier вернул bytes — кодируем в base64.
#         """
#         out = dict(result)
#         if "annotated_image_b64" in out:
#             return out
#         if "annotated_image_jpg" in out and isinstance(out["annotated_image_jpg"], (bytes, bytearray)):
#             out["annotated_image_b64"] = base64.b64encode(out.pop("annotated_image_jpg")).decode("utf-8")
#         return out

#     def register_routes(self) -> None:

#         @self.app.get("/health")
#         async def health() -> Dict[str, str]:
#             return {"status": "ok"}

#         @self.app.post("/classify-single")
#         async def classify_single(file: UploadFile = File(...), return_image: bool = True):
#             # базовая валидация
#             if not file.content_type or not file.content_type.startswith(("image/", "application/octet-stream")):
#                 raise HTTPException(status_code=400, detail="file must be an image")

#             image_bytes = await file.read()
#             if not image_bytes:
#                 raise HTTPException(status_code=400, detail="empty file")

#             # инференс
#             try:
#                 result = await self.classifier.classify(image_bytes, return_annotated=return_image)
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=f"inference error: {e}")

#             # гарантируем base64 (удобно для RN <Image source={{uri: 'data:image/jpeg;base64,'+...}}/>)
#             result = self._ensure_b64(result)

#             # опционально: не возвращать картинку
#             if not return_image:
#                 result.pop("annotated_image_b64", None)
#                 result.pop("annotated_image_jpg", None)

#             return JSONResponse(result)

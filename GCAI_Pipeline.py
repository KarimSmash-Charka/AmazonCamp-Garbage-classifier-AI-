# gcai_classifier.py
import io, math, asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from ultralytics import YOLO

try:
    import cv2
except ImportError:
    cv2 = None  # если нет opencv, уберем отрисовку

# ===================== CONFIG =====================
YOLO_WEIGHTS = "bestYolo4.pt"         # путь к YOLO чекпоинту
CLS_WEIGHTS  = "MaybeTheBest2.pth"    # путь к твоей классификационной модели
NUM_CLASSES  = 6
CLASS_NAMES  = ["cardboard","glass","metal","paper","plastic","trash"]

IMG_IN   = 256
IMG_CROP = 224
MARGIN   = 0.15
YOLO_CONF = 0.20

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ===================== HELPERS =====================
def crop_with_margin(pil_img: Image.Image, xyxy, margin_ratio=0.2):
    W, H = pil_img.size
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    mx = w * margin_ratio
    my = h * margin_ratio
    x1m = max(0, int(math.floor(x1 - mx)))
    y1m = max(0, int(math.floor(y1 - my)))
    x2m = min(W, int(math.ceil(x2 + mx)))
    y2m = min(H, int(math.ceil(y2 + my)))
    if (x2m - x1m) < 8 or (y2m - y1m) < 8:
        return None
    return pil_img.crop((x1m, y1m, x2m, y2m))


def build_transforms():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize((IMG_IN, IMG_IN), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_CROP),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_classifier(weights_path: str, n_classes: int):
    base_w = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=base_w)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes)
    )
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        model.load_state_dict(sd["model_state"])
    else:
        model.load_state_dict(sd)
    model.eval().to(DEVICE)
    return model


def draw_box_with_label(pil_img, xyxy, label, color=(0,255,0)):
    if cv2 is None:
        return pil_img  # opencv не установлен — пропускаем визуализацию
    img = np.array(pil_img).copy()
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return Image.fromarray(img)


def pil_to_jpg_bytes(pil_img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ===================== CLASSIFIER =====================
class GCAI_Classifier:
    """
    Принимает bytes изображения, делает:
    1) YOLO детекцию (bbox)
    2) классификацию кропов (EfficientNet)
    Возвращает JSON с детекциями и, опционально, аннотированное изображение (bytes).
    """

    def __init__(self) -> None:
        print("Device:", DEVICE)

        # Инициализируем модели ОДИН РАЗ
        if not Path(YOLO_WEIGHTS).exists():
            raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
        self.yolo = YOLO(YOLO_WEIGHTS)

        if not Path(CLS_WEIGHTS).exists():
            raise FileNotFoundError(f"Classifier weights not found: {CLS_WEIGHTS}")
        self.cls_model = load_classifier(CLS_WEIGHTS, NUM_CLASSES)
        self.tf = build_transforms()

        # Очередь/воркеры
        self.process_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._tasks: list[asyncio.Task] = []

    async def start(self, num_workers: int = 1) -> None:
        for _ in range(num_workers):
            task = asyncio.create_task(self._process_worker())
            self._tasks.append(task)

    async def classify(self, image_bytes: bytes, return_annotated: bool = True) -> Dict[str, Any]:
        """
        Асинхронный интерфейс:
        - на вход bytes
        - на выход словарь с детекциями и, опционально, аннотированное изображение в bytes
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self.process_queue.put((image_bytes, return_annotated, future))
        return await future

    async def _process_worker(self) -> None:
        while True:
            image_bytes, return_annotated, future = await self.process_queue.get()
            try:
                result = self._run_inference(image_bytes, return_annotated=return_annotated)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.process_queue.task_done()

    # Синхронная инференс-функция (внутри воркера)
    def _run_inference(self, image_bytes: bytes, return_annotated: bool = True) -> Dict[str, Any]:
        pil = bytes_to_pil(image_bytes)

        # 1) YOLO
        results = self.yolo.predict(source=pil, conf=YOLO_CONF, verbose=False)
        if not results or len(results[0].boxes) == 0:
            out = {
                "detections": [],
                "message": "no objects found by YOLO"
            }
            if return_annotated:
                out["annotated_image_jpg"] = pil_to_jpg_bytes(pil)
            return out

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores= results[0].boxes.conf.cpu().numpy()

        annotated = pil.copy()
        crops_tensors, crops_meta = [], []

        for bi, xy in enumerate(boxes):
            crop = crop_with_margin(pil, xy, margin_ratio=MARGIN)
            if crop is None:
                continue
            tensor = self.tf(crop)
            crops_tensors.append(tensor)
            crops_meta.append((xy, float(scores[bi])))

        if not crops_tensors:
            out = {
                "detections": [],
                "message": "all crops filtered (too small)"
            }
            if return_annotated:
                out["annotated_image_jpg"] = pil_to_jpg_bytes(pil)
            return out

        batch = torch.stack(crops_tensors).to(DEVICE)

        with torch.no_grad():
            logits = self.cls_model(batch)
            probs  = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

        detections: List[Dict[str, Any]] = []
        for (xy, det_conf), cls_id, pconf in zip(
            crops_meta, preds.cpu().tolist(), confs.cpu().tolist()
        ):
            x1, y1, x2, y2 = map(int, xy)
            label = CLASS_NAMES[cls_id]
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_conf": float(det_conf),
                "class": label,
                "class_conf": float(pconf)
            })
            if return_annotated:
                annotated = draw_box_with_label(annotated, xy, f"{label} {pconf*100:.1f}%")

        out: Dict[str, Any] = {
            "detections": detections,
            "num_detections": len(detections)
        }
        if return_annotated:
            out["annotated_image_jpg"] = pil_to_jpg_bytes(annotated, quality=95)
        return out

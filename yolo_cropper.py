import os, math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from ultralytics import YOLO
from PIL import Image
import numpy as np

# ===================== CONFIG =====================
SOURCE_FILE = "proverka3.jpg"
OUT_FILE    = "out_result.jpg"

YOLO_WEIGHTS = "bestYolo4.pt"          # YOLO веса (лучше натренированный мусорный чекпоинт)
CLS_WEIGHTS  = "MaybeTheBest2.pth"    # твой обученный классификатор
NUM_CLASSES  = 6
CLASS_NAMES  = ["cardboard","glass","metal","paper","plastic","trash"]

IMG_IN   = 256
IMG_CROP = 224 #224
MARGIN   = 0.15
YOLO_CONF = 0.20

# девайс
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# ===================== UTILS =====================
def crop_with_margin(pil_img: Image.Image, xyxy, margin_ratio=0.2):
    W, H = pil_img.size
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    mx = w * margin_ratio
    my = h * margin_ratio
    x1m = max(0, math.floor(x1 - mx))
    y1m = max(0, math.floor(y1 - my))
    x2m = min(W, math.ceil(x2 + mx))
    y2m = min(H, math.ceil(y2 + my))
    if (x2m - x1m) < 8 or (y2m - y1m) < 8:
        return None
    return pil_img.crop((x1m, y1m, x2m, y2m))

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

def build_transforms():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    val_tf = T.Compose([
        T.Resize((IMG_IN, IMG_IN), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_CROP),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return val_tf

def draw_box_with_label(pil_img, xyxy, label, color=(0,255,0)):
    import cv2
    img = np.array(pil_img).copy()
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return Image.fromarray(img)

# ===================== MAIN =====================
def main():
    p = Path(SOURCE_FILE)
    if not p.exists():
        print(f"⚠️ Файл не найден: {p}")
        return

    # 1) YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    # 2) Классификатор
    cls_model = load_classifier(CLS_WEIGHTS, NUM_CLASSES)
    tf = build_transforms()

    pil = Image.open(str(p)).convert("RGB")

    results = yolo.predict(source=str(p), conf=YOLO_CONF, verbose=False)
    if not results or len(results[0].boxes) == 0:
        print("⚠️ YOLO не нашёл объектов")
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores= results[0].boxes.conf.cpu().numpy()

    annotated = pil.copy()
    crops_tensors, crops_meta = [], []

    for bi, xy in enumerate(boxes):
        crop = crop_with_margin(pil, xy, margin_ratio=MARGIN)
        if crop is None: continue
        tensor = tf(crop)
        crops_tensors.append(tensor)
        crops_meta.append((xy, float(scores[bi])))

    if not crops_tensors:
        print("⚠️ Все кропы отфильтрованы (слишком маленькие)")
        return

    batch = torch.stack(crops_tensors).to(DEVICE)

    with torch.no_grad():
        logits = cls_model(batch)
        probs  = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

    # ===== Вывод в консоль =====
    print("\n=== Результаты классификации ===")
    for i, ((xy, det_conf), cls_id, pconf) in enumerate(
        zip(crops_meta, preds.cpu().tolist(), confs.cpu().tolist())
    ):
        x1, y1, x2, y2 = map(int, xy)
        print(f"Объект {i+1}: YOLO conf={det_conf:.2f} → "
              f"Класс={CLASS_NAMES[cls_id]} (уверенность {pconf*100:.1f}%), "
              f"bbox=({x1},{y1},{x2},{y2})")

        label = f"{CLASS_NAMES[cls_id]} {pconf*100:.1f}%"
        annotated = draw_box_with_label(annotated, xy, label)

    annotated.save(OUT_FILE, quality=95)
    print(f"\n✅ Результат сохранён в {OUT_FILE}")

if __name__ == "__main__":
    main()

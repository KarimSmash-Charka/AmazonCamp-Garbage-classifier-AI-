# 🏗️ Project Architecture Overview

## 📦 Modules

- efficientNet_model/ — EfficientNet classifier for detected items (developed by Karim)
- yolo_model/ — YOLOv8n object detector (developed by Daulet)
- backend/ — FastAPI server (developed by Stefan)
- app/ — Frontend/mobile interface (developed by Asylkhan)
- docs/ — Documentation folder (developed by Asylkhan)

## 🧠 Workflow
1. User uploads or captures image in the mobile app
2. Image sent to backend via FastAPI
3. Depending on mode:
   - YOLO detects multiple objects
   - EfficientNet classifies detected objects one by one.
4. Prediction returned to frontend

## 📥 Download
1. All the requirements and libraries for the project are in the **requirements.txt**  
2. Model weights for EfficientNet and Yolo are available here: [Google Drive Link](https://drive.google.com/drive/folders/1zjOuYtXwM5OrVcitNX3mJ4vauTer9j85?usp=share_link)


## 🙌 Team Contributions
- EfficientNet model: Karim
- YOLO model: Daulet
- Backend/API: Stefan
- Mobile App: Asylkhan
- Documentation: Asylkhan
---
---

# EfficientNetV2-S Fine-Tuning for Image Classification

This repository contains a training pipeline for **image classification** using **EfficientNetV2-S** with fine-tuning.  
The code supports **multi-stage training** (head → last block → pre-last block), **class balancing**, **advanced augmentations**, and **early stopping** based on macro-F1 score.

---

## 🚀 Features
- **Pretrained EfficientNetV2-S** (ImageNet weights)
- **Custom classifier head** with BatchNorm, Dropout, and Linear layers
- **Stage-wise fine-tuning**:
  - Stage 1: train only the classifier head
  - Stage 2: train head + last feature block
  - (Optional Stage 3: head + last two feature blocks)
- **WeightedRandomSampler** for imbalanced datasets
- **OneCycleLR scheduler** with different learning rates per parameter group
- **Macro-F1 evaluation** and early stopping
- **Heavy augmentations** for better generalization (rotation, affine, perspective, blur, jitter, erasing, etc.)

---

## 📂 Dataset Structure
Dataset should be organized in **ImageFolder format**:
```
Cropped_images_dataset/
│── class_0/(cardboard),
│── class_1/(glass),
│── class_2/(metal),
|── class_3/(paper),
|── class_4/(plastic),
│── class_5/(trash).
```

## ⚙️ Key Parameters
```
Input size: 288x288, Crop size: 256x256
Loss: CrossEntropyLoss with class weights + label smoothing (0.02)
Optimizer: AdamW
Scheduler: OneCycleLR
Early stopping: patience=10, delta=0.001
```

## 📌 Requirements
```
Python 3.9+
PyTorch 2.0+
torchvision
scikit-learn
numpy, pillow
```
# 📊 Evaluation
```
Evaluate accuracy and macro-F1:

get_accuracy(model, test_loader)
eval_macro_f1(model, test_loader, class_names=classes, device=device, verbose=True)
```

## 📊 Results
- **Approximate Accuracy**: ~85%  
  *(classification accuracy on detected objects)*  
  ⚠️ *This is an estimated value provided by the author — validation is still pending.*

---

## 👤 Author

- **Model trained by**: Karim
- **Documentation prepared by**: Karim
------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------

# 🧠 YOLOv8 Model for Garbage Detection

## 📌 Overview
This document describes the training and usage of a YOLOv8n object detection model designed for garbage classification.

- **Purpose**: Detect garbage items across 6 categories.
- **Model Used**: YOLOv8n (nano version)
- **Why YOLOv8n?** Lightweight, fast, suitable for real-time detection on mobile or embedded devices.
- **Training Source**: Dataset located at `/kaggle/input/garbage-detection/GARBAGE CLASSIFICATION/data.yaml`

---

## 🧠 Model Architecture
- **Base Model**: `yolov8n.pt`
- **Transfer Learning**: Yes (pretrained weights used)
- **Number of Classes**: 6
- **Class Labels**:
  - BIODEGRADABLE
  - CARDBOARD
  - GLASS
  - METAL
  - PAPER
  - PLASTIC

---

## 🧪 Training Details
- **Epochs**: 120  
- **Batch Size**: 32  
- **Image Size**: 640×640  
- **Learning Rate (initial)**: 0.01  
- **Weight Decay**: 0.0005  
- **Warmup Epochs**: 3.0  
- **Momentum**: 0.937  
- **Augmentations**:
  - Horizontal Flip: 0.5  
  - Translate: 0.1  
  - Scale: 0.5  
  - HSV Adjustments: Hue = 0.015, Saturation = 0.7, Value = 0.4  
  - Random Erasing: 0.4  
  - Mosaic: Enabled  
  - AutoAugment: RandAugment  
- **Optimizer**: Automatically selected by Ultralytics  
- **Pretrained**: True

---

## 📊 Results
- **Approximate Accuracy**: ~70%  
  *(Bounding box detection)*  
  ⚠️ *This is an estimated value provided by the author — validation is still pending.*

---

## 📥 How Prediction Works

This model is used in **object detection mode** to identify multiple types of garbage items in a single image.

### 🔄 Prediction Flow:
1. Input image is resized to **640×640**
2. Image is passed through the YOLOv8 detection engine
3. Model returns:
   - 📦 Bounding boxes (object locations)
   - 📈 Confidence scores for each prediction

---

## 👤 Author

- **Model trained by**: Daulet  
- **Documentation prepared by**: Asylkhan



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
1. All the requirements and libraries for the project are in the **requirements.txt** (no react_native libraries)
2. Model weights for EfficientNet(MaybeTheBest2.pth) and Yolo(bestYolo2.pt) are available here: [Google Drive Link](https://drive.google.com/drive/folders/1zjOuYtXwM5OrVcitNX3mJ4vauTer9j85?usp=share_link)
3. Demo Video of Mobile App [Google Drive Link](https://drive.google.com/file/d/1s14zmmTMxAKD6HVjtnieNHRnDSzfx_MK/view?usp=share_link)

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
- **Approximate Accuracy**: ~75%  
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

---
---
---

# Garbage Detection AI Server Documentation

 ## 1. Overview
 This document describes the functionality of the main component which serves the purpose of accepting client requests, processing them and returning the desired result in an efficient manner.  The server is composed of **3 parts**:
 

 - **API Layer,** which ensures connectivity between the clients and the processing layer
 - **Processing Layer,** which parses client requests and runs inference on the data received, sending back inference results
 - **Asset Collector,** which makes sure that model files are present and accesible for the processing layer

## 2. API Layer
By making use of the **FastAPI** framework, the API layer exposes an endpoint where the client would send the image data and receive the result of the inference process (detection label, bounding boxes).  
The API layer class instantiates the **classifier object** and passes the data received from the client for further processing by placing it in a queue and awaiting the results using asynchronous methods.

## 3. Processing Layer
This component is responsible for emptying the queue containing client data, running inference, and returning the results. A worker method constantly checks if the queue contains data, in which case it will pass the image data to the `run_inference` method where the detection models will be employed.  
The `run_inference` method is called at the API layer in order to obtain the results from the classifier class, passing them back to the client in JSON format.

## 4. Asset Collector
The asset collector object will be instantiated at the processing layer level in order to ensure the server has the necessary files for running inference (e.g. model files, post processing files) and to restrict the access to those files outside the scope of the server. By making use of custom exceptions and proper handling, the server will not start if model files are not present where they are supposed to be. The asset collector object encapsulates the paths to the models, making the accesible only at the processing layer level, therefore keeping sensitive information away from the clients.

---
---
---

# Documentation for ⁠ App.js ⁠ and running the app (Expo + real server)
>
	⁠In short: the React Native (Expo) client takes/makes a photo, shows a preview, *sends it to a real backend* and displays the *actual result*. In this version, the API address is set as a **constant in ⁠ App.js ⁠*:

 ```js
 const API_URL = 'http://<PC_IPV4>:8000/classify-image';
  ```
>
	⁠Replace ⁠ <PC_IPV4> ⁠ with the local IPv4 address of your computer, which must be on the same Wi-Fi network as your phone.

---

## 1) Structure of ⁠ App.js ⁠

*Purpose: the root component. Initializes camera/gallery permissions, renders the UI, prepares bytes (if needed), **sends the image to ⁠ API_URL ⁠* and displays the response.*

### 1.1 Imports (typical)

•⁠  ⁠⁠ react ⁠, ⁠ useState ⁠, ⁠ useEffect ⁠, ⁠ useMemo ⁠, ⁠ useCallback ⁠  
•⁠  ⁠⁠ expo-image-picker ⁠ (gallery/camera)  
•⁠  ⁠⁠ react-native ⁠ — ⁠ View ⁠, ⁠ Text ⁠, ⁠ Image ⁠, ⁠ Pressable ⁠, ⁠ SafeAreaView ⁠, ⁠ StyleSheet ⁠, ⁠ ActivityIndicator ⁠, ⁠ Alert ⁠  
•⁠  ⁠⁠ expo-file-system/legacy ⁠ — read file from filesystem  
•⁠  ⁠⁠ base64-js ⁠ — reliable conversion bytes ↔️ base64  

### 1.2 State

•⁠  ⁠⁠ imageUri ⁠ — path to the selected/captured file  
•⁠  ⁠⁠ phase ⁠ — screen stage: ⁠ CAPTURE ⁠ → ⁠ PREVIEW ⁠ → ⁠ PROCESSING ⁠ → ⁠ RESULT ⁠  
•⁠  ⁠⁠ loading ⁠ — request indicator  
•⁠  ⁠⁠ result ⁠ — response data (image/JSON depending on endpoint)  

### 1.3 Key functions

•⁠  ⁠⁠ pickImage() ⁠ / ⁠ takePhoto() ⁠ — select/capture and move to preview  
•⁠  ⁠⁠ readBytes(uri) ⁠ — read file and prepare *⁠ Uint8Array ⁠* or *base64* (both options exist in the project)  
•⁠  ⁠⁠ sendToServer() ⁠ — send to *⁠ API_URL ⁠. In this build, the path *⁠ /classify-image ⁠* is used, which returns an **image* (binary/octet-stream or image/*). Alternative endpoint — ⁠ /classify ⁠ with JSON (if enabled on the server)  
•⁠  ⁠⁠ reset() ⁠ — clear state for a new capture  

### 1.4 UI

•⁠  ⁠Buttons: “Pick from Gallery”, “Take Photo”, “Send”, “Retry”  
•⁠  ⁠Preview screen with mini-cards/color theme  
•⁠  ⁠Processing screen with loader  
•⁠  ⁠Result screen: response image *or* JSON fields — depending on the chosen endpoint  

---

## 2) API address configuration

At the top of ⁠ App.js ⁠ find the line:
```⁠js
const API_URL = "http://10.9.105.98:8000/classify-image";
 ```

and replace the IP *with your PC’s address* in the local network (example: ⁠ 192.168.0.12 ⁠). Make sure your phone and PC are connected to the *same* Wi-Fi network.

	⁠How to check IPv4 on Windows: ⁠ Win + R ⁠ → ⁠ cmd ⁠ → ⁠ ipconfig ⁠ → line ⁠ IPv4 Address ⁠. Take the address of your active network (usually 192.168.x.x).

If you need environment switching without editing code, you can replace the constant with reading from ⁠ .env ⁠ (⁠ EXPO_PUBLIC_API_URL ⁠) — but the current version uses a *hard-coded* ⁠ API_URL ⁠.

---

## 3) Sending to the server: two scenarios

### 3.1 Server returns an *image* (⁠ /classify-image ⁠)

```js
async function sendToServer() {
  try {
    setLoading(true);

    // prepare form-data with file
    const form = new FormData();
    form.append('file', { uri: imageUri, name: 'image.jpg', type: 'image/jpeg' });

    const res = await fetch(API_URL, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    // read binary response and show as image
    const blob = await res.blob();
    const reader = new FileReader();
    reader.onloadend = () => {
      // data URL: can be passed into <Image source={{ uri: reader.result }} />
      setResult({ previewUri: reader.result, type: 'image' });
    };
    reader.readAsDataURL(blob);
  } catch (e) {
    Alert.alert('Error', e?.message || 'Failed to send image');
  } finally {
    setLoading(false);
  }
}
```

	⁠In Expo (native) instead of ⁠ FileReader ⁠ you can use ⁠ expo-file-system ⁠ / ⁠ ImageManipulator ⁠, or save blob to a file and provide ⁠ uri ⁠ to the ⁠ <Image/> ⁠ component.

### 3.2 Server returns *JSON* (⁠ /classify ⁠)

```js
const API_JSON = 'http://<PC_IPV4>:8000/classify';

async function sendToServerJson() {
  setLoading(true);
  try {
    const form = new FormData();
    form.append('file', { uri: imageUri, name: 'image.jpg', type: 'image/jpeg' });

    const res = await fetch(API_JSON, { method: 'POST', body: form, headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const json = await res.json();
    setResult({ ...json, type: 'json' }); // expected: { label, score, ... }
  } catch (e) {
    Alert.alert('Error', e?.message || 'Request failed');
  } finally {
    setLoading(false);
  }
}
```

---

## 4) Quick start (Frontend, Expo)

⁠ bash
# install dependencies
npm install
# or
yarn install

# start Metro
npx expo start
 ⁠

Then open *Expo Go* on your phone → scan the QR code. If LAN does not work, switch to *Tunnel* in the Expo web panel.

*Common issues*: cache (⁠ npx expo start -c ⁠), camera/gallery permissions, VPN/firewall, router client isolation (AP isolation), connect to the same WiFi with computer.

---

## 5) Server (for reference)

Your real server is already running. If you need a local FastAPI mock — draft below:

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO

app = FastAPI()

@app.post('/classify')
async def classify(file: UploadFile = File(...)):
    data = await file.read()
    return JSONResponse({'label': 'demo', 'score': 1.0, 'bytes': len(data)})

@app.post('/classify-image')
async def classify_image(file: UploadFile = File(...)):
    data = await file.read()
    return StreamingResponse(BytesIO(data), media_type=file.content_type)
```

Run with: ⁠ uvicorn main:app --reload --host 0.0.0.0 --port 8000 ⁠.

---

## 6) Report checklist

•⁠  ⁠[ ] QR code ⁠ npx expo start ⁠ in terminal/browser  
•⁠  ⁠[ ] Source selection screen in Expo Go  
•⁠  ⁠[ ] File preview screen  
•⁠  ⁠[ ] Result screen:

  * image response for ⁠ /classify-image ⁠, *or*  
  * JSON fields (label/score/...) for ⁠ /classify ⁠  
•⁠  ⁠[ ] (Optional) screenshot of request to ⁠ http://<PC_IPV4>:8000/classify-image ⁠ or ⁠ /classify ⁠  

---

## 7) Short defense

*EN (ref): The Expo client handles image selection/capture, sends the file to a **live endpoint* ⁠ API_URL ⁠ (⁠ /classify-image ⁠ by default) and renders either an image or JSON response. The API address is hard-coded in ⁠ App.js ⁠ for fast LAN testing.*



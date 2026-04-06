# 🍝 Spaghetti MLOps 

This repository presents a **clean, independent implementation** of a spaghetti defect segmentation system using modern computer vision and MLOps practices.

⚠️ **Important Note**  
This project is built from scratch.  
The original model and dataset from the company where I previously worked are **not used at all**.

---

## 🚀 Overview

This project focuses on:

- Semantic segmentation for spaghetti defect detection  
- Clean dataset preparation pipeline  
- YOLO-based segmentation training  
- MLOps-ready structure for deployment and scaling  

---

## 📦 Dataset Preparation

The dataset was downloaded using the **Roboflow API**.

- Dataset exported in the required format for training  
- No proprietary or company dataset is used  
- Fully independent and reproducible setup  

### Download Script provided by Roboflow

```python
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="")
project = rf.workspace("").project("spaghetti-segmentation")
version = project.version()
dataset = version.download("yolo26")
```

---

## 📂 Dataset Structure  

```
data/
├── spaghetti-segmentation-dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml
│
```

This structure is fully compatible with YOLO segmentation training.

---

## ⚙️ Project Structure

```
spaghetti-mlops/
│
├── data/               # Dataset
├── src/                # Training and inference code
├── models/             # Saved models
├── configs/            # Config files
├── docker/             # Docker setup
├── mlops/              # CI/CD and monitoring
├── notebooks/          # Experiments
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🏋️ Training

### Step 1: Go to Dataset Folder

```bash
cd ~/spaghetti-segmentation-dataset/
```

👉 This step ensures you are inside the dataset directory where `data.yaml` and image folders are located.

---

### Step 2: Install YOLO

```bash
pip install ultralytics
```

👉 Installs the Ultralytics YOLO framework required for training and inference.

---

### Step 3: Train the Model

```bash
yolo task=segment mode=train \
model=yolo26n-seg.pt \
data=data.yaml \
epochs=100 \
imgsz=640 \
batch=8
```

👉 This command:
- Uses a segmentation model (`-seg`)  
- Loads your dataset using `data.yaml`  
- Trains for 100 epochs  
- Uses image size of 640 for training  
- Sets batch size to 8 based on GPU capacity  
- Automatically saves the best-performing model  

---

## 📁 Training Output

```
runs/segment/train/
```

👉 This folder contains:
- Training logs  
- Loss curves  
- Evaluation metrics  
- Model checkpoints  

Best model:

```
runs/segment/train/weights/best.pt
```

👉 This is the final trained model you should use for inference and deployment.

---

## ⚡ Monitor Training

During training, you will observe:

- Loss decreasing over epochs  
- mAP (accuracy) increasing  
- Performance stabilizing  

👉 These indicators show the model is learning properly.

---

## 🧪 Prediction

```bash
yolo task=segment mode=predict \
model=runs/segment/train/weights/best.pt \
source=test/images
```

👉 This step:
- Loads your trained model  
- Runs inference on test images  
- Generates segmentation outputs  

---

## 📌 Prediction Output

```
runs/segment/predict/
```

👉 This folder contains:
- Predicted images with segmentation masks  
- Visualization results  

---

## 🧠 Model

- YOLO-based segmentation model  
- Suitable for real-time applications  
- Lightweight and scalable  

---

## 🔥 Future Improvements

- ONNX export  
- TensorRT optimization  
- Real-time inference pipeline  
- Docker deployment  
- CI/CD integration  
- Monitoring system  

---

## 📌 Key Highlights

- Clean-room implementation  
- No dependency on previous company assets  
- Fully reproducible pipeline  
- MLOps-ready architecture  

---

## 🤝 Contribution

Open for improvements and extensions.

---

## 📄 License

MIT License

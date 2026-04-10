# 🍝 Spaghetti Failure Detection - MLOps 

This repository presents a **clean, independent implementation** of a spaghetti defect segmentation system using modern computer vision and MLOps practices.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c6fe0375-36d3-4d68-a65d-fa3f2522a27f" width="17%" />
  <img src="https://github.com/user-attachments/assets/7774b49b-2ee2-435c-9f93-4ea2a9bc856c" width="40%" />
  <img src="https://github.com/user-attachments/assets/3946fa68-4615-436c-b049-116d27f59841" width="40%" />
</p>



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

- Download the dataset you want to train  
- Dataset exported in the required format for training  
- No proprietary or company dataset is used  
- Fully independent and reproducible setup  

For this project, I downloaded a **spaghetti segmentation dataset** that was already provided in **YOLO26 format**, making it directly compatible with training.

### Download Script provided by Roboflow

```python
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("WORKSPACE_NAME").project("PROJECT_NAME")
version = project.version(VERSION_NUMBER)
dataset = version.download("FORMAT_NAME")
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
python3 src/train/train.py \
--data ./data/spaghetti-segmentation-dataset/data.yaml \
--epochs 100 \
--imgsz 640 \
--batch 8
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

After training completes, all results will be saved in:

```
runs/segment/train/
```

👉 This folder contains:
- Training logs  
- Evaluation metrics and plots  
- Model checkpoints 
- Loss curves  

Final trained weights will be saved in:

```
runs/segment/train/weights/
```

👉 The best model:

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
<img width="4000" height="1200" alt="results" src="https://github.com/user-attachments/assets/195e84fd-a819-4ee6-970e-4e256076c33d" />



---

## 🧪 Prediction

```bash
python3 src/inference/torch_infer.py \
--model models/best.pt \
--source data/spaghetti-segmentation-dataset/test/images \
--save
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

## ⚡ Model Export (ONNX & TensorRT)

After training, the model can be exported for optimized inference and deployment.

---

### 🚀 Step 1: Export to ONNX

```bash
python3 src/export/export_onnx.py \
--model models/best.pt \
--output models/best.onnx
```

👉 This step:
- Converts the trained PyTorch model (`.pt`) into ONNX format  
- Enables cross-platform deployment  
- Prepares the model for further optimization  

Output:

```
models/best.onnx
```

---

### 🚀 Step 2: Convert ONNX to TensorRT

```bash
python3 src/export/export_tensorrt.py \
--onnx models/best.onnx \
--engine models/best.trt \
--fp16
```

👉 This step:
- Converts ONNX model to TensorRT engine  
- Uses FP16 precision for faster inference  
- Optimized for NVIDIA GPUs  

Output:

```
models/best.trt
```

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

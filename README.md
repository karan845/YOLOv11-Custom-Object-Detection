

## **YOLOv11 Custom Object Detection**

Welcome to the **YOLOv11 Custom Object Detection ** repository! This repository provides a complete setup for training and running object detection on **images, webcam feeds, and YouTube videos** using **YOLOv11**.

---

## 📌 **Table of Contents**
- [1️⃣ Features](#1️⃣-features)
- [2️⃣ Installation Guide](#2️⃣-installation-guide)
- [3️⃣ Training the YOLOv11 Model](#3️⃣-training-the-yolov11-model)
- [4️⃣ Running Object Detection](#4️⃣-running-object-detection)
- [5️⃣ Using the Model](#5️⃣-using-the-model)
- [6️⃣ Troubleshooting](#6️⃣-troubleshooting)


---

## **1️⃣ Features**
✔ **Custom Object Detection** using YOLOv11  
✔ **Train on Custom Datasets**  
✔ **Supports Images, Webcam, and YouTube Videos**  
✔ **Saves processed images and videos**  
✔ **Supports GPU & CPU**  

---

## **2️⃣ Installation Guide**
Follow these steps to set up the project.

### **🔹 Step 1: Clone the Repository**
```bash
git clone https://github.com/karan845/yolov11_custom_object_detection.git
cd yolov11_custom_segmentation
```

### **🔹 Step 2: Set Up a Conda Environment**
```bash
conda create --name yolov11_env python=3.11 -y
conda activate yolov11_env
```

### **🔹 Step 3: Install Required Dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # For GPU
pip install opencv-python numpy yt-dlp ultralytics tqdm pillow scipy
```

---

## **3️⃣ Training the YOLOv11 Model**
To train the YOLOv11 model on your **custom dataset**, follow these steps.

### **🔹 Step 1: Prepare the Dataset**
- Structure your dataset inside the `train/` and `val/` folders.
- Create a `custom_dataset.yaml` file with:
```yaml
train: ./train  # Path to training data
val: ./val      # Path to validation data
nc: 1           # Number of object classes
names: ['your_class']  # Replace with your class name
```

### **🔹 Step 2: Train the Model**
Run the following command to start training:
```bash
yolo train model=yolov11_custom.pt data=custom_dataset.yaml epochs=100 imgsz=640
```
🔹 **Modify parameters**:
- `epochs=100` → Adjust training iterations  
- `imgsz=640` → Image resolution  

Once training is complete, the best model will be saved in `runs/train/exp/weights/best.pt`.

---

## **4️⃣ Running Object Detection**
Once the model is trained, you can use it to detect objects.

### **🔹 1️⃣ Run on an Image**
```bash
python predict.py --image path/to/image.jpg
```
✅ Output saved as: `output_image.jpg`

### **🔹 2️⃣ Run on a Webcam**
```bash
python predict.py --webcam
```
✅ Output video saved as: `output_webcam.avi`

### **🔹 3️⃣ Run on a YouTube Video**
```bash
python predict.py --youtube "https://www.youtube.com/watch?v=ZuZtQeGHxi8"
```
✅ Output video saved as: `output_youtube.avi`

---

## **5️⃣ Using the Model**
If you want to use your **trained model**, replace `yolov11_custom.pt` with `best.pt` from your training folder:
```bash
yolo predict model=runs/train/exp/weights/best.pt source=path/to/image.jpg
```

---

## **6️⃣ Troubleshooting**
### 🔹 **CUDA Error / GPU Not Detected**
Check if PyTorch detects your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If **False**, reinstall CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 🔹 **YouTube Video Not Working**
Update `yt-dlp`:
```bash
pip install --upgrade yt-dlp
```

---



🚀 **Happy Coding!** 🚀

import torch
from ultralytics import YOLO

# 1. Check if Apple Silicon GPU (MPS) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the YOLOv11s model
model = YOLO("yolo11s.pt")

# 3. Define dataset configuration
dataset_yaml = "custom_dataset.yaml"

# 4. Train the model with optimizations
model.train(
    data="custom_dataset.yaml",
    epochs=50,  
    batch=8,  
    imgsz=640,  
    device="mps",  
    workers=2,  
    optimizer="AdamW",  
    lr0=0.001,  # Lower LR for adaptive optimizers  
    patience=5,  
    weight_decay=0.0005,  
    augment=True,  
    cos_lr=True,  
    verbose=False,  
    cache=False,  
    iou=0.6,
    conf=0.25,
    max_det=25,
    amp=True,  
)

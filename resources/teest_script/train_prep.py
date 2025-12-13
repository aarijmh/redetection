import os
import torch
from ultralytics import YOLO

# Set environment variables for better memory management
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    # Empty cache to free up memory
    torch.cuda.empty_cache()

# Load model with device specified
model = YOLO("yolo11l.pt").to(device)

# Training configuration
try:
    results = model.train(
        data="data.yaml",
        epochs=40,
        batch=16,
        imgsz=640,
        freeze=10,          # freeze backbone
        lr0=0.0008,
        device=0,           # Use first GPU
        workers=4,          # Adjust based on your CPU cores
        cache='ram',        # Cache images in RAM for faster training
        single_cls=False,   # Set to True if you have only one class
        amp=True,           # Automatic Mixed Precision training
        optimizer='auto',   # Let YOLO choose the best optimizer
    )
except Exception as e:
    print(f"Error during training: {e}")
    if device == 'cuda':
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1E9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1E9:.2f} GB")

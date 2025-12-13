import os
import shutil
from pathlib import Path

def merge_coco_with_custom(
    coco_path="datasets/coco",
    custom_path="datasets/printer_custom",
    out_path="datasets/coco_plus_custom"
):
    coco_train_img = Path(coco_path)/"images/train2017"
    coco_train_lbl = Path(coco_path)/"labels/train2017"

    custom_train_img = Path(custom_path)/"images/train"
    custom_train_lbl = Path(custom_path)/"labels/train"

    custom_val_img = Path(custom_path)/"images/val"
    custom_val_lbl = Path(custom_path)/"labels/val"

    out_train_img = Path(out_path)/"images/train"
    out_train_lbl = Path(out_path)/"labels/train"
    out_val_img = Path(out_path)/"images/val"
    out_val_lbl = Path(out_path)/"labels/val"

    # Create directories
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)
    out_val_img.mkdir(parents=True, exist_ok=True)
    out_val_lbl.mkdir(parents=True, exist_ok=True)

    # Copy COCO train images + labels
    print("Copying COCO train set...")
    for img in coco_train_img.iterdir():
        shutil.copy(img, out_train_img)
    for lbl in coco_train_lbl.iterdir():
        shutil.copy(lbl, out_train_lbl)

    # Copy your custom training data
    print("Adding Custom Training Data...")
    for img in custom_train_img.iterdir():
        shutil.copy(img, out_train_img)
    for lbl in custom_train_lbl.iterdir():
        shutil.copy(lbl, out_train_lbl)

    # Copy your custom validation data
    print("Adding Custom Validation Data...")
    for img in custom_val_img.iterdir():
        shutil.copy(img, out_val_img)
    for lbl in custom_val_lbl.iterdir():
        shutil.copy(lbl, out_val_lbl)

    print("Done! Merged dataset is ready at:", out_path)

# Example usage:
merge_coco_with_custom()

# Resources Folder

Place your input videos and any required model files in this folder.

## Usage

When running the object counting pipeline, specify only the filename (not the full path):

```bash
python run_object_counting.py --video my_video.mp4 --objects chair desk
```

The system will automatically look for `my_video.mp4` in this `resources/` folder.

## Custom Models

To use a custom YOLO model, place the .pt file in this folder and specify it:

```bash
python run_object_counting.py --video my_video.mp4 --model yolov8s.pt --objects chair desk
```

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- And other formats supported by OpenCV
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load video and model
cap = cv2.VideoCapture('resources/videos/test3.mp4')
model = YOLO('yolo11l.pt')

# Track chair positions and IDs
chair_positions = defaultdict(list)
frame_count = 0
unique_positions = set()

print("Analyzing chair detection patterns...")

while frame_count < 100:  # Check first 100 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection with low confidence for chairs only
    results = model.predict(source=frame, conf=0.15, verbose=False)
    result = results[0]
    
    chairs_in_frame = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cls]
        
        if name == 'chair' and conf > 0.15:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Create a position key (rounded to group nearby detections)
            pos_key = (int(center_x/50)*50, int(center_y/50)*50)  # Group by 50px grid
            unique_positions.add(pos_key)
            
            chairs_in_frame.append({
                'frame': frame_count,
                'center': (center_x, center_y),
                'conf': conf,
                'bbox': xyxy
            })
    
    if chairs_in_frame:
        print(f"Frame {frame_count}: {len(chairs_in_frame)} chairs")
        for chair in chairs_in_frame:
            print(f"  Position: ({chair['center'][0]:.0f}, {chair['center'][1]:.0f}), Conf: {chair['conf']:.3f}")
    
    frame_count += 1

cap.release()

print(f"\nSummary:")
print(f"Unique chair positions (50px grid): {len(unique_positions)}")
print("Positions:")
for i, pos in enumerate(sorted(unique_positions)):
    print(f"  Chair {i+1}: {pos}")

# Save some frames with chair detections
cap = cv2.VideoCapture('resources/videos/test3.mp4')
frame_count = 0
saved_frames = 0

while frame_count < 200 and saved_frames < 5:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(source=frame, conf=0.15, verbose=False)
    result = results[0]
    
    chairs = [box for box in result.boxes if int(box.cls[0]) == 56]  # class 56 = chair
    
    if chairs:
        annotated_frame = result.plot()
        cv2.imwrite(f'chair_analysis_frame_{frame_count}.jpg', annotated_frame)
        saved_frames += 1
        print(f"Saved frame {frame_count} with {len(chairs)} chairs")
    
    frame_count += 1

cap.release()
print(f"\nSaved analysis frames as chair_analysis_frame_*.jpg")

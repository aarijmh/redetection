#!/usr/bin/env python3

import cv2
import numpy as np
from core.registry import build_detector, build_tracker, build_cmc
from motion.anchor_manager import AnchorManager
from utils.metrics import TrackCSVWriter

def debug_tracking():
    # Initialize components
    detector = build_detector('yolov8', 'yolo11l.pt', 0.76, 0.10, ['tv'], device='cuda')
    tracker = build_tracker('strongsort', reid=None, match_iou=0.10)
    cmc = build_cmc('superglue', device='cuda')
    anchor_mgr = AnchorManager(max_time_lost=120, dist_thresh=200.0, app_thresh=0.95, use_appearance=False)
    
    # Open video
    cap = cv2.VideoCapture('output.mp4')
    
    # Process frames around 1380 where we saw the issue
    for frame_idx in range(1370, 1390):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        print(f"\n=== Frame {frame_idx} ===")
        
        # Get detections
        dets = detector.detect(frame)
        print(f"Detections: {len(dets)}")
        for i, d in enumerate(dets):
            print(f"  {i}: bbox={d.xyxy}, conf={d.conf:.3f}, cls={d.cls}")
        
        # Get tracks from tracker
        tracks = tracker.update(dets, frame)
        print(f"Tracks from tracker: {len(tracks)}")
        for t in tracks:
            print(f"  Track {t.id}: bbox={t.xyxy}, cls={t.cls}")
        
        # Apply anchor manager
        remapped_tracks = anchor_mgr.remap_and_update(tracks, frame, frame_idx)
        print(f"Tracks after anchor manager: {len(remapped_tracks)}")
        for t in remapped_tracks:
            print(f"  Track {t.id}: bbox={t.xyxy}, cls={t.cls}")
            
        # Check for duplicate IDs
        track_ids = [t.id for t in remapped_tracks]
        if len(track_ids) != len(set(track_ids)):
            print(f"*** DUPLICATE TRACK IDs FOUND: {track_ids} ***")
    
    cap.close()

if __name__ == "__main__":
    debug_tracking()

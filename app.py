import argparse, cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from collections import defaultdict
from core.registry import build_detector, build_tracker, build_cmc, build_reid
from core.video_io import VideoReader, VideoWriter
from utils.metrics import TrackCSVWriter
from motion.anchor_manager import AnchorManager
import time
import torch

import sys, os
sys.path.append(os.path.abspath('SuperGluePretrainedNetwork'))  # adjust path

from motion.cmc_superglue import SuperPointSuperGlueCMC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='output.mp4')
    p.add_argument('--save-csv', default='tracks.csv')
    p.add_argument('--detector', default='yolov8', choices=['yolov8','rtdetr','dummy'])
    p.add_argument('--detector-weights', default='yolov8x.pt')
    p.add_argument('--tracker', default='strongsort', choices=['strongsort','botsort','deocsort','simplesort'])
    p.add_argument('--reid', default='osnet', choices=['osnet','none'])
    p.add_argument('--use-cmc', action='store_true')
    p.add_argument('--conf', type=float, default=0.76)
    p.add_argument('--iou', type=float, default=0.10)
    p.add_argument('--class-filter', nargs='*', default=None)
    p.add_argument('--cmc', default='homography', choices=['none','homography','superglue'])
    p.add_argument('--sample-rate', type=int, default=10, help='Sample every N frames for faster processing')
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'], help='Device to run inference on')
    return p.parse_args()


def print_summary(stats, processing_time, total_frames):
    print("\n" + "="*50)
    print("TRACKING SUMMARY")
    print("="*50)
    print(f"Total frames processed: {total_frames}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"FPS: {total_frames/processing_time:.2f}")
    print("\nObject Statistics:")
    print(f"- Total detections: {stats['total_detections']}")
    print(f"- Unique tracks: {stats['unique_tracks']}")
    print(f"- Average detections per frame: {stats['total_detections']/total_frames:.2f}")
    
    # Simple COCO class name mapping
    class_names = {
        0: 'person', 56: 'chair', 62: 'tv/monitor', 63: 'laptop', 64: 'book', 67: 'cell phone',
        73: 'laptop', 74: 'keyboard', 76: 'mouse', 45: 'bowl'
    }
    
    if stats['class_counts']:
        print("\nClass distribution:")
        for cls, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            unique_count = len(stats['class_unique_tracks'][cls])
            class_name = class_names.get(int(cls), f"class_{cls}")
            print(f"- {class_name} ({cls}): {count} detections, {unique_count} unique tracks")
    
    print("="*50 + "\n")

def main():
    args = parse_args()
    
    # Device detection and selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    detector = build_detector(args.detector, args.detector_weights, args.conf, args.iou, args.class_filter, device=device)
    reid = build_reid(args.reid)
    tracker = build_tracker(args.tracker, reid=reid, match_iou=args.iou, track_buffer=90)  # Slightly reduced buffer
    cmc = build_cmc(name=args.cmc, device=device)

    vr = VideoReader(args.video)
    vw = VideoWriter(args.output, vr.fps, vr.width, vr.height)
    csvw = TrackCSVWriter(args.save_csv)
    anchor_mgr = AnchorManager(max_time_lost=240, dist_thresh=30.0, app_thresh=0.5, use_appearance=True)  # More conservative settings

    # Initialize statistics
    stats = {
        'total_detections': 0,
        'unique_tracks': set(),
        'class_counts': defaultdict(int),
        'class_unique_tracks': defaultdict(set),  # Track unique track IDs per class
        'frame_times': []
    }

    prev = None
    t0 = time.time()
    
    # Get total frames for progress bar
    total_frames = int(vr.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(vr, 'cap') else 0
    
    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit='frames', ncols=100)
    
    for i, frame in enumerate(vr):
        # Frame sampling for faster processing
        if i % args.sample_rate != 0:
            continue
            
        frame_start = time.time()
        
        H = cmc.estimate(prev, frame) if prev is not None else None
        anchor_mgr.predict(H)
        tracker.set_cmc(H)
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame)
        # tracks = anchor_mgr.remap_and_update(tracks, frame, i)  # Disabled to reduce fragmentation
        
        # Update statistics
        stats['total_detections'] += len(tracks)
        for track in tracks:
            stats['unique_tracks'].add(track.id)  # Changed from track.track_id to track.id
            if hasattr(track, 'cls'):
                stats['class_counts'][track.cls] += 1
                stats['class_unique_tracks'][track.cls].add(track.id)
        
        anno = tracker.draw(frame.copy(), tracks)
        vw.write(anno)
        csvw.write(i, tracks)
        
        prev = frame
        
        # Update progress bar
        frame_time = time.time() - frame_start
        stats['frame_times'].append(frame_time)
        pbar.set_postfix({
            'fps': f"{1/frame_time:.1f}",
            'tracks': len(tracks),
            'dets': len(dets)
        })
        pbar.update(1)
    
    # Cleanup
    pbar.close()
    vw.close()
    csvw.close()
    vr.close()
    
    # Calculate and print summary
    processing_time = time.time() - t0
    stats['unique_tracks'] = len(stats['unique_tracks'])
    print_summary(stats, processing_time, i+1)

if __name__ == '__main__':
    main()

"""
Office Video Object Detection, Tracking and Re-identification
Uses custom re-identification system for consistent object tracking
"""

import cv2
import numpy as np
from collections import defaultdict
import argparse
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.registry import build_detector, build_tracker, build_reid
from motion.anchor_manager import AnchorManager

class OfficeTracker:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3, device='auto'):
        """
        Initialize the office tracker with proper re-identification system
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            device: Device for inference ('auto', 'cpu', 'cuda')
        """
        import torch
        # Device detection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Office-relevant classes (COCO dataset) - moved before component initialization
        self.office_classes = {
        
            63: 'laptop',
            64: 'mouse',
            66: 'keyboard',
            67: 'cell phone',
            56: 'chair'
        }
        
        # Initialize detection, tracking, and re-identification components
        self.detector = build_detector('yolov8', model_path, conf_threshold, 0.5, 
                                     list(self.office_classes.keys()), device=self.device)
        self.reid = build_reid('osnet')
        self.tracker = build_tracker('strongsort', reid=self.reid, match_iou=0.2, track_buffer=15)
        self.anchor_manager = AnchorManager(max_time_lost=240, dist_thresh=30.0, 
                                          app_thresh=0.5, use_appearance=True)
        
        # Track history for visualization
        self.track_history = defaultdict(lambda: [])
        self.track_colors = {}
        
    def get_color_for_track(self, track_id):
        """Generate consistent color for each track ID"""
        if track_id not in self.track_colors:
            np.random.seed(int(track_id))
            self.track_colors[track_id] = tuple(
                int(x) for x in np.random.randint(0, 255, 3)
            )
        return self.track_colors[track_id]
    
    def process_video(self, video_path, output_path='out.mp4', 
                     show_trails=True, save_video=True):
        """
        Process video with object detection, tracking and re-identification
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_trails: Whether to show movement trails
            save_video: Whether to save the output video
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        detected_objects = defaultdict(lambda: {'count': 0, 'frames': []})
        prev_frame = None
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {w}x{h}, FPS: {fps}")
        print(f"Using device: {self.device}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection step
            detections = self.detector.detect(frame)
            
            # Tracking step
            tracks = self.tracker.update(detections, frame)
            
            # Re-identification step - this is the key fix!
            tracks = self.anchor_manager.remap_and_update(tracks, frame, frame_count)
            
            # Process tracks for visualization
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.xyxy)
                track_id = int(track.id)
                cls = int(track.cls)
                conf = float(track.conf)
                
                # Get class name
                class_name = self.office_classes.get(cls, f'class_{cls}')
                
                # Track object statistics
                obj_key = f"{class_name}_{track_id}"
                detected_objects[obj_key]['count'] += 1
                detected_objects[obj_key]['frames'].append(frame_count)
                
                # Get color for this track
                color = self.get_color_for_track(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Calculate center point for trails
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                self.track_history[track_id].append(center)
                
                # Limit trail length
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Draw trails
                if show_trails and len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)
                
                # Draw label
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add object count
            unique_tracks = len(set(
                k.split('_')[1] for k in detected_objects.keys()
            ))
            cv2.putText(frame, f"Tracked Objects: {unique_tracks}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Office Tracking', frame)
            
            # Save frame
            if save_video:
                out.write(frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        self.print_statistics(detected_objects, frame_count)
    
    def print_statistics(self, detected_objects, total_frames):
        """Print detection and tracking statistics"""
        print("\n" + "="*60)
        print("TRACKING STATISTICS")
        print("="*60)
        print(f"Total frames processed: {total_frames}")
        
        # Calculate unique objects and their persistence
        unique_objects = {}
        for obj_key, data in detected_objects.items():
            class_name, track_id = obj_key.rsplit('_', 1)
            if track_id not in unique_objects:
                unique_objects[track_id] = {
                    'class': class_name,
                    'detections': 0,
                    'frames': [],
                    'first_frame': float('inf'),
                    'last_frame': 0
                }
            
            unique_objects[track_id]['detections'] += data['count']
            unique_objects[track_id]['frames'].extend(data['frames'])
            unique_objects[track_id]['first_frame'] = min(unique_objects[track_id]['first_frame'], data['frames'][0])
            unique_objects[track_id]['last_frame'] = max(unique_objects[track_id]['last_frame'], data['frames'][-1])
        
        print(f"\nUnique objects tracked: {len(unique_objects)}")
        
        # Summary by class
        class_summary = {}
        for track_id, obj_data in unique_objects.items():
            cls = obj_data['class']
            if cls not in class_summary:
                class_summary[cls] = {'count': 0, 'total_detections': 0, 'avg_persistence': 0}
            class_summary[cls]['count'] += 1
            class_summary[cls]['total_detections'] += obj_data['detections']
            persistence = obj_data['last_frame'] - obj_data['first_frame'] + 1
            class_summary[cls]['avg_persistence'] += persistence
        
        print("\nSummary by class:")
        print("-"*60)
        for class_name, summary in sorted(class_summary.items()):
            avg_persistence = summary['avg_persistence'] / summary['count'] if summary['count'] > 0 else 0
            print(f"{class_name}:")
            print(f"  - Unique objects: {summary['count']}")
            print(f"  - Total detections: {summary['total_detections']}")
            print(f"  - Average persistence: {avg_persistence:.1f} frames")
        
        print("\nDetailed object tracking:")
        print("-"*60)
        
        for track_id, obj_data in sorted(unique_objects.items(), key=lambda x: x[1]['first_frame']):
            class_name = obj_data['class']
            detections = obj_data['detections']
            first_frame = obj_data['first_frame']
            last_frame = obj_data['last_frame']
            persistence = last_frame - first_frame + 1
            
            # Calculate detection frequency
            detection_freq = detections / persistence if persistence > 0 else 0
            
            print(f"\n{class_name} (ID: {track_id}):")
            print(f"  - Detections: {detections}")
            print(f"  - First seen: frame {first_frame}")
            print(f"  - Last seen: frame {last_frame}")
            print(f"  - Persistence: {persistence} frames")
            print(f"  - Detection frequency: {detection_freq:.2f} detections/frame")
            print(f"  - Coverage: {(persistence/total_frames)*100:.1f}% of video")

def main():
    parser = argparse.ArgumentParser(
        description='Office Video Object Detection and Tracking'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--no-trails', action='store_true',
                       help='Disable movement trails')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = OfficeTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        device='auto'
    )
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        show_trails=not args.no_trails,
        save_video=not args.no_save
    )

if __name__ == "__main__":
    main()
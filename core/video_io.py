
import cv2
import numpy as np

class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise RuntimeError(f'Cannot open {path}')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check for video rotation metadata
        self.rotation_angle = self._detect_rotation()
        
        # Adjust dimensions if video is rotated
        if self.rotation_angle in [90, 270]:
            self.width, self.height = self.height, self.width
    
    def _detect_rotation(self):
        """Detect video rotation from metadata"""
        # Try to get rotation tag (common in smartphone videos)
        rotation_tag = self.cap.get(cv2.CAP_PROP_ORIENTATION_META)
        
        # Convert rotation tag to angle
        if rotation_tag == 3:  # 180 degrees
            return 180
        elif rotation_tag == 6:  # 90 degrees clockwise
            return 90
        elif rotation_tag == 8:  # 90 degrees counter-clockwise
            return 270
        
        # Alternative method: check actual frame dimensions vs reported dimensions
        ret, frame = self.cap.read()
        if ret:
            actual_height, actual_width = frame.shape[:2]
            reported_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            reported_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Reset video position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # If dimensions are swapped, likely 90 or 270 degree rotation
            if (actual_width == reported_height and actual_height == reported_width):
                return 90  # Default to 90 degrees
        
        return 0
    
    def _rotate_frame(self, frame):
        """Rotate frame according to detected rotation"""
        if self.rotation_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def __iter__(self): return self
    
    def __next__(self):
        ok, frame = self.cap.read()
        if not ok: raise StopIteration
        return self._rotate_frame(frame)
    
    def close(self): self.cap.release()

class VideoWriter:
    def __init__(self, path, fps, w, h):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.wr = cv2.VideoWriter(path, fourcc, fps, (w,h))
        self.width = w
        self.height = h
        print(f"VideoWriter initialized: {w}x{h} at {fps} FPS")
    
    def write(self, frame): 
        # Ensure frame dimensions match writer dimensions
        if frame.shape[:2] != (self.height, self.width):
            print(f"Warning: Frame size {frame.shape[:2]} doesn't match writer size {(self.height, self.width)}")
        self.wr.write(frame)
    
    def close(self): self.wr.release()

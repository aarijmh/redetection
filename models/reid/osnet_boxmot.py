from .base import BaseReID
import numpy as np

class OSNetBoxMOT(BaseReID):
    def __init__(self):
        try:
            from boxmot.appearance.backends import get_backend
            self.model = get_backend('osnet_x0_5_imagenet')
            self.weights = 'osnet_x0_5_imagenet'
        except Exception:
            self.model = None
            self.weights = None
    
    def embed(self, frame, xyxy):
        if self.model is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            embedding = self.model(crop)
            return embedding / (np.linalg.norm(embedding) + 1e-6)
        except Exception:
            return None

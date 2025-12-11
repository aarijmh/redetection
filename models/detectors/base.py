
import numpy as np
class Detection:
    def __init__(self, xyxy, conf, cls, cls_name=None):
        self.xyxy = np.array(xyxy, float)
        self.conf = float(conf); self.cls = int(cls) if cls is not None else -1
        self.cls_name = cls_name
class BaseDetector:
    def detect(self, frame): raise NotImplementedError
class DummyDetector(BaseDetector):
    def detect(self, frame): return []

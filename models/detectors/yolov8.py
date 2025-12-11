
from .base import BaseDetector, Detection
class YOLOv8Detector(BaseDetector):
    def __init__(self, weights='yolov11l.pt', conf=0.75, iou=0.5, class_filter=None, device='cpu'):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model.to(device)
        self.device = device
        self.conf, self.iou, self.class_filter = conf, iou, class_filter
    def detect(self, frame):
        res = self.model.predict(source=frame, conf=self.conf, iou=self.iou, verbose=False, device=self.device)[0]
        dets=[]; names = res.names; boxes = res.boxes
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy(); conf = float(b.conf[0].cpu().numpy()); cls = int(b.cls[0].cpu().numpy())
            name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
            if self.class_filter and (name not in self.class_filter and cls not in self.class_filter): continue
            dets.append(Detection(xyxy, conf, cls, name))
        return dets

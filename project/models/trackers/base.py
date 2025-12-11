
import numpy as np, cv2
class Track:
    def __init__(self, track_id, xyxy, conf, cls, cls_name=None):
        self.id=int(track_id); self.xyxy=np.array(xyxy,float); self.conf=float(conf); self.cls=int(cls); self.cls_name=cls_name
class BaseTracker:
    def update(self, detections, frame): raise NotImplementedError
    def set_cmc(self, H): pass
    def draw(self, frame, tracks):
        for t in tracks:
            x1,y1,x2,y2 = t.xyxy.astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f'ID {t.id} {t.cls}', (x1,max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        return frame

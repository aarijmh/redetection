
from .base import BaseTracker, Track
import numpy as np
class BoxMOTTracker(BaseTracker):
    def __init__(self, tracker_type='strongsort', reid=None, match_iou=0.5):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if tracker_type == 'strongsort':
            from boxmot import StrongSort as Impl
            self.impl = Impl(reid_weights=reid.weights if reid else None, 
                           device=device,
                           half=True)
        elif tracker_type == 'botsort':
            from boxmot import BotSort as Impl
            self.impl = Impl(reid_weights=reid.weights if reid else None,
                           device=device,
                           half=True)  # half precision
        elif tracker_type == 'deocsort':
            from boxmot import DeepOcSort as Impl
            self.impl = Impl(model_weights=reid.weights if reid else None,
                           device=device)
        else:
            raise ValueError('unknown tracker')
        self.H=None
    def set_cmc(self, H): self.H = H
    def update(self, detections, frame):
        dets = []
        for d in detections:
            x1,y1,x2,y2 = d.xyxy; dets.append([x1,y1,x2,y2,d.conf,d.cls])
        dets = np.array(dets, float) if len(dets)>0 else np.empty((0,6), float)
        out = self.impl.update(dets, frame)
        tracks=[]
        for row in out:
            x1,y1,x2,y2,tid,cls = row[:6]
            tracks.append(Track(tid,[x1,y1,x2,y2],1.0,int(cls)))
        return tracks

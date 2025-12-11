
from .base import BaseTracker, Track
import numpy as np
class BoxMOTTracker(BaseTracker):
    def __init__(self, tracker_type='strongsort', reid=None, match_iou=0.5):
        if tracker_type=='strongsort':
            from boxmot import StrongSORT as Impl
        elif tracker_type=='botsort':
            from boxmot import BotSORT as Impl
        elif tracker_type=='deocsort':
            from boxmot import DeepOCSORT as Impl
        else:
            raise ValueError('unknown tracker')
        self.impl = Impl(model=None)
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

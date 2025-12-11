
from .base import BaseTracker, Track
import numpy as np
class SimpleSORTTracker(BaseTracker):
    def __init__(self, match_iou=0.5): self.next_id=1; self.tracks=[]; self.match_iou=match_iou
    def iou(self,a,b):
        ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
        inter = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
        area_a=max(0,(ax2-ax1)*(ay2-ay1)); area_b=max(0,(bx2-bx1)*(by2-by1))
        return inter/(area_a+area_b-inter+1e-6)
    def update(self, detections, frame):
        det_boxes=[d.xyxy for d in detections]; det_clses=[d.cls for d in detections]
        assigned=set(); new=[]
        for tid,tbox,tcls in self.tracks:
            j=-1; best=0
            for idx,db in enumerate(det_boxes):
                if idx in assigned: continue
                v=self.iou(tbox,db)
                if v>best: best=v; j=idx
            if j!=-1 and best>=self.match_iou: new.append((tid,det_boxes[j],det_clses[j])); assigned.add(j)
            else: new.append((tid,tbox,tcls))
        for idx,db in enumerate(det_boxes):
            if idx in assigned: continue
            new.append((self.next_id,db,det_clses[idx])); self.next_id+=1
        self.tracks=new
        return [Track(tid,tb,1.0,tc) for tid,tb,tc in self.tracks]

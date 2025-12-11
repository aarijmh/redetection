
import csv
class TrackCSVWriter:
    def __init__(self, path):
        self.f=open(path,'w',newline=''); self.w=csv.writer(self.f)
        self.w.writerow(['frame','id','x1','y1','x2','y2','cls','conf'])
    def write(self, frame_idx, tracks):
        for t in tracks:
            x1,y1,x2,y2 = t.xyxy
            self.w.writerow([frame_idx,t.id,x1,y1,x2,y2,t.cls,t.conf])
    def close(self): self.f.close()


import cv2
class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise RuntimeError(f'Cannot open {path}')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def __iter__(self): return self
    def __next__(self):
        ok, frame = self.cap.read();
        if not ok: raise StopIteration
        return frame
    def close(self): self.cap.release()

class VideoWriter:
    def __init__(self, path, fps, w, h):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.wr = cv2.VideoWriter(path, fourcc, fps, (w,h))
    def write(self, frame): self.wr.write(frame)
    def close(self): self.wr.release()

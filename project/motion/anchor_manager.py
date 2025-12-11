
import numpy as np
import cv2

try:
    from models.trackers.base import Track
except Exception:
    class Track:
        def __init__(self, track_id, xyxy, conf, cls, cls_name=None):
            self.id = int(track_id)
            self.xyxy = np.array(xyxy, dtype=float)
            self.conf = float(conf)
            self.cls = int(cls) if cls is not None else -1
            self.cls_name = cls_name

def _box_bottom_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, y2], dtype=float)

def _hsv_hist_signature(img, xyxy, bins=(8, 8, 8)):
    x1, y1, x2, y2 = map(int, np.clip(xyxy, 0, [img.shape[1]-1, img.shape[0]-1, img.shape[1]-1, img.shape[0]-1]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((bins[0]*bins[1]*bins[2],), dtype=float)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((bins[0]*bins[1]*bins[2],), dtype=float)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180, 0,256, 0,256])
    hist = hist.flatten().astype(float)
    norm = np.linalg.norm(hist) + 1e-6
    return hist / norm

class Anchor:
    def __init__(self, canon_id, img_pt, cls, appearance=None, ema_alpha=0.5):
        self.canon_id = int(canon_id)
        self.img_pt = np.array(img_pt, dtype=float)
        self.cls = int(cls)
        self.appearance = appearance.copy() if appearance is not None else None
        self.ema_alpha = float(ema_alpha)
        self.last_frame = -1
        self.time_lost = 0
        self.active = True

    def predict_with_homography(self, H):
        if H is None:
            return self.img_pt
        p = np.array([self.img_pt[0], self.img_pt[1], 1.0], dtype=float)
        q = H @ p
        if abs(q[2]) < 1e-6:
            return self.img_pt
        self.img_pt = q[:2] / q[2]
        return self.img_pt

    def update_observation(self, img_pt, appearance=None):
        self.img_pt = np.array(img_pt, dtype=float)
        if appearance is not None:
            if self.appearance is None:
                self.appearance = appearance.copy()
            else:
                self.appearance = self.ema_alpha * appearance + (1 - self.ema_alpha) * self.appearance
                n = np.linalg.norm(self.appearance) + 1e-6
                self.appearance = self.appearance / n
        self.time_lost = 0

class AnchorManager:
    def __init__(self, max_time_lost=30, dist_thresh=60.0, app_thresh=0.5, use_appearance=True):
        self.anchors = {}
        self.next_canon_id = 1
        self.max_time_lost = int(max_time_lost)
        self.dist_thresh = float(dist_thresh)
        self.app_thresh = float(app_thresh)
        self.use_appearance = bool(use_appearance)
        self.local_to_canon = {}

    def _new_canon_id(self):
        cid = self.next_canon_id
        self.next_canon_id += 1
        return cid

    def _appearance(self, frame, xyxy):
        return _hsv_hist_signature(frame, xyxy)

    def predict(self, H):
        for a in self.anchors.values():
            a.predict_with_homography(H)
            a.time_lost += 1
            if a.time_lost > self.max_time_lost:
                a.active = False

    def _best_anchor_match(self, img_pt, cls, appearance):
        best = (None, float('inf'), float('inf'))
        for cid, a in self.anchors.items():
            if not a.active or a.cls != cls:
                continue
            d = np.linalg.norm(a.img_pt - img_pt)
            if d > self.dist_thresh:
                continue
            if self.use_appearance and (appearance is not None) and (a.appearance is not None):
                app_d = 1.0 - float(np.dot(a.appearance, appearance))
            else:
                app_d = 0.0
            score = d + 100.0 * app_d
            if score < (best[1] + 100.0 * best[2]):
                best = (cid, d, app_d)
        return best

    def remap_and_update(self, tracks, frame, frame_idx):
        remapped = []
        claimed_canon = set()

        # Pass 1: honor existing local->canon mappings if still valid
        for tr in tracks:
            local_id = int(tr.id)
            xyxy = tr.xyxy
            cls = int(tr.cls)
            img_pt = _box_bottom_center(xyxy)
            appr = self._appearance(frame, xyxy) if self.use_appearance else None

            if local_id in self.local_to_canon:
                cid = self.local_to_canon[local_id]
                a = self.anchors.get(cid, None)
                if a is not None and a.active and a.cls == cls:
                    a.update_observation(img_pt, appr)
                    a.last_frame = frame_idx
                    claimed_canon.add(cid)
                    tr.id = cid
                    remapped.append(tr)
                else:
                    self.local_to_canon.pop(local_id, None)

        # Pass 2: match new locals to anchors or create new anchors
        for tr in tracks:
            if int(tr.id) in self.local_to_canon:
                continue
            local_id = int(tr.id)
            xyxy = tr.xyxy
            cls = int(tr.cls)
            img_pt = _box_bottom_center(xyxy)
            appr = self._appearance(frame, xyxy) if self.use_appearance else None

            cid, d, app_d = self._best_anchor_match(img_pt, cls, appr)
            if cid is not None and (cid not in claimed_canon):
                if (not self.use_appearance) or (app_d <= self.app_thresh):
                    self.local_to_canon[local_id] = cid
                    a = self.anchors[cid]
                    a.update_observation(img_pt, appr)
                    a.last_frame = frame_idx
                    claimed_canon.add(cid)
                    tr.id = cid
                    remapped.append(tr)
                    continue

            cid_new = self._new_canon_id()
            self.local_to_canon[local_id] = cid_new
            a_new = Anchor(cid_new, img_pt, cls, appearance=appr)
            a_new.last_frame = frame_idx
            self.anchors[cid_new] = a_new
            tr.id = cid_new
            remapped.append(tr)

        # Deactivate long-lost anchors
        for cid, a in self.anchors.items():
            if a.time_lost > self.max_time_lost:
                a.active = False

        return remapped

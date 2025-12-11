
"""
SuperPoint + SuperGlue based Camera Motion Compensation (CMC)

Drop-in class `SuperPointSuperGlueCMC`:
- Estimates homography H(prev → curr) using SuperPoint keypoints and SuperGlue matching.
- Falls back to ORB-based HomographyCMC if SuperGlue/SuperPoint are not available.

Usage:
    from motion.cmc_superglue import SuperPointSuperGlueCMC
    cmc = SuperPointSuperGlueCMC(weights_dir='models/weights', model='indoor', device='cuda')
    H = cmc.estimate(prev_frame, curr_frame)
    tracker.set_cmc(H)
"""

import numpy as np, cv2
try:
    from motion.cmc import HomographyCMC  # fallback
except Exception:
    HomographyCMC = None

class SuperPointSuperGlueCMC:
    def __init__(self, weights_dir='models/weights', model='indoor', device='cuda',
                 max_keypoints=-1, keypoint_threshold=0.005, match_threshold=0.2, force_cpu=False):
        self.device = 'cpu' if force_cpu else device
        self.weights_dir = weights_dir
        self.model = model
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.match_threshold = match_threshold
        self._sp = None; self._sg = None; self._ready = False
        try:
            import torch
            from pathlib import Path
            # Expect Magic Leap SuperGlue repo in PYTHONPATH:
            from models.superpoint import SuperPoint
            from models.superglue import SuperGlue

            sp_w = Path(weights_dir) / 'superpoint_v1.pth'
            sg_w = Path(weights_dir) / f'superglue_{model}.pth'
            if not sp_w.exists() or not sg_w.exists():
                raise FileNotFoundError(f"Missing weights in '{weights_dir}'.")

            # Build models
            self._sp = SuperPoint({'keypoint_threshold': self.keypoint_threshold,
                                   'max_keypoints': self.max_keypoints}).to(self.device)
            self._sg = SuperGlue({'weights': self.model,
                                  'sinkhorn_iterations': 20,
                                  'match_threshold': self.match_threshold}).to(self.device)
            # Load weights
            self._sp.load_state_dict(torch.load(str(sp_w), map_location=self.device))
            self._sg.load_state_dict(torch.load(str(sg_w), map_location=self.device))
            self._sp.eval(); self._sg.eval()
            self._ready = True
        except Exception as e:
            self._ready = False
            self._fallback = HomographyCMC() if HomographyCMC is not None else None
            self._error = str(e)

    def estimate(self, prev, curr):
        if prev is None or curr is None:
            return None
        if not self._ready:
            return None if self._fallback is None else self._fallback.estimate(prev, curr)

        import torch
        def to_gray(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
            return (g.astype(np.float32) / 255.0)

        img0, img1 = to_gray(prev), to_gray(curr)
        t0 = torch.from_numpy(img0)[None, None].to(self.device)
        t1 = torch.from_numpy(img1)[None, None].to(self.device)

        sp0, sp1 = self._sp({'image': t0}), self._sp({'image': t1})
        k0 = sp0['keypoints'][0].detach().cpu().numpy()
        k1 = sp1['keypoints'][0].detach().cpu().numpy()
        if k0.shape[0] < 8 or k1.shape[0] < 8:
            return None

        out = self._sg({'image0': t0, 'image1': t1,
                        'keypoints0': sp0['keypoints'], 'keypoints1': sp1['keypoints'],
                        'descriptors0': sp0['descriptors'], 'descriptors1': sp1['descriptors'],
                        'scores0': sp0['scores'], 'scores1': sp1['scores']})
        matches0 = out['matches0'][0].detach().cpu().numpy()
        valid = matches0 > -1
        if valid.sum() < 8:
            return None
        pts0 = k0[valid]; pts1 = k1[matches0[valid]]

        H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        return H

    def ready(self):  # True if SuperGlue/SuperPoint are loaded
        return self._ready


import cv2, numpy as np
class NoCMC:
    def estimate(self, prev, curr): return None
class HomographyCMC:
    def __init__(self, max_features=2000, match_thresh=0.75):
        self.max_features=max_features; self.match_thresh=match_thresh
    def estimate(self, prev, curr):
        if prev is None or curr is None: return None
        orb=cv2.ORB_create(nfeatures=self.max_features)
        kp1,des1=orb.detectAndCompute(prev,None); kp2,des2=orb.detectAndCompute(curr,None)
        if des1 is None or des2 is None: return None
        bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches=bf.knnMatch(des1,des2,k=2)
        good=[m for m,n in matches if m.distance < self.match_thresh*n.distance]
        if len(good)<8: return None
        src=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,_=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        return H

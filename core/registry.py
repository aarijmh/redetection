from models.detectors.base import DummyDetector
from models.detectors.yolov8 import YOLOv8Detector
from models.detectors.rtdetr import RTDETRDetector
from models.trackers.boxmot_wrappers import BoxMOTTracker
from models.trackers.simple_sort import SimpleSORTTracker
from models.reid.base import DummyReID
from models.reid.osnet_boxmot import OSNetBoxMOT
from motion.cmc import HomographyCMC, NoCMC
from motion.cmc_superglue import SuperPointSuperGlueCMC


def build_detector(name='yolov8', weights='yolov8x.pt', conf=0.25, iou=0.5, class_filter=None, device='cpu'):
    if name == 'yolov8':
        try:
            return YOLOv8Detector(weights, conf, iou, class_filter, device)
        except Exception as e:
            print('[WARN] YOLOv8 unavailable:', e); return DummyDetector()
    if name == 'rtdetr':
        try:
            return RTDETRDetector(weights, conf, iou, class_filter, device)
        except Exception as e:
            print('[WARN] RT-DETR unavailable:', e); return DummyDetector()
    return DummyDetector()

def build_reid(name='osnet'):
    if name == 'osnet':
        try:
            return OSNetBoxMOT()
        except Exception as e:
            print('[WARN] OSNet unavailable:', e); return DummyReID()
    return DummyReID()

def build_tracker(name='strongsort', reid=None, match_iou=0.5):
    if name in ['strongsort','botsort','deocsort']:
        try:
            return BoxMOTTracker(name, reid=reid, match_iou=match_iou)
        except Exception as e:
            print('[WARN] BoxMOT unavailable:', e); return SimpleSORTTracker(match_iou)
    return SimpleSORTTracker(match_iou)

def build_cmc(name='none', device='cpu'):
    if name == 'superglue':
        return SuperPointSuperGlueCMC(weights_dir='SuperGluePretrainedNetwork/models/weights',
                                      model='outdoor', device=device)
    elif name == 'homography':
        return HomographyCMC()
    else:
        return NoCMC()


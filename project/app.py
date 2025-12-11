
import argparse, time
from core.registry import build_detector, build_tracker, build_cmc, build_reid
from core.video_io import VideoReader, VideoWriter
from utils.metrics import TrackCSVWriter
from motion.anchor_manager import AnchorManager


import sys, os
sys.path.append(os.path.abspath('SuperGluePretrainedNetwork'))  # adjust path

from motion.cmc_superglue import SuperPointSuperGlueCMC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='output.mp4')
    p.add_argument('--save-csv', default='tracks.csv')
    p.add_argument('--detector', default='yolov8', choices=['yolov8','rtdetr','dummy'])
    p.add_argument('--detector-weights', default='yolov8x.pt')
    p.add_argument('--tracker', default='strongsort', choices=['strongsort','botsort','deocsort','simplesort'])
    p.add_argument('--reid', default='osnet', choices=['osnet','none'])
    p.add_argument('--use-cmc', action='store_true')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--class-filter', nargs='*', default=None)
    p.add_argument('--cmc', default='homography', choices=['none','homography','superglue'])
    return p.parse_args()


def main():
    args = parse_args()
    detector = build_detector(args.detector, args.detector_weights, args.conf, args.iou, args.class_filter)
    reid = build_reid(args.reid)
    tracker = build_tracker(args.tracker, reid=reid, match_iou=args.iou)
    cmc = build_cmc(name=args.cmc)

    #cmc = build_cmc(args.use_cmc)

    vr = VideoReader(args.video)
    vw = VideoWriter(args.output, vr.fps, vr.width, vr.height)
    csvw = TrackCSVWriter(args.save_csv)
    anchor_mgr = AnchorManager(max_time_lost=30, dist_thresh=60.0, app_thresh=0.5, use_appearance=True)

    prev = None
    t0 = time.time()
    for i, frame in enumerate(vr):
        H = cmc.estimate(prev, frame) if prev is not None else None
        anchor_mgr.predict(H)
        tracker.set_cmc(H)
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame)
        tracks = anchor_mgr.remap_and_update(tracks, frame, i)
        anno = tracker.draw(frame.copy(), tracks)
        vw.write(anno)
        csvw.write(i, tracks)
        prev = frame

    vw.close(); csvw.close(); vr.close()
    print(f'Done in {time.time()-t0:.2f}s -> {args.output} / {args.save_csv}')

if __name__ == '__main__':
    main()

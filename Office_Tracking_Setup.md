
# Accuracy‑First Office Object Tracking — Setup & Options (Chairs, Laptops, Printers, Monitors, Desks)

This guide explains how to configure your **modular detection + tracking** pipeline to focus on office objects: **chair, laptop, printer, computer monitor, desk** (and similar). It covers **COCO classes**, **open‑vocabulary** options for non‑COCO objects, **fine‑tuning YOLO**, tracker settings (BoT‑SORT/StrongSORT), **camera motion compensation** (CMC), and recommended CLI commands.

---

## 1) Which classes exist in COCO already?

**COCO (80 classes)** contains: `chair`, `laptop`, `tv`, `dining table`, `keyboard`, `mouse`, etc. It does **not** have `printer`. In office scenes, common mappings are:

- **chair** → `chair` ✔️
- **laptop** → `laptop` ✔️
- **computer monitor** → **use `tv`** (closest COCO category for screens/monitors) ✔️
- **desk** → **use `dining table`** (COCO has no desk; tables are the closest) ✔️
- **printer** → ❌ **not in COCO**; use **open‑vocabulary detectors** or **custom training**.

**References:** COCO class lists and docs citeturn11search171turn11search174; YOLOv8 class mapping examples citeturn11search153.

---

## 2) Run your current pipeline with COCO-only classes (class filtering)

Your scaffold supports a `--class-filter` flag. Recommended **accuracy-first** command (YOLOv8x + BoT‑SORT + CMC):

```bash
python app.py --video input.mp4 --output out.mp4 \
  --detector yolov8 --detector-weights yolov8x.pt \
  --tracker botsort --use-cmc --save-csv office.csv \
  --class-filter chair laptop tv "dining table"
```

- **YOLOv8x** is a strong COCO detector with robust AP. citeturn11search174  
- **BoT‑SORT** adds appearance + **camera motion compensation**; excellent when the **camera moves** or the view changes. citeturn10search145turn10search149

> If you prefer **StrongSORT**, replace `--tracker botsort` with `--tracker strongsort` (note StrongSORT in BoxMOT does **not** include offline AFLink/GSI modules). citeturn10search119

---

## 3) Alias mapping (optional) for friendly labels

To keep CLI arguments intuitive (`monitor`, `desk`), you can map to COCO labels inside `models/detectors/yolov8.py`:

```python
# Alias mapping (so your --class-filter can use monitor/desk)
alias = {
    'monitor': 'tv',
    'screen': 'tv',
    'desk': 'dining table'
}
# Normalize filter terms
norm_filter = set()
if self.class_filter:
    for cf in self.class_filter:
        norm_filter.add(alias.get(str(cf).lower(), cf))
# Use norm_filter when checking
if self.class_filter and (name not in norm_filter and cls not in norm_filter):
    continue
```

This lets you run:

```bash
python app.py ... --class-filter chair laptop monitor desk
```

**Why?** Because COCO exposes `tv` and `dining table`, not `monitor` or `desk`; aliasing preserves usability. citeturn11search171

---

## 4) Detect **printer** (and precise monitor/desk) without labeling → **Open‑vocabulary detectors**

**Open‑vocabulary** models detect arbitrary categories at inference time via text prompts — ideal for **printer**, **computer monitor**, **office desk**:

- **Grounding DINO** — detects arbitrary objects via language prompts; strong **zero‑shot** AP; widely adopted with HF demos. citeturn11search177turn11search180
- **OWLv2 / OWL‑ST** — scales open‑vocabulary detection with web‑scale self‑training; excellent for rare/unseen categories. citeturn11search162turn11search158
- **Detic (Meta)** — expands vocabulary to **tens of thousands** of concepts using image‑level supervision & CLIP; includes ready code and checkpoints. citeturn11search164turn11search165

> We can add `models/detectors/grounding_dino.py` to your project, so `--detector groundingdino` accepts prompts like: `"chair. laptop. printer. computer monitor. desk."` and returns detections in your pipeline.

Example (after integration):

```bash
python app.py --video input.mp4 --output out.mp4 \
  --detector groundingdino --detector-weights groundingdino-tiny.pt \
  --tracker botsort --use-cmc --save-csv office_ov.csv \
  --class-filter "chair" "laptop" "printer" "computer monitor" "desk"
```

---

## 5) Highest precision for your exact office classes → **Fine‑tune YOLOv8**

If you can label a modest dataset (e.g., 200–1000 images), **fine‑tuning YOLOv8** on your target classes gives the most **precise, stable** detector:

1. Prepare `office.yaml` pointing to your train/val images and class names (`chair`, `laptop`, `printer`, `computer monitor`, `desk`).  
2. Train with Ultralytics:

```bash
yolo train model=yolov8m.pt data=office.yaml epochs=50 imgsz=640
```

3. Use the trained weights in your pipeline:

```bash
python app.py --video input.mp4 --output out.mp4 \
  --detector yolov8 --detector-weights runs/detect/train/weights/best.pt \
  --tracker botsort --use-cmc --save-csv office_ft.csv \
  --class-filter chair laptop printer "computer monitor" desk
```

**References:** Ultralytics training & model docs. citeturn11search146turn11search151

---

## 6) Tracking configuration (ID stability & camera motion)

### A) BoT‑SORT (recommended for moving cameras)

In `models/trackers/boxmot_wrappers.py`, configure BoT‑SORT with appearance + CMC:

```python
from boxmot import BotSORT as Impl
self.impl = Impl(
    reid_weights=Path('weights/osnet_x0_25_msmt17.pt'),
    device=device,
    with_reid=True,
    proximity_thresh=0.5,       # IoU for first association
    appearance_thresh=0.25,     # ReID distance threshold
    cmc_method='sof',           # internal camera motion compensation
    match_thresh=0.85,
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.7,
    frame_rate=30               # set to your video FPS
)
```

BoT‑SORT integrates **camera motion compensation** (GMC) and refined association, improving identity stability in dynamic views. citeturn10search145

### B) Per‑class tracking (separate ID spaces)

If your BoxMOT version supports it, enable **per‑class tracking** for BotSORT/DeepOCSORT/ByteTrack/etc. (StrongSORT **not** supported):

```python
self.impl = Impl(..., per_class=True, nr_classes=80)
```

This keeps **IDs separate per class** (chair IDs independent from laptop IDs), simplifying analytics. citeturn11search183

### C) Stronger CMC via **SuperPoint + SuperGlue** (module included)

For advanced camera motion, use `motion/cmc_superglue.py` — estimates homography with **SuperPoint + SuperGlue** and falls back to ORB if weights/repo are missing. Clone Magic Leap’s repo for weights:

```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork
# expects: models/weights/superpoint_v1.pth, superglue_{indoor|outdoor}.pth
```

Then in `app.py`:

```python
import sys, os
sys.path.append(os.path.abspath('SuperGluePretrainedNetwork'))
from motion.cmc_superglue import SuperPointSuperGlueCMC
cmc = SuperPointSuperGlueCMC(weights_dir='SuperGluePretrainedNetwork/models/weights',
                             model='outdoor', device='cuda') if args.use_cmc else None
```

**References:** Magic Leap’s official SuperGlue/SuperPoint repo and weights. citeturn8search110

---

## 7) StrongSORT++ (AFLink + GSI) — offline post‑processing

If you require **global relinking** and **trajectory smoothing** beyond online tracking, run **AFLink** (appearance‑free global linking) and **GSI** (Gaussian‑Smoothed Interpolation) **after** tracking. BoxMOT implements **StrongSORT** (online) but explicitly **not** the offline AFLink/GSI modules. citeturn10search119

- StrongSORT++ code (AFLink & GSI modules): citeturn10search131  
- Paper & summaries: citeturn10search126turn10search133

Minimal post‑processing outline:

```python
# Convert your tracks.csv -> MOTChallenge-like result, then
# run AFLink and GSI from dyhBUPT/StrongSORT to relink and smooth.
```

---

## 8) Decision guide

- **Fast & simple, COCO‑like scenes** → **YOLOv8x + BoT‑SORT** with alias mapping (`monitor` → `tv`, `desk` → `dining table`). citeturn11search174  
- **Must detect printer without labels** → add **Grounding DINO** (or **Detic/OWLv2**). citeturn11search177turn11search164turn11search158  
- **You can label 200–1000 images** → **fine‑tune YOLOv8** on exact office classes for best precision. citeturn11search146

---

## 9) Ready-to-use CLI recipes

**COCO-only (alias mapping):**
```bash
python app.py --video input.mp4 --output out.mp4 \
  --detector yolov8 --detector-weights yolov8x.pt \
  --tracker botsort --use-cmc --save-csv office.csv \
  --class-filter chair laptop monitor desk
```

**Open‑vocabulary (include printer via prompts):**
```bash
python app.py --video input.mp4 --output out.mp4 \
  --detector groundingdino --detector-weights groundingdino-tiny.pt \
  --tracker botsort --use-cmc --save-csv office_ov.csv \
  --class-filter "chair" "laptop" "printer" "computer monitor" "desk"
```

**Fine‑tuned YOLOv8 (exact office classes):**
```bash
yolo train model=yolov8m.pt data=office.yaml epochs=50 imgsz=640
python app.py --video input.mp4 --output out.mp4 \
  --detector yolov8 --detector-weights runs/detect/train/weights/best.pt \
  --tracker botsort --use-cmc --save-csv office_ft.csv \
  --class-filter chair laptop printer "computer monitor" desk
```

---

## 10) Sources & further reading

- **COCO classes & dataset:** Roboflow class list; COCO official docs. citeturn11search171turn11search172  
- **YOLOv8 / Ultralytics training & class mapping:** Ultralytics docs; SO example of names mapping. citeturn11search146turn11search153  
- **Open‑vocabulary detection:** Grounding DINO GitHub & HF docs; OWLv2 (NeurIPS 2023); Detic (ECCV 2022). citeturn11search177turn11search180turn11search162turn11search158turn11search164turn11search165  
- **Trackers:** BoT‑SORT paper & BoxMOT docs; BoxMOT tracker pages (BotSORT/StrongSORT). citeturn10search145turn10search149  
- **StrongSORT++ offline modules:** BoxMOT issue note; StrongSORT repo & paper. citeturn10search119turn10search131turn10search126  
- **Advanced CMC:** Magic Leap SuperGlue repo & weights. citeturn8search110

---

### Need help wiring **Grounding DINO** as a detector or creating `office.yaml` with a labeling checklist? 
Tell me your **video FPS/resolution** and hardware (GPU/CPU), and I’ll tailor thresholds (conf/IoU and BoT‑SORT params) for **chairs, laptops, printers, monitors, desks** in your exact footage.

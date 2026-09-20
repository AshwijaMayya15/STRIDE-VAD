"""
X3D-L video-level feature extraction for XD-Violence (STRIDE pipeline).

Input layout (your dataset):
    XD VIOLENCE/
        1-1004/        <video>.mp4 ...   |
        1005-2004/     <video>.mp4 ...   |  numbered folders = download batches
        2005-2804/     ...               |  = the 3954 TRAIN videos
        2805-3319/     ...               |
        3320-3954/     ...               |
        test_videos/   <video>.mp4 ...   -> the 800 TEST videos

The numbered folders are NOT classes. The label lives in the filename
(_label_A = normal; _label_B1/B2/B4/B5/B6/G = anomaly), so we ignore the
folder name and only use it to decide train vs test.

Output:
    X3D_Videos/train/<video_name>.npy   (192, 16, 10, 10)  -- filename preserved
    X3D_Videos/test/<video_name>.npy

Geometry is identical to UCF: crop 320 -> 320/32 = 10x10. Flow not extracted.
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pytorchvideo.models.hub import x3d_l

# ------------------------------- CONFIG -------------------------------
XD_ROOT     = "XD VIOLENCE"
OUTPUT_ROOT = "X3D_Videos"
MODEL_PATH  = "models/x3d_l.pyth"          # same checkpoint used for UCF

TEST_FOLDER_NAMES = {"test_videos"}        # anything NOT in here -> train

NUM_FRAMES      = 16
SAMPLING_RATE   = 5
FRAMES_PER_CLIP = NUM_FRAMES * SAMPLING_RATE   # 80
SIDE_SIZE       = 320
CROP_SIZE       = 320              # 320 / 32 = 10 -> (192,16,10,10). DO NOT CHANGE.
POOLING         = "max"            # "max" or "avg" aggregation across clips
VIDEO_EXTS      = (".mp4", ".avi", ".mkv", ".mov")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

# ------------------------------- MODEL --------------------------------
model = x3d_l(pretrained=False)
ckpt  = torch.load(MODEL_PATH, map_location="cpu")
state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
state = {k.replace("module.", "").replace("network.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.blocks[-1] = torch.nn.Identity()   # drop head -> 192-channel backbone output
model = model.eval().to(DEVICE)
print(f"Model loaded on {DEVICE}. Backbone output expected: (192, 16, 10, 10).")


# --------------------------- PREPROCESS CLIP --------------------------
def preprocess(frames):
    """frames: list of RGB uint8 (H,W,3), length 1..80 -> (1,3,16,320,320)."""
    x = torch.from_numpy(np.stack(frames)).float()        # (T,H,W,3)
    T = x.shape[0]
    idx = torch.linspace(0, T - 1, NUM_FRAMES).long()     # 16 uniform indices (repeats if T<16)
    x = x[idx]                                            # (16,H,W,3)
    x = x / 255.0
    x = x.permute(3, 0, 1, 2)                             # (3,16,H,W)
    x = (x - MEAN) / STD

    _, _, H, W = x.shape                                  # short-side scale
    scale = SIDE_SIZE / min(H, W)
    x = torch.nn.functional.interpolate(
        x, size=(int(round(H * scale)), int(round(W * scale))),
        mode="bilinear", align_corners=False)

    _, _, H, W = x.shape                                  # center crop
    top  = (H - CROP_SIZE) // 2
    left = (W - CROP_SIZE) // 2
    x = x[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]
    return x.unsqueeze(0)                                 # (1,3,16,320,320)


@torch.no_grad()
def extract_video(path):
    """Stream a video in 80-frame clips, extract per-clip feats, pool to one tensor."""
    cap = cv2.VideoCapture(path)
    clip_feats, buf = [], []

    def flush(b):
        inp = preprocess(b).to(DEVICE)
        f = model(inp).squeeze(0).cpu().numpy().astype(np.float32)  # (192,16,10,10)
        clip_feats.append(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(buf) == FRAMES_PER_CLIP:
            flush(buf)
            buf = []
    cap.release()

    if len(buf) > 0:          # tail clip: keep it instead of discarding
        flush(buf)
    if len(clip_feats) == 0:
        return None

    clips = np.stack(clip_feats, axis=0)                  # (N,192,16,10,10)
    return clips.max(0) if POOLING == "max" else clips.mean(0)


# ------------------------- DISCOVER ALL VIDEOS ------------------------
def gather_jobs():
    """Walk XD_ROOT; return list of (video_path, split) where split is train/test."""
    jobs = []
    for root, _, files in os.walk(XD_ROOT):
        # split decided by whether any path component is a test folder
        parts = {p.lower() for p in root.replace("\\", "/").split("/")}
        split = "test" if (parts & {n.lower() for n in TEST_FOLDER_NAMES}) else "train"
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                jobs.append((os.path.join(root, f), split))
    return jobs


# -------------------------------- LOOP --------------------------------
if __name__ == "__main__":
    jobs = gather_jobs()
    n_train = sum(1 for _, s in jobs if s == "train")
    n_test  = len(jobs) - n_train
    print(f"Found {len(jobs)} videos  ->  train: {n_train}, test: {n_test}")

    for split in ("train", "test"):
        os.makedirs(os.path.join(OUTPUT_ROOT, split), exist_ok=True)

    skipped = 0
    for vpath, split in tqdm(jobs, desc="Extracting"):
        name = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(OUTPUT_ROOT, split, name + ".npy")
        if os.path.exists(out_path):
            continue
        feat = extract_video(vpath)
        if feat is None:
            print(f"  skipped (no frames): {name}")
            skipped += 1
            continue
        np.save(out_path, feat)   # (192,16,10,10)

    print(f"Done -> {OUTPUT_ROOT}/  (skipped {skipped})")
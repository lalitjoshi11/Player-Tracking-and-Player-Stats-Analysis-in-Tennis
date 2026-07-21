import json
import os
import cv2

# -----------------------------
# Path configuration
# -----------------------------
json_path = "E:/MP/programs/Keypoint/tennis_court_det_dataset/data/data_train.json"
image_dir = "E:/MP/programs/Keypoint/tennis_court_det_dataset/data/images"
label_dir = "E:/MP/programs/Keypoint/tennis_court_yoloformat"

os.makedirs(label_dir, exist_ok=True)

# -----------------------------
# Load annotation JSON
# -----------------------------
with open(json_path, "r") as f:
    data = json.load(f)

print(f"[INFO] Loaded {len(data)} annotation entries")

# -----------------------------
# Iterate through annotations
# -----------------------------
for item in data:

    img_id = item.get("id")
    keypoints = item.get("kps")

    # ---------- Sanity check 1 ----------
    if img_id is None or keypoints is None:
        print("[WARNING] Missing image ID or keypoints. Skipping.")
        continue

    # ---------- Sanity check 2 ----------
    # Tennis court dataset has exactly 14 keypoints
    if len(keypoints) != 14:
        print(f"[WARNING] {img_id}: {len(keypoints)} keypoints found (expected 14). Skipping.")
        continue

    # ---------- Load image (PNG) ----------
    img_path = os.path.join(image_dir, img_id + ".png")
    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARNING] Image not found: {img_path}. Skipping.")
        continue

    H, W, _ = img.shape

    # ---------- Extract raw keypoint coordinates ----------
    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]

    # ---------- Compute bounding box from keypoints ----------
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # ---------- CLAMP bounding box to image boundaries ----------
    # This prevents invalid boxes caused by out-of-frame keypoints
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W - 1, xmax)
    ymax = min(H - 1, ymax)

    # ---------- Sanity check 3 ----------
    # If box collapses after clamping, skip
    if xmax <= xmin or ymax <= ymin:
        print(f"[WARNING] Degenerate bounding box for {img_id}. Skipping.")
        continue

    # ---------- Convert bounding box to YOLO format ----------
    cx = ((xmin + xmax) / 2) / W
    cy = ((ymin + ymax) / 2) / H
    bw = (xmax - xmin) / W
    bh = (ymax - ymin) / H

    # Final safety clamp (numerical stability)
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    # ---------- Start YOLO-Pose label ----------
    # Class ID = 0 (tennis court)
    label_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    # ---------- Append keypoints ----------
    # Format per keypoint: x y v
    # v = 2 → visible and labeled
    for x, y in keypoints:
        x = min(max(x, 0), W - 1)
        y = min(max(y, 0), H - 1)

        x_n = x / W
        y_n = y / H
        v = 2

        label_line += f" {x_n:.6f} {y_n:.6f} {v}"

    # ---------- Save label file ----------
    label_path = os.path.join(label_dir, img_id + ".txt")
    with open(label_path, "w") as f:
        f.write(label_line + "\n")

print("Conversion completed successfully with clamping.")

import os
import cv2
import math
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
IMG_DIR = r"E:\MP\programs\Keypoint\tennis_court_yoloformat\val\images"
GT_LABEL_DIR = r"E:\MP\programs\Keypoint\tennis_court_yoloformat\val\labels"
PRED_LABEL_DIR = r"E:\MP\runs\pose\predict15\labels" ## we need to infer val images to the trained model and generate predicted labels, this is the path to those predicted levels

NUM_KEYPOINTS = 14

# Indices of court diagonal keypoints
TOP_LEFT_IDX = 0
BOTTOM_RIGHT_IDX = 3


# -----------------------------
# Utility functions
# -----------------------------
def load_yolo_pose_label(label_path):
    """
    Loads YOLO-Pose label file.
    Returns:
        bbox: (cx, cy, w, h)
        keypoints: list of (x, y, v)
    """
    with open(label_path, "r") as f:
        line = f.readline().strip().split()

    # Bounding box
    cx, cy, bw, bh = map(float, line[1:5])

    # Keypoints
    keypoints = []
    kp_data = line[5:]

    for i in range(0, len(kp_data), 3):
        x = float(kp_data[i])
        y = float(kp_data[i + 1])
        v = float(kp_data[i + 2])
        keypoints.append((x, y, v))

    return (cx, cy, bw, bh), keypoints


def normalized_to_pixel(kp, W, H):
    """Convert normalized keypoint to pixel coordinates"""
    return kp[0] * W, kp[1] * H


def euclidean_distance(p1, p2):
    """Pixel distance"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# -----------------------------
# Main evaluation loop
# -----------------------------
total_pixel_error = 0.0
total_keypoints = 0
nke_values = []

image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png") or f.endswith(".jpg")]

for img_name in image_files:
    img_path = os.path.join(IMG_DIR, img_name)
    gt_label_path = os.path.join(GT_LABEL_DIR, img_name.replace(".png", ".txt").replace(".jpg", ".txt"))
    pred_label_path = os.path.join(PRED_LABEL_DIR, img_name.replace(".png", ".txt").replace(".jpg", ".txt"))

    # Sanity checks
    if not os.path.exists(gt_label_path) or not os.path.exists(pred_label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W, _ = img.shape

    # Load labels
    _, gt_kps = load_yolo_pose_label(gt_label_path)
    _, pred_kps = load_yolo_pose_label(pred_label_path)

    if len(gt_kps) != NUM_KEYPOINTS or len(pred_kps) != NUM_KEYPOINTS:
        continue

    # -----------------------------
    # Compute court diagonal (GT)
    # -----------------------------
    tl = normalized_to_pixel(gt_kps[TOP_LEFT_IDX], W, H)
    br = normalized_to_pixel(gt_kps[BOTTOM_RIGHT_IDX], W, H)

    court_diagonal = euclidean_distance(tl, br)

    if court_diagonal == 0:
        continue

    # -----------------------------
    # Keypoint-wise error
    # -----------------------------
    image_error = 0.0
    visible_kps = 0

    for i in range(NUM_KEYPOINTS):
        if gt_kps[i][2] == 0:
            continue  # skip invisible points

        gt_pt = normalized_to_pixel(gt_kps[i], W, H)
        pred_pt = normalized_to_pixel(pred_kps[i], W, H)

        dist = euclidean_distance(gt_pt, pred_pt)

        image_error += dist
        total_pixel_error += dist
        visible_kps += 1
        total_keypoints += 1

    # -----------------------------
    # Image-level NKE
    # -----------------------------
    if visible_kps > 0:
        image_mpe = image_error / visible_kps
        image_nke = image_mpe / court_diagonal
        nke_values.append(image_nke)


# -----------------------------
# Final metrics
# -----------------------------
MPE = total_pixel_error / total_keypoints
NKE = np.mean(nke_values)

print("=================================")
print(f"Mean Pixel Error (MPE): {MPE:.4f} pixels")
print(f"Normalized Keypoint Error (NKE): {NKE:.6f}")
print("=================================")

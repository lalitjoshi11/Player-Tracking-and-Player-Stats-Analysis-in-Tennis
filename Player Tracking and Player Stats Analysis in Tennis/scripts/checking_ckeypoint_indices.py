import cv2
import json
import os

# -----------------------------
# Paths (EDIT THESE)
# -----------------------------
IMAGE_PATH = "E:/MP/programs/Keypoint/tennis_court_det_dataset/data/images/PuXlxKdUIes_2450.png"
JSON_PATH = "E:/MP/programs/Keypoint/tennis_court_det_dataset/data/data_train.json"
IMAGE_ID = "PuXlxKdUIes_2450"

OUTPUT_PATH = "E:/MP/programs/Keypoint/prediction_of_validation_labels/keypoint_index_visualization.jpg"

# -----------------------------
# Load JSON annotations
# -----------------------------
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Find the annotation for this image
entry = None
for item in data:
    if item["id"] == IMAGE_ID:
        entry = item
        break

if entry is None:
    raise ValueError("Image ID not found in JSON")

keypoints = entry["kps"]    ##storing kepoints in a list so i can enumerate the index,x,y coordinate values of the keypoints.

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found")

# -----------------------------
# Draw keypoints with index
# -----------------------------
for idx, (x, y) in enumerate(keypoints):    ## x and y are integers because the OpenCV drawing functions need integer pixel positions and which are available in json file.
    x, y = int(x), int(y)

    cv2.circle(img, (x, y), 5, (0, 0, 255), -1) ## draws a red circle (0,0,255) in the image (img) at (x,y) and circle is filled i.e (-1)
    cv2.putText(
        img,
        str(idx),       #- Writes the index number (str(idx)) next to the circle.
        (x + 6, y - 6), #Position  offsets the text slightly so it doesn’t overlap the circle
        cv2.FONT_HERSHEY_SIMPLEX, #font
        0.5, #font size
        (255, 255, 0), #font color
        1,  #thickness 1 pix.
        cv2.LINE_AA #makes the text anti‑aliased (smooth edges).
    )

# -----------------------------
# Save visualization
# -----------------------------
cv2.imwrite(OUTPUT_PATH, img)

print(f"✅ Keypoint index visualization saved to: {OUTPUT_PATH}")

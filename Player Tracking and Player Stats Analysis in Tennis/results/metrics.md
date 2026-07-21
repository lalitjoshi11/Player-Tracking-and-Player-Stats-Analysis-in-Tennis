# Model Performance Metrics

Results reported below are from the minor project report *"Player Tracking and Player Stats Analysis in Tennis"* (Kantipur Engineering College, March 2026), evaluated on the held-out validation/test splits.

## 1\. Player Detection Model (YOLOv8)

**Dataset:** 4,166 images (Roboflow, racket-sports domain) — 72% train (3,000), 17% val (691), 11% test (475)

**Confusion Matrix (image-level, validation set, n = 691)**

||Predicted: Player|Predicted: Background|
|-|:-:|:-:|
|**Actual: Player**|474 (TP)|10 (FN)|
|**Actual: Background**|15 (FP)|192 (TN)|

**Derived Metrics**

|Metric|Value|
|-|:-:|
|Precision|96.9%|
|Recall|97.9%|
|F1 Score|97.4%|
|Accuracy|96.4%|

## 2\. Court Keypoint Detection Model (YOLOv8-Pose)

**Dataset:** 8,841 images (court keypoint dataset, Sergey Kosolapov) — 75% train, 25% val. 14 court keypoints per frame.

|Metric|Value|
|-|:-:|
|Pose mAP|97.32%|
|mAP50|97.48%|
|mAP75|97.34%|
|Mean Pixel Error (MPE)|3 px|
|Normalized Keypoint Error (NKE)|0.00375|

*The small gap (0.14%) between mAP50 and mAP75 indicates precise keypoint localization rather than approximate placement.*

## 3\. Training Configuration Summary

|Parameter|Player Detection|Court Keypoint|
|-|:-:|:-:|
|Base model|YOLOv8 (COCO pretrained)|yolov8s-pose.pt|
|Input size|640 × 640|640 × 640|
|Epochs|100|100|
|Batch size|—|8|
|Optimizer|SGD (momentum 0.937, weight decay 0.0005)|—|
|Initial LR|0.01 (cosine decay to 0.01×)|—|
|Warmup|3 epochs|—|

## Notes

* Loss curves (classification, box, DFL) all show convergence with a small, stable train/val gap.
* Precision and recall for the player detector both stabilize near 0.97–0.98 by mid-training.


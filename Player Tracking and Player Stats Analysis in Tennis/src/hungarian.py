import numpy as np
from scipy.optimize import linear_sum_assignment


# -------------------------------------------------------
# FIX 1: Rewrote both IoU and GIoU together in one clean
# function so GIoU always applies the enclosing-box penalty
# even when the boxes DO overlap (original code skipped it).
# -------------------------------------------------------
def get_iou_and_giou(boxA, boxB):
    """
    Computes both IoU and GIoU for two boxes [x1, y1, x2, y2].

    GIoU = IoU - (area_enclosing - area_union) / area_enclosing
    Range: [-1, 1]  (IoU range: [0, 1])

    Returns
    -------
    iou  : float
    giou : float
    """
    # Validate boxes — must have positive area
    if (boxA[2] <= boxA[0] or boxA[3] <= boxA[1] or
            boxB[2] <= boxB[0] or boxB[3] <= boxB[1]):
        return 0.0, -1.0

    # Intersection
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)

    # Individual areas
    aA    = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB    = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = aA + aB - inter
    iou   = inter / union if union > 0 else 0.0

    # Enclosing (smallest bounding) box
    x1_enc = min(boxA[0], boxB[0]);  y1_enc = min(boxA[1], boxB[1])
    x2_enc = max(boxA[2], boxB[2]);  y2_enc = max(boxA[3], boxB[3])
    a_enc  = (x2_enc - x1_enc) * (y2_enc - y1_enc)

    # GIoU — always subtract the enclosing-box penalty
    giou = iou - (a_enc - union) / a_enc if a_enc > 0 else -1.0

    return iou, giou


def solve_hungarian_association(trackers, detections, frame_idx=0):
    """
    Hungarian matching with multi-metric cost matrix.

    Cost branches
    -------------
    HIGH overlap  (iou > 0.1) : driven by IoU, distance, shape similarity
    NEAR miss     (norm_dist < 3.0) : driven by distance + GIoU penalty
    FAR           (otherwise) : high base cost, capped to avoid explosion

    2-PLAYER FIX
    ------------
    Confirmed tracks that have been lost for several frames get a larger
    acceptance threshold so a returning player re-attaches to the original
    track ID instead of spawning a new one.
    """
    num_tracks = len(trackers)
    num_dets   = len(detections)

    if num_tracks == 0 or num_dets == 0:
        return [], list(range(num_tracks)), list(range(num_dets))

    cost_matrix = np.full((num_tracks, num_dets), 1000.0)

    for i, track in enumerate(trackers):

        # FIX 4: Guard against bad / missing predicted state
        pred_state = track.get_predicted_state()
        if pred_state is None or len(pred_state) < 4:
            continue
        u, v, a, h = pred_state
        if h <= 0 or a <= 0:
            continue

        w        = max(a * h, 1.0)
        pred_box = [u - w / 2, v - h / 2, u + w / 2, v + h / 2]

        for j, det in enumerate(detections):
            det_bbox = det['bbox']
            det_conf = det.get('conf', 1.0)

            # FIX 1: use unified iou+giou function
            iou, giou = get_iou_and_giou(pred_box, det_bbox)

            det_u = (det_bbox[0] + det_bbox[2]) / 2
            det_v = (det_bbox[1] + det_bbox[3]) / 2
            det_h = max(det_bbox[3] - det_bbox[1], 1.0)
            det_w = det_bbox[2] - det_bbox[0]

            center_dist = np.sqrt((u - det_u) ** 2 + (v - det_v) ** 2)
            norm_dist   = center_dist / det_h

            size_ratio   = min(h, det_h) / max(h, det_h)
            det_a        = det_w / det_h
            aspect_ratio = min(a, det_a) / max(a, det_a) if max(a, det_a) > 0 else 1.0

            # FIX 2: cap missed_factor so long-lost tracks don't get
            # astronomically high costs (was unbounded: lost_frames/10)
            missed_factor = min(track.lost_frames / 10.0, 0.5)   # max 1.5× penalty

            # FIX 3: reduce track_age reward — old tracks shouldn't
            # automatically get halved costs (was up to 1.0 → cost/2)
            track_age_factor = min(track.age / 30.0, 1.0) * 0.2  # max 0.2 reduction

            # FIX 5: consistent cost scales across all three branches
            if iou > 0.1:
                # Good overlap — IoU-driven cost, range ≈ [0, 1]
                cost = (
                    (1.0 - iou)          * 0.6 +
                    norm_dist            * 0.2 +
                    (1.0 - size_ratio)   * 0.1 +
                    (1.0 - aspect_ratio) * 0.1
                )

            elif norm_dist < 3.0:
                # Near miss — distance + GIoU penalty, range ≈ [0.7, 1.3]
                giou_penalty = max(0.0, -giou)   # only penalise negative GIoU
                cost = (
                    0.7 +
                    norm_dist            * 0.1 +
                    (1.0 - size_ratio)   * 0.1 +
                    (1.0 - aspect_ratio) * 0.1 +
                    giou_penalty         * 0.2
                )

            else:
                # Far away — high base cost, capped to avoid explosion
                cost = min(1.0 + norm_dist * 0.2, 2.0)

            # Apply track-age reward and missed-frames penalty
            cost = cost * (1.0 + missed_factor) / (1.0 + track_age_factor)

            # Low-confidence detections are less trusted
            cost = cost / max(det_conf, 0.3)

            cost_matrix[i, j] = cost

    # Hungarian assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        return [], list(range(num_tracks)), list(range(num_dets))

    matches       = []
    assigned_rows = set()
    assigned_cols = set()

    for r, c in zip(row_ind, col_ind):
        cost  = cost_matrix[r, c]
        track = trackers[r]

        # FIX 6: raised base threshold to 1.0 to match the new cost scale
        base_threshold = 1.0

        if track.is_confirmed and track.total_hits > 10:
            # 2-PLAYER FIX: confirmed + strong tracks get a generous threshold
            # that grows with how long they've been lost so a returning player
            # re-attaches to the same track ID instead of spawning a new one.
            #   lost  5 frames → ~1.5 × base
            #   lost 20 frames → ~3.0 × base  (capped)
            lost_bonus = min(track.lost_frames / 8.0, 1.5)
            threshold  = base_threshold * (1.5 + lost_bonus)

        elif track.is_confirmed:
            threshold = base_threshold * 1.2

        else:
            # Unconfirmed tracks — tighter threshold to avoid false matches
            threshold = base_threshold * 0.8

        if cost < threshold:
            matches.append((r, c))
            assigned_rows.add(r)
            assigned_cols.add(c)

    unmatched_tracks     = [i for i in range(num_tracks) if i not in assigned_rows]
    unmatched_detections = [i for i in range(num_dets)   if i not in assigned_cols]

    return matches, unmatched_tracks, unmatched_detections
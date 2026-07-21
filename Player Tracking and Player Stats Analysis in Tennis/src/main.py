import cv2
import numpy as np
from ultralytics import YOLO


from homography    import ransac_homography, dst_pts, MINI_W, MINI_H, DOUBLES_W, COURT_H, ALLEY, SINGLES_W, padding, build_mini_court
from hungarian     import solve_hungarian_association
from kalmanfilter  import PlayerKalmanFilter
from player_stats  import PlayerStatsTracker, draw_speed_on_frame, draw_stats_panel, draw_fatigue_bar
from heatmap       import PlayerHeatmap

# ===============================
# CONFIG
# ===============================
PLAYER_MODEL_PATH  = r"E:\MP\main\lalit\best_of_player\best_of_player.pt"
COURT_MODEL_PATH   = r"E:\MP\runs\pose\100epoches\weights\best.pt"
VIDEO_PATH         = r"C:\Users\lalit\Downloads\mvs.mp4"
OUTPUT_PATH        = r"E:\MP\testing_output.mp4"

CONF_THRES  = 0.5
MAX_LOST    = 60    # remove a track after being lost this many frames, keep track alive for 2 seconds at 30fps
MIN_HITS    = 3     # confirm a track after this many detections
PADDING     = padding

# ── Player colours (P0 = red dot, P1 = green dot) ──────────
PLAYER_COLORS = [(0, 0, 255), (0, 255, 0)]

# ===============================
# TRACK CLASS
# Wraps PlayerKalmanFilter and
# holds the attributes that
# hungarian.py expects:
#   .age, .lost_frames,
#   .is_confirmed, .total_hits,
#   .get_predicted_state()
# ===============================
class Track:
    _id_counter = 0

    def __init__(self, bbox, conf=1.0):
        Track._id_counter += 1
        self.id = Track._id_counter

        u, v, a, h       = self._to_state(bbox)
        self.kf          = PlayerKalmanFilter(initial_state=[u, v, a, h])

        self.age         = 0       # frames since creation
        self.lost_frames = 0       # consecutive frames without a detection
        self.total_hits  = 0       # total times matched to a detection
        self.is_confirmed = False  # True after MIN_HITS matches
        self.last_bbox   = bbox

    # ── helpers ──────────────────────────────────────────
    @staticmethod
    def _to_state(bbox):
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        h = max(y2 - y1, 1.0)
        a = (x2 - x1) / h
        return u, v, a, h

    # called every frame BEFORE association
    def predict(self):
        self.age += 1
        return self.kf.predict()

    # called when this track is matched to a detection
    def update(self, bbox, conf=1.0):
        u, v, a, h = self._to_state(bbox)
        self.kf.update([u, v, a, h], confidence=conf)
        self.last_bbox    = bbox
        self.lost_frames  = 0
        self.total_hits  += 1
        if self.total_hits >= MIN_HITS:
            self.is_confirmed = True

    # called when no detection matched this track
    def mark_lost(self):
        self.lost_frames += 1
        self.kf.mark_lost()

    # required by hungarian.py
    def get_predicted_state(self):
        return self.kf.get_state()   # [u, v, a, h]

    # returns [x1, y1, x2, y2] for drawing
    def get_bbox(self):
        return self.kf.get_bbox()


def nms_detections(detections, iou_threshold=0.3):
    """
    Remove overlapping detections using Non-Maximum Suppression.
    Keeps the highest confidence box when two boxes overlap too much.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d['conf'], reverse=True)

    kept = []
    for det in detections:
        b1 = det['bbox']
        keep = True
        for kept_det in kept:
            b2 = kept_det['bbox']

            # Compute IoU
            xA = max(b1[0], b2[0]);  yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2]);  yB = min(b1[3], b2[3])
            inter = max(0, xB-xA) * max(0, yB-yA)
            aA = (b1[2]-b1[0]) * (b1[3]-b1[1])
            aB = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = aA + aB - inter
            iou = inter / union if union > 0 else 0.0

            if iou > iou_threshold:
                keep = False  # too much overlap — discard this detection
                break

        if keep:
            kept.append(det)

    return kept


# ===============================
# MAIN
# ===============================
def main():
    player_model = YOLO(PLAYER_MODEL_PATH)
    court_model  = YOLO(COURT_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video FPS: {fps}, Resolution: {W}x{H}")

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    mini_base = build_mini_court()
    tracks    = []

    stats_tracker = PlayerStatsTracker(
        fps                      = fps,
        mini_court_width_pixels  = MINI_W - 2 * PADDING,
        mini_court_height_pixels = MINI_H - 2 * PADDING,
    )
    print("[INFO] Processing...")

    prev_Hmat             = None
    player_mini_positions = {}
    player_margin_top     = 60
    player_margin_bottom  = 100
    player_margin_sides   = 30
    court_draw_w  = MINI_W - 2 * PADDING
    court_draw_h  = MINI_H - 2 * PADDING
    sx = court_draw_w / DOUBLES_W
    sy = court_draw_h / COURT_H

    x_court_left  = PADDING - player_margin_sides
    x_court_right = MINI_W - PADDING + player_margin_sides
    y_court_top   = PADDING - player_margin_top
    y_court_bot   = MINI_H - PADDING + player_margin_bottom

    # initializing heatmap
    heatmaps = {0: PlayerHeatmap(), 1: PlayerHeatmap()}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        court_results  = court_model.predict(frame, conf=CONF_THRES, verbose=False)
        player_results = player_model.predict(frame, conf=CONF_THRES, iou=0.3, verbose=False)

        # ── STEP 1: Court homography (homography.py) ─────────
        Hmat = None
        if court_results[0].keypoints is not None:
            kpts_all = court_results[0].keypoints.xy.cpu().numpy()
            if kpts_all.shape[0] > 0:
                court_kpts = kpts_all[0][:14]
                raw_Hmat = ransac_homography(court_kpts, dst_pts)
                if raw_Hmat is not None:                            ##removing jitters in minimap
                    if prev_Hmat is not None:
                        # Blend 70% previous + 30% new — smooths out frame-to-frame jumps
                        Hmat = 0.85 * prev_Hmat + 0.15 * raw_Hmat
                    else:
                        Hmat = raw_Hmat  # reuse last good homography if current frame fails
                    prev_Hmat = Hmat
                else:
                    Hmat = prev_Hmat  # RANSAC failed — reuse last good one
        else:   # court not visible
            prev_Hmat = None

        # ── STEP 2: Collect player detections ────────────────
        detections = []
        if Hmat is not None and player_results[0].boxes is not None:        #First condition Only collect detections when court is visible
            for box in player_results[0].boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        # call nms
        detections = nms_detections(detections, iou_threshold=0.3)

        # ── STEP 3: Kalman predict (kalmanfilter.py via Track) 
        for t in tracks:
            t.predict()

        # ── STEP 4: Hungarian association (hungarian.py) ─────
        matches, unmatched_trks, unmatched_dets = \
            solve_hungarian_association(tracks, detections)

        # Update matched
        for trk_idx, det_idx in matches:
            tracks[trk_idx].update(
                detections[det_idx]['bbox'],
                detections[det_idx]['conf']
            )

        # Mark unmatched tracks as lost
        for trk_idx in unmatched_trks:
            tracks[trk_idx].mark_lost()

        # Spawn new tracks for unmatched detections — only if fewer than 2 total tracks exist
        total_tracks = len(tracks)
        for det_idx in unmatched_dets:
            if total_tracks < 2:    # spawn guard — never more than 2 tracks
                tracks.append(Track(
                    detections[det_idx]['bbox'],
                    detections[det_idx]['conf']
                ))
                total_tracks += 1

        # Remove stale tracks
        for t in tracks:
            if t.lost_frames > MAX_LOST:
                stats_tracker.reset(t.id)
        tracks = [t for t in tracks if t.lost_frames <= MAX_LOST]

        # Kill unconfirmed tracks that are lost quickly
        tracks = [t for t in tracks if t.is_confirmed or t.lost_frames <= 5]

        # ── STEP 5: Draw on frame + mini court ───────────────
        mini = mini_base.copy()

        # Build confirmed list — hard cap at 2
        confirmed = [t for t in tracks if t.is_confirmed]
        if len(confirmed) > 2:
            confirmed = sorted(confirmed, key=lambda t: t.total_hits, reverse=True)[:2]

        # Reset heatmap for any confirmed track that just died
        for idx, ct in enumerate(confirmed):
            if ct.lost_frames > MAX_LOST:
                heatmaps[idx].reset()

        for i, track in enumerate(confirmed):
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            bx1, by1, bx2, by2 = [int(v) for v in track.get_bbox()]

            # bounding box + label on main frame
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(frame, f"P{i}",
                        (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # project feet onto mini court via Hmat
            if Hmat is not None:
                feet    = np.array([[[(bx1+bx2)/2, float(by2)]]], dtype=np.float32)
                mini_pt = cv2.perspectiveTransform(feet, Hmat)
                mx, my  = int(mini_pt[0][0][0]), int(mini_pt[0][0][1])

                # Smooth dot position with previous position
                if track.id in player_mini_positions:
                    prev_mx, prev_my = player_mini_positions[track.id]
                    mx = int(0.7 * prev_mx + 0.3 * mx)
                    my = int(0.7 * prev_my + 0.3 * my)
                player_mini_positions[track.id] = (mx, my)

                # Record position + compute realtime stats
                stats_tracker.record_position(
                    track.id, mx, my,
                    detected=(track.lost_frames == 0)
                )
                stats_tracker.compute_stats_realtime(track.id)

                # Draw on frame
                draw_speed_on_frame(frame, bx1, by2,
                                    stats_tracker.get(track.id), color)

                # Update heatmap — real detections only
                if track.lost_frames == 0:
                    heatmaps[i].update(mx, my)

                if x_court_left <= mx <= x_court_right and y_court_top <= my <= y_court_bot:
                    # Clamp to canvas before drawing
                    draw_mx = max(0, min(MINI_W - 1, mx))
                    draw_my = max(0, min(MINI_H - 1, my))

                    cv2.circle(mini, (draw_mx, draw_my), 6, color, -1)
                    cv2.putText(mini, f"P{i}",
                                (draw_mx + 6, draw_my - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw heatmap onto mini court
                mini = heatmaps[i].draw(mini, player_id=i)

        # ── STEP 6: Overlay mini court ────────────────────────
        if Hmat is not None:
            ys, ye = 10, 10 + MINI_H
            xs, xe = W - MINI_W - 10, W - 10
            roi    = frame[ys:ye, xs:xe]
            frame[ys:ye, xs:xe] = cv2.addWeighted(mini, 0.85, roi, 0.15, 0)

        draw_stats_panel(frame, confirmed, stats_tracker,
                         PLAYER_COLORS,
                         panel_x=W - MINI_W - 10,
                         panel_y=MINI_H + 20)   # confirmed instead of tracks
        
        for i, track in enumerate(confirmed):
            draw_fatigue_bar(frame, i,
                             stats_tracker.get(track.id),
                             PLAYER_COLORS[i],
                             panel_x = W - MINI_W - 10,
                             panel_y = MINI_H + 145 + i * 35)

        out.write(frame)

    cap.release()
    out.release()
    print("[INFO] Done:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
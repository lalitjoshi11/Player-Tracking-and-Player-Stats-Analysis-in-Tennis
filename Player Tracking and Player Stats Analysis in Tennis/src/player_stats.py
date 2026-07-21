# player_stats.py
import numpy as np
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
REAL_COURT_WIDTH  = 10.97   # DOUBLES_W in metres
SPEED_WINDOW      = 5       # frames between measurements (same as his approach)
FATIGUE_WINDOW_SEC = 5   # measure fatigue every 5 seconds


# ── Utilities ─────────────────────────────────────────────────────────────────

def measure_distance(p1, p2):
    """Euclidean distance between two (x,y) points."""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def convert_pixel_distance_to_meters(pixel_distance, mini_court_width_pixels,
                                      real_court_width_meters=REAL_COURT_WIDTH):
    """
    Same formula as abdullahtarek:
        metres = pixels * (real_width / pixel_width)
    """
    return pixel_distance * (real_court_width_meters / mini_court_width_pixels)


def interpolate_positions(positions):
    """
    Fill None gaps using linear interpolation — his key technique.
    positions: list of (mx, my) or None, one entry per frame.
    Returns: list of (mx, my) with no None values.
    """
    # Find first and last valid position
    valid_indices = [i for i, p in enumerate(positions) if p is not None]
    if not valid_indices:
        return positions

    # Forward fill before first detection
    for i in range(valid_indices[0]):
        positions[i] = positions[valid_indices[0]]

    # Backward fill after last detection
    for i in range(valid_indices[-1]+1, len(positions)):
        positions[i] = positions[valid_indices[-1]]

    # Linear interpolation between valid positions
    for i in range(len(valid_indices)-1):
        start_idx = valid_indices[i]
        end_idx   = valid_indices[i+1]
        gap       = end_idx - start_idx

        if gap <= 1:
            continue

        start_pos = np.array(positions[start_idx], dtype=float)
        end_pos   = np.array(positions[end_idx],   dtype=float)

        for j in range(1, gap):
            t = j / gap
            positions[start_idx + j] = (
                int(start_pos[0] + t * (end_pos[0] - start_pos[0])),
                int(start_pos[1] + t * (end_pos[1] - start_pos[1]))
            )

    return positions


# ── Main Class ────────────────────────────────────────────────────────────────

class PlayerStatsTracker:
    """
    1. Collect minimap positions every frame (None if not detected)
    2. Interpolate missing positions
    3. Every SPEED_WINDOW frames, measure distance between
       position[i - SPEED_WINDOW] and position[i]
    4. Convert pixel distance → metres using court width
    5. Speed = distance / (SPEED_WINDOW / fps)
    """
    
    def __init__(self, fps, mini_court_width_pixels, mini_court_height_pixels,
                 real_court_width=REAL_COURT_WIDTH):
        self.fps                      = fps
        self.mini_court_width_pixels  = mini_court_width_pixels
        self.mini_court_height_pixels = mini_court_height_pixels
        self.real_court_width         = real_court_width

        self._speed_windows  = {}   # track_id → list of (window_num, avg_speed)
        self._window_speeds  = {}   # track_id → speeds in current window
        self._frame_count    = {}   # track_id → frame counter

        # Per-player position history — list of (mx,my) or None per frame
        self._positions  = {}   # track_id → [pos_frame0, pos_frame1, ...]

        # Per-player computed stats
        self._stats      = {}   # track_id → {distance, current_speed, max_speed}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_player(self, track_id):
        self._positions[track_id] = []
        self._stats[track_id]     = {
            'distance'     : 0.0,
            'current_speed': 0.0,
            'max_speed'    : 0.0,
        }

        self._speed_windows[track_id]  = []
        self._window_speeds[track_id]  = []
        self._frame_count[track_id]    = 0
        self._stats[track_id]['fatigue_index'] = 0.0   # 0=fresh, 1=fatigued
        self._stats[track_id]['avg_speed_now'] = 0.0
        self._stats[track_id]['avg_speed_peak']= 0.0
    # ── Public API ────────────────────────────────────────────────────────────

    def record_position(self, track_id, mx, my, detected):
        """
        Call every frame for every confirmed track.

        Parameters
        ----------
        track_id : track.id
        mx, my   : minimap pixel position (smoothed)
        detected : True if real YOLO detection, False if KF-only prediction
        """
        if track_id not in self._positions:
            self._init_player(track_id)

        # Record position if detected, else None (will be interpolated)
        if detected:
            self._positions[track_id].append((mx, my))
        else:
            self._positions[track_id].append(None)

    def compute_stats(self):
        """
        Call once after the video loop (or periodically).
        Interpolates positions then calculates distance + speed.
        His approach — processes all frames at once.
        """
        for track_id, positions in self._positions.items():
            # Step 1 — interpolate missing frames
            positions = interpolate_positions(positions[:])   # work on copy

            stats = self._stats[track_id]
            total_distance = 0.0
            speeds         = []

            # Step 2 — measure every SPEED_WINDOW frames
            for i in range(SPEED_WINDOW, len(positions)):
                p1 = positions[i - SPEED_WINDOW]
                p2 = positions[i]

                if p1 is None or p2 is None:
                    continue

                pixel_dist = measure_distance(p1, p2)

                # Convert to metres using court width scale
                real_dist = convert_pixel_distance_to_meters(
                    pixel_dist,
                    self.mini_court_width_pixels,
                    self.real_court_width
                )

                total_distance += real_dist

                # Speed over this window
                time_seconds = SPEED_WINDOW / self.fps
                speed_ms     = real_dist / time_seconds
                speed_kmh    = speed_ms * 3.6
                speeds.append(speed_kmh)

            stats['distance']      = total_distance
            stats['max_speed']     = max(speeds) if speeds else 0.0
            stats['current_speed'] = float(np.mean(speeds)) if speeds else 0.0

    def compute_stats_realtime(self, track_id):
        """
        Call every frame for real-time speed display.
        Only uses last SPEED_WINDOW frames — no full recompute.
        """
        if track_id not in self._positions:
            return

        positions = self._positions[track_id]

        if len(positions) < SPEED_WINDOW + 1:
            return

        # Get last window — interpolate just this slice
        window_slice = interpolate_positions(
            positions[-(SPEED_WINDOW+1):]
        )

        p1 = window_slice[0]
        p2 = window_slice[-1]

        if p1 is None or p2 is None:
            return

        pixel_dist = measure_distance(p1, p2)
        real_dist  = convert_pixel_distance_to_meters(
            pixel_dist,
            self.mini_court_width_pixels,
            self.real_court_width
        )

        time_seconds = SPEED_WINDOW / self.fps
        speed_kmh    = (real_dist / time_seconds) * 3.6

        stats = self._stats[track_id]
        stats['distance']      += real_dist / SPEED_WINDOW   # incremental
        stats['current_speed']  = speed_kmh
        stats['max_speed']      = max(stats['max_speed'], speed_kmh)

        # ── Fatigue tracking ──────────────────────────────────
        if track_id not in self._frame_count:
            self._frame_count[track_id]   = 0
            self._window_speeds[track_id] = []
            self._speed_windows[track_id] = []

        self._window_speeds[track_id].append(speed_kmh)
        self._frame_count[track_id] += 1

        window_frames = int(FATIGUE_WINDOW_SEC * self.fps)

        # Every 5 seconds, close the current window and start a new one
        if self._frame_count[track_id] % window_frames == 0:
            if self._window_speeds[track_id]:
                window_avg = float(np.mean(self._window_speeds[track_id]))
                window_num = self._frame_count[track_id] // window_frames
                self._speed_windows[track_id].append((window_num, window_avg))
                self._window_speeds[track_id] = []   # reset for next window
        
        # Fatigue index — compare current window avg to peak window avg
        if len(self._speed_windows[track_id]) >= 2:
            all_avgs  = [w[1] for w in self._speed_windows[track_id]]
            peak_avg  = max(all_avgs)
            now_avg = float(np.mean(all_avgs[-3:]))     # rolling average of last 3 windows (15 seconds)

            stats['avg_speed_now']  = now_avg
            stats['avg_speed_peak'] = peak_avg

            # Fatigue index: 0.0 = same as peak, 1.0 = completely stopped
            if peak_avg > 0:
                stats['fatigue_index'] = max(0.0, min(1.0, 1.0 - (now_avg / peak_avg)))

    def get(self, track_id):
        """Returns stats dict or None."""
        return self._stats.get(track_id, None)

    def reset(self, track_id):
        """Call when track is deleted."""
        self._positions.pop(track_id, None)
        self._stats.pop(track_id, None)


# ── Drawing Helpers ───────────────────────────────────────────────────────────

def draw_speed_on_frame(frame, bx1, by2, stats, color):
    """Draw current speed + total distance below bounding box."""
    if stats is None:
        return
    cv2.putText(frame,
                f"{stats['current_speed']:.1f} km/h",
                (bx1, by2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame,
                f"{stats['distance']:.1f} m",
                (bx1, by2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)


def draw_stats_panel(frame, tracks, stats_tracker, player_colors,
                     panel_x, panel_y):
    """Summary panel showing speed + distance for each player."""
    confirmed = [t for t in tracks if t.is_confirmed]
    for i, track in enumerate(confirmed):
        stats = stats_tracker.get(track.id)
        if stats is None:
            continue
        color = player_colors[i % len(player_colors)]
        lines = [
            f"P{i}  Spd:  {stats['current_speed']:.1f} km/h",
            f"P{i}  Max:  {stats['max_speed']:.1f} km/h",
            f"P{i}  Dist: {stats['distance']:.1f} m",
        ]
        for j, line in enumerate(lines):
            cv2.putText(frame, line,
                        (panel_x, panel_y + i * 70 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_fatigue_bar(frame, track_idx, stats, color, panel_x, panel_y):
    """
    Draws a fatigue bar below the stats panel.
    Green = fresh, Yellow = moderate, Red = fatigued
    """
    if stats is None:
        return

    fatigue = stats.get('fatigue_index', 0.0)
    label   = f"P{track_idx} Fatigue:"

    bar_w   = 100
    bar_h   = 10
    filled  = int(bar_w * fatigue)

    # Bar background
    cv2.rectangle(frame,
                  (panel_x, panel_y),
                  (panel_x + bar_w, panel_y + bar_h),
                  (80, 80, 80), -1)

    # Bar fill — green → yellow → red based on fatigue
    if fatigue < 0.3:
        bar_color = (0, 255, 0)      # green — fresh
    elif fatigue < 0.6:
        bar_color = (0, 200, 255)    # yellow — moderate
    else:
        bar_color = (0, 0, 255)      # red — fatigued

    if filled > 0:
        cv2.rectangle(frame,
                      (panel_x, panel_y),
                      (panel_x + filled, panel_y + bar_h),
                      bar_color, -1)

    # Label
    cv2.putText(frame, label,
                (panel_x, panel_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Percentage
    cv2.putText(frame, f"{int(fatigue*100)}%",
                (panel_x + bar_w + 4, panel_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
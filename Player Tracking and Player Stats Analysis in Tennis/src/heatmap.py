# heatmap.py
# ─────────────────────────────────────────────────────────────────────────────
# Court position heatmap for CourtVision.
# Renders across the FULL minimap canvas (including outside court lines)
# so player movement behind baselines and near sidelines is visible.
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
from collections import defaultdict
from homography import MINI_W, MINI_H, padding

# ── Constants ─────────────────────────────────────────────────────────────────
GRID_SIZE   = 10     # each grid cell = 10×10 pixels on minimap
MAX_HISTORY = 2000   # max positions stored per player before oldest is dropped

# Full minimap canvas size — heatmap covers entire minimap, not just court area
CANVAS_W = MINI_W
CANVAS_H = MINI_H

# Player colormaps
PLAYER_CMAPS = {
    0: cv2.COLORMAP_JET,     # P0 — warm red/yellow
    1: cv2.COLORMAP_WINTER,  # P1 — cool blue/green
}

# ── Spread kernel — manual 3×3 Gaussian ───────────────────────────────────────
# Center=0.36, direct neighbours=0.12, diagonals=0.05, total=1.0
_SPREAD = [
    ( 0,  0, 0.36),
    ( 1,  0, 0.12), (-1,  0, 0.12),
    ( 0,  1, 0.12), ( 0, -1, 0.12),
    ( 1,  1, 0.05), (-1,  1, 0.05),
    ( 1, -1, 0.05), (-1, -1, 0.05),
]


# ── PlayerHeatmap class ────────────────────────────────────────────────────────

class PlayerHeatmap:
    """
    Tracks one player's position history and renders a heatmap
    overlay onto the full minimap canvas (including outside court lines).

    Usage in main.py:
        # Init (once, outside loop)
        heatmaps = {0: PlayerHeatmap(), 1: PlayerHeatmap()}

        # Update (inside loop, in Step 5)
        if track.lost_frames == 0:   # real detection only
            heatmaps[i].update(mx, my)

        # Draw (inside loop, after drawing dot)
        mini = heatmaps[i].draw(mini, player_id=i)
    """

    def __init__(self):
        # Sparse grid: {(gx, gy): accumulated_weight}
        # Uses full canvas coordinates — no clamping to court area
        self.position_frequencies = defaultdict(float)

        # Raw position history for interpolation
        self.heatmap_positions = []

        # Cache to avoid rebuilding every frame
        self._heatmap_cache = None
        self._heatmap_dirty = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_point(self, rx, ry, weight=1.0):
        """Add a weighted point to the sparse grid with spread kernel."""
        gx = int(rx / GRID_SIZE)
        gy = int(ry / GRID_SIZE)
        for dx, dy, wt in _SPREAD:
            self.position_frequencies[(gx + dx, gy + dy)] += wt * weight

    def _remove_point(self, rx, ry):
        """Subtract oldest point contribution (sliding window decay)."""
        gx = int(rx / GRID_SIZE)
        gy = int(ry / GRID_SIZE)
        for dx, dy, wt in _SPREAD:
            k = (gx + dx, gy + dy)
            self.position_frequencies[k] = max(0.0, self.position_frequencies[k] - wt)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, mx, my):
        """
        Call every frame when player is detected (lost_frames == 0).

        mx, my: smoothed minimap pixel position (absolute canvas coordinates)
                No clamping — full canvas including outside court lines.
        """
        # Use absolute canvas coordinates directly (no offset subtraction)
        rx = float(mx)
        ry = float(my)

        # Only clamp to full canvas bounds — allows outside-court positions
        rx = max(0, min(CANVAS_W - 1, rx))
        ry = max(0, min(CANVAS_H - 1, ry))

        # Interpolate between last and current position for smooth trails
        if self.heatmap_positions:
            px, py = self.heatmap_positions[-1]
            dist    = np.sqrt((rx - px)**2 + (ry - py)**2)
            n_steps = max(1, int(dist / (GRID_SIZE * 0.8)))
            if n_steps > 1:
                for step in range(1, n_steps):
                    t  = step / n_steps
                    ix = px + t * (rx - px)
                    iy = py + t * (ry - py)
                    self._add_point(ix, iy, weight=0.6)  # interpolated = less weight

        # Add current position at full weight
        self._add_point(rx, ry, weight=1.0)
        self.heatmap_positions.append((rx, ry))

        # Sliding window — drop oldest point when history is full
        if len(self.heatmap_positions) > MAX_HISTORY:
            ox, oy = self.heatmap_positions.pop(0)
            self._remove_point(ox, oy)

        self._heatmap_dirty = True

    def reset(self):
        """Call when track is deleted to clear history."""
        self.position_frequencies.clear()
        self.heatmap_positions.clear()
        self._heatmap_cache = None
        self._heatmap_dirty = True

    def _build_overlay(self):
        """
        Converts sparse dict → dense numpy array → blurred heatmap image.
        Covers full canvas (CANVAS_H x CANVAS_W) so outside-court areas show.
        Returns None if no data or data too sparse.
        Caches result until next update() call.
        """
        if not self.position_frequencies:
            self._heatmap_cache = None
            return None

        # Cache hit — no need to rebuild
        if not self._heatmap_dirty and self._heatmap_cache is not None:
            return self._heatmap_cache

        max_freq = max(self.position_frequencies.values())
        if max_freq <= 0:
            self._heatmap_cache = None
            return None

        # Build dense float array over FULL canvas
        heatmap = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

        for (gx, gy), freq in self.position_frequencies.items():
            xs = max(0, gx * GRID_SIZE)
            xe = min(CANVAS_W, xs + GRID_SIZE)
            ys = max(0, gy * GRID_SIZE)
            ye = min(CANVAS_H, ys + GRID_SIZE)
            if xe > xs and ye > ys:
                # sqrt scaling — makes low-frequency areas more visible
                heatmap[ys:ye, xs:xe] = np.maximum(
                    heatmap[ys:ye, xs:xe],
                    np.sqrt(freq / max_freq) * 200.0
                )

        # Gaussian blur for smooth appearance
        blurred = cv2.GaussianBlur(heatmap, (21, 21), 0)

        self._heatmap_cache = blurred
        self._heatmap_dirty = False
        return blurred

    def draw(self, mini, player_id=0, alpha=0.28):
        """
        Draws heatmap overlay onto the full mini court canvas.
        Covers entire minimap including outside-court areas.

        Parameters
        ----------
        mini      : copy of mini court canvas (MINI_H × MINI_W × 3)
        player_id : 0 or 1 — selects colormap
        alpha     : blend strength (0=invisible, 1=fully opaque)

        Returns
        -------
        mini with heatmap blended across full canvas
        """
        heatmap = self._build_overlay()

        # Skip if no data or too faint
        if heatmap is None or heatmap.max() < 8:
            return mini

        # Convert to uint8 and apply colormap
        hm_u8  = np.uint8(np.clip(heatmap, 0, 255))
        cmap   = PLAYER_CMAPS.get(player_id, cv2.COLORMAP_JET)
        hm_col = cv2.applyColorMap(hm_u8, cmap)

        # Mask — only blend pixels above threshold
        mask = (hm_u8 > 25).astype(np.uint8) * 255

        # Blend over the FULL minimap canvas (no ROI restriction)
        mb      = mask > 0
        blended = mini.copy()
        blended[mb] = cv2.addWeighted(mini, 1 - alpha, hm_col, alpha, 0)[mb]

        return blended
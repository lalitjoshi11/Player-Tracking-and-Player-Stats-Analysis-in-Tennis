import numpy as np


class PlayerKalmanFilter:
    def __init__(self, initial_state=None):
        """
        8D State vector: [u, v, a, h, du, dv, da, dh]
        u, v  : bounding box center (x, y)
        a     : aspect ratio (width / height)
        h     : height
        du,dv : velocities for u, v
        da,dh : velocities for a, h
        """

        self.x = np.zeros((8, 1))

        # -------------------------------------------------------
        # State Transition Matrix (constant velocity model)
        # pos_new = pos_old + vel * dt
        # -------------------------------------------------------
        self.F = np.eye(8)
        dt = 1.0
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = dt

        # -------------------------------------------------------
        # Measurement Matrix
        # Maps 8D state → 4D measurement [u, v, a, h]
        # -------------------------------------------------------
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # -------------------------------------------------------
        # Initial Covariance (high uncertainty at start)
        # -------------------------------------------------------
        self.P = np.eye(8) * 100.0
        self.P[4:, 4:] *= 50.0     # even higher uncertainty for initial velocities

        # -------------------------------------------------------
        # Process Noise (Q): how much can the player deviate from physics?
        # High for position (u,v) to allow sudden direction changes.
        # Low for aspect ratio — it shouldn't change wildly frame-to-frame.
        # -------------------------------------------------------
        self.Q_base = np.diag([
            20.0,   # u
            20.0,   # v
            0.01,   # a (aspect ratio — very stable)
            3.0,    # h
            8.0,    # du
            8.0,    # dv
            0.001,  # da
            1.5,    # dh
        ])
        self.Q = self.Q_base.copy()

        # -------------------------------------------------------
        # FIX 5: Measurement Noise (R) — realistic detector noise
        # Increased from 1.5 → 4.0 for position since YOLO can
        # easily be off by 5-10 pixels on fast-moving players.
        # -------------------------------------------------------
        self.R = np.diag([4.0, 4.0, 6.0, 4.0])

        # -------------------------------------------------------
        # Velocity history for motion change detection
        # -------------------------------------------------------
        self.velocity_history = []
        self.sudden_change_detected = False

        # Counts how many frames the player has been lost
        self.frames_lost = 0

        if initial_state is not None:
            self.x[:4] = np.array(initial_state, dtype=float).reshape(4, 1)

    # -----------------------------------------------------------
    # FIX 1: mark_lost now actually does something useful
    # -----------------------------------------------------------
    def mark_lost(self):
        """
        Called when no detection is found for this player.
        Increases uncertainty and decays velocity so the predicted
        box doesn't drift far off-screen during occlusion.
        """
        self.frames_lost += 1

        # Grow uncertainty the longer the player is missing
        self.P *= (1.0 + 0.3 * self.frames_lost)

        # Decay velocity — box should slow down, not fly off-screen
        self.x[4:6] *= max(0.5, 0.8 ** self.frames_lost)
        self.x[6:8] *= 0.85

    def reset_lost(self):
        """Call this when the player is found again after being lost."""
        self.frames_lost = 0

    # -----------------------------------------------------------
    # FIX 3: detect_motion_change requires at least 3 entries
    # and is safe when called with a stale/empty history
    # -----------------------------------------------------------
    def detect_motion_change(self):
        """
        Analyzes recent velocity history to detect if the player
        is sprinting or lunging (sudden direction/speed change).
        Returns True if a sudden change is detected.
        """
        if len(self.velocity_history) < 3:
            return False

        recent_vel = np.array(self.velocity_history[-2:])
        older_vel  = (np.array(self.velocity_history[-4:-2])
                      if len(self.velocity_history) >= 4
                      else recent_vel)

        vel_change    = np.linalg.norm(
            np.mean(recent_vel, axis=0) - np.mean(older_vel, axis=0)
        )
        current_speed = np.linalg.norm(recent_vel[-1])
        prev_speed    = np.linalg.norm(recent_vel[-2]) if len(recent_vel) > 1 else current_speed
        speed_change_ratio = abs(current_speed - prev_speed) / max(prev_speed, 1.0)

        return vel_change > 15.0 or speed_change_ratio > 0.7

    # -----------------------------------------------------------
    # PREDICT
    # -----------------------------------------------------------
    def predict(self):
        """
        Advances the state by one time step using the physics model.
        Adaptive Q increases uncertainty during sudden movements so
        the filter trusts YOLO more than its own prediction.
        """
        self.sudden_change_detected = self.detect_motion_change()

        # Propagate state
        self.x = np.dot(self.F, self.x)

        # Adaptive process noise
        if self.sudden_change_detected:
            self.Q = self.Q_base * 3.0   # relax physics rules — follow YOLO
        else:
            self.Q = self.Q_base * 1.2

        # Propagate covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Friction / damping — prevents runaway predictions when player is lost
        if self.sudden_change_detected:
            self.x[4:6] *= 0.85   # slow down faster during erratic moves
        else:
            self.x[4:6] *= 0.97   # slight drag for stable predictions
        self.x[6:8] *= 0.90

        return self.x[:4].flatten()

    # -----------------------------------------------------------
    # UPDATE
    # -----------------------------------------------------------
    def update(self, measurement, confidence=1.0):
        """
        Corrects the prediction using a new YOLO detection.

        Parameters
        ----------
        measurement : array-like, shape (4,)
            [u, v, a, h] from the detector.
        confidence  : float in (0, 1]
            YOLO detection confidence score.

        Returns
        -------
        mahalanobis : float
            Statistical distance of the detection from the prediction.
            High value → detection is an outlier.
        """

        z = np.array(measurement, dtype=float).reshape(4, 1)

        # FIX 6: Guard against NaN / zero / negative measurements
        if np.any(np.isnan(z)) or np.any(np.isinf(z)) or z[3, 0] <= 0:
            return 0.0

        # Innovation (error between measurement and prediction)
        y = z - np.dot(self.H, self.x)

        # Dynamic R: low confidence → trust detector less
        R_weighted = self.R / max(confidence, 0.2)

        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + R_weighted

        # Mahalanobis distance — how statistically surprising is this detection?
        try:
            inv_S = np.linalg.inv(S)
            mahalanobis = float((y.T @ inv_S @ y)[0, 0])
        except np.linalg.LinAlgError:
            mahalanobis = 0.0

        # FIX 4: Softer alpha — was too aggressive (up to 8x), now max 3x
        # Gradually boosts the filter's responsiveness for large outliers
        # instead of a hard jump that could cause overcorrection.
        alpha = 1.0
        if mahalanobis > 5.0:
            alpha = min(1.0 + (mahalanobis - 5.0) * 0.2, 3.0)

        temp_P = self.P + alpha * self.Q

        # Kalman Gain
        S_boosted = np.dot(self.H, np.dot(temp_P, self.H.T)) + R_weighted
        try:
            K = np.dot(np.dot(temp_P, self.H.T), np.linalg.inv(S_boosted))
        except np.linalg.LinAlgError:
            K = np.dot(temp_P, self.H.T) * 0.5

        # Update state
        self.x = self.x + np.dot(K, y)

        # Joseph form covariance update (numerically stable)
        I_KH = np.eye(8) - np.dot(K, self.H)
        self.P = (np.dot(np.dot(I_KH, temp_P), I_KH.T)
                  + np.dot(np.dot(K, R_weighted), K.T))

        # FIX 2: Save velocity AFTER the state has been updated,
        # not before (old code saved the pre-update/predicted velocity)
        updated_vel = self.x[4:6].flatten()
        self.velocity_history.append(updated_vel.copy())
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)

        # Player is visible again — reset lost counter
        self.reset_lost()

        return mahalanobis

    def get_state(self):
        """Returns current [u, v, a, h] estimate."""
        return self.x[:4].flatten()

    def get_bbox(self):
        """
        Converts internal [u, v, a, h] state to
        bounding box [x1, y1, x2, y2] format.
        """
        u, v, a, h = self.x[:4].flatten()
        w  = a * h
        x1 = u - w / 2
        y1 = v - h / 2
        x2 = u + w / 2
        y2 = v + h / 2
        return np.array([x1, y1, x2, y2])
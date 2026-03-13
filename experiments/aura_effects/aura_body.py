"""
Finger Aura Effect — Marvel-style energy effects around fingertips.

Hand tracking only (no body segmentation). Fingertips emit glowing particle
trails.  Rotating the hand clockwise (right) triggers an energy spiral;
rotating counter-clockwise (left) triggers a shockwave burst.

Usage:
    python aura_body.py            # webcam 0
    python aura_body.py --camera 1
"""

import argparse
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import numpy as np

from config import (
    BURST_COLOR,
    BURST_DURATION,
    BURST_MAX_RADIUS,
    BURST_PARTICLE_COUNT,
    BURST_RING_COUNT,
    CHAR_FONT_SCALE,
    CHAR_FONT_THICKNESS,
    CHAR_SET_DEFAULT,
    CHAR_SETS,
    DARKEN_FACTOR,
    GLOW_BLEND_ALPHA,
    GLOW_KERNEL_SIZE,
    MAX_PARTICLES,
    ROTATION_COOLDOWN,
    ROTATION_THRESHOLD,
    ROTATION_WINDOW,
    SPIRAL_COLOR,
    SPIRAL_DURATION,
    SPIRAL_INNER_R,
    SPIRAL_OUTER_R,
    SPIRAL_RING_COLORS,
    SPIRAL_SPOKES,
    TRAIL_CHAR_DENSITY,
    TRAIL_COLOR,
    TRAIL_DOT_SIZE,
    TRAIL_DRIFT_SPEED,
    TRAIL_LIFETIME,
    TRAIL_NOISE,
    TRAIL_SPAWN_PER_TIP,
)

# ─── Model path ──────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(_DIR, "models", "hand_landmarker.task")

# MediaPipe fingertip landmark indices
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
WRIST_ID = 0
MID_BASE_ID = 9  # middle-finger MCP — used with wrist to compute hand angle


# ═══════════════════════════════════════════════════════════════════════════════
# Particle pool
# ═══════════════════════════════════════════════════════════════════════════════

class ParticlePool:
    def __init__(self, capacity: int = MAX_PARTICLES):
        self.cap = capacity
        self.x     = np.zeros(capacity, np.float32)
        self.y     = np.zeros(capacity, np.float32)
        self.vx    = np.zeros(capacity, np.float32)
        self.vy    = np.zeros(capacity, np.float32)
        self.r     = np.zeros(capacity, np.uint8)
        self.g     = np.zeros(capacity, np.uint8)
        self.b     = np.zeros(capacity, np.uint8)
        self.age   = np.zeros(capacity, np.float32)
        self.life  = np.ones(capacity,  np.float32)
        self.size  = np.ones(capacity,  np.uint8)
        self.char_idx = np.full(capacity, -1, np.int8)
        self.alive = np.zeros(capacity, bool)

    def _alloc(self, n: int) -> np.ndarray:
        dead = np.where(~self.alive)[0]
        n = min(n, len(dead))
        return dead[:n] if n > 0 else np.array([], dtype=int)

    def spawn_at(self, cx: float, cy: float, n: int, color: tuple,
                 speed: float, lifetime: tuple, size_range: tuple,
                 char_density: float, direction: np.ndarray | None = None):
        """Spawn *n* particles at (cx, cy) with random outward velocity."""
        idx = self._alloc(n)
        if len(idx) == 0:
            return
        k = len(idx)
        self.alive[idx] = True
        self.age[idx] = 0.0
        self.life[idx] = np.random.uniform(*lifetime, size=k)
        self.size[idx] = np.random.randint(size_range[0], size_range[1] + 1, size=k)

        self.x[idx] = cx + np.random.uniform(-2, 2, k)
        self.y[idx] = cy + np.random.uniform(-2, 2, k)

        if direction is not None:
            # biased direction
            angles = np.arctan2(direction[1], direction[0]) + np.random.uniform(-0.8, 0.8, k)
        else:
            angles = np.random.uniform(0, 2 * math.pi, k)
        spd = np.random.uniform(speed * 0.4, speed * 1.3, k)
        self.vx[idx] = np.cos(angles) * spd
        self.vy[idx] = np.sin(angles) * spd

        self.b[idx] = color[0]
        self.g[idx] = color[1]
        self.r[idx] = color[2]

        is_char = np.random.random(k) < char_density
        charset = CHAR_SETS.get(CHAR_SET_DEFAULT, CHAR_SETS["mixed"])
        self.char_idx[idx] = np.where(is_char, np.random.randint(0, len(charset), k), -1)

    def spawn_ring(self, cx: float, cy: float, n: int, radius: float,
                   color: tuple, speed: float, lifetime: tuple):
        """Spawn particles in a ring expanding outward."""
        idx = self._alloc(n)
        if len(idx) == 0:
            return
        k = len(idx)
        self.alive[idx] = True
        self.age[idx] = 0.0
        self.life[idx] = np.random.uniform(*lifetime, size=k)
        self.size[idx] = 1

        angles = np.linspace(0, 2 * math.pi, k, endpoint=False)
        self.x[idx] = cx + np.cos(angles) * radius
        self.y[idx] = cy + np.sin(angles) * radius
        spd = np.random.uniform(speed * 0.8, speed * 1.2, k)
        self.vx[idx] = np.cos(angles) * spd
        self.vy[idx] = np.sin(angles) * spd

        self.b[idx] = color[0]
        self.g[idx] = color[1]
        self.r[idx] = color[2]
        self.char_idx[idx] = -1

    def update(self, dt: float):
        alive = self.alive
        self.x[alive] += self.vx[alive] * dt
        self.y[alive] += self.vy[alive] * dt
        self.age[alive] += dt
        self.alive[alive & (self.age >= self.life)] = False

    def draw(self, overlay: np.ndarray, h: int, w: int):
        idx = np.where(self.alive)[0]
        if len(idx) == 0:
            return
        progress = self.age[idx] / self.life[idx]
        alpha = (1.0 - progress) ** 1.5

        xs = self.x[idx].astype(int)
        ys = self.y[idx].astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        idx, xs, ys, alpha = idx[valid], xs[valid], ys[valid], alpha[valid]

        charset = CHAR_SETS.get(CHAR_SET_DEFAULT, CHAR_SETS["mixed"])
        for k in range(len(idx)):
            i = idx[k]
            a = float(alpha[k])
            col = (int(self.b[i] * a), int(self.g[i] * a), int(self.r[i] * a))
            if self.char_idx[i] < 0:
                cv2.circle(overlay, (xs[k], ys[k]), int(self.size[i]), col, -1, cv2.LINE_AA)
            else:
                ch = charset[self.char_idx[i] % len(charset)]
                cv2.putText(overlay, ch, (xs[k], ys[k]),
                            cv2.FONT_HERSHEY_SIMPLEX, CHAR_FONT_SCALE,
                            col, CHAR_FONT_THICKNESS, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Gesture effects
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveEffect:
    """One active gesture effect (spiral or burst) with a position and timer."""
    def __init__(self, kind: str, cx: float, cy: float, t_start: float):
        self.kind = kind        # "spiral" or "burst"
        self.cx = cx
        self.cy = cy
        self.t_start = t_start
        self.duration = SPIRAL_DURATION if kind == "spiral" else BURST_DURATION
        self.burst_spawned = False

    def progress(self, t: float) -> float:
        return (t - self.t_start) / self.duration

    def alive(self, t: float) -> bool:
        return (t - self.t_start) < self.duration


def draw_spiral(overlay: np.ndarray, cx: int, cy: int, t: float, progress: float):
    """Draw an animated spinning spiral / portal ring."""
    fade = max(0.0, 1.0 - progress)
    angle_offset = t * 3.0  # fast spin
    n_spokes = SPIRAL_SPOKES
    # rings expand as the effect progresses
    inner = int(SPIRAL_INNER_R * (0.5 + progress * 0.5))
    outer = int(SPIRAL_OUTER_R * (0.3 + progress * 0.7))
    for i in range(n_spokes):
        angle = angle_offset + (2 * math.pi * i / n_spokes)
        color = SPIRAL_RING_COLORS[i % len(SPIRAL_RING_COLORS)]
        col = tuple(int(c * fade) for c in color)
        x1 = int(cx + inner * math.cos(angle))
        y1 = int(cy + inner * math.sin(angle))
        x2 = int(cx + outer * math.cos(angle))
        y2 = int(cy + outer * math.sin(angle))
        cv2.line(overlay, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
    # concentric rings
    for r_frac in [0.4, 0.7, 1.0]:
        r = int(outer * r_frac)
        col = tuple(int(c * fade * 0.7) for c in SPIRAL_RING_COLORS[0])
        cv2.circle(overlay, (cx, cy), r, col, 1, cv2.LINE_AA)


def draw_burst_rings(overlay: np.ndarray, cx: int, cy: int, progress: float):
    """Draw expanding shockwave rings."""
    fade = max(0.0, 1.0 - progress)
    for i in range(BURST_RING_COUNT):
        p = max(0.0, progress - i * 0.15)
        r = int(BURST_MAX_RADIUS * p)
        if r < 2:
            continue
        thickness = max(1, int(3 * (1.0 - p)))
        col = tuple(int(c * fade) for c in BURST_COLOR)
        cv2.circle(overlay, (cx, cy), r, col, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Hand rotation detector
# ═══════════════════════════════════════════════════════════════════════════════

class RotationDetector:
    """Track wrist→middle-finger angle over time to detect CW / CCW turns."""

    def __init__(self):
        self.angles: deque[float] = deque(maxlen=ROTATION_WINDOW)
        self.last_trigger_time = 0.0

    def push(self, wrist, mid_base, t: float):
        angle = math.atan2(mid_base[1] - wrist[1], mid_base[0] - wrist[0])
        self.angles.append(angle)

    def detect(self, t: float) -> str | None:
        """Return 'cw', 'ccw', or None."""
        if len(self.angles) < ROTATION_WINDOW:
            return None
        if t - self.last_trigger_time < ROTATION_COOLDOWN:
            return None

        # cumulative angular change (handling wrap-around)
        total = 0.0
        angles = list(self.angles)
        for i in range(1, len(angles)):
            d = angles[i] - angles[i - 1]
            # normalise to [-pi, pi]
            d = (d + math.pi) % (2 * math.pi) - math.pi
            total += d

        if total > ROTATION_THRESHOLD:
            self.last_trigger_time = t
            self.angles.clear()
            return "ccw"  # math convention: positive = counter-clockwise
        if total < -ROTATION_THRESHOLD:
            self.last_trigger_time = t
            self.angles.clear()
            return "cw"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Glow helper
# ═══════════════════════════════════════════════════════════════════════════════

def apply_glow(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    glow = cv2.GaussianBlur(overlay, GLOW_KERNEL_SIZE, 0)
    blended = cv2.addWeighted(frame, 1.0, glow, GLOW_BLEND_ALPHA, 0)
    blended = cv2.add(blended, overlay // 2)
    return blended


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Finger Aura Effect")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    hands = HandLandmarker.create_from_options(hand_options)

    pool = ParticlePool(MAX_PARTICLES)
    # one rotation detector per hand slot
    rot_detectors = [RotationDetector(), RotationDetector()]
    active_effects: list[ActiveEffect] = []

    t0 = time.time()
    prev_t = t0

    print("Press 'q' to quit.  Rotate hand RIGHT → spiral.  Rotate LEFT → burst.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        t = now - t0
        dt = min(now - prev_t, 0.05)
        prev_t = now
        ts_ms = int(t * 1000)

        overlay = np.zeros_like(frame)

        # ── Hand detection ────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hands.detect_for_video(mp_image, ts_ms)

        if result.hand_landmarks:
            for hi, hand_lm in enumerate(result.hand_landmarks):
                det = rot_detectors[min(hi, 1)]

                wrist = (hand_lm[WRIST_ID].x * w, hand_lm[WRIST_ID].y * h)
                mid_base = (hand_lm[MID_BASE_ID].x * w, hand_lm[MID_BASE_ID].y * h)

                # ── Fingertip particle trails ─────────────────────────────
                for tip_id in FINGERTIP_IDS:
                    tx = hand_lm[tip_id].x * w
                    ty = hand_lm[tip_id].y * h
                    # direction = tip away from wrist
                    dx = tx - wrist[0]
                    dy = ty - wrist[1]
                    ln = math.hypot(dx, dy) + 1e-6
                    direction = np.array([dx / ln, dy / ln])
                    pool.spawn_at(
                        tx, ty, TRAIL_SPAWN_PER_TIP, TRAIL_COLOR,
                        TRAIL_DRIFT_SPEED, TRAIL_LIFETIME, TRAIL_DOT_SIZE,
                        TRAIL_CHAR_DENSITY, direction,
                    )

                # ── Rotation detection ────────────────────────────────────
                det.push(wrist, mid_base, t)
                gesture = det.detect(t)

                palm_cx = (wrist[0] + mid_base[0]) / 2
                palm_cy = (wrist[1] + mid_base[1]) / 2

                if gesture == "cw":
                    active_effects.append(ActiveEffect("spiral", palm_cx, palm_cy, t))
                elif gesture == "ccw":
                    active_effects.append(ActiveEffect("burst", palm_cx, palm_cy, t))

        # ── Update particles ──────────────────────────────────────────────
        pool.update(dt)

        # ── Draw particles ────────────────────────────────────────────────
        pool.draw(overlay, h, w)

        # ── Draw & update active effects ──────────────────────────────────
        alive_effects = []
        for fx in active_effects:
            if not fx.alive(t):
                continue
            alive_effects.append(fx)
            p = fx.progress(t)
            cx_i, cy_i = int(fx.cx), int(fx.cy)

            if fx.kind == "spiral":
                draw_spiral(overlay, cx_i, cy_i, t, p)
                # spawn ring particles as spiral grows
                if int(p * 10) % 2 == 0:
                    pool.spawn_ring(fx.cx, fx.cy, 30,
                                    SPIRAL_OUTER_R * p,
                                    SPIRAL_RING_COLORS[0], 40, (0.3, 0.8))

            elif fx.kind == "burst":
                draw_burst_rings(overlay, cx_i, cy_i, p)
                if not fx.burst_spawned:
                    # one-time outward particle explosion
                    pool.spawn_at(fx.cx, fx.cy, BURST_PARTICLE_COUNT,
                                  BURST_COLOR, 120, (0.4, 1.2), (1, 2), 0.05)
                    fx.burst_spawned = True

        active_effects = alive_effects

        # ── Composite ─────────────────────────────────────────────────────
        darkened = (frame.astype(np.float32) * DARKEN_FACTOR).astype(np.uint8)
        output = apply_glow(darkened, overlay)

        cv2.imshow("Aura Effect", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

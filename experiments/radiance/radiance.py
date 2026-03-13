"""
Radiance — gesture-driven radial mandala / color wheel.

A glowing circular structure tracks your palm and reacts to finger geometry:
  - Fingers spread → large expanded color wheel with wide wedges
  - Fingers pinched → tight spiral with curved arms
  - Hand rotation → the whole mandala rotates
  - Concentric dotted rings + floating symbol markers orbit the structure
  - Scattered particles drift around the edges

Usage:
    python radiance.py              # webcam 0
    python radiance.py --camera 1
"""

import argparse
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from config import (
    ARCS,
    BAR_COLORS,
    BAR_INNER_RATIO,
    BAR_OUTER_RATIO,
    BAR_WIDTH,
    BRACKET_CLUSTERS,
    CLOCK_COLOR,
    CLOCK_FONT_SCALE,
    CLOCK_NUMERALS,
    CLOCK_RATIO,
    CLOCK_SPIN_SPEED,
    COLORED_ARCS,
    DARKEN_FACTOR,
    FINGERTIP_IDS,
    GLOW_BLEND_ALPHA,
    GLOW_KERNEL_SIZE,
    MAX_PARTICLES,
    MAX_RADIUS,
    MID_BASE_ID,
    MIN_RADIUS,
    NUM_BARS,
    PARTICLE_LIFETIME,
    PARTICLE_SIZE,
    PARTICLE_SPAWN_RATE,
    PARTICLE_SPEED,
    RINGS,
    SCATTER_CHARS,
    SCATTER_COLORS,
    SCATTER_COUNT,
    SCATTER_DRIFT_SPEED,
    SCATTER_FONT_RANGE,
    SCATTER_RADIUS_RANGE,
    SPIRAL_ARM_COUNT,
    SPIRAL_DOT_COUNT,
    SPIRAL_DOT_SIZE,
    SPIRAL_TURNS,
    SPREAD_MAX,
    SPREAD_MIN,
    SYMBOL_RINGS,
    TRAIL_ARCS,
    WRIST_ID,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(_DIR, "models", "hand_landmarker.task")


# ═══════════════════════════════════════════════════════════════════════════════
# Hand geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def palm_center(lm, w, h):
    """Average of wrist + all MCP bases as a stable palm center."""
    ids = [0, 5, 9, 13, 17]  # wrist + 4 MCP bases
    cx = sum(lm[i].x for i in ids) / len(ids) * w
    cy = sum(lm[i].y for i in ids) / len(ids) * h
    return cx, cy


def hand_rotation(lm, w, h):
    """Angle (radians) from wrist to middle-finger MCP."""
    wx, wy = lm[WRIST_ID].x * w, lm[WRIST_ID].y * h
    mx, my = lm[MID_BASE_ID].x * w, lm[MID_BASE_ID].y * h
    return math.atan2(my - wy, mx - wx)


def finger_spread(lm, w, h):
    """Normalised spread: average fingertip distance from palm / hand_size.
    Returns 0.0 (fully pinched) → 1.0 (fully open)."""
    cx, cy = palm_center(lm, w, h)
    tips = [(lm[i].x * w, lm[i].y * h) for i in FINGERTIP_IDS]
    dists = [math.hypot(tx - cx, ty - cy) for tx, ty in tips]
    avg_dist = sum(dists) / len(dists)
    # hand size ≈ wrist-to-middle-MCP distance
    wx, wy = lm[WRIST_ID].x * w, lm[WRIST_ID].y * h
    mx, my = lm[MID_BASE_ID].x * w, lm[MID_BASE_ID].y * h
    hand_size = math.hypot(mx - wx, my - wy) + 1e-6
    ratio = avg_dist / hand_size
    t = (ratio - SPREAD_MIN) / (SPREAD_MAX - SPREAD_MIN + 1e-6)
    return max(0.0, min(1.0, t))


def pinch_factor(lm, w, h):
    """0 = fingers wide apart, 1 = all fingertips touching.
    Based on max pairwise distance among fingertips."""
    tips = [(lm[i].x * w, lm[i].y * h) for i in FINGERTIP_IDS]
    max_d = 0
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            d = math.hypot(tips[i][0] - tips[j][0], tips[i][1] - tips[j][1])
            if d > max_d:
                max_d = d
    wx, wy = lm[WRIST_ID].x * w, lm[WRIST_ID].y * h
    mx, my = lm[MID_BASE_ID].x * w, lm[MID_BASE_ID].y * h
    hand_size = math.hypot(mx - wx, my - wy) + 1e-6
    norm = max_d / hand_size
    # invert: small max_d → high pinch
    t = 1.0 - max(0.0, min(1.0, (norm - 0.3) / 0.8))
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing functions
# ═══════════════════════════════════════════════════════════════════════════════

def draw_bars(overlay, cx, cy, radius, rotation, spread_t):
    """Compact rainbow color-wheel spokes at the core."""
    inner_r = radius * BAR_INNER_RATIO
    outer_r = radius * BAR_OUTER_RATIO
    brightness = 0.5 + spread_t * 0.5

    for i in range(NUM_BARS):
        angle = rotation + i * (2 * math.pi / NUM_BARS)
        color = BAR_COLORS[i % len(BAR_COLORS)]
        col = tuple(int(c * brightness) for c in color)
        x1 = int(cx + inner_r * math.cos(angle))
        y1 = int(cy + inner_r * math.sin(angle))
        x2 = int(cx + outer_r * math.cos(angle))
        y2 = int(cy + outer_r * math.sin(angle))
        cv2.line(overlay, (x1, y1), (x2, y2), col, BAR_WIDTH, cv2.LINE_AA)


def draw_rings(overlay, cx, cy, radius, rotation, t):
    """Dense concentric geometric rings: dots, ticks, dashes, thin circles."""
    for ri, (ratio, rtype, color, count) in enumerate(RINGS):
        r = int(radius * ratio)
        if r < 3:
            continue
        ring_spin = rotation + t * (0.10 + ri * 0.03)

        if rtype == "circle":
            cv2.circle(overlay, (int(cx), int(cy)), r, color, 1, cv2.LINE_AA)
        elif rtype == "dots":
            for di in range(count):
                a = ring_spin + 2 * math.pi * di / count
                px = int(cx + r * math.cos(a))
                py = int(cy + r * math.sin(a))
                cv2.circle(overlay, (px, py), 2, color, -1, cv2.LINE_AA)
        elif rtype == "ticks":
            tick_len = max(3, int(radius * 0.03))
            for di in range(count):
                a = ring_spin + 2 * math.pi * di / count
                x1 = int(cx + (r - tick_len) * math.cos(a))
                y1 = int(cy + (r - tick_len) * math.sin(a))
                x2 = int(cx + (r + tick_len) * math.cos(a))
                y2 = int(cy + (r + tick_len) * math.sin(a))
                cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        elif rtype == "dashes":
            arc_span = int(360 / count * 0.6)
            for di in range(count):
                a_deg = int(math.degrees(ring_spin) + 360 * di / count) % 360
                cv2.ellipse(overlay, (int(cx), int(cy)), (r, r),
                            0, a_deg, a_deg + arc_span, color, 1, cv2.LINE_AA)


def draw_colored_arcs(overlay, cx, cy, radius, rotation, t):
    """Partial arcs of individually-colored dots (pink, orange, green, cyan…)."""
    for ratio, color, dot_count, arc_span_deg, dot_r, spin_spd in COLORED_ARCS:
        r = int(radius * ratio)
        if r < 5:
            continue
        base_a = rotation + t * spin_spd
        arc_span_rad = math.radians(arc_span_deg)
        for di in range(dot_count):
            frac = di / max(1, dot_count - 1)
            a = base_a + frac * arc_span_rad
            px = int(cx + r * math.cos(a))
            py = int(cy + r * math.sin(a))
            cv2.circle(overlay, (px, py), dot_r, color, -1, cv2.LINE_AA)


def draw_clock_numerals(overlay, cx, cy, radius, rotation, t):
    """Roman numeral clock ring (XII, I, II, … XI) like an astrolabe."""
    r = int(radius * CLOCK_RATIO)
    if r < 15:
        return
    clock_rot = rotation + t * CLOCK_SPIN_SPEED
    for i, numeral in enumerate(CLOCK_NUMERALS):
        a = clock_rot + 2 * math.pi * i / 12
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        (tw, th), _ = cv2.getTextSize(numeral, cv2.FONT_HERSHEY_SIMPLEX,
                                       CLOCK_FONT_SCALE, 1)
        cv2.putText(overlay, numeral, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, CLOCK_FONT_SCALE,
                    CLOCK_COLOR, 1, cv2.LINE_AA)


def draw_symbol_rings(overlay, cx, cy, radius, rotation, t):
    """Multiple independently-rotating text/symbol rings."""
    for ratio, chars, count, color, fscale, spin_spd in SYMBOL_RINGS:
        r = int(radius * ratio)
        if r < 8:
            continue
        ring_angle = rotation + t * spin_spd
        for si in range(count):
            a = ring_angle + 2 * math.pi * si / count
            px = int(cx + r * math.cos(a))
            py = int(cy + r * math.sin(a))
            sym = chars[si % len(chars)]
            (tw, th), _ = cv2.getTextSize(sym, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
            cv2.putText(overlay, sym, (px - tw // 2, py + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, color, 1, cv2.LINE_AA)


def _init_scatter_table():
    """Pre-generate random positions for scattered symbols (called once)."""
    rng = np.random.RandomState(42)
    table = []
    for _ in range(SCATTER_COUNT):
        ratio = rng.uniform(*SCATTER_RADIUS_RANGE)
        angle_offset = rng.uniform(0, 2 * math.pi)
        char = SCATTER_CHARS[rng.randint(0, len(SCATTER_CHARS))]
        color = SCATTER_COLORS[rng.randint(0, len(SCATTER_COLORS))]
        fscale = rng.uniform(*SCATTER_FONT_RANGE)
        drift_dir = rng.choice([-1, 1])
        table.append((ratio, angle_offset, char, color, fscale, drift_dir))
    return table

_SCATTER_TABLE = _init_scatter_table()


def draw_scattered_symbols(overlay, cx, cy, radius, rotation, t):
    """Jittered symbols floating at random positions (not on perfect rings)."""
    for ratio, angle_off, char, color, fscale, drift_dir in _SCATTER_TABLE:
        r = int(radius * ratio)
        if r < 5:
            continue
        a = rotation + angle_off + t * SCATTER_DRIFT_SPEED * drift_dir
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        (tw, th), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        cv2.putText(overlay, char, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, color, 1, cv2.LINE_AA)


def draw_bracket_clusters(overlay, cx, cy, radius, rotation, t):
    """Groups of brackets ))) ((( }} [[ scattered at fixed slots."""
    for ratio, angle_off, text, color, fscale, spin_spd in BRACKET_CLUSTERS:
        r = int(radius * ratio)
        if r < 8:
            continue
        a = rotation + angle_off + t * spin_spd
        px = int(cx + r * math.cos(a))
        py = int(cy + r * math.sin(a))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        cv2.putText(overlay, text, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, color, 1, cv2.LINE_AA)


def draw_arcs(overlay, cx, cy, radius, rotation, t):
    """Curved arc connectors between ring layers."""
    for inner_r, outer_r, arc_count, color, thickness in ARCS:
        r_in = int(radius * inner_r)
        r_out = int(radius * outer_r)
        if r_in < 3:
            continue
        arc_spin = rotation + t * 0.05
        for ai in range(arc_count):
            base_a = arc_spin + 2 * math.pi * ai / arc_count
            x1 = int(cx + r_in * math.cos(base_a))
            y1 = int(cy + r_in * math.sin(base_a))
            x2 = int(cx + r_out * math.cos(base_a + 0.10))
            y2 = int(cy + r_out * math.sin(base_a + 0.10))
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            a_deg_start = int(math.degrees(base_a + 0.10)) % 360
            a_deg_end = a_deg_start + int(360 / arc_count * 0.35)
            cv2.ellipse(overlay, (int(cx), int(cy)), (r_out, r_out),
                        0, a_deg_start, a_deg_end, color, thickness, cv2.LINE_AA)


def draw_trail_arcs(overlay, cx, cy, radius, rotation, t):
    """Dotted spiral trails curving outward like comet tails."""
    for start_r, end_r, angle_off, dot_count, color, dot_sz, spin_spd in TRAIL_ARCS:
        base_a = rotation + angle_off + t * spin_spd
        for di in range(dot_count):
            frac = di / max(1, dot_count - 1)
            r = int(radius * (start_r + (end_r - start_r) * frac))
            # spiral: angle advances as radius grows
            a = base_a + frac * 1.8
            px = int(cx + r * math.cos(a))
            py = int(cy + r * math.sin(a))
            # fade outer dots
            alpha = 1.0 - frac * 0.5
            col = tuple(int(c * alpha) for c in color)
            cv2.circle(overlay, (px, py), dot_sz, col, -1, cv2.LINE_AA)


def draw_spiral_arms(overlay, cx, cy, radius, rotation, pinch_t, t):
    """Curved spiral arms when pinched."""
    if pinch_t < 0.2:
        return
    alpha = pinch_t
    for arm in range(SPIRAL_ARM_COUNT):
        base_a = rotation + 2 * math.pi * arm / SPIRAL_ARM_COUNT
        color_idx = arm * (len(BAR_COLORS) // SPIRAL_ARM_COUNT)
        color = BAR_COLORS[color_idx % len(BAR_COLORS)]
        col = tuple(int(c * alpha) for c in color)
        prev_pt = None
        for di in range(SPIRAL_DOT_COUNT):
            frac = di / SPIRAL_DOT_COUNT
            r = radius * 0.10 + radius * 1.5 * frac
            a = base_a + SPIRAL_TURNS * 2 * math.pi * frac + t * 0.5
            px = int(cx + r * math.cos(a))
            py = int(cy + r * math.sin(a))
            cv2.circle(overlay, (px, py), SPIRAL_DOT_SIZE, col, -1, cv2.LINE_AA)
            if prev_pt is not None:
                cv2.line(overlay, prev_pt, (px, py), col, 1, cv2.LINE_AA)
            prev_pt = (px, py)


def draw_center_glow(overlay, cx, cy, radius):
    """Bright core at the center."""
    core_r = max(4, int(radius * 0.05))
    cv2.circle(overlay, (int(cx), int(cy)), core_r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, (int(cx), int(cy)), core_r + 3, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(overlay, (int(cx), int(cy)), core_r + 7, (140, 140, 200), 1, cv2.LINE_AA)


def draw_fingertip_accents(overlay, lm, w, h):
    """Bright dots at each fingertip."""
    for tip_id in FINGERTIP_IDS:
        tx = int(lm[tip_id].x * w)
        ty = int(lm[tip_id].y * h)
        cv2.circle(overlay, (tx, ty), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (tx, ty), 9, (200, 200, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Particle system (ambient floating particles around the mandala)
# ═══════════════════════════════════════════════════════════════════════════════

class ParticlePool:
    def __init__(self, cap=MAX_PARTICLES):
        self.x     = np.zeros(cap, np.float32)
        self.y     = np.zeros(cap, np.float32)
        self.vx    = np.zeros(cap, np.float32)
        self.vy    = np.zeros(cap, np.float32)
        self.b     = np.zeros(cap, np.uint8)
        self.g     = np.zeros(cap, np.uint8)
        self.r     = np.zeros(cap, np.uint8)
        self.age   = np.zeros(cap, np.float32)
        self.life  = np.ones(cap,  np.float32)
        self.alive = np.zeros(cap, bool)

    def spawn_around(self, cx, cy, radius, n, color):
        dead = np.where(~self.alive)[0]
        n = min(n, len(dead))
        if n == 0:
            return
        idx = dead[:n]
        self.alive[idx] = True
        self.age[idx] = 0.0
        self.life[idx] = np.random.uniform(*PARTICLE_LIFETIME, size=n)
        angles = np.random.uniform(0, 2 * math.pi, n)
        radii = np.random.uniform(radius * 0.5, radius * 1.4, n)
        self.x[idx] = cx + np.cos(angles) * radii
        self.y[idx] = cy + np.sin(angles) * radii
        spd = np.random.uniform(PARTICLE_SPEED * 0.3, PARTICLE_SPEED, n)
        drift_a = angles + np.random.uniform(-1.0, 1.0, n)
        self.vx[idx] = np.cos(drift_a) * spd
        self.vy[idx] = np.sin(drift_a) * spd
        self.b[idx] = color[0]
        self.g[idx] = color[1]
        self.r[idx] = color[2]

    def update(self, dt):
        a = self.alive
        self.x[a] += self.vx[a] * dt
        self.y[a] += self.vy[a] * dt
        self.age[a] += dt
        self.alive[a & (self.age >= self.life)] = False

    def draw(self, overlay, h, w):
        idx = np.where(self.alive)[0]
        if len(idx) == 0:
            return
        progress = self.age[idx] / self.life[idx]
        alpha = (1.0 - progress) ** 1.5
        xs = self.x[idx].astype(int)
        ys = self.y[idx].astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        idx, xs, ys, alpha = idx[valid], xs[valid], ys[valid], alpha[valid]
        for k in range(len(idx)):
            i = idx[k]
            a = float(alpha[k])
            col = (int(self.b[i] * a), int(self.g[i] * a), int(self.r[i] * a))
            sz = np.random.randint(PARTICLE_SIZE[0], PARTICLE_SIZE[1] + 1)
            cv2.circle(overlay, (xs[k], ys[k]), sz, col, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Glow
# ═══════════════════════════════════════════════════════════════════════════════

def apply_glow(frame, overlay):
    glow = cv2.GaussianBlur(overlay, GLOW_KERNEL_SIZE, 0)
    blended = cv2.addWeighted(frame, 1.0, glow, GLOW_BLEND_ALPHA, 0)
    blended = cv2.add(blended, overlay // 2)
    return blended


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothing helpers
# ═══════════════════════════════════════════════════════════════════════════════

class Smoother:
    """Exponential moving average for float values."""
    def __init__(self, alpha=0.3, init=0.0):
        self.alpha = alpha
        self.val = init

    def update(self, raw):
        self.val += self.alpha * (raw - self.val)
        return self.val


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Radiance — gesture-driven mandala")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    hands = HandLandmarker.create_from_options(hand_opts)

    pool = ParticlePool(MAX_PARTICLES)

    # smoothers for stable visuals
    s_cx = Smoother(0.35)
    s_cy = Smoother(0.35)
    s_rot = Smoother(0.25)
    s_spread = Smoother(0.3, 0.5)
    s_pinch = Smoother(0.3, 0.0)

    t0 = time.time()
    prev_t = t0

    print("Press 'q' to quit.  Move your fingers to control the mandala.")

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
            lm = result.hand_landmarks[0]  # use first hand

            # ── Compute hand geometry ─────────────────────────────────────
            raw_cx, raw_cy = palm_center(lm, w, h)
            raw_rot = hand_rotation(lm, w, h)
            raw_spread = finger_spread(lm, w, h)
            raw_pinch = pinch_factor(lm, w, h)

            cx = s_cx.update(raw_cx)
            cy = s_cy.update(raw_cy)
            # smooth rotation with angle wrapping
            rot_diff = raw_rot - s_rot.val
            rot_diff = (rot_diff + math.pi) % (2 * math.pi) - math.pi
            rot = s_rot.val + s_rot.alpha * rot_diff
            s_rot.val = rot
            spread_t = s_spread.update(raw_spread)
            p_pinch = s_pinch.update(raw_pinch)

            radius = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * spread_t

            # ── Draw layers (back to front) ───────────────────────────
            draw_trail_arcs(overlay, cx, cy, radius, rot, t)
            draw_rings(overlay, cx, cy, radius, rot, t)
            draw_colored_arcs(overlay, cx, cy, radius, rot, t)
            draw_arcs(overlay, cx, cy, radius, rot, t)
            draw_scattered_symbols(overlay, cx, cy, radius, rot, t)
            draw_bracket_clusters(overlay, cx, cy, radius, rot, t)
            draw_symbol_rings(overlay, cx, cy, radius, rot, t)
            draw_clock_numerals(overlay, cx, cy, radius, rot, t)
            draw_bars(overlay, cx, cy, radius, rot, spread_t)
            draw_spiral_arms(overlay, cx, cy, radius, rot, p_pinch, t)
            draw_center_glow(overlay, int(cx), int(cy), radius)
            draw_fingertip_accents(overlay, lm, w, h)

            # ── Spawn ambient particles ───────────────────────────────────
            spawn_color = BAR_COLORS[int(t * 3) % len(BAR_COLORS)]
            pool.spawn_around(cx, cy, radius, PARTICLE_SPAWN_RATE, spawn_color)

        # ── Update & draw particles ───────────────────────────────────────
        pool.update(dt)
        pool.draw(overlay, h, w)

        # ── Composite ─────────────────────────────────────────────────────
        darkened = (frame.astype(np.float32) * DARKEN_FACTOR).astype(np.uint8)
        output = apply_glow(darkened, overlay)

        cv2.imshow("Radiance", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

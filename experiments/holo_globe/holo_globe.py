"""
Holo Globe — holographic globe interaction with hand tracking.

Usage:
    python experiments/holo_globe/holo_globe.py
    python experiments/holo_globe/holo_globe.py --camera 1
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

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
    ATMOSPHERE_LAYERS,
    BACKGROUND_GRID_STEP,
    BACKGROUND_STAR_COUNT,
    BASE_RADIUS,
    CURSOR_COLOR,
    CURSOR_HIT_RADIUS,
    DARKEN_FACTOR,
    DATA_ARC_LIFT,
    DATA_ARC_POINT_COUNT,
    DETAIL_CARD_SIZE,
    FINGERTIP_IDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GLOBE_BACK_LINE_COLOR,
    GLOBE_CENTER_RATIO,
    GLOBE_CORE_COLOR,
    GLOBE_LAT_STEPS,
    GLOBE_LINE_COLOR,
    GLOBE_LON_STEPS,
    GLOBE_SEGMENTS,
    GLOW_BLEND_ALPHA,
    GLOW_KERNEL_SIZE,
    HUD_COLOR,
    HUD_RING_COUNT,
    HUD_RING_SPEED,
    INDEX_MCP_ID,
    INDEX_TIP_ID,
    LOCATION_PINS,
    MAX_HANDS,
    MAX_ZOOM,
    MIDDLE_MCP_ID,
    MIDDLE_TIP_ID,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIN_ZOOM,
    OPEN_HAND_ORBIT_THRESHOLD,
    OPEN_HAND_RATIO_MAX,
    OPEN_HAND_RATIO_MIN,
    OPENNESS_ZOOM_DEADZONE,
    OPENNESS_ZOOM_SENSITIVITY,
    ORBIT_RADIUS_FACTOR,
    PANEL_ACCENT,
    PANEL_COLOR,
    HOVER_LOCK_SECONDS,
    HOVER_STICKY_RADIUS,
    PINCH_CLOSED_RATIO,
    PINCH_OPEN_RATIO,
    PINCH_ORBIT_THRESHOLD,
    PINCH_RELEASE_THRESHOLD,
    PINCH_SELECT_THRESHOLD,
    ROTATION_SENSITIVITY,
    SCANLINE_SPEED,
    SELECT_COOLDOWN_SECONDS,
    SECONDARY_PINCH_CLOSED_RATIO,
    SECONDARY_PINCH_OPEN_RATIO,
    SECONDARY_PINCH_RELEASE,
    SECONDARY_PINCH_THRESHOLD,
    TEXT_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THUMB_TIP_ID,
    PORTRAIT_ALPHA,
    PORTRAIT_FADE_WIDTH,
    PORTRAIT_TINT,
    WARNING_COLOR,
    WINDOW_TITLE,
    WRIST_ID,
    ZOOM_DEADZONE,
    ZOOM_DELTA_SENSITIVITY,
    ZOOM_SMOOTHING,
)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def locate_hand_model() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "models", "hand_landmarker.task"),
        os.path.join(here, "..", "radiance", "models", "hand_landmarker.task"),
        os.path.join(here, "..", "aura_effects", "models", "hand_landmarker.task"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    return None


def create_selfie_segmenter():
    try:
        from mediapipe.python.solutions import selfie_segmentation
    except Exception:
        return None
    return selfie_segmentation.SelfieSegmentation(model_selection=1)


def point(lm, idx: int, w: int, h: int):
    return (lm[idx].x * w, lm[idx].y * h)


def palm_center(lm, w: int, h: int):
    ids = [WRIST_ID, INDEX_MCP_ID, MIDDLE_MCP_ID, 13, 17]
    return (
        sum(lm[i].x for i in ids) / len(ids) * w,
        sum(lm[i].y for i in ids) / len(ids) * h,
    )


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) + 1e-6
    return vec / norm


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:
        return normalize((1.0 - t) * a + t * b)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    return normalize(w1 * a + w2 * b)


def location_metrics(pin, idx: int):
    signal = 62 + (idx * 9) % 31
    coverage = 48 + int(abs(pin.lat) * 0.7) % 44
    stability = 54 + int(abs(pin.lon) * 0.4) % 40
    return signal, coverage, stability


@dataclass
class HandState:
    palm: tuple[float, float]
    thumb_tip: tuple[float, float]
    index_tip: tuple[float, float]
    middle_tip: tuple[float, float]
    pinch_center: tuple[float, float]
    hand_size: float
    pinch_ratio: float
    pinch_strength: float
    span_strength: float
    openness: float
    close_strength: float


@dataclass
class GlobePin:
    name: str
    region: str
    lat: float
    lon: float
    color: tuple[int, int, int]
    detail: str


class EMA:
    def __init__(self, alpha: float, initial: float = 0.0):
        self.alpha = alpha
        self.value = initial
        self.initialized = False

    def update(self, sample: float) -> float:
        if not self.initialized:
            self.value = sample
            self.initialized = True
            return self.value
        self.value += self.alpha * (sample - self.value)
        return self.value


class HoloGlobe:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center = (
            int(width * GLOBE_CENTER_RATIO[0]),
            int(height * GLOBE_CENTER_RATIO[1]),
        )
        self.base_radius = BASE_RADIUS
        self.yaw = -0.55
        self.pitch = 0.30
        self.zoom = 1.05
        self.target_zoom = self.zoom
        self.cursor = (width * 0.5, height * 0.5)
        self.cursor_x = EMA(0.33, self.cursor[0])
        self.cursor_y = EMA(0.33, self.cursor[1])
        self.zoom_filter = EMA(ZOOM_SMOOTHING, self.zoom)
        self.pins = [GlobePin(**pin) for pin in LOCATION_PINS]
        self.hovered_pin = None
        self.selected_pin = None
        self.display_pin = None
        self.panel_progress = 0.0
        self.visible_pins = []
        self.orbit_anchor = None
        self.orbit_velocity = [0.0, 0.0]
        self.pinch_engaged = False
        self.close_engaged = False
        self.last_select_time = 0.0
        self.hover_candidate = None
        self.hover_started = 0.0
        self.star_field = self._build_star_field()

    @property
    def radius(self) -> float:
        return self.base_radius * self.zoom

    def _build_star_field(self):
        rng = np.random.RandomState(7)
        stars = []
        for _ in range(BACKGROUND_STAR_COUNT):
            stars.append(
                (
                    int(rng.uniform(0, self.width)),
                    int(rng.uniform(0, self.height)),
                    float(rng.uniform(0.4, 1.2)),
                    float(rng.uniform(0.25, 0.8)),
                )
            )
        return stars

    def geo_vector(self, lat_deg: float, lon_deg: float) -> np.ndarray:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        return np.array(
            [
                math.cos(lat) * math.cos(lon),
                math.sin(lat),
                math.cos(lat) * math.sin(lon),
            ],
            dtype=np.float32,
        )

    def rotate_vector(self, vec: np.ndarray) -> np.ndarray:
        x, y, z = vec
        cosy = math.cos(self.yaw)
        siny = math.sin(self.yaw)
        xz = cosy * x + siny * z
        zz = -siny * x + cosy * z

        cosp = math.cos(self.pitch)
        sinp = math.sin(self.pitch)
        yz = cosp * y - sinp * zz
        zz2 = sinp * y + cosp * zz
        return np.array([xz, yz, zz2], dtype=np.float32)

    def project_vec(self, vec: np.ndarray, scale: float = 1.0):
        rotated = self.rotate_vector(vec * scale)
        sx = int(self.center[0] + rotated[0] * self.radius)
        sy = int(self.center[1] - rotated[1] * self.radius)
        return sx, sy, float(rotated[2])

    def project(self, lat_deg: float, lon_deg: float):
        return self.project_vec(self.geo_vector(lat_deg, lon_deg))

    def update_hand_presence(self, hand: HandState | None):
        if hand is None:
            self.orbit_anchor = None
            self.pinch_engaged = False
            self.close_engaged = False
            self.orbit_velocity[0] *= 0.90
            self.orbit_velocity[1] *= 0.90
            return

        self.cursor = (
            self.cursor_x.update(hand.pinch_center[0]),
            self.cursor_y.update(hand.pinch_center[1]),
        )

    def update_visible_pins(self, t: float, now_abs: float):
        candidate = None
        candidate_dist = float("inf")
        self.visible_pins = []
        for idx, pin in enumerate(self.pins):
            sx, sy, depth = self.project(pin.lat, pin.lon)
            if depth <= 0.02:
                continue
            pulse = 0.72 + 0.28 * math.sin(t * 3.0 + idx * 0.9)
            base_r = int(6 + 7 * depth + 4 * pulse)
            dist = math.hypot(self.cursor[0] - sx, self.cursor[1] - sy)
            self.visible_pins.append(
                {
                    "idx": idx,
                    "pin": pin,
                    "sx": sx,
                    "sy": sy,
                    "depth": depth,
                    "base_r": base_r,
                    "dist": dist,
                }
            )
            sticky_radius = HOVER_STICKY_RADIUS if self.hover_candidate == idx else CURSOR_HIT_RADIUS + base_r
            if dist <= sticky_radius and dist < candidate_dist:
                candidate = idx
                candidate_dist = dist

        self.visible_pins.sort(key=lambda item: item["depth"])
        if candidate != self.hover_candidate:
            self.hover_candidate = candidate
            self.hover_started = now_abs

        if candidate is not None and (now_abs - self.hover_started) >= HOVER_LOCK_SECONDS:
            self.hovered_pin = candidate
        else:
            self.hovered_pin = self.hover_candidate if self.hover_candidate == self.selected_pin else None

    def update_interaction(self, hand: HandState | None, dt: float, now_abs: float):
        if hand is None:
            self.yaw += self.orbit_velocity[0]
            self.pitch += self.orbit_velocity[1]
            self.pitch = clamp(self.pitch, -1.10, 1.10)
            self.zoom = self.zoom_filter.update(clamp(self.target_zoom, MIN_ZOOM, MAX_ZOOM))
            self.panel_progress += ((1.0 if self.selected_pin is not None else 0.0) - self.panel_progress) * min(1.0, dt * 6.5)
            if self.selected_pin is not None:
                self.display_pin = self.selected_pin
            if self.selected_pin is None and self.panel_progress <= 0.02:
                self.display_pin = None
            return

        palm_over_globe = math.hypot(hand.palm[0] - self.center[0], hand.palm[1] - self.center[1]) <= self.radius * ORBIT_RADIUS_FACTOR
        cursor_over_globe = math.hypot(self.cursor[0] - self.center[0], self.cursor[1] - self.center[1]) <= self.radius * ORBIT_RADIUS_FACTOR
        deep_pinch = hand.pinch_strength >= PINCH_SELECT_THRESHOLD
        orbit_pinch = PINCH_ORBIT_THRESHOLD <= hand.pinch_strength < PINCH_SELECT_THRESHOLD
        open_hand_orbit = hand.openness >= OPEN_HAND_ORBIT_THRESHOLD and palm_over_globe
        close_gesture = hand.close_strength >= SECONDARY_PINCH_THRESHOLD

        if deep_pinch and not self.pinch_engaged and (now_abs - self.last_select_time) >= SELECT_COOLDOWN_SECONDS:
            self.pinch_engaged = True
            self.last_select_time = now_abs
            if self.hovered_pin is not None:
                if self.selected_pin == self.hovered_pin:
                    self.selected_pin = None
                else:
                    self.selected_pin = self.hovered_pin
        if close_gesture and self.selected_pin is not None and not self.close_engaged:
            self.selected_pin = None
            self.close_engaged = True

        orbit_pointer = None
        orbit_metric = None
        if open_hand_orbit:
            orbit_pointer = hand.palm
            orbit_metric = hand.openness
        elif orbit_pinch and cursor_over_globe:
            orbit_pointer = self.cursor
            orbit_metric = hand.pinch_ratio

        if orbit_pointer is not None:
            if self.orbit_anchor is None:
                self.orbit_anchor = {
                    "pointer": orbit_pointer,
                    "metric": orbit_metric,
                }
            else:
                dx = orbit_pointer[0] - self.orbit_anchor["pointer"][0]
                dy = orbit_pointer[1] - self.orbit_anchor["pointer"][1]
                self.orbit_velocity[0] = dx * ROTATION_SENSITIVITY
                self.orbit_velocity[1] = -dy * ROTATION_SENSITIVITY
                self.yaw += self.orbit_velocity[0]
                self.pitch += self.orbit_velocity[1]

                if open_hand_orbit:
                    zoom_delta = orbit_metric - self.orbit_anchor["metric"]
                    if abs(zoom_delta) > OPENNESS_ZOOM_DEADZONE:
                        self.target_zoom = clamp(
                            self.target_zoom + zoom_delta * OPENNESS_ZOOM_SENSITIVITY,
                            MIN_ZOOM,
                            MAX_ZOOM,
                        )
                else:
                    zoom_delta = orbit_metric - self.orbit_anchor["metric"]
                    if abs(zoom_delta) > ZOOM_DEADZONE:
                        self.target_zoom = clamp(
                            self.target_zoom + zoom_delta * ZOOM_DELTA_SENSITIVITY,
                            MIN_ZOOM,
                            MAX_ZOOM,
                        )

                self.orbit_anchor = {
                    "pointer": orbit_pointer,
                    "metric": orbit_metric,
                }
        else:
            self.orbit_anchor = None
            self.yaw += self.orbit_velocity[0]
            self.pitch += self.orbit_velocity[1]
            self.orbit_velocity[0] *= 0.86
            self.orbit_velocity[1] *= 0.86

        if hand.pinch_strength <= PINCH_RELEASE_THRESHOLD:
            self.pinch_engaged = False
        if hand.close_strength <= SECONDARY_PINCH_RELEASE:
            self.close_engaged = False

        self.pitch = clamp(self.pitch, -1.10, 1.10)
        self.zoom = self.zoom_filter.update(clamp(self.target_zoom, MIN_ZOOM, MAX_ZOOM))
        if self.selected_pin is not None:
            self.display_pin = self.selected_pin
        self.panel_progress += ((1.0 if self.selected_pin is not None else 0.0) - self.panel_progress) * min(1.0, dt * 6.5)
        if self.selected_pin is None and self.panel_progress <= 0.02:
            self.display_pin = None

    def draw_presence_layer(self, canvas: np.ndarray, frame: np.ndarray, segmentation_mask: np.ndarray | None):
        h, w = frame.shape[:2]
        if segmentation_mask is None:
            mask = np.ones((h, w), dtype=np.float32)
        else:
            mask = cv2.GaussianBlur(segmentation_mask.astype(np.float32), (0, 0), 11)
        fade_width = max(1, int(w * PORTRAIT_FADE_WIDTH))
        x_fade = np.linspace(1.0, 0.0, fade_width, dtype=np.float32)
        x_fade = np.pad(x_fade, (0, max(0, w - fade_width)), constant_values=0.0)
        x_fade = x_fade.reshape(1, w)
        alpha = np.clip(mask * x_fade * PORTRAIT_ALPHA, 0.0, 1.0)

        tinted = frame.astype(np.float32)
        tint = np.zeros_like(tinted)
        tint[..., 0] = PORTRAIT_TINT[0]
        tint[..., 1] = PORTRAIT_TINT[1]
        tint[..., 2] = PORTRAIT_TINT[2]
        subject = cv2.addWeighted(tinted, 0.88, tint, 0.28, 0)

        alpha_3 = alpha[..., None]
        canvas[:] = (canvas.astype(np.float32) * (1.0 - alpha_3) + subject * alpha_3).astype(np.uint8)

        aura = cv2.GaussianBlur(alpha, (0, 0), 25) * 0.65
        aura_overlay = np.zeros_like(canvas, dtype=np.float32)
        aura_overlay[..., 0] = TEXT_ACCENT[0] * aura
        aura_overlay[..., 1] = TEXT_ACCENT[1] * aura
        aura_overlay[..., 2] = TEXT_ACCENT[2] * aura
        canvas[:] = np.clip(canvas.astype(np.float32) + aura_overlay, 0, 255).astype(np.uint8)

    def draw_background(self, canvas: np.ndarray, t: float):
        h, w = canvas.shape[:2]
        yy, xx = np.indices((h, w))
        cx = self.center[0] / w
        cy = self.center[1] / h
        dx = xx / w - cx
        dy = yy / h - cy
        radial = np.clip(1.0 - np.sqrt(dx * dx + dy * dy) * 1.6, 0.0, 1.0)
        base = np.zeros_like(canvas)
        base[..., 0] = (10 + radial * 18).astype(np.uint8)
        base[..., 1] = (14 + radial * 28).astype(np.uint8)
        base[..., 2] = (18 + radial * 48).astype(np.uint8)
        canvas[:] = cv2.addWeighted(canvas, 0.20, base, 0.80, 0)

        for x in range(0, self.width, BACKGROUND_GRID_STEP):
            alpha = 0.13 if x % (BACKGROUND_GRID_STEP * 4) == 0 else 0.05
            col = tuple(int(c * alpha) for c in HUD_COLOR)
            cv2.line(canvas, (x, 0), (x, self.height), col, 1, cv2.LINE_AA)
        for y in range(0, self.height, BACKGROUND_GRID_STEP):
            alpha = 0.13 if y % (BACKGROUND_GRID_STEP * 4) == 0 else 0.05
            col = tuple(int(c * alpha) for c in HUD_COLOR)
            cv2.line(canvas, (0, y), (self.width, y), col, 1, cv2.LINE_AA)

        for x, y, size, alpha in self.star_field:
            pulse = 0.6 + 0.4 * math.sin(t * 0.8 + (x * 0.01) + (y * 0.02))
            col = tuple(int(c * alpha * pulse * 0.45) for c in TEXT_ACCENT)
            cv2.circle(canvas, (x, y), int(size), col, -1, cv2.LINE_AA)

        scan_y = int(((math.sin(t * SCANLINE_SPEED) * 0.5) + 0.5) * self.height)
        cv2.line(canvas, (0, scan_y), (self.width, scan_y), (24, 42, 62), 2, cv2.LINE_AA)
        cv2.line(canvas, (0, scan_y + 3), (self.width, scan_y + 3), (12, 20, 30), 1, cv2.LINE_AA)

    def draw_globe(self, overlay: np.ndarray, t: float):
        cx, cy = self.center
        r = int(self.radius)

        for layer in range(ATMOSPHERE_LAYERS, 0, -1):
            rr = int(r * (1.02 + layer * 0.025))
            alpha = 0.18 * (layer / ATMOSPHERE_LAYERS)
            col = tuple(int(c * alpha) for c in GLOBE_LINE_COLOR)
            cv2.circle(overlay, (cx, cy), rr, col, 1, cv2.LINE_AA)

        for li in range(1, GLOBE_LAT_STEPS):
            lat = -90 + 180 * li / GLOBE_LAT_STEPS
            self._draw_latitude(overlay, lat)

        for li in range(GLOBE_LON_STEPS):
            lon = -180 + 360 * li / GLOBE_LON_STEPS
            self._draw_longitude(overlay, lon)

        self._draw_equator_band(overlay, t)
        cv2.circle(overlay, (cx, cy), r, GLOBE_CORE_COLOR, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), max(8, int(r * 0.045)), (255, 252, 236), -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), max(20, int(r * 0.11)), tuple(int(c * 0.25) for c in GLOBE_CORE_COLOR), 1, cv2.LINE_AA)

    def _draw_equator_band(self, overlay: np.ndarray, t: float):
        prev = None
        prev_depth = None
        for step in range(GLOBE_SEGMENTS + 1):
            lon = -180 + 360 * step / GLOBE_SEGMENTS
            point = self.project(0.0, lon)
            if prev is not None:
                depth = (prev_depth + point[2]) * 0.5
                color = tuple(int(c * (0.8 + 0.2 * math.sin(t * 2.0))) for c in TEXT_ACCENT) if depth > 0 else GLOBE_BACK_LINE_COLOR
                cv2.line(overlay, prev, point[:2], color, 2 if depth > 0 else 1, cv2.LINE_AA)
            prev = point[:2]
            prev_depth = point[2]

    def _draw_latitude(self, overlay: np.ndarray, lat: float):
        prev = None
        prev_depth = None
        for step in range(GLOBE_SEGMENTS + 1):
            lon = -180 + 360 * step / GLOBE_SEGMENTS
            point = self.project(lat, lon)
            if prev is not None:
                depth = (prev_depth + point[2]) * 0.5
                color = GLOBE_LINE_COLOR if depth > 0 else GLOBE_BACK_LINE_COLOR
                thickness = 1 if depth > 0 else 1
                cv2.line(overlay, prev, point[:2], color, thickness, cv2.LINE_AA)
            prev = point[:2]
            prev_depth = point[2]

    def _draw_longitude(self, overlay: np.ndarray, lon: float):
        prev = None
        prev_depth = None
        for step in range(GLOBE_SEGMENTS + 1):
            lat = -90 + 180 * step / GLOBE_SEGMENTS
            point = self.project(lat, lon)
            if prev is not None:
                depth = (prev_depth + point[2]) * 0.5
                color = GLOBE_LINE_COLOR if depth > 0 else GLOBE_BACK_LINE_COLOR
                cv2.line(overlay, prev, point[:2], color, 1, cv2.LINE_AA)
            prev = point[:2]
            prev_depth = point[2]

    def draw_hud(self, overlay: np.ndarray, t: float):
        cx, cy = self.center
        r = int(self.radius)

        cv2.circle(overlay, (cx, cy), int(r * 1.05), GLOBE_CORE_COLOR, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), int(r * 1.18), (120, 90, 30), 1, cv2.LINE_AA)

        for i in range(HUD_RING_COUNT):
            ring_r = int(r * (1.18 + i * 0.11))
            start = int((math.degrees(t * (HUD_RING_SPEED + i * 0.05)) + i * 42) % 360)
            sweep = 42 + i * 10
            cv2.ellipse(overlay, (cx, cy), (ring_r, ring_r), 0, start, start + sweep, HUD_COLOR, 1, cv2.LINE_AA)
            cv2.ellipse(overlay, (cx, cy), (ring_r, ring_r), 0, start + 180, start + 180 + sweep // 2, WARNING_COLOR, 1, cv2.LINE_AA)

        bracket_r = int(r * 1.44)
        for angle in [40, 140, 220, 320]:
            rad = math.radians(angle)
            px = int(cx + math.cos(rad) * bracket_r)
            py = int(cy + math.sin(rad) * bracket_r)
            cv2.line(overlay, (px - 22, py), (px + 22, py), HUD_COLOR, 1, cv2.LINE_AA)
            cv2.line(overlay, (px, py - 22), (px, py + 22), HUD_COLOR, 1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 3, TEXT_ACCENT, -1, cv2.LINE_AA)

        title = "AURA / HOLOGRAPHIC EARTH"
        subtitle = "OPEN HAND ORBIT / INDEX+THUMB PINCH SELECT / MIDDLE+THUMB PINCH DISMISS"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(overlay, title, (cx - tw // 2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_PRIMARY, 1, cv2.LINE_AA)
        cv2.putText(overlay, subtitle, (cx - sw // 2, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_SECONDARY, 1, cv2.LINE_AA)

    def draw_data_arcs(self, overlay: np.ndarray, t: float):
        if self.selected_pin is not None:
            pairs = [(self.selected_pin, idx) for idx in range(len(self.pins)) if idx != self.selected_pin]
        else:
            pairs = [(i, (i + 1) % len(self.pins)) for i in range(len(self.pins))]

        for pair_idx, (a_idx, b_idx) in enumerate(pairs):
            vec_a = self.geo_vector(self.pins[a_idx].lat, self.pins[a_idx].lon)
            vec_b = self.geo_vector(self.pins[b_idx].lat, self.pins[b_idx].lon)
            prev = None
            prev_depth = None
            for step in range(DATA_ARC_POINT_COUNT):
                frac = step / max(1, DATA_ARC_POINT_COUNT - 1)
                lift = 1.0 + DATA_ARC_LIFT * math.sin(math.pi * frac)
                vec = slerp(vec_a, vec_b, frac) * lift
                point = self.project_vec(vec)
                if prev is not None:
                    depth = (prev_depth + point[2]) * 0.5
                    if depth > -0.20:
                        alpha = 0.28 + 0.24 * math.sin(t * 1.8 + pair_idx + frac * math.pi)
                        color = tuple(int(c * max(0.16, alpha)) for c in TEXT_ACCENT)
                        cv2.line(overlay, prev, point[:2], color, 1, cv2.LINE_AA)
                prev = point[:2]
                prev_depth = point[2]

    def draw_pins(self, overlay: np.ndarray, t: float):
        for item in self.visible_pins:
            idx = item["idx"]
            pin = item["pin"]
            sx = item["sx"]
            sy = item["sy"]
            base_r = item["base_r"]
            pulse = 0.76 + 0.24 * math.sin(t * 3.2 + idx)
            color = pin.color
            col_outer = tuple(int(c * 0.60) for c in color)
            cv2.circle(overlay, (sx, sy), base_r + 10, col_outer, 1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), base_r, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (sx, sy), base_r + 16, tuple(int(c * 0.30) for c in color), 1, cv2.LINE_AA)

            stem_end = (
                int(lerp(sx, self.center[0], -0.10)),
                int(lerp(sy, self.center[1], -0.10)),
            )
            cv2.line(overlay, (sx, sy), stem_end, tuple(int(c * 0.22) for c in color), 1, cv2.LINE_AA)

            if self.hovered_pin == idx or self.selected_pin == idx:
                highlight = tuple(int(c * pulse) for c in TEXT_ACCENT)
                cv2.circle(overlay, (sx, sy), base_r + 22, highlight, 1, cv2.LINE_AA)
                cv2.line(overlay, (sx + base_r + 18, sy - 12), (sx + 108, sy - 12), highlight, 1, cv2.LINE_AA)
                cv2.putText(overlay, pin.name.upper(), (sx + 114, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_PRIMARY, 1, cv2.LINE_AA)
                if self.hovered_pin == idx and self.hover_candidate == idx:
                    progress = clamp((time.time() - self.hover_started) / HOVER_LOCK_SECONDS, 0.0, 1.0)
                    cv2.ellipse(overlay, (sx, sy), (base_r + 26, base_r + 26), 0, -90, int(-90 + progress * 360), TEXT_ACCENT, 2, cv2.LINE_AA)

    def draw_cursor(self, overlay: np.ndarray, hand: HandState | None):
        if hand is None:
            return

        thumb = (int(hand.thumb_tip[0]), int(hand.thumb_tip[1]))
        index = (int(hand.index_tip[0]), int(hand.index_tip[1]))
        middle = (int(hand.middle_tip[0]), int(hand.middle_tip[1]))
        center = (int(self.cursor[0]), int(self.cursor[1]))

        cv2.line(overlay, thumb, index, CURSOR_COLOR, 1, cv2.LINE_AA)
        cv2.line(overlay, center, middle, tuple(int(c * 0.45) for c in CURSOR_COLOR), 1, cv2.LINE_AA)
        cv2.circle(overlay, thumb, 4, CURSOR_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, index, 4, CURSOR_COLOR, -1, cv2.LINE_AA)

        pinch_r = int(26 - 11 * hand.pinch_strength)
        outer_r = pinch_r + 14
        cv2.circle(overlay, center, outer_r, CURSOR_COLOR, 1, cv2.LINE_AA)
        cv2.circle(overlay, center, pinch_r, CURSOR_COLOR, 1, cv2.LINE_AA)
        cv2.line(overlay, (center[0] - 12, center[1]), (center[0] + 12, center[1]), CURSOR_COLOR, 1, cv2.LINE_AA)
        cv2.line(overlay, (center[0], center[1] - 12), (center[0], center[1] + 12), CURSOR_COLOR, 1, cv2.LINE_AA)

        if self.hovered_pin is not None:
            item = next((it for it in self.visible_pins if it["idx"] == self.hovered_pin), None)
            if item is not None:
                cv2.line(overlay, center, (item["sx"], item["sy"]), TEXT_ACCENT, 1, cv2.LINE_AA)

    def draw_panel(self, overlay: np.ndarray, canvas: np.ndarray):
        if self.panel_progress <= 0.01 or self.display_pin is None:
            return

        pin = self.pins[self.display_pin]
        signal, coverage, stability = location_metrics(pin, self.display_pin)
        sx, sy, _ = self.project(pin.lat, pin.lon)
        dx = sx - self.center[0]
        dy = sy - self.center[1]
        norm = math.hypot(dx, dy) + 1e-6
        nx = dx / norm
        ny = dy / norm
        card_w = int(DETAIL_CARD_SIZE[0] * self.panel_progress)
        card_h = int(DETAIL_CARD_SIZE[1] * self.panel_progress)
        offset = int(self.radius * 0.72)
        anchor_x = int(sx + nx * offset)
        anchor_y = int(sy + ny * offset)
        x0 = int(clamp(anchor_x - (30 if nx > 0 else card_w - 30), 36, self.width - card_w - 36))
        y0 = int(clamp(anchor_y - card_h * 0.5, 92, self.height - card_h - 64))
        x1 = x0 + card_w
        y1 = y0 + card_h

        cv2.rectangle(canvas, (x0, y0), (x1, y1), PANEL_COLOR, -1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), HUD_COLOR, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x1, min(y1, y0 + 44)), PANEL_ACCENT, -1)
        cv2.line(overlay, (x0, y0 + 44), (x1, y0 + 44), HUD_COLOR, 1, cv2.LINE_AA)
        cv2.line(overlay, (sx, sy), (anchor_x, anchor_y), TEXT_ACCENT, 1, cv2.LINE_AA)
        cv2.circle(overlay, (anchor_x, anchor_y), 6, TEXT_ACCENT, -1, cv2.LINE_AA)
        cv2.line(overlay, (anchor_x, anchor_y), (x0 if nx > 0 else x1, anchor_y), TEXT_ACCENT, 1, cv2.LINE_AA)
        cv2.line(overlay, (x0 + 18, y1 - 52), (x1 - 18, y1 - 52), (80, 120, 150), 1, cv2.LINE_AA)

        if self.panel_progress < 0.82:
            return

        cv2.putText(overlay, "LOCATION LOCK", (x0 + 18, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(overlay, pin.name.upper(), (x0 + 18, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.86, TEXT_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(overlay, pin.region, (x0 + 18, y0 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_SECONDARY, 1, cv2.LINE_AA)
        coords = f"LAT {pin.lat:+06.2f}   LON {pin.lon:+07.2f}"
        cv2.putText(overlay, coords, (x0 + 18, y0 + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_ACCENT, 1, cv2.LINE_AA)

        text_y = y0 + 174
        for line in wrap_text(pin.detail, 34):
            cv2.putText(overlay, line, (x0 + 18, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_SECONDARY, 1, cv2.LINE_AA)
            text_y += 22

        metric_y = y0 + 200
        self._draw_metric_bar(overlay, x0 + 18, metric_y, "SIGNAL", signal)
        self._draw_metric_bar(overlay, x0 + 18, metric_y + 22, "COVERAGE", coverage)
        self._draw_metric_bar(overlay, x0 + 18, metric_y + 44, "STABILITY", stability)

        cv2.putText(overlay, "THUMB + MIDDLE PINCH TO DISMISS", (x0 + 18, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WARNING_COLOR, 1, cv2.LINE_AA)

    def _draw_metric_bar(self, overlay: np.ndarray, x: int, y: int, label: str, value: int):
        bar_w = 170
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_SECONDARY, 1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x + 96, y - 10), (x + 96 + bar_w, y - 2), (60, 78, 92), 1)
        fill = int(bar_w * clamp(value / 100.0, 0.0, 1.0))
        cv2.rectangle(overlay, (x + 96, y - 10), (x + 96 + fill, y - 2), TEXT_ACCENT, -1)
        cv2.putText(overlay, f"{value:02d}", (x + 96 + bar_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_PRIMARY, 1, cv2.LINE_AA)

    def draw_status(self, overlay: np.ndarray, hand: HandState | None):
        lines = [
            "OPEN HAND DRAG TO ORBIT  |  INDEX + THUMB PINCH TO OPEN  |  MIDDLE + THUMB PINCH TO DISMISS",
        ]
        if hand is None:
            lines.append("HAND STATUS: SEARCHING")
        else:
            mode = "DISMISS" if hand.close_strength >= SECONDARY_PINCH_THRESHOLD else ("SELECT" if hand.pinch_strength >= PINCH_SELECT_THRESHOLD else ("ORBIT" if hand.openness >= OPEN_HAND_ORBIT_THRESHOLD or hand.pinch_strength >= PINCH_ORBIT_THRESHOLD else "TRACK"))
            lines.append(
                f"MODE {mode}   ORBIT OPEN {hand.openness:0.2f}   SELECT {hand.pinch_strength:0.2f}   DISMISS {hand.close_strength:0.2f}"
            )

        y = self.height - 48
        for i, line in enumerate(lines):
            color = TEXT_SECONDARY if i == 0 else TEXT_ACCENT
            (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.putText(overlay, line, (self.center[0] - lw // 2, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def hand_state_from_landmarks(lm, w: int, h: int) -> HandState:
    thumb_tip = point(lm, THUMB_TIP_ID, w, h)
    index_tip = point(lm, INDEX_TIP_ID, w, h)
    middle_tip = point(lm, MIDDLE_TIP_ID, w, h)
    wrist = point(lm, WRIST_ID, w, h)
    middle_mcp = point(lm, MIDDLE_MCP_ID, w, h)
    palm = palm_center(lm, w, h)
    hand_size = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]) + 1e-6

    pinch_center = (
        (thumb_tip[0] + index_tip[0]) * 0.5,
        (thumb_tip[1] + index_tip[1]) * 0.5,
    )
    pinch_distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
    pinch_ratio = pinch_distance / hand_size
    pinch_strength = 1.0 - clamp(
        (pinch_ratio - PINCH_CLOSED_RATIO) / (PINCH_OPEN_RATIO - PINCH_CLOSED_RATIO + 1e-6),
        0.0,
        1.0,
    )
    close_distance = math.hypot(middle_tip[0] - thumb_tip[0], middle_tip[1] - thumb_tip[1])
    close_ratio = close_distance / hand_size
    close_strength = 1.0 - clamp(
        (close_ratio - SECONDARY_PINCH_CLOSED_RATIO) / (SECONDARY_PINCH_OPEN_RATIO - SECONDARY_PINCH_CLOSED_RATIO + 1e-6),
        0.0,
        1.0,
    )
    span_strength = smoothstep(PINCH_CLOSED_RATIO, PINCH_OPEN_RATIO, pinch_ratio)
    avg_tip_distance = sum(
        math.hypot(point(lm, idx, w, h)[0] - palm[0], point(lm, idx, w, h)[1] - palm[1])
        for idx in FINGERTIP_IDS
    ) / len(FINGERTIP_IDS)
    openness_ratio = avg_tip_distance / hand_size
    openness = smoothstep(OPEN_HAND_RATIO_MIN, OPEN_HAND_RATIO_MAX, openness_ratio)

    return HandState(
        palm=palm,
        thumb_tip=thumb_tip,
        index_tip=index_tip,
        middle_tip=middle_tip,
        pinch_center=pinch_center,
        hand_size=hand_size,
        pinch_ratio=pinch_ratio,
        pinch_strength=pinch_strength,
        span_strength=span_strength,
        openness=openness,
        close_strength=close_strength,
    )


def apply_glow(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    glow = cv2.GaussianBlur(overlay, GLOW_KERNEL_SIZE, 0)
    blended = cv2.addWeighted(frame, 1.0, glow, GLOW_BLEND_ALPHA, 0)
    return cv2.add(blended, overlay // 2)


def main():
    parser = argparse.ArgumentParser(description="Holographic globe interaction")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model_path = locate_hand_model()
    if model_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        print("Error: could not find hand_landmarker.task.")
        print("Checked:")
        print(os.path.abspath(os.path.join(here, "models", "hand_landmarker.task")))
        print(os.path.abspath(os.path.join(here, "..", "radiance", "models", "hand_landmarker.task")))
        print(os.path.abspath(os.path.join(here, "..", "aura_effects", "models", "hand_landmarker.task")))
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    hands = HandLandmarker.create_from_options(options)
    selfie = create_selfie_segmenter()
    globe = None
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    t0 = time.time()
    prev_t = t0

    print("Press 'q' to quit. Open hand drags orbit. Thumb+index pinch opens focus. Thumb+middle pinch dismisses focus.")
    if selfie is None:
        print("Selfie segmentation is unavailable in this MediaPipe build. Using raw portrait blend fallback.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if globe is None or globe.width != w or globe.height != h:
            globe = HoloGlobe(w, h)

        now_abs = time.time()
        t = now_abs - t0
        dt = min(now_abs - prev_t, 0.05)
        prev_t = now_abs

        base = (frame.astype(np.float32) * DARKEN_FACTOR).astype(np.uint8)
        canvas = base.copy()
        overlay = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hands.detect_for_video(mp_image, int(t * 1000))
        seg_mask = None
        if selfie is not None:
            seg_result = selfie.process(rgb)
            seg_mask = seg_result.segmentation_mask if seg_result is not None else None

        hand_state = None
        if result.hand_landmarks:
            hand_state = hand_state_from_landmarks(result.hand_landmarks[0], w, h)

        globe.update_hand_presence(hand_state)
        globe.update_visible_pins(t, now_abs)
        globe.update_interaction(hand_state, dt, now_abs)

        globe.draw_background(canvas, t)
        globe.draw_presence_layer(canvas, frame, seg_mask)
        globe.draw_data_arcs(overlay, t)
        globe.draw_globe(overlay, t)
        globe.draw_hud(overlay, t)
        globe.draw_pins(overlay, t)
        globe.draw_cursor(overlay, hand_state)
        globe.draw_panel(overlay, canvas)
        globe.draw_status(overlay, hand_state)

        output = apply_glow(canvas, overlay)
        cv2.imshow(WINDOW_TITLE, output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if selfie is not None:
        selfie.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Finger aura effect configuration — Marvel-style hand effects.
"""

import numpy as np


# ─── Fingertip trail particles ───────────────────────────────────────────────

MAX_PARTICLES = 8000
TRAIL_LIFETIME = (0.5, 1.8)        # seconds
TRAIL_DRIFT_SPEED = 25.0           # px/s outward drift
TRAIL_NOISE = 15.0                 # px/s random lateral wiggle
TRAIL_SPAWN_PER_TIP = 12           # particles per fingertip per frame
TRAIL_DOT_SIZE = (1, 2)
TRAIL_CHAR_DENSITY = 0.03          # fraction of particles that are digits

# Default fingertip trail color (BGR)
TRAIL_COLOR = (80, 255, 80)        # green

# ─── Gesture detection ───────────────────────────────────────────────────────

ROTATION_WINDOW = 15               # frames to track for rotation detection
ROTATION_THRESHOLD = 0.35          # radians — minimum cumulative rotation to trigger
ROTATION_COOLDOWN = 0.5            # seconds between gesture triggers

# ─── Right-turn effect: Energy spiral (Doctor Strange portal style) ──────────

SPIRAL_COLOR = (50, 180, 255)      # orange
SPIRAL_SPOKES = 32
SPIRAL_INNER_R = 20
SPIRAL_OUTER_R = 100
SPIRAL_DURATION = 2.0              # seconds the spiral stays active
SPIRAL_RING_COLORS = [
    (0, 200, 255),
    (0, 255, 200),
    (50, 180, 255),
    (100, 255, 255),
]

# ─── Left-turn effect: Energy burst / shockwave ─────────────────────────────

BURST_COLOR = (255, 100, 200)      # magenta/pink
BURST_DURATION = 1.5               # seconds
BURST_MAX_RADIUS = 160             # max expansion radius
BURST_RING_COUNT = 3               # concentric rings
BURST_PARTICLE_COUNT = 200         # particles emitted per burst

# ─── Character sets ──────────────────────────────────────────────────────────

CHAR_SETS = {
    "digits":  list("0123456789"),
    "symbols": list("⟡⊛✦)}>⊙◉"),
    "mixed":   list("0123456789✦⟡)}>"),
}
CHAR_SET_DEFAULT = "mixed"
CHAR_FONT_SCALE = 0.28
CHAR_FONT_THICKNESS = 1

# ─── Glow / blur ─────────────────────────────────────────────────────────────

GLOW_KERNEL_SIZE = (21, 21)
GLOW_BLEND_ALPHA = 0.75

# ─── Background darkening ────────────────────────────────────────────────────

DARKEN_FACTOR = 0.40

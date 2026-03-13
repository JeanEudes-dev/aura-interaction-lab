"""
Radiance — configuration for gesture-driven geometric mandala.
"""

import math

# ─── Hand tracking ────────────────────────────────────────────────────────────

FINGERTIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
WRIST_ID = 0
MID_BASE_ID = 9                       # middle-finger MCP

# ─── Core sizing ──────────────────────────────────────────────────────────────
# The mandala fills a huge area — max radius almost half the screen width
MIN_RADIUS = 80
MAX_RADIUS = 320

# ─── Radial bars (tight compact color wheel at the core) ─────────────────────

NUM_BARS = 36
BAR_INNER_RATIO = 0.06                # very close to center
BAR_OUTER_RATIO = 0.28                # short — stays compact
BAR_WIDTH = 4

# Rainbow palette (BGR) — 36 entries for smooth gradient
BAR_COLORS = [
    (30, 30, 220),   (0, 40, 240),   (0, 70, 255),   (0, 110, 255),
    (0, 160, 255),   (0, 200, 255),  (0, 240, 255),  (0, 255, 220),
    (0, 255, 180),   (0, 255, 130),  (0, 255, 80),   (0, 255, 30),
    (30, 255, 0),    (80, 255, 0),   (140, 255, 0),  (200, 255, 0),
    (255, 255, 0),   (255, 230, 0),  (255, 200, 0),  (255, 170, 0),
    (255, 130, 0),   (255, 90, 0),   (255, 50, 0),   (255, 10, 0),
    (255, 0, 40),    (255, 0, 90),   (255, 0, 150),  (255, 0, 210),
    (255, 0, 255),   (210, 0, 255),  (160, 0, 255),  (110, 0, 255),
    (60, 0, 255),    (40, 20, 255),  (60, 50, 255),  (80, 70, 240),
]

# ─── Concentric geometric rings (many, dense, varied) ────────────────────────

# (radius_ratio, type, color_bgr, count)
# types: "dots", "ticks", "dashes", "circle"
RINGS = [
    # inner structure
    (0.15, "circle",  (60, 60, 60),     0),
    (0.22, "ticks",   (100, 160, 100),  36),
    (0.30, "circle",  (70, 70, 70),     0),
    (0.35, "dots",    (100, 180, 100),  48),
    (0.42, "dashes",  (130, 130, 180),  24),
    (0.48, "circle",  (60, 70, 80),     0),
    # mid structure
    (0.55, "ticks",   (120, 120, 160),  54),
    (0.62, "dots",    (140, 180, 120),  72),
    (0.70, "circle",  (80, 80, 80),     0),
    (0.78, "dashes",  (160, 120, 140),  32),
    (0.85, "dots",    (160, 140, 100),  80),
    (0.92, "circle",  (90, 90, 90),     0),
    # outer structure
    (1.00, "ticks",   (130, 130, 170),  72),
    (1.08, "dots",    (100, 160, 180),  96),
    (1.15, "circle",  (70, 70, 90),     0),
    (1.25, "dashes",  (140, 100, 160),  40),
    (1.35, "dots",    (180, 130, 130),  108),
    (1.42, "circle",  (60, 60, 80),     0),
    (1.50, "ticks",   (100, 120, 140),  80),
    (1.60, "dots",    (120, 140, 100),  120),
]

# ─── Colored dot arcs (partial arcs of colored dots, like the reference) ─────

# (radius_ratio, color_bgr, dot_count, arc_span_deg, dot_radius, spin_speed)
COLORED_ARCS = [
    # inner colored arcs
    (0.48, (60, 60, 220),    20, 120, 2, 0.15),   # red
    (0.55, (0, 130, 255),    24, 140, 2, -0.10),   # orange
    (0.65, (0, 220, 255),    18, 100, 3, 0.08),    # yellow
    # mid colored arcs
    (0.80, (0, 200, 100),    26, 150, 2, -0.12),   # green
    (0.88, (200, 200, 0),    22, 130, 2, 0.09),    # cyan
    (0.95, (200, 100, 60),   20, 110, 3, -0.07),   # blue-ish
    # outer colored arcs — big sweeping dotted curves
    (1.12, (80, 80, 255),    30, 160, 3, 0.06),    # red-pink
    (1.22, (0, 180, 255),    28, 140, 2, -0.11),   # orange
    (1.32, (100, 255, 200),  32, 170, 2, 0.13),    # green-yellow
    (1.45, (220, 180, 60),   24, 120, 3, -0.08),   # cyan-blue
    (1.55, (180, 80, 200),   26, 150, 2, 0.10),    # purple
    (1.65, (60, 100, 255),   30, 180, 2, -0.05),   # deep red
]

# ─── Roman numeral clock ring ────────────────────────────────────────────────

CLOCK_RATIO = 0.75                    # radius for the clock face
CLOCK_NUMERALS = ["XII", "I", "II", "III", "IV", "V",
                  "VI", "VII", "VIII", "IX", "X", "XI"]
CLOCK_COLOR = (200, 200, 220)
CLOCK_FONT_SCALE = 0.55
CLOCK_SPIN_SPEED = 0.04               # slow rotation

# ─── Symbol / number rings (more of them, varied) ────────────────────────────

# (radius_ratio, characters, count, color, font_scale, spin_speed)
SYMBOL_RINGS = [
    (0.32, list("0123456789"),         16, (130, 200, 130), 0.28, 0.10),
    (0.52, list("(){}[]<>"),           12, (170, 170, 210), 0.30, -0.07),
    (0.72, list("0123456789"),         20, (180, 160, 140), 0.26, 0.12),
    (1.05, list("(){}[]<>"),           16, (140, 160, 200), 0.32, -0.09),
    (1.28, list("0123456789"),         24, (160, 130, 170), 0.24, 0.06),
    (1.48, list(".|:;.,:|"),           28, (110, 110, 130), 0.22, -0.04),
]

# ─── Scattered floating symbols (jittered, not on perfect rings) ─────────────

SCATTER_COUNT = 60                    # how many scattered symbols
SCATTER_RADIUS_RANGE = (0.35, 1.70)   # min/max ratio from center
SCATTER_CHARS = list("0123456789") + list("(){}[]<>") + ["XI", "IV", "00"]
SCATTER_COLORS = [
    (120, 200, 120), (180, 180, 220), (200, 200, 200),
    (160, 140, 180), (100, 180, 180), (200, 160, 100),
]
SCATTER_FONT_RANGE = (0.28, 0.50)     # random font sizes
SCATTER_DRIFT_SPEED = 0.03            # how fast they slowly orbit

# ─── Bracket clusters (groups of ))  (( scattered around) ────────────────────

BRACKET_CLUSTERS = [
    # (radius_ratio, angle_offset, text, color, font_scale, spin_speed)
    (0.45, 0.0,   ")))",   (160, 180, 200), 0.35, 0.06),
    (0.60, 1.2,   "(((",   (140, 200, 160), 0.30, -0.08),
    (0.82, 2.5,   "}}",    (200, 160, 140), 0.32, 0.05),
    (1.00, 0.8,   "{{",    (160, 160, 200), 0.28, -0.10),
    (1.20, 3.8,   "]]",    (180, 140, 180), 0.30, 0.07),
    (1.35, 5.0,   "[[",    (140, 180, 140), 0.28, -0.06),
    (1.50, 1.5,   ")))",   (200, 180, 160), 0.35, 0.04),
    (1.62, 4.2,   "(((",   (160, 200, 180), 0.32, -0.09),
]

# ─── Arc connectors (curved lines between ring layers) ───────────────────────

# (inner_ratio, outer_ratio, arc_count, color, thickness)
ARCS = [
    (0.30, 0.48,  6,  (70, 110, 70),   1),
    (0.50, 0.70,  8,  (80, 80, 120),   1),
    (0.72, 0.92,  6,  (110, 70, 100),  1),
    (0.95, 1.15, 10,  (70, 90, 110),   1),
    (1.18, 1.40,  8,  (100, 70, 110),  1),
    (1.42, 1.62,  6,  (80, 100, 80),   1),
]

# ─── Dotted trail arcs (curved outward comet tails) ──────────────────────────

# (start_ratio, end_ratio, spiral_offset, dot_count, color, dot_size, spin_speed)
TRAIL_ARCS = [
    (0.40, 1.20, 0.8,  40, (100, 180, 100), 2, 0.05),
    (0.50, 1.40, 1.5,  50, (180, 140, 100), 2, -0.07),
    (0.35, 1.10, 2.2,  35, (100, 140, 200), 2, 0.06),
    (0.55, 1.50, 3.0,  45, (180, 100, 160), 2, -0.04),
    (0.45, 1.30, 3.8,  40, (140, 180, 120), 2, 0.08),
    (0.60, 1.60, 4.5,  55, (120, 120, 200), 1, -0.06),
    (0.38, 1.25, 5.2,  38, (200, 160, 100), 2, 0.03),
    (0.50, 1.55, 5.9,  48, (100, 200, 180), 2, -0.05),
]

# ─── Spiral arms (when fingers pinch close) ──────────────────────────────────

SPIRAL_ARM_COUNT = 6
SPIRAL_TURNS = 2.5
SPIRAL_DOT_COUNT = 140
SPIRAL_DOT_SIZE = 1

# ─── Floating particles ──────────────────────────────────────────────────────

MAX_PARTICLES = 4000
PARTICLE_SPAWN_RATE = 50
PARTICLE_LIFETIME = (0.6, 3.0)
PARTICLE_SPEED = 20.0
PARTICLE_SIZE = (1, 3)

# ─── Spread thresholds (normalised to hand size) ─────────────────────────────

SPREAD_MIN = 0.15
SPREAD_MAX = 0.65

# ─── Glow ─────────────────────────────────────────────────────────────────────

GLOW_KERNEL_SIZE = (25, 25)
GLOW_BLEND_ALPHA = 0.70

# ─── Background ──────────────────────────────────────────────────────────────

DARKEN_FACTOR = 0.20

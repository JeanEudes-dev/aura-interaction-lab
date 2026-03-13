"""
Holo Globe configuration.
"""

WINDOW_TITLE = "Holo Globe"

# Hand landmarks
WRIST_ID = 0
THUMB_TIP_ID = 4
INDEX_MCP_ID = 5
INDEX_TIP_ID = 8
MIDDLE_MCP_ID = 9
MIDDLE_TIP_ID = 12
RING_TIP_ID = 16
PINKY_TIP_ID = 20
FINGERTIP_IDS = [THUMB_TIP_ID, INDEX_TIP_ID, MIDDLE_TIP_ID, RING_TIP_ID, PINKY_TIP_ID]

# Camera / detection
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 1

# Globe placement
GLOBE_CENTER_RATIO = (0.54, 0.52)
BASE_RADIUS = 180
MIN_ZOOM = 0.72
MAX_ZOOM = 1.65
ZOOM_DELTA_SENSITIVITY = 1.8
ZOOM_SMOOTHING = 0.16

# Gesture thresholds are normalised to hand size
PINCH_CLOSED_RATIO = 0.18
PINCH_OPEN_RATIO = 0.70
PINCH_ORBIT_THRESHOLD = 0.58
PINCH_SELECT_THRESHOLD = 0.84
PINCH_RELEASE_THRESHOLD = 0.34
CURSOR_HIT_RADIUS = 28
ORBIT_RADIUS_FACTOR = 1.28
ROTATION_SENSITIVITY = 0.010
ZOOM_DEADZONE = 0.01
SELECT_COOLDOWN_SECONDS = 0.30
OPEN_HAND_RATIO_MIN = 1.30
OPEN_HAND_RATIO_MAX = 2.35
OPEN_HAND_ORBIT_THRESHOLD = 0.52
OPENNESS_ZOOM_SENSITIVITY = 1.45
OPENNESS_ZOOM_DEADZONE = 0.02
HOVER_LOCK_SECONDS = 0.12
HOVER_STICKY_RADIUS = 54
SECONDARY_PINCH_CLOSED_RATIO = 0.20
SECONDARY_PINCH_OPEN_RATIO = 0.70
SECONDARY_PINCH_THRESHOLD = 0.82
SECONDARY_PINCH_RELEASE = 0.32

# Visuals
DARKEN_FACTOR = 0.16
GLOW_KERNEL_SIZE = (41, 41)
GLOW_BLEND_ALPHA = 0.92
BACKGROUND_GRID_STEP = 36
BACKGROUND_STAR_COUNT = 120
HUD_RING_SPEED = 0.52
HUD_RING_COUNT = 6
GLOBE_LAT_STEPS = 11
GLOBE_LON_STEPS = 22
GLOBE_SEGMENTS = 144
ATMOSPHERE_LAYERS = 8
SCANLINE_SPEED = 0.52
DATA_ARC_POINT_COUNT = 72
DATA_ARC_LIFT = 0.34

GLOBE_CORE_COLOR = (255, 245, 180)
GLOBE_LINE_COLOR = (255, 190, 70)
GLOBE_BACK_LINE_COLOR = (90, 70, 30)
HUD_COLOR = (255, 220, 120)
PANEL_COLOR = (18, 28, 40)
PANEL_ACCENT = (34, 60, 78)
TEXT_PRIMARY = (255, 244, 220)
TEXT_SECONDARY = (185, 198, 214)
TEXT_ACCENT = (120, 235, 255)
CURSOR_COLOR = (130, 245, 255)
WARNING_COLOR = (90, 160, 255)
PORTRAIT_TINT = (34, 72, 120)
PORTRAIT_ALPHA = 0.88
PORTRAIT_FADE_WIDTH = 0.52
DETAIL_CARD_SIZE = (360, 286)

LOCATION_PINS = [
    {
        "name": "Bucharest",
        "region": "Romania",
        "lat": 44.4268,
        "lon": 26.1025,
        "color": (255, 210, 120),
        "detail": (
            "Command hub with dense sensor coverage, urban energy patterns, "
            "and real-time public space telemetry."
        ),
    },
    {
        "name": "Reykjavik",
        "region": "Iceland",
        "lat": 64.1466,
        "lon": -21.9426,
        "color": (255, 170, 90),
        "detail": (
            "High-latitude node suited for aurora-linked visual overlays, "
            "volcanic terrain mapping, and extreme atmosphere studies."
        ),
    },
    {
        "name": "Tokyo",
        "region": "Japan",
        "lat": 35.6764,
        "lon": 139.6500,
        "color": (255, 120, 120),
        "detail": (
            "Megacity interaction target for transit density, gesture-driven "
            "navigation layers, and high-frequency motion visualization."
        ),
    },
    {
        "name": "Nairobi",
        "region": "Kenya",
        "lat": -1.2921,
        "lon": 36.8219,
        "color": (120, 255, 190),
        "detail": (
            "East Africa relay for wildlife corridors, climate-linked motion "
            "visuals, and conservation data storytelling."
        ),
    },
    {
        "name": "Sao Paulo",
        "region": "Brazil",
        "lat": -23.5505,
        "lon": -46.6333,
        "color": (110, 220, 255),
        "detail": (
            "Expanded southern hemisphere cluster tracking mobility, cultural "
            "events, and layered urban pulse analysis."
        ),
    },
    {
        "name": "San Francisco",
        "region": "USA",
        "lat": 37.7749,
        "lon": -122.4194,
        "color": (180, 170, 255),
        "detail": (
            "Prototype launch zone for XR systems, satellite-linked interaction "
            "design, and coastal signal experiments."
        ),
    },
]

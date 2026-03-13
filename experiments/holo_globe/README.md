# Holo Globe

Gesture-controlled holographic globe with selectable location nodes.

## What it does

- Tracks one hand with **MediaPipe Hands**
- Uses a **thumb-index pinch midpoint** as a precision cursor
- Uses an **open-hand drag** or **light pinch drag** to rotate the globe
- Uses **hand openness** or **pinch span changes** while orbiting to zoom in and out
- Uses an **index-thumb pinch** to open a location and a **middle-thumb pinch** to dismiss it
- Presents a **fullscreen blended composition** where the user remains visible in the scene on the left and the globe stays central
- Adds denser HUD rings, scan lines, data arcs, atmosphere glow, and telemetry bars for a more polished Iron Man-style feel

## Run

```bash
pip install -r requirements.txt
python experiments/holo_globe/holo_globe.py
```

Optional camera selection:

```bash
python experiments/holo_globe/holo_globe.py --camera 1
```

## Controls

- Aim with the midpoint between your thumb and index finger
- Use an open hand over the globe to drag and rotate it
- You can also use a light pinch and drag over the globe to rotate it
- Open or close your hand while orbiting to zoom, or vary the pinch span during a light pinch orbit
- Hover the pinch cursor over a node
- Use an index-thumb pinch to open the hovered node
- Use a middle-thumb pinch to dismiss the active location
- Press `q` to quit

## Notes

`holo_globe.py` reuses the hand landmarker model already present in sibling experiments. If that model is missing, the script will print the expected path candidates and exit.

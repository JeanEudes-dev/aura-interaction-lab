# Holo Globe Web

Web-based fullscreen holographic globe experiment using **CesiumJS** and **MediaPipe Hand Landmarker**.

## What this version does

- Runs in the browser with a real 3D globe
- Keeps the user visually present on the left through a blended live camera layer
- Uses a dedicated **three-finger pinch gesture** to rotate the globe
- Uses **index + thumb pinch** to focus a place while held
- Uses **middle + thumb pinch** to lock or unlock the active place
- Uses a **closed fist** to reset to the globe overview
- Shows floating location cards anchored to the selected city

## Setup

```bash
cd web-demo/holo_globe_web
npm install
npm run dev
```

Then open the local Vite URL, usually:

```text
http://localhost:5173
```

## Build

```bash
npm run build
```

## Notes

- The MediaPipe hand model is expected at `public/models/hand_landmarker.task`.
- Cesium static assets are copied during build using `vite-plugin-static-copy`.
- Terrain is requested at runtime when available, but the app still runs without it.

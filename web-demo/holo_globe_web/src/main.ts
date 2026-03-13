import "./style.css";

import {
  Cartesian2,
  Cartesian3,
  Color,
  ConstantProperty,
  Entity,
  LabelStyle,
  Math as CesiumMath,
  OpenStreetMapImageryProvider,
  SceneMode,
  SceneTransforms,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  Viewer,
  createWorldTerrainAsync,
} from "cesium";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { PLACES, type PlaceRecord } from "./places";

type GestureState = {
  cursorX: number;
  cursorY: number;
  orbitX: number;
  orbitY: number;
  orbitStrength: number;
  selectStrength: number;
  lockStrength: number;
  resetStrength: number;
  isOrbiting: boolean;
  selectTrigger: boolean;
  lockTrigger: boolean;
  resetTrigger: boolean;
  debugText: string;
};

const CAMERA_WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";
const HAND_MODEL_PATH = "/models/hand_landmarker.task";
const PLACE_PICK_RADIUS = 58;
const PLACE_PICK_STICKY_RADIUS = 84;
const SELECT_HOLD_MS = 140;
const LOCK_HOLD_MS = 180;
const OVERVIEW_DESTINATION = Cartesian3.fromDegrees(10, 22, 18_000_000);
const FOCUS_ALTITUDE_FAR = 5_200_000;
const FOCUS_ALTITUDE_NEAR = 1_850_000;

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

const cameraVideo = requireElement<HTMLVideoElement>("#camera-feed");
const cesiumRoot = requireElement<HTMLDivElement>("#cesium-root");
const reticle = requireElement<HTMLDivElement>("#reticle");
const placeCard = requireElement<HTMLDivElement>("#place-card");
const placeTitle = requireElement<HTMLHeadingElement>("#place-title");
const placeRegion = requireElement<HTMLDivElement>("#place-region");
const placeCoords = requireElement<HTMLDivElement>("#place-coords");
const placeDescription = requireElement<HTMLParagraphElement>("#place-description");
const placeSignal = requireElement<HTMLSpanElement>("#place-signal");
const placeCoverage = requireElement<HTMLSpanElement>("#place-coverage");
const placeStability = requireElement<HTMLSpanElement>("#place-stability");
const orbitMeter = requireElement<HTMLDivElement>("#orbit-meter");
const selectMeter = requireElement<HTMLDivElement>("#select-meter");
const dismissMeter = requireElement<HTMLDivElement>("#dismiss-meter");
const resetMeter = requireElement<HTMLDivElement>("#reset-meter");
const gestureText = requireElement<HTMLDivElement>("#gesture-text");
const trackingPill = requireElement<HTMLSpanElement>("#tracking-pill");

const viewer = new Viewer(cesiumRoot, {
  animation: false,
  timeline: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  baseLayerPicker: false,
  fullscreenButton: false,
  infoBox: false,
  selectionIndicator: false,
  requestRenderMode: true,
  sceneMode: SceneMode.SCENE3D,
});

viewer.imageryLayers.removeAll();
viewer.imageryLayers.addImageryProvider(
  new OpenStreetMapImageryProvider({
    url: "https://tile.openstreetmap.org/",
  }),
);
(viewer.cesiumWidget.creditContainer as HTMLElement).style.display = "none";
viewer.scene.screenSpaceCameraController.enableRotate = false;
viewer.scene.screenSpaceCameraController.enableZoom = false;
viewer.scene.screenSpaceCameraController.enableTilt = false;
viewer.scene.screenSpaceCameraController.enableLook = false;
viewer.scene.screenSpaceCameraController.enableTranslate = false;
viewer.scene.globe.enableLighting = true;
if (viewer.scene.skyAtmosphere) {
  viewer.scene.skyAtmosphere.show = true;
}
viewer.scene.globe.showGroundAtmosphere = true;
viewer.scene.requestRender();

createWorldTerrainAsync()
  .then((terrain) => {
    viewer.scene.terrainProvider = terrain;
    viewer.scene.requestRender();
  })
  .catch(() => {
    // Terrain is optional for the first experiment.
  });

const placeEntityMap = new Map<string, Entity>();
const placeMetaMap = new Map<string, PlaceRecord & ReturnType<typeof computeMetrics>>();

function computeMetrics(place: PlaceRecord, index: number) {
  return {
    signal: 62 + ((index * 9) % 31),
    coverage: 48 + (Math.round(Math.abs(place.lat)) % 43),
    stability: 54 + (Math.round(Math.abs(place.lon)) % 39),
  };
}

PLACES.forEach((place, index) => {
  const entity = viewer.entities.add({
    id: place.id,
    name: place.name,
    position: Cartesian3.fromDegrees(place.lon, place.lat, 160000),
    point: {
      pixelSize: 14,
      color: Color.fromCssColorString(place.accent),
      outlineColor: Color.WHITE,
      outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text: place.name.toUpperCase(),
      font: '600 14px "Space Grotesk"',
      fillColor: Color.fromCssColorString("#f7ead1"),
      outlineColor: Color.fromCssColorString("#09111e"),
      outlineWidth: 3,
      style: LabelStyle.FILL_AND_OUTLINE,
      pixelOffset: new Cartesian2(0, -26),
      showBackground: true,
      backgroundColor: Color.fromCssColorString("rgba(9, 15, 23, 0.62)"),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  });
  placeEntityMap.set(place.id, entity);
  placeMetaMap.set(place.id, { ...place, ...computeMetrics(place, index) });
});

viewer.camera.flyTo({
  destination: OVERVIEW_DESTINATION,
  duration: 0,
});

let handLandmarker: HandLandmarker | null = null;
let activePlace: Entity | null = null;
let hoveredPlace: Entity | null = null;
let lastOrbitSample: { x: number; y: number } | null = null;
let lastGestureTime = 0;
let selectLatch = false;
let lockLatch = false;
let resetLatch = false;
let hoveredPlaceCandidate: Entity | null = null;
let hoveredPlaceSince = 0;
let lockHoldSince = 0;
let placePinned = false;
let focusGestureActive = false;

const gestureState: GestureState = {
  cursorX: window.innerWidth * 0.5,
  cursorY: window.innerHeight * 0.5,
  orbitX: window.innerWidth * 0.5,
  orbitY: window.innerHeight * 0.5,
  orbitStrength: 0,
  selectStrength: 0,
  lockStrength: 0,
  resetStrength: 0,
  isOrbiting: false,
  selectTrigger: false,
  lockTrigger: false,
  resetTrigger: false,
  debugText: "Awaiting hand feed",
};

function setMeter(element: HTMLDivElement, amount: number) {
  element.style.width = `${Math.max(0, Math.min(1, amount)) * 100}%`;
}

function setHoveredPlace(next: Entity | null) {
  if (hoveredPlace === next) {
    return;
  }

  if (hoveredPlace?.point) {
    hoveredPlace.point.pixelSize = new ConstantProperty(14);
  }
  hoveredPlace = next;
  if (hoveredPlace?.point) {
    hoveredPlace.point.pixelSize = new ConstantProperty(18);
  }
  viewer.scene.requestRender();
}

function flyToOverview(duration = 0.85) {
  viewer.camera.flyTo({
    destination: OVERVIEW_DESTINATION,
    duration,
  });
}

function focusPlace(entity: Entity, amount = 0.78, duration = 0) {
  const meta = placeMetaMap.get(String(entity.id));
  if (!meta) {
    return;
  }
  const altitude = FOCUS_ALTITUDE_FAR + (FOCUS_ALTITUDE_NEAR - FOCUS_ALTITUDE_FAR) * amount;
  if (duration > 0) {
    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(meta.lon, meta.lat, altitude),
      duration,
    });
  } else {
    viewer.camera.setView({
      destination: Cartesian3.fromDegrees(meta.lon, meta.lat, altitude),
    });
  }
}

function setActivePlace(next: Entity | null) {
  activePlace = next;
  placeCard.classList.toggle("hidden", !activePlace);

  for (const entity of placeEntityMap.values()) {
    if (entity.point) {
      entity.point.outlineWidth = new ConstantProperty(entity === activePlace ? 4 : 2);
      entity.point.pixelSize = new ConstantProperty(entity === activePlace ? 20 : entity === hoveredPlace ? 18 : 14);
    }
  }

  if (!activePlace) {
    placePinned = false;
    viewer.scene.requestRender();
    return;
  }

  const meta = placeMetaMap.get(String(activePlace.id));
  if (!meta) {
    return;
  }
  placeTitle.textContent = activePlace.name ?? "";
  placeRegion.textContent = meta.region;
  const position = activePlace.position?.getValue(viewer.clock.currentTime);
  if (position) {
    const cartographic = viewer.scene.globe.ellipsoid.cartesianToCartographic(position);
    placeCoords.textContent = `LAT ${CesiumMath.toDegrees(cartographic.latitude).toFixed(2)}  LON ${CesiumMath.toDegrees(cartographic.longitude).toFixed(2)}`;
  }
  placeDescription.textContent = meta.description;
  placeSignal.textContent = String(meta.signal);
  placeCoverage.textContent = String(meta.coverage);
  placeStability.textContent = String(meta.stability);
  viewer.scene.requestRender();
}

function updatePlaceCardPosition() {
  if (!activePlace || !activePlace.position) {
    return;
  }
  const position = activePlace.position.getValue(viewer.clock.currentTime);
  if (!position) {
    return;
  }

  const screen = SceneTransforms.worldToWindowCoordinates(viewer.scene, position);
  if (!screen) {
    placeCard.classList.add("hidden");
    return;
  }

  placeCard.classList.remove("hidden");
  const cardWidth = placeCard.offsetWidth || 360;
  const cardHeight = placeCard.offsetHeight || 250;
  const x = Math.min(window.innerWidth - cardWidth - 24, Math.max(24, screen.x + 36));
  const y = Math.min(window.innerHeight - cardHeight - 90, Math.max(100, screen.y - cardHeight * 0.5));
  placeCard.style.left = `${x}px`;
  placeCard.style.top = `${y}px`;
}

function pickPlaceAtScreen(x: number, y: number): Entity | null {
  let bestEntity: Entity | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const entity of placeEntityMap.values()) {
    const position = entity.position?.getValue(viewer.clock.currentTime);
    if (!position) {
      continue;
    }

    const screen = SceneTransforms.worldToWindowCoordinates(viewer.scene, position);
    if (!screen) {
      continue;
    }

    const radius = entity === hoveredPlace || entity === activePlace ? PLACE_PICK_STICKY_RADIUS : PLACE_PICK_RADIUS;
    const distance = Math.hypot(screen.x - x, screen.y - y);
    if (distance <= radius && distance < bestDistance) {
      bestDistance = distance;
      bestEntity = entity;
    }
  }

  return bestEntity;
}

function handleGestureInteraction(state: GestureState) {
  const now = performance.now();
  const orbitScale = state.isOrbiting ? 0.0032 : 0;

  if (state.isOrbiting) {
    if (lastOrbitSample) {
      const dx = state.orbitX - lastOrbitSample.x;
      const dy = state.orbitY - lastOrbitSample.y;
      viewer.camera.rotateRight(dx * orbitScale);
      viewer.camera.rotateUp(dy * orbitScale * 0.9);
    }
    lastOrbitSample = { x: state.orbitX, y: state.orbitY };
  } else {
    lastOrbitSample = null;
  }

  const hovered = pickPlaceAtScreen(state.cursorX, state.cursorY);
  setHoveredPlace(hovered);

  if (hovered !== hoveredPlaceCandidate) {
    hoveredPlaceCandidate = hovered;
    hoveredPlaceSince = now;
  }

  const selectHoldReady =
    hovered !== null &&
    !state.isOrbiting &&
    state.selectStrength > 0.72 &&
    now - hoveredPlaceSince >= SELECT_HOLD_MS;

  const selectEngaged = hovered !== null && !state.isOrbiting && state.selectStrength > 0.58;
  if (selectEngaged && hovered) {
    focusGestureActive = true;
    if (activePlace !== hovered) {
      setActivePlace(hovered);
      placePinned = false;
    }
    focusPlace(hovered, smoothstep(0.58, 0.98, state.selectStrength));
  } else if (focusGestureActive && !placePinned && state.selectStrength < 0.24) {
    focusGestureActive = false;
    setActivePlace(null);
    flyToOverview();
  } else if (!selectEngaged && state.selectStrength < 0.24) {
    focusGestureActive = false;
  }

  if ((state.selectTrigger || selectHoldReady) && hovered && now - lastGestureTime > 260) {
    setActivePlace(hovered);
    focusPlace(hovered, smoothstep(0.58, 0.98, Math.max(state.selectStrength, 0.72)), 0.42);
    lastGestureTime = now;
    hoveredPlaceSince = now;
  }

  if (!state.isOrbiting && state.lockStrength > 0.84) {
    if (lockHoldSince === 0) {
      lockHoldSince = now;
    }
  } else {
    lockHoldSince = 0;
  }

  const lockHoldReady = lockHoldSince > 0 && now - lockHoldSince >= LOCK_HOLD_MS;
  if (!state.isOrbiting && (state.lockTrigger || lockHoldReady) && activePlace && now - lastGestureTime > 260) {
    placePinned = !placePinned;
    if (!placePinned && state.selectStrength < 0.4) {
      setActivePlace(null);
      flyToOverview(0.65);
    }
    lastGestureTime = now;
    lockHoldSince = 0;
  }

  if (!state.isOrbiting && state.resetTrigger && now - lastGestureTime > 260) {
    focusGestureActive = false;
    setActivePlace(null);
    flyToOverview(0.8);
    lastGestureTime = now;
  }

  reticle.classList.remove("hidden");
  reticle.style.left = `${state.cursorX}px`;
  reticle.style.top = `${state.cursorY}px`;
  setMeter(orbitMeter, state.orbitStrength);
  setMeter(selectMeter, state.selectStrength);
  setMeter(dismissMeter, state.lockStrength);
  setMeter(resetMeter, state.resetStrength);
  gestureText.textContent = state.debugText;
  trackingPill.textContent = "Hand tracking live";
  trackingPill.classList.add("live");
  viewer.scene.requestRender();
}

function handDistance(a: { x: number; y: number }, b: { x: number; y: number }) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function averagePoint(points: Array<{ x: number; y: number }>) {
  const total = points.reduce(
    (acc, point) => {
      acc.x += point.x;
      acc.y += point.y;
      return acc;
    },
    { x: 0, y: 0 },
  );
  return {
    x: total.x / points.length,
    y: total.y / points.length,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function landmarksToGesture(result: Awaited<ReturnType<HandLandmarker["detectForVideo"]>>): GestureState | null {
  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    selectLatch = false;
    lockLatch = false;
    resetLatch = false;
    trackingPill.textContent = "Searching for hand";
    trackingPill.classList.remove("live");
    gestureText.textContent = "Show one hand to activate spatial controls.";
    reticle.classList.add("hidden");
    setMeter(orbitMeter, 0);
    setMeter(selectMeter, 0);
    setMeter(dismissMeter, 0);
    setMeter(resetMeter, 0);
    return null;
  }

  const thumb = landmarks[4];
  const index = landmarks[8];
  const middle = landmarks[12];
  const ring = landmarks[16];
  const pinky = landmarks[20];
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const handSize = handDistance(wrist, middleMcp) + 1e-4;
  const palm = averagePoint([landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]);

  const pinchCenterX = (thumb.x + index.x) * 0.5;
  const pinchCenterY = (thumb.y + index.y) * 0.5;
  const cursorX = (1 - pinchCenterX) * window.innerWidth;
  const cursorY = pinchCenterY * window.innerHeight;
  const steerCenterX = (thumb.x + index.x + middle.x) / 3;
  const steerCenterY = (thumb.y + index.y + middle.y) / 3;
  const orbitX = (1 - steerCenterX) * window.innerWidth;
  const orbitY = steerCenterY * window.innerHeight;

  const selectRatio = handDistance(thumb, index) / handSize;
  const lockRatio = handDistance(thumb, middle) / handSize;
  const selectStrength = 1 - clamp((selectRatio - 0.18) / (0.72 - 0.18), 0, 1);
  const lockStrength = 1 - clamp((lockRatio - 0.20) / (0.74 - 0.20), 0, 1);

  const opennessSource = [thumb, index, middle, ring, pinky]
    .map((point) => handDistance(point, palm))
    .reduce((sum, value) => sum + value, 0) / 5;
  const opennessRatio = opennessSource / handSize;
  const trioPinchRatio =
    (handDistance(thumb, index) + handDistance(thumb, middle) + handDistance(index, middle)) /
    (3 * handSize);
  const orbitStrength = 1 - clamp((trioPinchRatio - 0.16) / (0.58 - 0.16), 0, 1);
  const resetStrength = 1 - smoothstep(0.82, 1.36, opennessRatio);
  const isOrbiting = orbitStrength > 0.7;

  const rotateDominant = orbitStrength > 0.7;
  const selectTrigger = selectStrength > 0.72 && !rotateDominant && !selectLatch;
  const lockTrigger = lockStrength > 0.88 && selectStrength < 0.45 && !rotateDominant && !lockLatch;
  const resetTrigger = resetStrength > 0.92 && selectStrength < 0.24 && lockStrength < 0.24 && !rotateDominant && !resetLatch;
  selectLatch = selectStrength > 0.28;
  lockLatch = lockStrength > 0.42;
  resetLatch = resetStrength > 0.35;

  return {
    cursorX,
    cursorY,
    orbitX,
    orbitY,
    orbitStrength,
    selectStrength,
    lockStrength,
    resetStrength,
    isOrbiting,
    selectTrigger,
    lockTrigger,
    resetTrigger,
    debugText: `Rotate ${orbitStrength.toFixed(2)}  Focus ${selectStrength.toFixed(2)}  Lock ${lockStrength.toFixed(2)}  Reset ${resetStrength.toFixed(2)}${placePinned ? "  PINNED" : ""}${selectEngagedLabel(selectStrength)}${rotateDominant ? "  TRIPLE-PINCH" : ""}`,
  };
}

function selectEngagedLabel(selectStrength: number) {
  return selectStrength > 0.58 ? "  FOCUS" : "";
}

async function initHandTracker() {
  const vision = await FilesetResolver.forVisionTasks(CAMERA_WASM_ROOT);
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_PATH,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.55,
    minTrackingConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
  });
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });
  cameraVideo.srcObject = stream;
  await cameraVideo.play();
}

async function animate() {
  if (handLandmarker && cameraVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const result = handLandmarker.detectForVideo(cameraVideo, performance.now());
    const gesture = landmarksToGesture(result);
    if (gesture) {
      Object.assign(gestureState, gesture);
      handleGestureInteraction(gestureState);
    }
  }

  updatePlaceCardPosition();
  requestAnimationFrame(animate);
}

function bindResize() {
  window.addEventListener("resize", () => {
    viewer.scene.requestRender();
  });
}

function suppressPointerCamera() {
  const handler = new ScreenSpaceEventHandler(viewer.canvas);
  handler.setInputAction(() => undefined, ScreenSpaceEventType.LEFT_DOWN);
  handler.setInputAction(() => undefined, ScreenSpaceEventType.LEFT_UP);
  handler.setInputAction(() => undefined, ScreenSpaceEventType.MOUSE_MOVE);
  handler.setInputAction(() => undefined, ScreenSpaceEventType.WHEEL);
}

async function boot() {
  await Promise.all([initCamera(), initHandTracker()]);
  bindResize();
  suppressPointerCamera();
  animate();
}

boot().catch((error) => {
  console.error(error);
  gestureText.textContent = "Initialization failed. Check camera permissions and dependency install.";
  trackingPill.textContent = "Boot failed";
});

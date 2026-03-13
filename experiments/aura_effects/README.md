# Body Aura Effect

Real-time glowing "aura" overlay around your body silhouette using webcam input.

## What it does

- Detects your body using **MediaPipe Selfie Segmentation**
- Expands the silhouette contour into multiple offset layers
- Scatters **colored digits and symbols** along each layer
- Detects hands via **MediaPipe Hands** and draws a **radial spoke burst** at the wrist
- Applies a **soft glow pass** (Gaussian blur blend) for the neon look

## Quick start

```bash
# from the repo root
pip install -r requirements.txt

# run the aura effect
cd experiments/aura_effects
python aura_body.py
```

Press **q** to quit.

## Configuration

Edit `config.py` to tweak:

| Parameter                | What it controls                         |
| ------------------------ | ---------------------------------------- |
| `AURA_LAYERS`            | Number of layers, offsets, colors, chars |
| `CONTOUR_SAMPLE_STEP`    | Density of characters along the contour  |
| `RADIAL_SPOKES`          | Number of radial lines at the wrist      |
| `GLOW_KERNEL_SIZE`       | Blur radius for the glow composite       |
| `GLOW_BLEND_ALPHA`       | Glow intensity                           |
| `SEGMENTATION_THRESHOLD` | Body detection sensitivity               |

## How it works

```
webcam frame
  ↓
MediaPipe Selfie Segmentation → binary body mask
  ↓
cv2.findContours → body silhouette contour
  ↓
offset_contour (dilate) × N layers → expanded contours
  ↓
scatter digits/symbols along each contour layer
  ↓
MediaPipe Hands → wrist position → radial burst
  ↓
Gaussian blur glow + additive blend → final output
```

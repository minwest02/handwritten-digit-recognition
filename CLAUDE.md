# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- Created: 2026-09-29 22:53:00 -->

## Project Overview

A multi-platform handwritten digit recognition application with shared ML model and separate platform implementations:
- **Web version** (Flask web app with HTML5 canvas)
- **Desktop version** (Tkinter GUI)

Users draw digits (0-9) and receive predictions from a CNN trained on MNIST. The model trains automatically on first run (5 epochs) and is cached to `mnist_model.h5` for fast subsequent launches.

## Project Structure

```
/
├── CLAUDE.md (this file - root guide)
├── model.py (shared ML model - used by both versions)
├── requirements.txt (shared core dependencies)
├── mnist_model.h5 (shared trained model, generated at runtime)
│
├── desktop_version/
│   ├── CLAUDE.md (desktop-specific guide)
│   ├── main.py (Tkinter GUI)
│   └── requirements.txt (desktop dependencies)
│
└── web_version/
    ├── CLAUDE.md (web-specific guide)
    ├── app.py (Flask server with /predict API)
    ├── requirements.txt (web dependencies)
    └── templates/
        └── index.html (web UI with HTML5 canvas)
```

## Shared Architecture

**`model.py`** — ML model (used by both versions)
- `build_model()`: Builds CNN with Conv2D→MaxPool→Dense layers
- `train_model(epochs=5)`: Loads MNIST, trains, saves to `mnist_model.h5`
- `load_model()`: Loads pre-trained model or trains if missing; handles corrupted files
- `predict_digit(image_array)`: Takes 28×28 grayscale array, returns (digit, confidence)

**Data flow (both versions):**
1. User draws on canvas (280×280, white bg, black strokes)
2. Capture/encode canvas image
3. Convert to 28×28 grayscale
4. Invert colors (white→black, black→white) for MNIST
5. Call `predict_digit()` → get prediction
6. Display result (digit + confidence)

## Platform-Specific Details

- **Desktop:** See `desktop_version/CLAUDE.md` for Tkinter implementation
- **Web:** See `web_version/CLAUDE.md` for Flask + HTML5 Canvas implementation

## Commands

### Desktop Version

**Run desktop app** (requires `tf-env` conda environment):
```bash
cd desktop_version
/Users/youth/miniconda3/envs/tf-env/bin/python3 main.py
```

### Web Version

**Run web app** (requires `tf-env` conda environment):
```bash
cd web_version
/Users/youth/miniconda3/envs/tf-env/bin/python3 app.py
# Visit http://localhost:5000 in your browser
```

### Shared Setup

**Install dependencies** (both versions use shared model):
```bash
conda activate tf-env
pip install tensorflow numpy pillow flask
```

**Check if model file exists** (shared by both versions):
```bash
ls -la mnist_model.h5
```

**View model training output** (printed to console on first run):
Training progress shows epochs, loss, and accuracy for all 5 epochs.

## Environment

The project uses a conda environment `tf-env` with Python 3.11 and TensorFlow 2.16+. Both versions (desktop and web) use the same environment.

If you need to set up the environment from scratch:
```bash
conda create -n tf-env python=3.11
conda activate tf-env
pip install tensorflow numpy pillow flask
```

## Critical Implementation Details

1. **Shared model logic:** Both web and desktop versions use the same `model.py` with identical preprocessing:
   - Load/train MNIST model
   - Cache to `mnist_model.h5`
   - Handle missing or corrupted files

2. **Color inversion for MNIST:** Canvas has white background with black strokes (user-friendly), but MNIST expects black background with white digits. Both versions invert via `255 - image_array` before inference.

3. **Empty canvas detection:** Uses `np.max(inverted_array) < 50` to verify user drew something. Same check in both versions.

4. **Platform differences:**
   - **Desktop:** Uses `PIL.ImageGrab.grab(bbox)` to capture actual Tkinter canvas
   - **Web:** Uses HTML5 Canvas API with JavaScript to capture drawing

5. **Model persistence:** On first launch, training may take 2-3 minutes and is cached to `mnist_model.h5`. Subsequent launches are instant.

## Testing

**Desktop version:**
1. Run `desktop_version/main.py`
2. Draw digit on Tkinter canvas
3. Click "Recognize"
4. Verify prediction appears

**Web version:**
1. Run `web_version/app.py`
2. Open http://localhost:5000
3. Draw digit on HTML5 canvas
4. Click "Recognize"
5. Verify prediction appears

## Documentation

- `model.py` - ML model code (~95 lines, shared)
- `requirements.txt` - Shared core dependencies
- `desktop_version/CLAUDE.md` - Desktop implementation guide
- `web_version/CLAUDE.md` - Web implementation guide
- `docs/superpowers/specs/` - Overall design specification
- `docs/superpowers/plans/` - Implementation plan

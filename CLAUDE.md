# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python desktop application for handwritten digit recognition. Users draw digits (0-9) on a Tkinter canvas, and the app predicts the digit using a CNN trained on MNIST. The model trains automatically on first run (5 epochs) and is saved to `mnist_model.h5` for fast subsequent launches.

## Architecture

**Two-file modular design:**

- **`model.py`** — All ML logic (no UI dependencies)
  - `build_model()`: Builds CNN with Conv2D→MaxPool→Dense layers
  - `train_model(epochs=5)`: Loads MNIST, trains, saves to `mnist_model.h5`
  - `load_model()`: Loads pre-trained model or trains if missing; handles corrupted files
  - `predict_digit(image_array)`: Takes 28×28 grayscale array, returns (digit, confidence)

- **`main.py`** — Tkinter UI and application logic
  - `DigitRecognitionApp`: Main class with canvas, buttons, results display
  - Canvas: 280×280 white background, black brush for drawing
  - Buttons: "Clear" (reset canvas), "Recognize" (run inference)
  - Key detail: Uses `PIL.ImageGrab` to capture actual canvas pixels (not a mirror image)
  - Color inversion: Canvas has white bg/black strokes; inverted to black bg/white strokes before MNIST inference

**Data flow:** Draw → Capture canvas with ImageGrab → Convert to 28×28 grayscale → Invert colors → Predict → Display result

## Commands

**Run the app** (requires `tf-env` conda environment with TensorFlow):
```bash
cd /Users/youth/Desktop/VibeCoding/Study-01
/Users/youth/miniconda3/envs/tf-env/bin/python3 main.py
```

**Install dependencies** (in tf-env):
```bash
pip install tensorflow numpy pillow
```

**Check if model file exists** (will trigger training on first launch if missing):
```bash
ls -la mnist_model.h5
```

**View model training output** (printed to console on first run):
Training progress shows epochs, loss, and accuracy for all 5 epochs.

## Environment

The project uses a conda environment `tf-env` with Python 3.11 and TensorFlow 2.16+. This is necessary because TensorFlow has strict compatibility requirements.

If you need to set up the environment from scratch:
```bash
conda create -n tf-env python=3.11
conda activate tf-env
pip install tensorflow numpy pillow
```

## Critical Implementation Details

1. **Canvas capture with ImageGrab:** The app uses `PIL.ImageGrab.grab(bbox)` to capture the actual Tkinter canvas pixels. This is essential because a mirror PIL Image buffer would not properly sync with the visual canvas. Without this, drawn strokes wouldn't be recognized.

2. **Color inversion for MNIST:** The canvas has white background with black strokes (user-friendly), but MNIST expects black background with white digits. The `recognize()` method inverts via `255 - image_array` before inference.

3. **Empty canvas detection:** Uses `np.max(inverted_array) < 50` to check if user actually drew something. This is more reliable than checking mean pixel value.

4. **Model persistence:** On first launch, training may take 2-3 minutes. The trained model is cached in `mnist_model.h5` (2.6 MB), so subsequent launches are instant. If the file gets corrupted, `load_model()` catches the exception and retrains automatically.

5. **Error handling:** The app shows messagebox alerts for:
   - Empty canvas: "Please draw a digit first!"
   - Any exception during inference: Shows error message

## Testing

Manual test procedure:
1. Draw a digit (0-9) on the canvas
2. Click "Recognize"
3. Verify prediction and confidence score appear
4. Click "Clear" and repeat with different digits
5. On second run, verify model loads instantly (no retraining console output)

## Files

- `model.py` - ML model code (~95 lines)
- `main.py` - Tkinter UI code (~165 lines)
- `requirements.txt` - Dependencies
- `.gitignore` - Excludes `mnist_model.h5` and Python artifacts
- `docs/superpowers/specs/` - Design specification
- `docs/superpowers/plans/` - Implementation plan

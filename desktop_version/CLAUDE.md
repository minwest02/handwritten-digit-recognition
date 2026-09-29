# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the desktop version.

<!-- Created: 2026-09-29 22:52:00 -->

## Project: Handwritten Digit Recognition - Desktop Version

A Python desktop application using Tkinter for handwritten digit recognition. Part of a multi-platform project that includes web and desktop versions sharing a common ML model.

## Architecture

**Desktop-specific files in this directory:**

- **`main.py`** — Tkinter GUI application
  - `DigitRecognitionApp`: Main class with canvas, buttons, results display
  - Canvas: 280×280 white background, black brush for drawing
  - Uses `PIL.ImageGrab` to capture actual canvas pixels
  - Color inversion: white bg/black strokes → black bg/white strokes for MNIST
  - Buttons: "Clear" (reset), "Recognize" (inference)

- **`requirements.txt`** — Desktop dependencies (tkinter is built-in, only need TensorFlow, NumPy, Pillow)

**Shared files (in parent directory):**

- **`../model.py`** — ML model (shared between desktop and web versions)
- **`../mnist_model.h5`** — Trained model (cached after first run)

## Commands

**Run desktop app** (requires `tf-env` conda environment):
```bash
cd desktop_version
/Users/youth/miniconda3/envs/tf-env/bin/python3 main.py
```

**Install dependencies:**
```bash
conda activate tf-env
pip install -r requirements.txt
```

## Implementation Details

1. **Canvas capture:** Uses `PIL.ImageGrab` to capture actual Tkinter canvas (not a mirror image buffer)
2. **Color inversion:** `255 - image_array` converts UI format to MNIST format
3. **Empty canvas check:** `np.max(inverted_array) < 50` detects if user drew anything
4. **Model loading:** Shared `load_model()` from parent `model.py` handles training/loading
5. **Error handling:** Messagebox alerts for empty canvas and exceptions

## Imports from Shared Code

```python
from model import predict_digit, load_model
```

The `model.py` file is in the parent directory. Python's import system will find it automatically since the parent is in the path.

## Desktop-Specific Testing

1. Draw a digit on the canvas
2. Click "Recognize" 
3. Verify prediction and confidence appear
4. Click "Clear" and repeat
5. Close window (⌘Q or window close button)

## macOS Considerations

- Tkinter works natively on macOS
- `PIL.ImageGrab` captures from native window coordinates
- App may need display permissions on newer macOS versions

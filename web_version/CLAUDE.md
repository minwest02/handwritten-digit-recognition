# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the web version.

<!-- Created: 2026-09-29 22:52:30 -->

## Project: Handwritten Digit Recognition - Web Version

A Flask web application for handwritten digit recognition. Users draw digits in a browser canvas and receive predictions from a shared MNIST CNN model.

## Architecture

**Web-specific files in this directory:**

- **`app.py`** — Flask application server (to be implemented)
  - Routes: `/` (homepage), `/predict` (API endpoint for digit prediction)
  - Serves `templates/index.html`
  - API accepts canvas image as base64 or binary
  - Returns JSON with prediction result and confidence

- **`templates/index.html`** — Web UI (to be implemented)
  - HTML5 canvas for drawing (280×280)
  - JavaScript for mouse tracking and canvas rendering
  - "Clear" and "Recognize" buttons
  - Results display area

- **`requirements.txt`** — Web dependencies (Flask, shared ML dependencies)

**Shared files (in parent directory):**

- **`../model.py`** — ML model (shared between desktop and web versions)
- **`../mnist_model.h5`** — Trained model (cached after first run)

## Commands

**Install dependencies:**
```bash
pip install flask numpy pillow tensorflow
```

**Run web app** (development):
```bash
cd web_version
python3 app.py
# Visit http://localhost:5000
```

## API Design

**GET `/`**
- Returns `index.html`

**POST `/predict`**
- Request body: JSON with `image` field (base64-encoded 28×28 grayscale image)
- Response: `{"digit": 7, "confidence": 0.985}`

## Implementation Guidelines

1. **Canvas handling in JavaScript:**
   - Track mouse down/move/up events
   - Draw on HTML5 canvas with black strokes on white background
   - Convert canvas to 28×28 grayscale before sending to server

2. **Image preprocessing on server:**
   - Decode base64 or binary image
   - Convert to 28×28 if needed
   - Invert colors (white→black, black→white) for MNIST
   - Pass to `predict_digit()` from shared `model.py`

3. **Error handling:**
   - Empty canvas: return error message "Please draw a digit first"
   - Prediction errors: return error with details

4. **Model loading:**
   - Load model once on app startup using `load_model()`
   - Cache in memory for fast inference

## Imports from Shared Code

```python
from model import predict_digit, load_model
```

The `model.py` file is in the parent directory. Add parent to Python path:

```python
import sys
sys.path.insert(0, '..')
from model import predict_digit, load_model
```

## Web-Specific Testing

1. Start Flask dev server
2. Open http://localhost:5000 in browser
3. Draw a digit on the canvas
4. Click "Recognize"
5. Verify prediction appears
6. Test with different digits
7. Verify results update correctly

## Frontend Technologies

- HTML5 Canvas API for drawing
- Fetch API for server requests
- Base64 encoding for image transmission
- CSS for styling

## Browser Compatibility

Target modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

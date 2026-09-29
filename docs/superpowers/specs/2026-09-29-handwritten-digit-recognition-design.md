# Handwritten Digit Recognition Desktop App - Design Spec

**Date:** 2026-09-29  
**Status:** Approved  
**Approach:** Modular separation (Approach B)

## Overview

A Python desktop application built with Tkinter that allows users to draw handwritten digits on a canvas and receive recognition predictions powered by an MNIST-trained neural network. The model trains on first run (5 epochs) and is saved for subsequent uses.

## Architecture

### Three Main Modules

**1. `model.py` — Model training and inference**
- Builds a CNN using TensorFlow/Keras
- Loads MNIST dataset from `tf.keras.datasets`
- Trains model for 5 epochs on first run
- Saves trained weights to `mnist_model.h5`
- Provides `load_model()` function to load pre-trained weights
- Provides `predict_digit(image)` function that takes a 28×28 grayscale array and returns (predicted_digit, confidence_score)

**2. `main.py` — Tkinter UI and application logic**
- Creates main application window
- Implements drawing canvas: 280×280 pixels, white background, black brush strokes
- Implements two buttons: "Clear" and "Recognize"
- Results display area: shows "Predicted: X" and "Confidence: Y%"
- On startup: checks if `mnist_model.h5` exists; if not, trains model via `model.py`
- On "Recognize" click: 
  - Captures canvas image (280×280)
  - Resizes to 28×28 pixels
  - Inverts colors (white→black, black→white) to match MNIST format
  - Calls `predict_digit()` from `model.py`
  - Displays result

**3. `requirements.txt` — Dependencies**
- tensorflow
- numpy
- pillow (for image handling)

## Data Flow

```
User draws on canvas (280×280, white bg, black strokes)
    ↓
User clicks "Recognize"
    ↓
Canvas image captured and converted to 28×28 grayscale
    ↓
Image colors inverted (MNIST expects black bg, white digits)
    ↓
Model inference (predict_digit)
    ↓
Display result: "Predicted: 7, Confidence: 98.5%"
```

## Model Details

**Architecture:** Convolutional Neural Network
- Input layer: 28×28 grayscale (784 values)
- Conv2D(32 filters, 3×3) + ReLU + MaxPooling(2×2)
- Conv2D(64 filters, 3×3) + ReLU + MaxPooling(2×2)
- Flatten
- Dense(128) + ReLU + Dropout(0.5)
- Dense(10) + Softmax (output: probability for each digit 0-9)

**Training:**
- Dataset: MNIST (60,000 training images)
- Epochs: 5
- Batch size: 128 (default)
- Loss: categorical_crossentropy
- Optimizer: Adam

**Persistence:**
- After training, save weights to `mnist_model.h5` in project root
- On app startup, load from `mnist_model.h5` if it exists (skip training)

## UI Layout

```
┌─────────────────────────────────────────┐
│  Handwritten Digit Recognition         │
├─────────────────────────────────────────┤
│                                         │
│         [White Drawing Canvas]          │
│              (280×280px)                │
│                                         │
├─────────────────────────────────────────┤
│  [Clear]              [Recognize]       │
├─────────────────────────────────────────┤
│  Predicted: 7                           │
│  Confidence: 98.5%                      │
└─────────────────────────────────────────┘
```

## Error Handling

- **Model file missing:** Train on first run (displays training progress in console)
- **Invalid canvas input:** If canvas is empty, show "Please draw a digit first"
- **Model load failure:** Catch exceptions and retrain if weights file is corrupted

## Testing Scope

- Manual test: Draw digits 0-9 and verify recognition accuracy
- Edge cases: Very light strokes, very thick strokes, off-center digits
- Model persistence: Verify model loads correctly on second run (no retraining)

## Future Extensions (Out of Scope)

- Batch prediction from multiple drawings
- Model accuracy metrics display
- Adjustable brush size in UI
- Dataset augmentation for better accuracy

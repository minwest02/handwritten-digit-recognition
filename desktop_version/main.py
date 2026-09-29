import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
from model import predict_digit

# Constants
CANVAS_SIZE = 280
CANVAS_BG_COLOR = 'white'
BRUSH_COLOR = 'black'
BRUSH_WIDTH = 3


class DigitRecognitionApp:
    """
    Tkinter application for handwritten digit recognition.
    """

    def __init__(self, root):
        """Initialize the app with UI components."""
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("400x550")
        self.root.resizable(False, False)

        # Create drawing canvas
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=CANVAS_BG_COLOR,
            cursor="cross"
        )
        self.canvas.pack(pady=10)

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)

        # Image for capturing canvas content
        self.image = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw_obj = ImageDraw.Draw(self.image)

        # Last position for drawing lines
        self.last_x = None
        self.last_y = None

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Clear button
        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            width=15,
            height=2
        )
        self.clear_button.grid(row=0, column=0, padx=5)

        # Recognize button
        self.recognize_button = tk.Button(
            button_frame,
            text="Recognize",
            command=self.recognize,
            width=15,
            height=2
        )
        self.recognize_button.grid(row=0, column=1, padx=5)

        # Results display area
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.prediction_label = tk.Label(
            self.results_frame,
            text="Predicted: -",
            font=("Arial", 16, "bold")
        )
        self.prediction_label.pack()

        self.confidence_label = tk.Label(
            self.results_frame,
            text="Confidence: -",
            font=("Arial", 14)
        )
        self.confidence_label.pack()

    def start_draw(self, event):
        """Record starting position for drawing."""
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        """Draw on canvas when mouse is dragged."""
        if self.last_x and self.last_y:
            # Draw on Tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=BRUSH_COLOR, width=BRUSH_WIDTH, capstyle=tk.ROUND
            )

            # Draw on PIL image (for saving)
            self.draw_obj.line(
                [(self.last_x, self.last_y), (event.x, event.y)],
                fill=BRUSH_COLOR, width=BRUSH_WIDTH
            )

        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.delete("all")
        self.image = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw_obj = ImageDraw.Draw(self.image)

    def recognize(self):
        """
        Capture canvas, preprocess image, and run digit recognition.
        Display prediction and confidence on UI.
        """
        try:
            # Get canvas position on screen
            canvas_x = self.canvas.winfo_rootx()
            canvas_y = self.canvas.winfo_rooty()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            # Capture the actual drawn canvas from screen
            bbox = (canvas_x, canvas_y, canvas_x + canvas_w, canvas_y + canvas_h)
            captured_image = ImageGrab.grab(bbox=bbox)

            # Convert to grayscale
            gray_image = captured_image.convert('L')

            # Resize to 28x28 (MNIST format)
            resized = gray_image.resize((28, 28))

            # Convert to NumPy array
            image_array = np.array(resized)

            # Invert colors: white background becomes black (MNIST format)
            # In UI: white bg with black strokes
            # After invert: black bg with white strokes (what MNIST expects)
            inverted_array = 255 - image_array

            # Check if canvas is blank (look for any non-black pixels after invert)
            if np.max(inverted_array) < 50:  # If no significant white strokes, canvas is empty
                messagebox.showwarning("Empty Canvas", "Please draw a digit first!")
                return

            # Run prediction
            predicted_digit, confidence = predict_digit(inverted_array)

            # Update UI with results
            self.prediction_label.config(
                text=f"Predicted: {predicted_digit}"
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence*100:.1f}%"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

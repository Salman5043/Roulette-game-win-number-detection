from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pyautogui
import time
import easyocr
from ultralytics import YOLO
import pandas as pd
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("models/best.pt")  # Update with your trained model path
reader = easyocr.Reader(['en'])

# Global variables
detection_active = False  # Control flag for detection
last_detected_number = None  # Store the last detected winning number

# Create results folder
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
EXCEL_FILE = os.path.join(RESULTS_FOLDER, "detected_numbers.xlsx")

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def preprocess_image(img_file):
    """Preprocess the image for better OCR results."""
    if isinstance(img_file, str):
        # Read image from file path
        img = cv2.imread(img_file)
        if img is None:
            raise ValueError(f"Could not read the image: {img_file}")
    elif isinstance(img_file, np.ndarray):
        # Image is already read into memory
        img = img_file
    else:
        raise ValueError("Unsupported input type for preprocess_image")

    # Resize the image to have a minimum height of 400 pixels
    height, width, _ = img.shape
    if height < 400:
        new_height = 400
        new_width = int((new_height / height) * width)
        img = cv2.resize(img, (new_width, new_height))

    # Apply bilateral filtering to reduce noise while preserving edges
    img = cv2.bilateralFilter(img, 7, 50, 50)

    # Apply proper unsharp masking to enhance edges
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply sharpening filter
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray


def capture_right_half_screen():
    """Capture the right 50% of the screen using PyAutoGUI."""
    try:
        screen_width, screen_height = pyautogui.size()
        left = screen_width // 2  # Start from the middle
        top = 0
        width = screen_width // 2  # Capture right half
        height = screen_height

        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error capturing screen: {e}")
        return None


def process_frame(frame):
    """Run YOLO detection and apply OCR on detected objects."""
    global last_detected_number  # Access the last detected number

    results = model(frame)[0]  # Get the first (and only) result

    best_detection = None  # Store the best detection

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = map(float, box)  # Extract box details
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers

        # Extract region of interest (ROI)
        roi = frame[y1:y2, x1:x2]
        gray = preprocess_image(roi)

        # Perform OCR on the ROI
        ocr_results = reader.readtext(gray)
        detected_text = ocr_results[0][1] if ocr_results else "Not Found"

        # Handle "Not Found" or invalid numbers
        if detected_text == "Not Found":
            continue  # Skip this detection

        try:
            detected_number = int(detected_text)
            if detected_number < 37:
                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Track the best detection
                if best_detection is None or score > best_detection["confidence"]:
                    best_detection = {
                        "text": detected_text,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")  # Save detection time
                    }
        except ValueError:
            # Skip if detected_text is not a valid integer
            continue

    # Save the best detection to Excel **only if it's different from the last detected number**
    if best_detection and best_detection["text"] != last_detected_number:
        save_detected_data(best_detection)  # Save new number
        last_detected_number = best_detection["text"]  # Update last detected number

    # Encode the frame as a JPEG image
    _, buffer = cv2.imencode('.jpg', frame)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def save_detected_data(data):
    """Save detected data to an Excel file."""
    if not data:
        return

    try:
        # Create a DataFrame with new data
        new_data = pd.DataFrame([data])

        # Load existing Excel file if it exists, else create a new one
        if os.path.exists(EXCEL_FILE):
            existing_data = pd.read_excel(EXCEL_FILE)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data

        # Save to Excel file
        updated_data.to_excel(EXCEL_FILE, index=False)
    except Exception as e:
        logging.error(f"Error saving detected data: {e}")


def generate_frames():
    """Capture and process frames continuously."""
    global detection_active
    while detection_active:
        try:
            frame = capture_right_half_screen()
            if frame is None:
                continue
            yield process_frame(frame)
            time.sleep(0.05)
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
            break  # Exit the loop on error


@app.route('/')
def index():
    """Render the web interface."""
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    """Stream processed video to the UI."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_detection():
    """Start detection on right half of the screen."""
    global detection_active
    if not detection_active:
        detection_active = True
        return jsonify({"status": "Detection started"})
    return jsonify({"status": "Detection already running"})


@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop detection."""
    global detection_active
    if detection_active:
        detection_active = False
        return jsonify({"status": "Detection stopped"})
    return jsonify({"status": "Detection is not running"})


@app.route('/detected_numbers', methods=['GET'])
def get_detected_numbers():
    """Fetch detected numbers from the Excel file."""
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])


if __name__ == '__main__':
    app.run(debug=True)
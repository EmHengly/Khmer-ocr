from flask import Flask, request, render_template, redirect, send_from_directory
import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import fuzz
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, ViTFeatureExtractor, NllbTokenizer
import asyncio
from Levenshtein import distance as levenshtein_distance
from multiprocessing import cpu_count
import re
import unicodedata

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLOv8 model (Load once when the server starts)
detection_model = YOLO('train18/weights/best.pt')

# Load TRoCR model (Load once when the server starts)
recognition_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-21180')
recognition_model.eval()

# Load Processor (Load once when the server starts)
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/trocr-small-stage1')
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

def normalize_text(text):
    """Normalize the text to remove unwanted characters and standardize the format."""
    return unicodedata.normalize('NFC', text)

def preprocess_image(image_path):
    """Read and preprocess the image for text detection."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image

def detect_text(image, model):
    """Detect text boxes in the image using the YOLO model."""
    results = model(image)
    boxes = [box for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()) if conf >= 0.4]  # Adjusted confidence threshold
    return boxes

def merge_boxes(boxes, threshold=10):
    """Merge adjacent text boxes into one."""
    if not boxes:
        return []
    merged_boxes = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):  # Sort by y1, then x1
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            last_box = merged_boxes[-1]
            if box[1] <= last_box[3] + threshold and box[0] <= last_box[2] + threshold:
                merged_boxes[-1] = (
                    min(last_box[0], box[0]), min(last_box[1], box[1]),
                    max(last_box[2], box[2]), max(last_box[3], box[3])
                )
            else:
                merged_boxes.append(box)
    return merged_boxes

async def recognize_text(image, detections, model, processor, target_phrase, threshold=50):  # Lower threshold to 50
    """Recognize text from detected boxes and check if it matches the target phrase."""
    matching_boxes = []
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, recognize_single_box, image, box, model, processor, target_phrase, threshold) for box in detections]
    for result in await asyncio.gather(*tasks):
        if result:
            matching_boxes.append(result)
    return matching_boxes

def recognize_single_box(image, box, model, processor, target_phrase, threshold):
    """Recognize text from a single box and check if it matches the target phrase."""
    try:
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        roi = image[y1:y2, x1:x2]
        # Preprocess ROI for TRoCR (resize and normalize)
        roi = cv2.resize(roi, (256, 256))  # Adjust as per TRoCR's input size
        pixel_values = processor(images=roi, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=50)  # Increased to 50 tokens
        recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_text = post_process_text(normalize_text(recognized_text))
        print(f"Detected text: {recognized_text}")  # Print recognized text
        
        # Levenshtein distance similarity
        levenshtein_sim = 100 - (levenshtein_distance(recognized_text, target_phrase) / max(len(recognized_text), len(target_phrase)) * 100)
        print(f"Levenshtein Similarity: {levenshtein_sim}")  # Print similarity score
        
        if levenshtein_sim >= threshold:
            return (box, recognized_text)
    except Exception as e:
        print(f"Error recognizing text from box {box}: {e}")
    return None

def post_process_text(text):
    """Post-process recognized text to correct spelling and remove unwanted characters."""
    text = correct_spelling(text)
    text = re.sub(r'[^\u1780-\u17FF\s]', '', text)  # Keep only Khmer characters and whitespace
    text = normalize_text(text)
    return text

def correct_spelling(text):
    """A placeholder function for spelling correction."""
    # Implement a simple spell checker or use an existing library for Khmer language
    # For demonstration purposes, we will just return the text as is
    return text

def draw_highlight_boxes(image, detections):
    """Draw bounding boxes around detected text on the image."""
    if not detections:
        print("No matching words found.")
    for box, _ in detections:
        x1, y1, x2, y2 = map(int, box)
        # Create a semi-transparent green rectangle to highlight the detected text area
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green rectangle for highlighting
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Blend the rectangle with the original image

    return image

def merge_detected_words(detected_texts):
    """Merge detected words into a single string."""
    if not detected_texts:
        return ""
    
    merged_text = " ".join([text for _, text in detected_texts])
    return merged_text

def preprocess_target_phrase(target_phrase):
    """Preprocess the target phrase to handle variations such as spaces, punctuation, and normalization."""
    target_phrase = normalize_text(target_phrase)
    target_phrase = re.sub(r'\s+', ' ', target_phrase)  # Replace multiple spaces with a single space
    target_phrase = target_phrase.strip()  # Remove leading and trailing whitespace
    return target_phrase

def process_image(image_path, target_phrase, threshold):
    """Process the image to detect text and check against the target phrase."""
    try:
        image = preprocess_image(image_path)
        detections = detect_text(image, detection_model)
        merged_detections = merge_boxes(detections)
        matching_detections = asyncio.run(recognize_text(image, merged_detections, recognition_model, processor, target_phrase, threshold))
        
        if matching_detections:
            detected_words = [(box, text) for box, text in matching_detections]
            merged_detected_text = merge_detected_words(detected_words)
            print(f"Merged detected text: {merged_detected_text}")  # Print merged detected text
            
            # Adjusted similarity check to handle cases where recognized text might have slight variations
            if fuzz.ratio(merged_detected_text, target_phrase) >= threshold:
                result_image_path = draw_highlight_boxes(image.copy(), matching_detections)
                result_image_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
                cv2.imwrite(result_image_path, cv2.cvtColor(result_image_path, cv2.COLOR_RGB2BGR))
                return result_image_path
        
        print("No matching words found.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the upload form and handle file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        target_phrase = request.form['target_word'].strip()
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            target_phrase = preprocess_target_phrase(target_phrase)  # Process target phrase
            result_image_path = process_image(filepath, target_phrase, 50)  # Lower threshold to 50
            if result_image_path:
                return render_template('index.html', filename=os.path.basename(result_image_path), target_word=target_phrase)
            else:
                return render_template('index.html', error="No matching words found.", target_word=target_phrase)
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    """Serve the processed image result."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    # Adjust for your available CPUs/Threads
    executor = ThreadPoolExecutor(cpu_count())
    app.run(debug=True)

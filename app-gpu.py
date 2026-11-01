import logging
from flask import Flask, request, render_template, redirect, send_from_directory
import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from fuzzywuzzy import fuzz
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, ViTFeatureExtractor, NllbTokenizer
import asyncio
from multiprocessing import cpu_count
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

# Load YOLOv8 model (Load once when the server starts)
detection_model = YOLO('train18/weights/best.pt').to(device)

# Load TRoCR model (Load once when the server starts)
recognition_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-21180').to(device)
recognition_model.eval()

# Load Processor (Load once when the server starts)
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/trocr-small-printed')
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image

def detect_text(image, model, confidence_threshold=0.3):
    results = model(image)
    boxes = [box for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()) if conf > confidence_threshold]
    logger.debug(f"Detected {len(boxes)} text regions")
    return boxes

def cluster_text_regions(boxes, eps=15, min_samples=1):
    # Prepare data for DBSCAN clustering
    box_centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(box_centers)
    labels = dbscan.labels_
    clustered_boxes = {}
    for box, label in zip(boxes, labels):
        if label not in clustered_boxes:
            clustered_boxes[label] = []
        clustered_boxes[label].append(box)
    logger.debug(f"Clustered into {len(clustered_boxes)} groups")
    return clustered_boxes

def merge_text_regions(boxes, overlap_threshold=0.5):
    merged_boxes = []
    for box in boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            overlap = False
            for existing_box in merged_boxes:
                if boxes_overlap(existing_box, box, overlap_threshold):
                    existing_box = merge_boxes(existing_box, box)
                    overlap = True
                    break
            if not overlap:
                merged_boxes.append(box)
    logger.debug(f"Merged into {len(merged_boxes)} regions")
    return merged_boxes

def boxes_overlap(box1, box2, overlap_threshold):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    x_overlap = max(0, min(x2, x2b) - max(x1, x1b))
    y_overlap = max(0, min(y2, y2b) - max(y1, y1b))
    intersection_area = x_overlap * y_overlap
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area > overlap_threshold

def merge_boxes(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return (x1, y1, x2, y2)

async def recognize_text(image, clustered_boxes, model, processor, target_word, threshold=70):
    matching_boxes = []
    for cluster_boxes in clustered_boxes.values():
        for box in cluster_boxes:
            result = await recognize_single_box(image, box, model, processor, target_word, threshold)
            if result:
                matching_boxes.append(result)
    # Post-process results
    matching_boxes = post_process_text(matching_boxes, target_word)
    return matching_boxes

async def recognize_single_box(image, box, model, processor, target_word, threshold):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, single_box_recognition, image, box, model, processor, target_word, threshold)

def single_box_recognition(image, box, model, processor, target_word, threshold):
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
    roi = image[y1:y2, x1:x2]
    # Preprocess ROI for TRoCR (resize and normalize)
    roi = cv2.resize(roi, (384, 384))  # Adjust as per TRoCR's input size
    pixel_values = processor(images=roi, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=50)  # Increased max_new_tokens for longer text
    keyword_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.debug(f"Detected text: '{keyword_text}'")
    # Fuzzy matching
    similarity = fuzz.ratio(keyword_text, target_word)
    if similarity >= threshold:
        return (box, keyword_text)
    return None

def post_process_text(detections, target_word):
    processed_detections = []
    for box, text in detections:
        if target_word in text:
            processed_detections.append((box, text))
    logger.debug(f"Found {len(processed_detections)} matches for '{target_word}'")
    return processed_detections

def draw_highlight_boxes(image, detections):
    if not detections:
        logger.debug("No matching words found.")
    for box, _ in detections:
        x1, y1, x2, y2 = map(int, box)
        # Create a semi-transparent green rectangle to highlight the detected text area
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green rectangle for highlighting
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Blend the rectangle with the original image

    return image

def process_image(image_path, target_word, threshold):
    image = preprocess_image(image_path)
    detections = detect_text(image, detection_model, confidence_threshold=0.3)
    clustered_boxes = cluster_text_regions(detections, eps=15, min_samples=1)  # Adjusted DBSCAN parameters
    matching_detections = asyncio.run(recognize_text(image, clustered_boxes, recognition_model, processor, target_word, threshold))
    result_image = draw_highlight_boxes(image.copy(), matching_detections)
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return result_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        target_word = request.form['target_word']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_image_path = process_image(filepath, target_word, 70)
            return render_template('index.html', filename=os.path.basename(result_image_path), target_word=target_word)
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=cpu_count())  # Use all available CPU cores
    app.run(debug=True)

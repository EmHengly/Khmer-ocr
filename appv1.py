# from flask import Flask, request, render_template, redirect, send_from_directory
# import torch
# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from concurrent.futures import ThreadPoolExecutor
# from fuzzywuzzy import fuzz
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor, ViTFeatureExtractor, NllbTokenizer
# import asyncio

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Load YOLOv8 model
# detection_model = YOLO('train18/weights/best.pt')

# # Load TRoCR model
# recognition_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-21180')
# recognition_model.eval()

# # Load Processor
# tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/trocr-small-printed')
# processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# if not os.path.exists(RESULT_FOLDER):
#     os.makedirs(RESULT_FOLDER)

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image not found at path: {image_path}")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     return image

# def detect_text(image, model):
#     results = model(image)
#     boxes = [box for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()) if conf > 0.5]
#     return boxes

# async def recognize_text(image, detections, model, processor, target_word, threshold=80):
#     matching_boxes = []
#     tasks = []
#     for box in detections:
#         tasks.append(recognize_single_box(image, box, model, processor, target_word, threshold))
#     results = await asyncio.gather(*tasks)
#     for result in results:
#         if result:
#             matching_boxes.append(result)
#     return matching_boxes

# async def recognize_single_box(image, box, model, processor, target_word, threshold):
#     x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
#     roi = image[y1:y2, x1:x2]
#     # Preprocess ROI for TRoCR (resize and normalize)
#     roi = cv2.resize(roi, (384, 384))  # Adjust as per TRoCR's input size
#     pixel_values = processor(images=roi, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     keyword_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fuzzy matching
#     similarity = fuzz.ratio(keyword_text, target_word)
#     if similarity >= threshold:
#         return (box, keyword_text)
#     return None

# def draw_highlight_boxes(image, detections):
#     if not detections:
#         print("No matching words found.")
#     for box, _ in detections:
#         x1, y1, x2, y2 = map(int, box)
#         # Create a semi-transparent green rectangle to highlight the detected text area
#         overlay = image.copy()
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green rectangle for highlighting
#         alpha = 0.4  # Transparency factor
#         cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Blend the rectangle with the original image

#     return image

# async def process_image(image_path, target_word, threshold):
#     image = preprocess_image(image_path)
#     detections = detect_text(image, detection_model)
#     matching_detections = await recognize_text(image, detections, recognition_model, processor, target_word, threshold)
#     result_image = draw_highlight_boxes(image.copy(), matching_detections)
#     result_image_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
#     cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
#     return result_image_path

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files or request.files['file'].filename == '':
#             return redirect(request.url)
#         file = request.files['file']
#         target_word = request.form['target_word']
#         if file:
#             filename = file.filename
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             result_image_path = loop.run_until_complete(process_image(filepath, target_word, 80))
#             return render_template('index.html', filename=os.path.basename(result_image_path), target_word=target_word)
#     return render_template('index.html')

# @app.route('/result/<filename>')
# def result(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)





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

def detect_text(image, model):
    results = model(image)
    boxes = [box for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()) if conf > 0.5]
    return boxes

def recognize_single_box(image, box, model, processor, target_word, threshold):
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
    roi = image[y1:y2, x1:x2]
    # Preprocess ROI for TRoCR (resize and normalize)
    roi = cv2.resize(roi, (384, 384))  # Adjust as per TRoCR's input size
    pixel_values = processor(images=roi, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    keyword_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Fuzzy matching
    similarity = fuzz.ratio(keyword_text, target_word)
    if similarity >= threshold:
        return (box, keyword_text)
    return None

def recognize_text(image, detections, model, processor, target_word, threshold=80):
    matching_boxes = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(recognize_single_box, image, box, model, processor, target_word, threshold) for box in detections]
        for future in as_completed(futures):
            result = future.result()
            if result:
                matching_boxes.append(result)
    return matching_boxes

def draw_highlight_boxes(image, detections):
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

def process_image(image_path, target_word, threshold):
    image = preprocess_image(image_path)
    detections = detect_text(image, detection_model)
    matching_detections = recognize_text(image, detections, recognition_model, processor, target_word, threshold)
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
            result_image_path = process_image(filepath, target_word, 80)
            return render_template('index.html', filename=os.path.basename(result_image_path), target_word=target_word)
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, render_template, redirect, send_from_directory
# import torch
# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from fuzzywuzzy import fuzz
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, ViTImageProcessor, NllbTokenizer
# import unicodedata
# import time
# from multiprocessing import cpu_count

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# # Check if CUDA is available, else default to CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load detection model
# detection_model = YOLO('train18/weights/best.pt').to(device)

# # Use half precision only if using GPU
# if device.type == 'cuda':
#     detection_model.half()  # Ensure the model is in half precision only on GPU

# # Load recognition model
# recognition_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-119112').to(device)
# recognition_model.eval()

# # Only use half precision if running on GPU
# if device.type == 'cuda':
#     recognition_model.half()


# tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# feature_extractor = ViTImageProcessor.from_pretrained('checkpoint-119112')

# processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)


# # processor = TrOCRProcessor.from_pretrained('fine-tuned-small-printed-V2-checkpoint-with100000data/checkpoint-170160')


# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# if not os.path.exists(RESULT_FOLDER):
#     os.makedirs(RESULT_FOLDER)

# def normalize_text(text):
#     return unicodedata.normalize('NFC', text)

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image not found at path: {image_path}")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
#     return image

# def detect_text(image, model):
#     results = model(image)
#     boxes = [box for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()) if conf >= 0.3]
#     return boxes

# def calculate_iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2
#     xi1 = max(x1, x3)
#     yi1 = max(y1, y3)
#     xi2 = min(x2, x4)
#     yi2 = min(y2, y4)
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x4 - x3) * (y4 - y3)
#     union_area = box1_area + box2_area - inter_area
#     return inter_area / union_area if union_area > 0 else 0

# def group_boxes_by_lines(boxes, iou_threshold=0.5):
#     lines = []
#     boxes = [list(box) for box in boxes]
#     while boxes:
#         current_box = boxes.pop(0)
#         line_group = [current_box]
#         for other_box in boxes[:]:
#             if calculate_iou(current_box, other_box) > iou_threshold:
#                 line_group.append(other_box)
#                 boxes.remove(other_box)
#         lines.append(line_group)
#     return lines

# def concatenate_boxes(line_group):
#     x_min = min(box[0] for box in line_group)
#     y_min = min(box[1] for box in line_group)
#     x_max = max(box[2] for box in line_group)
#     y_max = max(box[3] for box in line_group)
#     return [x_min, y_min, x_max, y_max]

# def recognize_text_by_lines(image, line_groups, model, processor, target_word, threshold=70):
#     detected_boxes = []
#     for line_group in line_groups:
#         line_box = concatenate_boxes(line_group)
#         x1, y1, x2, y2 = map(int, line_box)
#         roi = image[y1:y2, x1:x2]
#         roi = cv2.resize(roi, (384, 384))
#         pixel_values = processor(images=roi, return_tensors="pt").pixel_values.to(device)
#         generated_ids = model.generate(pixel_values, max_new_tokens=50)
#         recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         recognized_text = normalize_text(recognized_text)
#         if fuzz.partial_ratio(recognized_text, target_word) >= threshold:
#             detected_boxes.append((line_box, target_word))
#     return detected_boxes

# def detect_and_recognize(image, target_word, threshold):
#     detections = detect_text(image, detection_model)
#     line_groups = group_boxes_by_lines(detections)
#     with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
#         futures = [executor.submit(recognize_text_by_lines, image, [line_group], recognition_model, processor, target_word, threshold) for line_group in line_groups]
#         matching_detections = [result for future in as_completed(futures) for result in future.result()]
#     return matching_detections

# def draw_highlight_boxes(image, detections):
#     if not detections:
#         print("No matching words found.")
#     for box, _ in detections:
#         x1, y1, x2, y2 = map(int, box)
#         overlay = image.copy()
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
#         alpha = 0.4
#         cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
#     return image

# def process_image(image_path, target_word, threshold):
#     image = preprocess_image(image_path)
#     matching_detections = detect_and_recognize(image, target_word, threshold)
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
#             start_time = time.time()
#             result_image_path = process_image(filepath, target_word, 70)
#             end_time = time.time()
#             print(f"Processing time: {end_time - start_time} seconds")
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
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, NllbTokenizer, ViTImageProcessor
import unicodedata
import time
from multiprocessing import cpu_count

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Check if CUDA is available, else default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load detection model
detection_model = YOLO('train18/weights/best.pt').to(device)
if device.type == 'cuda':
    detection_model.half()  # Ensure the model is in half precision only on GPU

# Load recognition model
recognition_model = VisionEncoderDecoderModel.from_pretrained('checkpoint-119112').to(device)
recognition_model.eval()
if device.type == 'cuda':
    recognition_model.half()



tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
feature_extractor = ViTImageProcessor.from_pretrained('checkpoint-119112')
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

def normalize_text(text):
    return unicodedata.normalize('NFC', text)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return image

def detect_text(image, model):
    results = model(image)
    boxes = [box for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()) if conf >= 0.3]
    return boxes

def recognize_text_boxes(image, boxes, model, processor):
    pixel_values_list = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        roi = cv2.resize(roi, (384, 384))
        pixel_values = processor(images=roi, return_tensors="pt").pixel_values.to(device)
        pixel_values_list.append(pixel_values)
    
    # Batch process
    all_recognized_texts = []
    for pixel_values in pixel_values_list:
        generated_ids = model.generate(pixel_values, max_new_tokens=50)
        recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_recognized_texts.append(normalize_text(recognized_text))
    
    return all_recognized_texts

def highlight_matches(image, detections, target_word):
    matched_boxes = []
    for box, recognized_text in detections:
        if fuzz.partial_ratio(recognized_text, target_word) >= 70:  # Use your defined threshold
            matched_boxes.append(box)
            x1, y1, x2, y2 = map(int, box)
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image, matched_boxes

def process_image(image_path, target_word):
    image = preprocess_image(image_path)
    detections = detect_text(image, detection_model)
    recognized_texts = recognize_text_boxes(image, detections, recognition_model, processor)

    detections_with_text = list(zip(detections, recognized_texts))
    highlighted_image, matched_boxes = highlight_matches(image.copy(), detections_with_text, target_word)
    
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(result_image_path, cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
    
    return result_image_path, matched_boxes

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
            start_time = time.time()
            result_image_path, matched_boxes = process_image(filepath, target_word)
            end_time = time.time()
            print(f"Processing time: {end_time - start_time} seconds")
            return render_template('index.html', filename=os.path.basename(result_image_path), target_word=target_word, matches=matched_boxes)
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

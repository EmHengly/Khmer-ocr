from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import onnxruntime as ort
import cv2
import numpy as np
from fuzzywuzzy import fuzz
from transformers import (
    TrOCRProcessor,
    ViTFeatureExtractor,
    NllbTokenizer,
    VisionEncoderDecoderModel
)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load ONNX models
detection_session = ort.InferenceSession('train18/weights/best.onnx')

# Load TRoCR components
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/trocr-small-printed')
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Define the TRoCR model class for inference
class TRoCRModel:
    def __init__(self, model_name):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, pixel_values):
        with torch.no_grad():
            # Prepare inputs for the model
            outputs = self.model(pixel_values=pixel_values)
            # Decode the generated ids to text
            generated_ids = torch.argmax(outputs.logits, dim=-1)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Initialize the model with your checkpoint directory
trocr_model = TRoCRModel('checkpoint-21180')

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

def detect_text(image, session):
    image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Prepare inputs for ONNX model
    inputs = {'images': image}
    outputs = session.run(None, inputs)
    
    # Extract bounding boxes and confidences
    boxes, confidences = outputs[0], outputs[1]
    boxes = [box for box, conf in zip(boxes, confidences) if conf > 0.5]
    return boxes

def recognize_text(image, detections, model, processor, target_word, threshold=80):
    matching_boxes = []
    for box in detections:
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        roi = image[y1:y2, x1:x2]
        # Preprocess ROI for TRoCR (resize and normalize)
        roi = cv2.resize(roi, (384, 384))  # Adjust as per TRoCR's input size
        pixel_values = processor(images=roi, return_tensors="pt").pixel_values

        # Use the TRoCR model for prediction
        keyword_text = model.predict(pixel_values)
        
        # Fuzzy matching
        similarity = fuzz.ratio(keyword_text, target_word)
        if similarity >= threshold:
            matching_boxes.append((box, keyword_text))
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
    detections = detect_text(image, detection_session)
    matching_detections = recognize_text(image, detections, trocr_model, processor, target_word, threshold)
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
            result_image_path = process_image(filepath, target_word, threshold=80)
            return render_template('index.html', filename=os.path.basename(result_image_path), target_word=target_word)
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

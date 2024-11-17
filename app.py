from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Configure the Google Gemini API
genai.configure(api_key="AIzaSyBjooTOhwlZmslBwpq584KzNu7hY13lIkY")

# Set directory for uploaded images
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')  # Ensure you have the 'yolov8m.pt' model file

# Define confidence threshold
threshold = 0.5

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Get the story features input (optional)
            story_features = request.form.get('story_features', '').strip()

            return redirect(url_for('display_results', img_name=file.filename, story_features=story_features))
    return render_template('upload.html')

@app.route('/results/<img_name>')
def display_results(img_name):
    story_features = request.args.get('story_features', '')  # Get the story features from the URL parameter
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    img = Image.open(img_path)

    # Run inference with YOLOv8
    results = model(img)
    detections = results[0].boxes  # YOLOv8 stores predictions in `boxes`
    filtered_preds = [det for det in detections if det.conf > threshold]

    # Calculate precision
    precision = len(filtered_preds) / len(detections) if len(detections) > 0 else 0

    # Extract labels of detected objects
    detected_objects = [model.names[int(det.cls)] for det in filtered_preds]
    objects = ', '.join(detected_objects)
    
    # Generate the story prompt
    prompt = f"""You are a story generator. Your task is to 
    generate a story given a list of words. The story must contain all 
    the words in the list. The story can use synonyms of the words given in the list. If a person 
    is mentioned in the list, avoid using any gender-specific references 
    for them. Generate a short story given the following list of words: {objects}"""

    if story_features:
        prompt += f"\nAdditionally, the story should include the following features: {story_features}"

    # Generate story using Google Gemini API
    response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
    story = response.text if response else "Story generation failed."

    # Return the results as HTML
    return render_template('results.html', img_name=img_name, precision=precision, prompt=prompt, story=story)

@app.route('/history')
def history():
    # List all images in the upload folder
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('history.html', images=images)

@app.route('/delete/<img_name>')
def delete_image(img_name):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    if os.path.exists(img_path):
        os.remove(img_path)
    return redirect(url_for('history'))

@app.route('/recheck/<img_name>')
def recheck_image(img_name):
    return redirect(url_for('display_results', img_name=img_name))

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

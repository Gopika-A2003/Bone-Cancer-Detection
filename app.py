import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend for rendering images

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from io import BytesIO
import base64
from flask import Flask, request, jsonify, render_template
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)

# Create the inference client with the API URL and key
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="t6T0pTV4rzmFKX0siajY"
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file to a temporary location
        image_path = 'temp_image.jpg'
        file.save(image_path)

        # Run the inference
        result = CLIENT.infer(image_path, model_id="bone-cancer-segmentation/1")

        # Check for cancer or no cancer in the result
        if result['predictions'] and 'cancer' in result['predictions'][0]['class']:
            diagnosis = "Cancer Detected"
        else:
            diagnosis = "No Cancer Detected"

        # Load the image
        img = mpimg.imread(image_path)

        # Create a plot with the image and the polygon
        fig, ax = plt.subplots()
        ax.imshow(img)

        if result['predictions']:
            points = result['predictions'][0]['points']
            points = [(pt['x'], pt['y']) for pt in points]

            polygon = patches.Polygon(points, closed=True, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(polygon)

        plt.title(f"Segmentation Result: {diagnosis}")

        # Save the plot to a BytesIO object
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        plt.close(fig)
        img_stream.seek(0)

        # Convert the image to base64
        img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

        # Return the result template with the image and diagnosis
        return render_template('result.html', diagnosis=diagnosis, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)

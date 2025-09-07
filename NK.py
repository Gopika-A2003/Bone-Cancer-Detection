from flask import Flask, request, jsonify, render_template_string
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from io import BytesIO
import base64
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Create the inference client with the API URL and key
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="t6T0pTV4rzmFKX0siajY"
)


@app.route('/', methods=['GET'])
def index():
    return '''
       <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Detection - Single Page</title>
    <!-- Bootstrap CDN -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            margin: 0;
            padding: 0;
        }
        .navbar {
            background-color: #343a40;
        }
        .navbar a {
            color: white !important;
        }
        section {
            padding: 60px 0;
        }
        #about {
            background-color: #ffffff;
        }
        #case {
            background-color: #f1f1f1;
        }
        h2 {
            text-align: center;
            margin-bottom: 40px;
            color: #343a40;
        }
        .footer {
            background-color: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
        }
    </style>
</head>

<body>

    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="#">Cancer Detection</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#home">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#about">About</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#case">Case</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#contact">Contact Us</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Home Section -->
    <section id="home" style="background: url('https://max-website20-images.s3.ap-south-1.amazonaws.com/Bone_Cancer_9e4c9187d7.jpg') no-repeat center center / cover; height: 100vh; display: flex; align-items: center; justify-content: center;">
        <div class="container text-center">
            <h1 class="display-4 text-white">Welcome to Cancer Detection System</h1>
            <p class="lead text-white">Upload and detect cancerous images using advanced AI techniques.</p>
        </div>
    </section>

    <!-- About Section -->
    <section id="about">
        <div class="container">
            <h2>About the Project</h2>
            <p class="text-center">
                This project utilizes cutting-edge machine learning and deep learning techniques to enhance cancer detection capabilities. Specifically, we have integrated 
                <strong>YOLOV10</strong> (You Only Look Once, version 10), a state-of-the-art object detection model, with <strong>EfficientNet B3</strong>, a highly efficient 
                convolutional neural network for segmentation. The combination of these two powerful models allows us to precisely segment and identify cancerous regions in medical 
                images, such as X-rays, CT scans, and MRIs.
            </p>

            <p class="text-center">
                By leveraging YOLOV10’s ability to detect multiple objects in real time and EfficientNet B3’s optimized performance for image classification and segmentation, 
                we aim to significantly improve the accuracy and speed of detecting cancerous tissues. This approach helps in early diagnosis, reduces the number of false positives 
                and false negatives, and minimizes the need for unnecessary biopsies or surgeries.
            </p>

            <p class="text-center">
                Our team is continuously refining this system, focusing on enhancing model accuracy, reducing computational complexity, and making the solution more scalable 
                and accessible. The ultimate goal is to provide a tool that can assist medical professionals in diagnosing bone cancer and other critical conditions at an earlier 
                stage, leading to better patient outcomes and more effective treatment plans.
            </p>
            <div class="col-md-6 image-section">
                <img src="your-segmentation-image-url.jpg" alt="Segmentation Process">
                <p class="text-center">Segmentation of medical images using YOLOV10 and EfficientNet B3.</p>
            </div>
        </div>

        <!-- Second Row for Loss and Accuracy -->
        <div class="row mt-4">
            <div class="col-md-6">
                <img src="Figure_1.png" alt="Loss and Accuracy" class="img-fluid">
                <p class="text-center">Loss and Accuracy Graph</p>
            </div>
            <div class="col-md-6">
                <p>
                    <strong>Loss and Accuracy Analysis:</strong> 
                    During the training process, we monitored both the loss and accuracy of the model. The goal was to minimize the loss function while 
                    maximizing accuracy. Through iterative training, we were able to achieve remarkable performance, with accuracy consistently improving over time. 
                    The graph on the left shows the reduction in loss and the increase in accuracy as the model learns to make more precise predictions.
                </p>
            </div>
        </div>
        
        <div class="row text-center mt-5">
            <div class="col-md-6">
                <h4>Web Developer</h4>
                <p>Naresh Krishnan.M</p>
            </div>
            <div class="col-md-6">
                <h4>Python Developer</h4>
                <p>Team Member Name</p>
            </div>
        </div>
    </section>

    <!-- Case Section -->
    <section id="case">
        <div class="container">
            <h2>The Impact of Bone Cancer</h2>
            <p class="text-center">
                Bone cancer is a rare but serious form of cancer that originates in the cells of the bone. It typically affects the long bones of the body, such as those in the arms and legs, but it can develop in any bone. The condition weakens the bone structure, making it prone to fractures, causing severe and persistent pain, swelling, and limited mobility.
            </p>

            <p class="text-center">
                There are different types of bone cancer, such as osteosarcoma, chondrosarcoma, and Ewing's sarcoma, each of which affects various parts of the bone and responds to treatment differently. Osteosarcoma, for example, is most commonly found in children and adolescents, while chondrosarcoma is more prevalent in adults. Early diagnosis is critical because bone cancer can spread to other parts of the body, such as the lungs, making treatment more challenging and reducing the chances of a full recovery.
            </p>

            <p class="text-center">
                Symptoms can vary depending on the type of bone cancer and its location, but common signs include swelling and tenderness around the affected area, persistent bone pain, and an increase in fractures. Diagnostic procedures such as X-rays, MRIs, CT scans, and biopsies are essential for confirming the presence of cancer and determining the best treatment options.
            </p>

            <p class="text-center">
                Our project aims to aid in the early detection of bone cancer using machine learning techniques, which can analyze imaging data more quickly and accurately than traditional methods. This technology could lead to more timely interventions, ultimately improving the chances of successful treatment and recovery.
            </p>
        </div>
    </section>

    <!-- Contact Section -->
    <section id="contact" style="background-color: #ffffff;">
        <div class="container">
            <h2>Contact Us</h2>
            <form>
                <div class="form-group">
                    <label for="name">Name:</label>
                    <input type="text" class="form-control" id="name" required>
                </div>
                <div class="form-group">
                    <label for="email">Email:</label>
                    <input type="email" class="form-control" id="email" required>
                </div>
                <div class="form-group">
                    <label for="message">Message:</label>
                    <textarea class="form-control" id="message" rows="5" required></textarea>
                </div>
                <button type="submit" class="btn btn-primary">Submit</button>
            </form>
        </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>© 2024 Cancer Detection Project. All rights reserved.</p>
        </div>
    </footer>

    <!-- Bootstrap JS and dependencies -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>

</html>


    '''

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

        # Return the image with the result
        return render_template_string('''
            <html>
                <body>
                    <h2>Result: {{ diagnosis }}</h2>
                    <img src="data:image/png;base64,{{ img_base64 }}" />
                    <br><br>
                    <a href="/">Upload another image</a>
                </body>
            </html>
        ''', diagnosis=diagnosis, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)

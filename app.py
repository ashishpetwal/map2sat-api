from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import time

app = Flask(__name__)

os.makedirs('outputs', exist_ok=True)

model = tf.keras.models.load_model('model/map2sat_final.h5')

def preprocess_image(image_data):
    """Prepare the image for model input"""
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1
    img_array = np.expand_dims(img_array, 0)
    return img_array


def postprocess_image(generated_image):
    """Convert model output to viewable image"""
    img = (generated_image[0] * 0.5 + 0.5)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Map to Satellite Image Generator API!"

@app.route('/generate', methods=['POST'])
def generate_satellite_image():
    print(request.files)
    if 'map_image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['map_image']
        img_data = file.read()
        
        input_image = preprocess_image(img_data)
        
        generated_image = model(input_image, training=False)
        
        output_image = postprocess_image(generated_image)

        timestamp = int(time.time())
        filename = f'outputs/satellite_image_{timestamp}.jpg'
        output_image.save(filename)
        print(f"Saved generated image to {filename}")
        
        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'satellite_image': img_str})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
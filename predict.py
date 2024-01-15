from flask import Flask, request, jsonify, render_template
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
import tflite_runtime.interpreter as tflite
import numpy as np
import io
from PIL import Image

# Declare a flask app
app = Flask(__name__)


# Load the tensorflowlite model
# interpreter = tf.lite.Interpreter(model_path="MobileNetv2.tflite")
interpreter = tflite.Interpreter(model_path="MobileNetv2.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method =='POST':
        # Check if a valid image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the file into a BytesIO object
        filestream = io.BytesIO(file.read())
        
        # using PIL library instead of tf
        with Image.open(filestream) as img:
            img = img.resize((224, 224), Image.NEAREST)

        def prepare_input(x):
            return x / 255.0
        
        # img = image.load_img(filestream, target_size=(224,224))
        # x = image.img_to_array(img) / 255.0
        x = np.array(img, dtype=np.float32)
        img_array = np.array([x])
        img_array = prepare_input(img_array)

        interpreter.set_tensor(input_index, img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)

        # predictions = model.predict(img_array)
        index = np.argmax(predictions[0])

        classes = {0: 'dew',
                    1: 'fog/smog',
                    2: 'frost',
                    3: 'glaze',
                    4: 'hail',
                    5: 'lightning',
                    6: 'rain',
                    7: 'rainbow',
                    8: 'rime',
                    9: 'sandstorm',
                    10: 'snow'}
        
        top_prediction = classes[index]

        return jsonify({'prediction': top_prediction})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

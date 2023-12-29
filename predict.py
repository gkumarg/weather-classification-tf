from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io


# print a nice greeting.
def say_hello():
    return '<p>Weather Classification Project</p>\n' 

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service to predict weather from Image! </p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'


app = Flask(__name__)

# add a rule for the index page.
app.add_url_rule('/', 'index', (lambda: header_text +
    say_hello() + instructions + footer_text))

# Load the TensorFlow model
# model = tf.keras.models.load_model('xception_v3_28_0.825.h5')

# Load the tensorflowlite model
interpreter = tf.lite.Interpreter(model_path="MobileNetv2.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the file into a BytesIO object
    filestream = io.BytesIO(file.read())
    img = image.load_img(filestream, target_size=(224,224))
    x = image.img_to_array(img) / 255.0
    img_array = np.array([x])

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)

    # predictions = model.predict(img_array)
    index = np.argmax(predictions[0])

    classes = {0: 'dew',
                1: 'fogsmog',
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Architecture_keras_cnn.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    img_buff = np.zeros((1, 256, 256, 3))
    img_file_path = img_path
    img = cv2.imread(img_file_path)
    img = cv2.resize(img,(256, 256))
    img_buff[0] = img/255
    
    preds = model.predict(img_buff)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('templates/index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        res = 0
        for i in range(6):
            if preds[0][i] > res:
                res = i
        res_class = ['Chuan', 'Hui', 'Jin', 'Jing', 'Min', 'Su']  
        result = res_class[res]             
        return result
    return None


if __name__ == '__main__':
     app.run(port=5002, debug=True)
    # app.run(host='0.0.0.0', threaded=False)
    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()

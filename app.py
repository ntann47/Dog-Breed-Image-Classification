from flask import Flask, redirect, url_for, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import os
app = Flask(__name__)
cwd = os.getcwd()
path = os.path.join(cwd, "..\\static\\images")
app.config['IMAGES_FOLDER'] = path
# model = load_model(
#     'DogBreedClassificatiion_VGG16.h5')


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


def resize_image(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    width, height = image.size
    if width == height:
        image = image.resize((224, 224), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            image = image.crop((left, top, right, bottom))
            image = image.resize((224, 224), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            image = image.crop((left, top, right, bottom))
            image = image.resize((224, 224), Image.ANTIALIAS)
    numpy_image = np.array(image)
    return numpy_image


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = 'C:/Users/An Ngo/Desktop/Neural_D/DogBreedClassification/static/images/' + imagefile.filename
    imagefile.save(image_path)
    img_arr = cv2.imread(image_path)
    im = resize_image(img_arr)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 224, 224, 3)
    test_predictions = model.predict(ar)
# get model predictions
    maxnum = np.argmax(test_predictions, axis=1)
    pred_prob = test_predictions.max() * 100
    if maxnum == 0:
        result = f'Bernese Mountain Dog - Confidence: {pred_prob:.1f}%'
    elif maxnum == 1:
        result = f'Border Collie - Confidence: {pred_prob:.1f}%'
    elif maxnum == 2:
        result = f' Chihuahua - Confidence: {pred_prob:.1f}%'
    elif maxnum == 3:
        result = f'Corgi - Confidence: {pred_prob:.1f}%'
    elif maxnum == 4:
        result = f'Dachshund - Confidence: {pred_prob:.1f}%'
    elif maxnum == 5:
        result = f'Golden Retriever - Confidence: {pred_prob:.1f}%'
    elif maxnum == 6:
        result = f'Jack Russell Terrier - Confidence: {pred_prob:.1f}%'
    elif maxnum == 7:
        result = f'Labrador - Confidence: {pred_prob:.1f}%'
    elif maxnum == 8:
        result = f'Pug - Confidence: {pred_prob:.1f}%'
    elif maxnum == 9:
        result = f'Siberian Husky - Confidence: {pred_prob:.1f}%'
    else:
        result = 'Not Recognize'
    return render_template('index.html', img=imagefile.filename, predict=result)


if __name__ == "__main__":
    app.run(debug=True)

# print(app)

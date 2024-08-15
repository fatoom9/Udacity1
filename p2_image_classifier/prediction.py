import json
import time
import numpy as np
import os
import argparse as arg
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import sys
# from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing image

#model_path = sys.argv[2]
#model = load_model(model_pathp)

# define helper functions
def load_model(path):
    model_path = './' + path
    model = tf.keras.models.load_model(model_path ,custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    return model
                                                 
def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image    
def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    preds = model.predict(processed_test_image)
    probs = - np.partition(-preds[0], top_k)[:top_k]
    classes = np.argpartition(-preds[0], top_k)[:top_k]
    return probs, classes
#argument parsing
parser = arg.ArgumentParser()
parser.add_argument('img_path', type=str, help='/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/test_images')
parser.add_argument('model_path', type=str, help='/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/model_1723579057.h5')
parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to return')
parser.add_argument('--category_names', type=str, required=True, help='/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/label_map.json')
args = parser.parse_args()
#load model
model = load_model(args.model_path)
#make prediction
prob, classes = predict(image_path=args.img_path, model=model, top_k=args.top_k)
#load category namess
with open(args.category_names, 'r') as f:
    class_names = json.load(f)
    classes = [class_names[str(i)] for i in classes]
    
print('Predictions:', classes)
print('Probability:', prob)
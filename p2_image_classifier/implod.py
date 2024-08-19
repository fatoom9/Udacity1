
import json
import numpy as np
import os
import argparse as ap
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# to train model model
def trained(model_file):
    model_path = './' + model_file
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    return model

# to preprocess the image
def prepro_image(img):
    target_size = 224
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (target_size, target_size))
    img /= 255.0
    img = img.numpy()
    return img

#  to perform prediction
def performs_prediction(image_file, model, top_n=5):
    img = Image.open(image_file)
    img_array = np.asarray(img)
    processed_img = prepro_image(img_array)
    processed_img = np.expand_dims(processed_img, axis=0)
    predictions = model.predict(processed_img)
    probs = -np.partition(-predictions[0], top_n)[:top_n]
    classes = np.argpartition(-predictions[0], top_n)[:top_n]
    return probs, classes

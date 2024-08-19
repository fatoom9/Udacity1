import json
import numpy as np
import os
import argparse as ap
import sys
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from implod import trained, performs_prediction, prepro_image
parser = ap.ArgumentParser(description='')
parser.add_argument('img_path', type=str, help='/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/test_images.')
parser.add_argument('model_path', type=str, help='/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/images_classifier_model.h5')
parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
parser.add_argument('--category_names', type=str, required=True, help='/workspace/intro-to-ml-tensorflow/projects/p2_image_classifier/label_map.json')
args = parser.parse_args()
model = trained(args.model_path)
probabilities, class_indices = performs_prediction(image_file=args.img_path, model=model, top_n=args.top_k)
with open(args.category_names, 'r') as file:
    label_map = json.load(file)
    class_names = [label_map[str(i)] for i in class_indices]
print('Predictied Classes:', class_names)
print('class Probability:', probabilities)
##OUTPUT##
#1/1 [==============================] - 1s 670ms/step
#Predictied Classes: ['orange dahlia', 'english marigold', 'oxeye daisy', 'barbeton daisy', 'bishop of llandaff']
#class Probability: [0.29069862 0.12756142 0.03205685 0.04335629 0.07347481]
#p2_image_classifier student$ 
#---------------------------------
#*
#1/1 [==============================] - 1s 669ms/step
#Predictied Classes: ['pink primrose', 'balloon flower', 'wild pansy', 'clematis', 'morning glory']
#class Probability: [0.03345407 0.03750329 0.6954854  0.02298914 0.0127093 ]
#p2_image_classifier student$ 

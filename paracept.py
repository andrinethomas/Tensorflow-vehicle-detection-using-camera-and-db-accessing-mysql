#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:55:47 2018

@author: tensorflow-cuda
"""

import numpy as np
import os
import sys
import tensorflow as tf

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytesseract

from custom_plate import allow_needed_values as anv
from custom_plate import do_image_conversion as dic
from custom_plate import sql_inserter_fetcher as sif

sys.path.append("..")


MODEL_NAME = 'numplate'
PATH_TO_CKPT = MODEL_NAME + '/graph-200000/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
NUM_CLASSES = 1



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


pytesseract.tesseract_cmd = '/home/tensorflow-cuda/dharun_custom/models/research/object_detection/tessdata/'

def accept_and_die(image_path, dandt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
            ymin = boxes[0,0,0]
            xmin = boxes[0,0,1]
            ymax = boxes[0,0,2]
            xmax = boxes[0,0,3]
            (im_width, im_height) = image.size
            (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
            img_data = sess.run(cropped_image)
            count = 0
            filename = dic.yo_make_the_conversion(img_data, count)
            count+=1
            text = pytesseract.image_to_string(Image.open(filename),lang=None)
            yo = anv.catch_rectify_plate_characters(text)
            print('CHARACTER RECOGNITION : ',yo)
            if yo!='':
#               sif.store_whatever(yo, dandt)
               os.rename(image_path, "folder_processings/images/copy/{}.jpg".format(dandt))
            os.remove(image_path)
            return yo

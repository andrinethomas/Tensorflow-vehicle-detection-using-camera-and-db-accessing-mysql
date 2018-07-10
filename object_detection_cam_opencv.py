
import cv2
import numpy as np
import os
import tensorflow as tf


from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


cap = cv2.VideoCapture('rtsp://admin:kgisl@123@10.100.10.192/doc/page/preview.asp')
#cap = cv2.VideoCapture('http://192.168.43.83:8080/video')
#cap.set(cv2.CAP_PROP_FPS,300)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
    ret, image_np = cap.read()
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
    cv2.imshow('image',cv2.resize(image_np,(1280,960)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        cv2.destroyAllWindows
        cap.release


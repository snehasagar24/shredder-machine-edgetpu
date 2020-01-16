# Utilities for object detector.

from edgetpu.detection.engine import DetectionEngine
import numpy as np
#import tensorflow as tf
import os
import cv2
from utils import label_map_util
from utils import alertcheck
from PIL import Image
#detection_graph = tf.Graph()

TRAINED_MODEL_DIR = '/frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.getcwd() + TRAINED_MODEL_DIR + '/hand_edgetpu.tflite'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.getcwd() + TRAINED_MODEL_DIR + '/Glove_label_map.txt'

NUM_CLASSES = 2
# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}
#print(PATH_TO_LABELS, PATH_TO_CKPT)


# loop over the class labels file
for row in open(PATH_TO_LABELS):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()

a=b=0

# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== Loading frozen graph into memory")
    model = DetectionEngine(PATH_TO_CKPT)
    print(">  ====== Inference graph loaded.")
    return model

# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

def draw_box_on_image(results, im_width, im_height, image_np, Line_Position2, Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b
    hand_cnt=0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
   
    for r in results:
      
        # extract the bounding box and box and predicted class label
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        label = labels[r.label_id]

        # draw the bounding box and label on the image
        cv2.rectangle(image_np, (startX, startY), (endX, endY),
                      (0, 255, 0), 1)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = "{}: {:.2f}%".format(label, r.score * 100)
       

        cv2.putText(image_np, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        (left, right, top, bottom) = (box[0] , box[2] ,box[1] , box[3] )
                                      
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        
        dist = distance_to_camera(avg_width, focalLength, int(right - left))
        if dist:
            hand_cnt = hand_cnt + 1
        a = alertcheck.drawboxtosafeline(image_np, p1, p2, Line_Position2, Orientation)
        if hand_cnt==0:
            b=0
        else:
            b=1

    return a,b

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, model, score_thresh):
    image_np = Image.fromarray(image_np)
    # make predictions on the input frame
    results = model.DetectWithImage(image_np, threshold = score_thresh,
        keep_aspect_ratio=True, relative_coord=False)
    return results

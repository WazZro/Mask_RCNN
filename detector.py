import os
import sys
import cv2 as cv
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

import mrcnn.visualize as visualize
import mrcnn.visualize_video as visualize_video
import mrcnn.model as modellib
import mrcnn.config as configM
from mrcnn import utils
from mrcnn.model import MaskRCNN

ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='Video file for detection')
parser.add_argument('-w', '--web', help='Web cam for detection')
parser.add_argument('-i', '--image', help='Image file for detection')
parser.add_argument('-c', '--cpu', action='store_true',
                    help='Use CPU for detection')

params = parser.parse_args()
if params.video is not None:
    print('Video')
    SOURCE = params.video
    isVideo = True
elif params.image is not None:
    print('Image')
    SOURCE = params.image
    isVideo = False
elif params.web is not None:
    print('WEB')
    SOURCE = int(params.web)
    isVideo = True
else:
    isVideo = True
    SOURCE = 0

if params.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class MaskRCNNConfig(configM.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    # в датасете COCO находится 80 классов + 1 фоновый класс.
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.7

def counts(result):
    objects = {}

    for res in result:
        try:
            objects[res] += 1
        except:
            objects[res] = 1

    return objects


def out(res):
    for name, value in zip(res.keys(), res.values()):
        print(class_names[name], ":", value)


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load weights trained on MS-COCO
if not os.path.isfile(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

model.load_weights(COCO_MODEL_PATH, by_name=True)

if (isVideo):
    video_capture = cv.VideoCapture(SOURCE)
    colors = visualize_video.random_colors(len(class_names))
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        #frame = cv.resize(frame, None, fx=0.3, fy=0.3)
        image_batch = [frame]

        results = model.detect(image_batch, verbose=0)
        r = results[0]

        masked_image_batch = []
        for i in range(len(results)):
            r = results[i]
            im = image_batch[i]
            masked_image = visualize_video.get_masked_fixed_color(im, r['rois'], r['masks'], r['class_ids'],
                                                                  class_names, colors, r['scores'], show=False)
            #masked_image = cv.resize(masked_image, None, fx=3, fy=3)
            masked_image_batch.append(masked_image)

        cv.imshow('Masked image', masked_image_batch[0])

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()
else:
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, SOURCE))
    image = cv.imread(os.path.join(IMAGE_DIR, SOURCE), 1)
    image = image[:, :, ::-1]
    results = model.detect([image], verbose=0)

    r = results[0]
    counts = counts(r['class_ids'])
    out(counts)

    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
import sys, os, distutils.core
import torch, detectron2
import detectron2
from detectron2.utils.logger import setup_logger
import json

!python -m pip install pyyaml==5.1
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))


!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

setup_logger()

cfg = get_cfg()
cfg.merge_from_file("/content/drive/MyDrive/Detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ('short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 'sling_dress',)
cfg.DATASETS.TEST = ()  
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000   # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = "model path"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ('short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 'sling_dress',)
predictor = DefaultPredictor(cfg)


MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress', 'long_sleeve_dress', 'vest_dress', 'sling_dress',])

path = "/content/drive/MyDrive/DeepFashion2/test/9.png"

im = cv2.imread(path)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(v.get_image()[:, :, ::-1])

first_instance = outputs["instances"].to("cpu")[0]

bbox = first_instance.pred_boxes.tensor.cpu().numpy().squeeze()
x1, y1, x2, y2 = bbox
roi = im[int(y1):int(y2), int(x1):int(x2)]


metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

predictions = outputs["instances"].to("cpu")
category_names = [metadata.thing_classes[i] for i in predictions.pred_classes]
confidence_scores = predictions.scores.tolist()

results = []
for category, score in zip(category_names, confidence_scores):
    result = {
        "category": category,
        "score": score
    }
    results.append(result)

results.sort(key=lambda x: x["score"], reverse=True)

highest_confidence_result = results[0]["category"]

categories_dict = {"categories": [highest_confidence_result]} 
categories_json = json.dumps(categories_dict, indent=4)
"""
Initialization of the person detector model and methods for detection

@author: Angel Villar-Corrales
"""

import os

import numpy as np
import cv2
import torch
from torch.nn import DataParallel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from lib.logger import log_function, print_
from lib.visualizations import visualize_bbox


@log_function
def setup_detector():
    """
    Initializing person detector and loading pretrained model paramters
    """

    # intializing model skeleton
    print_("Initializing Person Detector")
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model = DataParallel(model)

    # loading pretrained weights
    print_("Loading detector pretrained parameters ")
    pretrained_path = os.path.join(os.getcwd(), "resources", "coco_person_detector.pth")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


@log_function
def person_detection(img_path, model):
    """
    Computing a forward pass throught the model to detect the persons in the image
    """

    # loading image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = torch.Tensor(img.transpose(2,0,1)[np.newaxis,:])
    print_(img.shape)

    # forward pass
    print_("Computing forward pass through person detector")
    outputs = model(img / 255)
    boxes, labels, scores = bbox_filtering(outputs, filter_=1, thr=0.7)

    # saving intermediate result
    print_("Obtaining intermediate detector visualization")
    title = "Person Detection Bboxes"
    img_vis = img[0,:].cpu().numpy().transpose(1,2,0) / 255
    visualize_bbox(img=img_vis, boxes=boxes[0], labels=labels[0],
                   scores=scores[0], title=title, savefig=True)

    return


@log_function
def bbox_filtering(predictions, filter_=1, thr=0.6):
    """
    Filtering predicitions in order to keep only the relevant bounding boxes #
    (people in our particular case)

    Args:
    -----
    predictions: list
        list containign a dictionary with all predicted bboxes, labels and scores:
            bbox: numpy array
                Array of shape (N, 4) where N is the number of boxes detected.
                The 4 corresponds to x_min, y_min, x_max, y_max
            label: numpy array
                Array containing the ID for the predicted labels
            scores: numpy array
                Array containing the prediction confident scores
    filter: list
        list containing label indices that we wnat to keep
    thr: float
        score threshold for considering a bounding box
    """

    filtered_bbox, filtered_labels, filtered_scores = [], [], []
    for pred in predictions:
        bbox, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
        cur_bbox, cur_labels, cur_scores = [], [], []
        for i, _ in enumerate(labels):
            if(labels[i] == filter_ and scores[i] > thr):
                aux = bbox[i].cpu().detach().numpy()
                reshaped_bbox = [aux[0], aux[1], aux[2], aux[3]]
                cur_bbox.append(reshaped_bbox)
                # cur_bbox.append(bbox[i].cpu().detach().numpy())
                cur_labels.append(labels[i])
                cur_scores.append(scores[i])
        # if(len(cur_bbox) == 0):
            # continue
        filtered_bbox.append(cur_bbox)
        filtered_labels.append(cur_labels)
        filtered_scores.append(cur_scores)


    return filtered_bbox, filtered_labels, filtered_scores


#

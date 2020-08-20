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
from lib.visualizations import visualize_bbox, visualize_img
from lib.transforms import TransformDetection


DETS_EXTRACTOR = None

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
    model = model.eval()

    # intiializing object for extracting detections
    global DETS_EXTRACTOR
    DETS_EXTRACTOR = TransformDetection(det_width=192, det_height=256)

    return model


@log_function
def person_detection(img_path, model):
    """
    Computing a forward pass throught the model to detect the persons in the image

    Args:
    -----
    img_path: string
        path to the image to extract the detections from
    model: NN Module
        Person detector model already Initialized

    Returns:
    --------
    savepath: string
        path where the image with the detected instances and bounding boxes is stored
    det_paths: list
        list with the path where the person detection images are stored
    """

    # loading image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = torch.Tensor(img.transpose(2,0,1)[np.newaxis,:])

    # forward pass through person detector
    print_("Computing forward pass through person detector...")
    outputs = model(img / 255)
    boxes, labels, scores = bbox_filtering(outputs, filter_=1, thr=0.7)

    # saving image with bounding boxes as intermediate results and for displaying
    # on the client side
    print_("Obtaining intermediate detector visualization...")
    img = img[0,:].cpu().numpy().transpose(1,2,0) / 255
    img_name = os.path.basename(img_path)
    savepath = os.path.join(os.getcwd(), "data", "intermediate_results", img_name)
    visualize_bbox(img=img, boxes=boxes[0], labels=labels[0],
                   scores=scores[0], savefig=True, savepath=savepath)

    # extracting the detected person instances and saving them as independent images
    print_("Extracting person detections from image...")
    detections, centers, scales = DETS_EXTRACTOR(img=img, list_coords=boxes[0])
    n_dets = detections.shape[0]
    det_paths = []
    for i, det in enumerate(detections):
        det_name = img_name.split(".")[0] + f"_det_{i}." + img_name.split(".")[1]
        det_path = os.path.join(os.getcwd(), "data", "final_results", "detection", det_name)
        det_paths.append(det_path)
        visualize_img(img=det.transpose(1,2,0), savefig=True, savepath=det_path)

    return savepath, det_paths


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

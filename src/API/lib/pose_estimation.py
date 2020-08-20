"""
Methods for pose estimation and keypoint detection

@author: Angel Villar-Corrales
"""

import os

import numpy as np
import cv2
import torch
from torch.nn import DataParallel
import torchvision.transforms as transforms

KEYPOINT_ESTIMATOR = None
NORMALIZER = None

def setup_pose_estimator():
    """
    Initializing the pretrained pose estimation model
    """

    print_("Initializing Pose Estimation model...")
    model = models.PoseHighResolutionNet(is_train=False)

    print_("Loading pretrained model parameters...")
    pretrained_path = os.path.join(os.getcwd(), "resources", "coco_hrnet_w32_256x192.pth")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.eval()

    # intiializing preprocessing method
    global NORMALIZER
    NORMALIZER = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

    return model


def pose_estimation(detections, centers, scales):
    """
    Detecting the keypoints for the input images and parsing the human poses

    Args:
    -----
    detections: numpy array
        person dectections from the input images. Shape is (n_dets, 3, 256, 192)
    centers: numpy array
        center coordinates of each person detection
    scales: numpy array
        scale factor for each person detection

    Returns:
    --------
    TODO
    """

    # initializing the model if necessary
    global KEYPOINT_ESTIMATOR
    if(KEYPOINT_ESTIMATOR is None):
        KEYPOINT_ESTIMATOR = setup_pose_estimator()

    # preprocessing the detections
    print_("Preprocessing person detections...")
    norm_detections = [normalize(torch.Tensor(det)).numpy() for det in detections]

    # forward pass through the keypoint detector model
    print_("Computing forward pass through the keypoint detector model...")
    keypoint_dets = KEYPOINT_ESTIMATOR(torch.Tensor(norm_detections).float())

    

    return

#

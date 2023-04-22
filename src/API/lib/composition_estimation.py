"""
Initialization of the person detector model, and pose estimation for composition estimation

@author: Prathmesh Madhu
"""

import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from lib.logger import log_function, print_
from lib.visualizations import visualize_bbox, visualize_img, draw_pose
from lib.transforms import TransformDetection
from lib.neural_nets.EfficientDet import EfficientDetBackbone as EfficientDet
from lib.neural_nets.HRNet import PoseHighResolutionNet
from lib.utils import create_directory
from lib.pose_parsing import get_final_preds_hrnet, get_max_preds_hrnet, create_pose_entries
from lib.pose_utils import hrnet_to_compoelem_poses, get_pose_lines, get_pose_lines_hrnet

DETECTOR_NAME = None
DETECTOR = None
DETS_EXTRACTOR = None

ESTIMATOR_NAME = None
KEYPOINT_ESTIMATOR = None
NORMALIZER = None


@log_function
def setup_detector(detector_name, database=None):
    """
    Initializing person detector and loading pretrained model paramters
    """

    # intializing model skeleton for the given model type
    print_(f"Initializing Person Detector: {detector_name}")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.eval()

    # loading pretrained weights
    print_("Loading detector pretrained parameters...")
    if(detector_name == "Faster R-CNN"):
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_faster_rcnn.pth")
        print_(f"    Loading: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
    else:
        raise ValueError("Detector not supported.")

    model.load_state_dict(checkpoint)
    model = model.eval()

    # intiializing object for extracting detections
    global DETS_EXTRACTOR
    DETS_EXTRACTOR = TransformDetection(det_width=192, det_height=256)

    return model


@log_function
def setup_pose_estimator(estimator_name):
    """
    Initializing the pretrained pose estimation model
    """

    print_("Initializing Pose Estimation model...")
    model = PoseHighResolutionNet(is_train=False)

    print_("Loading pretrained model parameters...")
    if(estimator_name == "Baseline HRNet"):
        pretrained_path = os.path.join(os.getcwd(), "resources", "coco_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = DataParallel(model)
    elif(estimator_name == "Styled HRNet"):
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "styled_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(checkpoint)
    elif(estimator_name == "Tuned HRNet"):
        # TODO: replace with actual tuned HRNet model
        model = DataParallel(model)
        pretrained_path = os.path.join(os.getcwd(), "resources", "styled_hrnet_w32_256x192.pth")
        checkpoint = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
        model.load_state_dict(checkpoint)
    model = model.eval()

    # intiializing preprocessing method
    global NORMALIZER
    NORMALIZER = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

    return model


@log_function
def composition_estimation(img_path, person_detector, keypoint_detector, database=None):
    """
    Computing a forward pass from the person-detection and pose-estimation model to
    generate image-compositions

    Args:
    -----
    img_path: string
        path to the image to extract the detections from
    person_detector: string
        name of the person detector model to use ['faster_rcnn']
    keypoint_detector: string
        name of the pose estimation model to use ['HRNet']

    Returns:
    --------
    pose_savepath: string
        path where the image with the pose-estimation instances highlighted is stored
    obj_savepath: string
        path where the image with the object-detection instances highlighted is stored
    pose_data: dictionary
        dict containing the results from the pose-estimator.
        Remember detections have fixed size of (256, 192)
    """
    # Detector Setup
    global DETECTOR
    global DETECTOR_NAME
    if (DETECTOR is None or DETECTOR_NAME != person_detector):
        DETECTOR_NAME = person_detector
        DETECTOR = setup_detector(detector_name=person_detector, database=database)

    # loading image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = torch.Tensor(img.transpose(2, 0, 1)[np.newaxis, :])

    # forward pass through person detector
    print_("Computing forward pass through person detector...")
    outputs = DETECTOR(img / 255)
    boxes, labels, scores = bbox_filtering(outputs, filter_=1, thr=0.7)

    # saving image with bounding boxes as intermediate results and for displaying
    # on the client side
    print_("Obtaining intermediate detector visualization...")
    img = img[0, :].cpu().numpy().transpose(1, 2, 0) / 255
    img_name = os.path.basename(img_path)
    create_directory(os.path.join(os.getcwd(), "data", "icc", "intermediate_bbox_results"))
    obj_savepath = os.path.join(os.getcwd(), "data", "icc", "intermediate_bbox_results", img_name)

    # case for image with no detections
    if (len(labels[0]) == 0 and os.path.exists(savepath)):
        os.remove(savepath)
        visualize_img(img=img, savefig=True, savepath=obj_savepath)
    # case for image with bounding boxes
    else:
        visualize_bbox(img=img, boxes=boxes[0], labels=labels[0],
                       scores=scores[0], savefig=True, savepath=obj_savepath)

    # extracting the detected person instances and saving them as independent images
    print_("Extracting person detections from image...")
    try:
        detections, centers, scales = DETS_EXTRACTOR(img=img, list_coords=boxes[0])
    except Exception as e:
        detections, centers, scales = [], [], []

    # skipping images with no person detections
    if (len(detections) == 0):
        pose_data = {
            "indep_pose_entries": np.array([]),
            "indep_all_keypoints": np.array([]),
            "pose_entries": np.array([]),
            "all_keypoints": np.array([]),
            "pose_paths": []
        }
        return pose_data

    # Pose-estimator Setup
    global KEYPOINT_ESTIMATOR
    global ESTIMATOR_NAME
    if (KEYPOINT_ESTIMATOR is None or ESTIMATOR_NAME != keypoint_detector):
        ESTIMATOR_NAME = keypoint_detector
        KEYPOINT_ESTIMATOR = setup_pose_estimator(keypoint_detector)

    # preprocessing the detections
    print_("Preprocessing person detections...")
    norm_detections = [NORMALIZER(torch.Tensor(det)).numpy() for det in detections]

    # forward pass through the keypoint detector model
    print_("Computing forward pass through the keypoint detector model...")
    keypoint_dets = KEYPOINT_ESTIMATOR(torch.Tensor(norm_detections).float())
    scaled_dets = F.interpolate(keypoint_dets.clone(), (256, 192),
                                mode="bilinear", align_corners=True)

    # extracting keypoint coordinates and confidence values from heatmaps
    print_("Extracting keypoints from heatmaps...")
    keypoint_coords, \
    max_vals_coords = get_max_preds_hrnet(scaled_heats=scaled_dets.detach().cpu().numpy())
    keypoints, max_vals, _ = get_final_preds_hrnet(heatmaps=keypoint_dets.detach().cpu().numpy(),
                                                   center=centers, scale=scales)

    # parsing poses by combining and joining keypoits
    print_("Parsing human poses...")
    indep_pose_entries, indep_all_keypoints = create_pose_entries(keypoints=keypoint_coords,
                                                                  max_vals=max_vals_coords,
                                                                  thr=0.1)
    indep_all_keypoints = [indep_all_keypoints[:, 1], indep_all_keypoints[:, 0], \
                           indep_all_keypoints[:, 2], indep_all_keypoints[:, 3]]
    indep_all_keypoints = np.array(indep_all_keypoints).T
    pose_entries, all_keypoints = create_pose_entries(keypoints=keypoints,
                                                      max_vals=max_vals,
                                                      thr=0.1)
    all_keypoints = [all_keypoints[:, 1], all_keypoints[:, 0], \
                     all_keypoints[:, 2], all_keypoints[:, 3]]
    all_keypoints = np.array(all_keypoints).T
    # all_individual_keypoints = np.split(all_keypoints, int(len(all_keypoints)/17))
    # print(f"All keypoints are: {all_keypoints}")
    # # print(f"All individual keypoints array is {all_individual_keypoints}")

    # print_(all_keypoints)
    # all_keypoints_hrnet = hrnet_to_compoelem_poses(all_keypoints)
    # print_(all_keypoints_hrnet)
    # all_poselines = get_pose_lines_hrnet(all_individual_keypoints)
    # print_(all_poselines)

    # creating pose visualizations and saving in the corresponding directory
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    create_directory(os.path.join(os.getcwd(), "data", "icc", "final_results", "pose_estimation"))
    pose_savepath = os.path.join(os.getcwd(), "data", "icc", "final_results", "pose_estimation", img_name)
    draw_pose(img / 255, pose_entries, all_keypoints, savefig=True, savepath=pose_savepath)
    pose_paths = []
    for i, det in enumerate(detections):
        det_name = img_name.split(".")[0] + f"_det_{i}." + img_name.split(".")[1]
        pose_i_savepath = os.path.join(os.getcwd(), "data", "icc", "final_results",
                                "pose_estimation", det_name)
        pose_paths.append(pose_i_savepath)
        draw_pose(norm_detections[i], [indep_pose_entries[i]], indep_all_keypoints,
                  preprocess=True, savefig=True, savepath=pose_i_savepath)

    # returning pose data in correct format
    pose_data = {
        "indep_pose_entries": indep_pose_entries,
        "indep_all_keypoints": indep_all_keypoints,
        "pose_entries": pose_entries,
        "all_keypoints": all_keypoints,
        "pose_paths": pose_paths
    }

    return pose_savepath, obj_savepath, pose_data


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

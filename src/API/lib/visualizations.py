"""
Methods for visualization and generation of images

@author: Angel Villar-Corrales
"""

import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import torch

from lib.logger import log_function, print_
from lib.transforms import unnormalize

# colors for drawing the human poses. Similar parts have similar colors, e.g.,
#   left-ankle2left-knee and left-hip2left-knee have redish hues
#   face lines have yellowish colors
COLORS = {
    "0": "red",             # left ankle -  left knee
    "1": "tomato",          # left knee  -  left hip
    "2": "lime",            # left hip   - left shoulder
    "3": "royalblue",       # right hip  - right knee
    "4": "navy",            # right knee - right ankle
    "5": "green",           # right hip  - right shoulder
    "6": "bisque",
    "7": "palegoldenrod",
    "8": "khaki",
    "9": "moccasin",
    "10": "wheat",
    "11": "fuchsia",        # left elbow     - left wrist
    "12": "deeppink",       # left shouldsr  - left elbow
    "13": "lawngreen",      # left shoulder  - right shoulder
    "14": "aqua",           # right shoulder - right elbow
    "15": "turquoise",      # right elbow    - right wrist
    "16": "darkorange",     # left shoulder  - left ear
    "17": "orange",         # right shoudler - right ear
    "18": "saddlebrown"
}

# connections between keypoints for parsing the pose skeleton
SKELETON = [[15, 13], [13, 11], [11, 5], [12, 14], [14, 16], [12, 6], [3, 1], [1, 2],
            [1, 0], [0, 2], [2, 4], [9, 7], [7, 5], [5, 6], [6, 8], [8, 10], [3, 5],
            [4, 6]]


@log_function
def visualize_img(img, **kwargs):
    """
    Visualizing an image accouting for the BGR format
    """

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8, 8)

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    if("preprocess" in kwargs):
        img = unnormalize(torch.Tensor(img))
        img = img.numpy().transpose(1,2,0)

    ax.imshow(img)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])

    if("savefig" in kwargs and kwargs["savefig"]==True):
        if("savepath" not in kwargs):
            savepath = os.path.join(os.getcwd(), "data", "final_results", "cur_img.png")
        else:
            savepath = kwargs["savepath"]
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)

    return


@log_function
def visualize_bbox(img, boxes, labels=None, scores=None, ax=None, **kwargs):
    """
    Visualizing the bounding boxes and scores predicted by the faster rcnn model

    Args:
    -----
    img: numpy array
        RGB image that has been predicted
    boxes: numpy array
        Array of shape (N, 4) where N is the number of boxes detected.
        The 4 corresponds to y_min, x_min, y_max, x_max
    labels: numpy array
        Array containing the ID for the predicted labels
    scores: numpy array
        Array containing the prediction confident scores
    """

    # initializing axis object if necessary
    if(ax is None):
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(6,6)

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    if("preprocess" in kwargs and kwargs["preprocess"] == True):
        img = unnormalize(torch.Tensor(img))
        img = img.numpy().transpose(1,2,0)
    ax.imshow(img)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])

    # in case of no detections
    if len(boxes) == 0:
        return ax

    # displaying BBs
    for i, bb in enumerate(boxes):
        x, y = bb[0], bb[1]
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        ax.add_patch(plt.Rectangle((x, y), width, height, fill=False,
                                    edgecolor='red', linewidth=2))

        message = None
        if(scores is not None):
            cur_score = scores[i]
            message = "Score: {:.2f}".format(cur_score)
        if(message is not None):
            ax.text(bb[0], bb[1], message, style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    if("savefig" in kwargs and kwargs["savefig"]==True):
        if("savepath" not in kwargs):
            savepath = os.path.join(os.getcwd(), "data", "intermediate_results", "bbox_img.png")
        else:
            savepath = kwargs["savepath"]
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)

    return


@log_function
def draw_pose(img, poses, all_keypoints, **kwargs):
    """
    Overlaying the predicted poses on top of the images
    """

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8, 8)

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    if("preprocess" in kwargs and kwargs["preprocess"] == True):
        img = unnormalize(torch.Tensor(img))
        img = img.numpy().transpose(1,2,0)
    ax.imshow(img)

    for pose in poses:
        for idx, limb in enumerate(SKELETON):
            idx_a, idx_b = int(pose[limb[0]]), int(pose[limb[1]])
            if(idx_a == -1 or idx_b == -1):
                continue
            a, b = all_keypoints[idx_a], all_keypoints[idx_b]
            if(a[-1] == 0 or b[-1] == 0):
                continue
            color = COLORS[str(idx)]
            line = mlines.Line2D(
                    np.array([a[1], b[1]]), np.array([a[0], b[0]]),
                    ls='-', lw=5, alpha=1, color=color)
            circle1 = mpatches.Circle(np.array([a[1], a[0]]), radius=5,
                                     ec='black', fc=color,
                                     alpha=1, linewidth=5)
            circle2 = mpatches.Circle(np.array([b[1], b[0]]), radius=5,
                                     ec='black', fc=color,
                                     alpha=1, linewidth=5)
            line.set_zorder(1)
            circle1.set_zorder(2)
            circle2.set_zorder(2)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.add_line(line)

    if("savefig" in kwargs and kwargs["savefig"]==True):
        if("savepath" not in kwargs):
            savepath = os.path.join(os.getcwd(), "data", "final_results", "cur_skeleton.png")
        else:
            savepath = kwargs["savepath"]
        plt.axis("off")
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)

    return

#

"""
Methods for visualization and generation of images

@author: Angel Villar-Corrales
"""

import os

import numpy as np
from matplotlib import pyplot as plt
import torch

from lib.logger import log_function, print_
from lib.transforms import unnormalize


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

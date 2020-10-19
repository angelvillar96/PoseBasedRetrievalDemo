"""
Auxiliary methods for util purposes

@author: Angel Villar-Corrales
"""

import os
import base64
import json
import datetime

from PIL import Image
import numpy as np


def create_directory(path):
    """
    Checking if a directory exists, and creating it if it doesnt
    """

    if(not os.path.exists(path)):
        os.makedirs(path)

    return


def timestamp():
    """
    Obtaining the current timestamp in an human-readable way

    Returns:
    --------
    timestamp: string
        current timestamp in format hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0] \
                                            .replace(' ', '_') \
                                            .replace(':', '-')

    return timestamp


def save_image_from_post(data, path):
    """
    Saving image obtained from the POST Json into the given path. The image might
    be given as a .png/.jpg file or as a blob

    Args:
    ----
    data: FileStorage object
        Image contained in a FileStorage object. It might be an image format or a blob
    path: string
        path where the image will be saved
    """

    if(data.content_type == "blob"):
        img = Image.open(data.stream)
        img.save(path)
    else:
        data.save(path)

    return


def encode_img(path):
    """
    Encoding an image into binary to send it to the clien side in the Json response

    Args:
    -----
    path: string
        path where the final image is stored

    Returns:
    --------
    encoded_img: string
        encode version of the input data and casted as a string to fit into a json object
    """

    if(not os.path.exists(path)):
        return None
    with open(path, "rb") as file:
        encoded_img = str(base64.b64encode(file.read()))
        encoded_img = encoded_img[2:-1]

    return encoded_img


def preprocess_pose_arrays(pose_vector, keypoints):
    """
    Processing pose and keypoints to convert them from strings (from the post form)
    to numpy arrays or lists
    """

    pose_vector = pose_vector.split(",")[:-2]
    pose_vector = [int(p) for p in pose_vector]

    keypoints = keypoints.split(",")
    keypoints = np.array([int(k) for k in keypoints])
    n_keypoints = len(keypoints) // 4
    keypoints = keypoints.reshape(n_keypoints, 4)

    return pose_vector, keypoints

#

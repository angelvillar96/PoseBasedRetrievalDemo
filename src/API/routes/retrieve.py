"""
Receiving pose and keypoint vectors from the client side and retrieving the most
similar ones from the database based on some metric

@author: Angel Villar-Corrales
"""

import os
import base64
import cv2

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.retrieve import RetrieveSchema
# from models.retrive import RetriveModel
from lib.logger import log_function, print_
from lib.pose_based_retrieval import pose_based_retrieval
from lib.utils import preprocess_pose_arrays, encode_img


retrieve_api = Blueprint('api/retrieve', __name__)
@retrieve_api.route('/', methods=['POST'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Data processed successfully!',
            'schema': RetrieveSchema
        }
    }
})
@log_function
def receive_data():
    """
    Receives a pose vector and keypoint coordinates to retrieve the similar poses from
    a database
    Receives a pose vector and keypoint coordinates to retrieve the similar poses from
    a database
    ---
    """

    # reading variables from the POST data and formatting them
    print_("Route '/api/retrieve' was called...")
    data = request.form
    det_idx = data["det_idx"]
    pose_vector = data["pose_vector"]
    keypoints = data["keypoints"]
    print_(f"Detection {det_idx} was selected for retrieval...")
    pose_vector, keypoints = preprocess_pose_arrays(pose_vector, keypoints)

    # pose based image retrieval
    pose_based_retrieval(kpt_idx=pose_vector, all_keypoints=keypoints)
    

    # for debugging purposes we return a placehodler image
    img_path = os.path.join(os.getcwd(), "resources", "science.jpg")
    images = [encode_img(path=img_path) for i in range(10)]
    import numpy as np
    dist = np.arange(10).tolist()
    json_data = {
        "images": images,
        "metrics": dist
    }

    response = jsonify(json_data)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200

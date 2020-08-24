"""
Receiving the data from the client side and processing the image

@author: Angel Villar-Corrales
"""

import os
import base64
import cv2

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.upload import UploadSchema
# from models.upload import UploadModel
from lib.utils import timestamp, save_image_from_post, encode_img
from lib.logger import log_function, print_
from lib.person_detection import setup_detector, person_detection
from lib.pose_estimation import setup_pose_estimator, pose_estimation


upload_api = Blueprint('api/upload', __name__)
@upload_api.route('/', methods=['POST'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Data processed successfully!',
            'schema': UploadSchema
        }
    }
})
@log_function
def receive_data():
    """
    Receives the image and metadata from a JSON and processing it
    Receives the image and metadata from a JSON and processing it
    ---
    """

    # relevant variables extracted from the POST data
    print_("Route '/api/upload' was called...")
    data = request.form
    files = request.files
    file_name = data["file_name"]

    # relevant paths
    file_path = os.path.join(os.getcwd(), "data", "imgs", file_name)
    final_path = os.path.join(os.getcwd(), "data", "final_results",
                              "others", file_name)

    # saving data in directory, cause why not?
    print_("Storing query in data directory...")
    storage_element = files["file"]
    save_image_from_post(data=storage_element, path=file_path)

    # person detection
    det_img_path, det_instances_path, det_data = person_detection(img_path=file_path)

    # pose estimation
    pose_data = pose_estimation(detections=det_data["detections"],
                                centers=det_data["centers"],
                                scales=det_data["scales"],
                                img_path=file_path)

    # saving final results
    img = cv2.imread(det_img_path, cv2.IMREAD_COLOR)
    cv2.imwrite(final_path, img)

    # sending the data back to the user. Encoding images to binary format and arrays
    # as lists so as to form a json response
    print_("Encoding results and returning response...")
    encoded_img = encode_img(path=final_path)
    encoded_dets = [encode_img(path=det_path) for det_path in det_instances_path]
    encoded_poses = [encode_img(path=pose_path) for pose_path in pose_data["pose_paths"]]
    indep_pose_entries = [entry.tolist() for entry in pose_data['indep_pose_entries']]
    indep_all_keypoints = pose_data['indep_all_keypoints'].tolist()
    json_data = {
        "img_name": os.path.basename(file_path),
        "img_url": file_path,
        "img_binary": encoded_img,
        "detections": encoded_dets,
        "poses": encoded_poses,
        "pose_vectors": indep_pose_entries,
        "keypoints": indep_all_keypoints
    }
    response = jsonify(json_data)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200


#

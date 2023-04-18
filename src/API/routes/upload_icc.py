"""
Receiving the data from the client side and processing the image

@author: Prathmesh Madhu
"""

import os
import base64
import cv2

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.upload import UploadSchema
# from models.upload import UploadModel
from lib.utils import timestamp, save_image_from_post, encode_img, create_directory
from lib.logger import log_function, print_
from lib.person_detection import setup_detector, person_detection
from lib.composition_estimation import composition_estimation


upload_icc_api = Blueprint('api/upload_icc', __name__)
@upload_icc_api.route('/', methods=['POST'])
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
    print_("Route '/api/upload_icc' was called...")
    data = request.form
    files = request.files
    file_name = data["file_name"]
    keypoint_detector = data["keypoint_detector"]
    database = data["retrieval_database"]
    # print(list(data.keys()), database, data["database"])

    # relevant paths/dirs for storing results
    imgs_path = os.path.join(os.getcwd(), "data", "imgs")
    create_directory(imgs_path)
    full_poses_path = os.path.join(os.getcwd(), "data", "icc", "final_results", "full_poses")
    create_directory(full_poses_path)
    file_path = os.path.join(imgs_path, file_name)
    final_path = os.path.join(full_poses_path, file_name)

    # saving data in directory, cause why not?
    print_("Storing query in data directory...")
    storage_element = files["file"]
    save_image_from_post(data=storage_element, path=file_path)


    # composition estimation
    person_detector = "Faster R-CNN"
    pose_img_path, obj_savepath, \
    pose_data = composition_estimation(img_path=file_path,
                                      person_detector=person_detector,
                                      keypoint_detector=keypoint_detector,
                                      database=database)
    pose_instances_path = pose_data['pose_paths']

    # saving final results
    img = cv2.imread(pose_img_path, cv2.IMREAD_COLOR)
    cv2.imwrite(final_path, img)

    # sending the data back to the user. Encoding images to binary format and arrays
    # as lists so as to form a json response
    print_("Encoding results and returning response...")
    encoded_img = encode_img(path=final_path)
    encoded_poses = [encode_img(path=pose_path) for pose_path in pose_instances_path]
    indep_pose_entries = [entry.tolist() for entry in pose_data['indep_pose_entries']]
    indep_all_keypoints = pose_data['indep_all_keypoints'].tolist()

    # print(encoded_dets)
    json_data = {
        "img_name": os.path.basename(file_path),
        "img_url": file_path,
        "img_binary": encoded_img,
        "poses": encoded_poses,
        "pose_vectors": indep_pose_entries,
        "keypoints": indep_all_keypoints
    }
    response = jsonify(json_data)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200


#

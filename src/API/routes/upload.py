"""
Receiving the data from the client side and processing the image

@author: Angel Villar-Corrales
"""

import os
import base64
import cv2
from PIL import Image

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.upload import UploadSchema
# from models.upload import UploadModel
from lib.utils import timestamp
from lib.logger import log_function, print_
from lib.person_detection import setup_detector, person_detection

DETECTOR = None
X = 200

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

    global DETECTOR

    print_("Route '/api/upload' was called...")
    data = request.form
    files = request.files
    file_name = data["file_name"]
    file_path = os.path.join(os.getcwd(), "data", "imgs", file_name)
    final_path = os.path.join(os.getcwd(), "data", "final_results",
                              "others", file_name)

    # saving data in directory, cause why not?
    print_("Storing query in data directory...")
    storage_element = files["file"]
    if(storage_element.content_type == "blob"):
        # with open(file_path, "wb") as file:
            # file.write(base64.decodebytes(storage_element.read()))
        img = Image.open(storage_element.stream)
        img.save(file_path)
    else:
        storage_element.save(file_path)

    # person detection
    # if(DETECTOR is None):
        # DETECTOR = setup_detector()
    # person_detection(img_path=file_path, model=DETECTOR)

    # TODO: pose estimation
    # TODO: retrieval

    # dummy processing
    global X
    X = X + 100
    print_("Dummy processing...")
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    cv2.putText(img, "Writing something for testing", (X,X), 0, 3, (0,0,255), 3)
    cv2.imwrite(final_path, img)

    # testing sending the data back to the user
    print_("Encoding result and returning response...")
    with open(final_path, "rb") as file:
        encoded_img = str(base64.b64encode(file.read()))
        encoded_img = encoded_img[2:-1]

    json_data = {
        "data_name": os.path.basename(file_path),
        "data_url": file_path,
        "data_binary": encoded_img
    }
    response = jsonify(json_data)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200


#

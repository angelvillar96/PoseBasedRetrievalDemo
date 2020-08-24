from http import HTTPStatus
from flask import Blueprint, jsonify
from flasgger import swag_from

from schemas.progress import ProgressSchema
from lib.logger import log_function, print_

progress_api = Blueprint('api/progress', __name__)

@progress_api.route('/')
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Progress was successfully fetched!',
            'schema': ProgressSchema
        }
    }
})
@log_function
def get_progress():
    """
    Obtaining current progress to update progress bar
    Obtaining current progress to update progress bar
    ---
    """

    print_("Route '/api/progress' was called...")

    progress = "50"
    task = "testing progress bar"

    result = {
        "progress": progress,
        "task": task
    }
    response = jsonify(result)

    return response, 200

from http import HTTPStatus
from flask import Blueprint, jsonify
from flasgger import swag_from

from schemas.home import ExampleSchema
from lib.logger import log_function, print_

example_api = Blueprint('api', __name__)

@example_api.route('/')
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'This is an example of how to create a route!',
            'schema': HomeSchema
        }
    }
})
@log_function
def example():
    """
    Example of how to create your own route
    Example of how to create your own route
    ---
    """

    print_("Route '/example' was callied...")


    return response, 200

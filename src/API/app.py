"""
Main app for the Flaks API for the 'Pose-based image retrieval' project

@author: Angel Villar-Corrales
"""

import os
from argparse import ArgumentParser

from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from routes.home import home_api
from routes.upload import upload_api
from routes.upload_object import upload_object_api
from routes.upload_icc import upload_icc_api
from routes.retrieve import retrieve_api
from routes.progress import progress_api
from lib.logger import Logger, log_function, print_

app = Flask(__name__)
cors = CORS(app)

@log_function
def create_app():
    """
    Initialization of the API
    """

    app.config['SWAGGER'] = {
        'title': 'Flask API Starter Kit',
    }
    swagger = Swagger(app)

    app.register_blueprint(home_api, url_prefix='/')
    app.register_blueprint(upload_api, url_prefix='/api/upload')
    app.register_blueprint(upload_object_api, url_prefix='/api/upload_object')
    app.register_blueprint(upload_icc_api, url_prefix='/api/upload_icc')
    app.register_blueprint(retrieve_api, url_prefix='/api/retrieve')
    app.register_blueprint(progress_api, url_prefix='/api/get_progress')
    app.run(debug=True,host='0.0.0.0')
    #
    return


if __name__ == '__main__':
    os.system("clear")
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    # Initializing logger
    logger = Logger(exp_path=os.getcwd())
    message = f"Starting API..."
    logger.log_info(message=message, message_type="start")

    app = create_app()


#

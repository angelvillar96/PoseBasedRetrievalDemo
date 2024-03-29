# Pose-Based Retrieval API

In this README, we explain in detail the installation procedure, the functionalities of the API and how to extend it for your own demos.


## Contents

 * [1. Getting Started](#getting-started)
 * [2. Directory Structure](#directory-structure)
 * [3. Adding New Routes](#adding-new-routes)
 * [4. Contact](#contact)


## Getting Started

Before running the API, make sure to install all required packages (e.g. Flask, PyTorch, Numpy or OpenCV). This can be done quickly by creating a virtual environment and installing all packages using the *requirements.txt* file.

To do so, run the following commands in the API root directory. This will install the *virtualenv* package, create a virtual environment named *ApiEnvironment*, activate it and install all requirements.

```bash
# installing the package
$ pip3 install virtualenv
# creating virtual environment
$ virtualenv ApiEnvironment
# activating on windows
$ .\ApiEnvironment\Scripts\activate
# activating on Linux/Mac
$ source ApiEnvironment/bin/activate
# installing requirements
(ApiEnvironment)$ pip3 install -r requirements.txt
```

Now you should be able to run the API. By default, the API will run in your local machine under port 5000.

```bash
(ApiEnvironment)$ python3 app.py
```

You can test that the API is running properly by typing `localhost:5000` in your browser. If g, you should be able to see a message confirming that the API is up and running.


## Directory Structure

The following directory tree illustrates the structure of the API.
**TODO**: Add *CONFIG:py* file where routes and directories are defined.

**Note**: Large files and directories (i.e, pretrained models, kNN trees or databases) are missing in the repository. You can download them from [here]() (Coming Soon!).

```
API
├── data/
│   ├── final_results/
│   ├── imgs/
│   └── intermediate_results/
|
├── database/
│   ├── imgs/
│   |   ├── arch_data
|   |   └── ...
│   └── knns/
|
├── lib/
|   ├── neural_nets/
|   |   ├── EfficeintDet.py
|   |   └── ...
|   ├── person_detection.py
|   ├── pose_based_retrieval.py
|   ├── pose_parsing.py
|   └── ...
|   
├── models/
|   ├── home.py
|   ├── retrieve.py
|   └── ...
|
├── resources/
|   ├── arch_faster_rcnn.pth
|   └── ...
|
├── routes/
|   ├── home.py
|   ├── retrieve.py
|   └── ...
|
├── schemas/
|   ├── home.py
|   ├── retrieve.py
|   └── ...
|
├── app.py
├── README.md
└── requirements.txt
```

We now give a short overview of the different directories and files, emphasizing their role and purpose:

 - **data/:** This directory is generated automatically during the execution of the API. Here, the query images as well as the intermediate and final results are stored. This automatic storage of all data allows us to perform loggin, debugging and visualization purposes. **TODO** Store metadata.

- **database/**: Under this directory we store the databases, images from datasets and objects used for retrieval purposes (e.g., kNN trees and graphs).

- **lib/**: Source code libraries containing methods and classes for different purposes: person detection, pose estimation, image retrueval, utils, visualization, ...  

- **models/** Coming soon

- **resources/**: Pretrained deep learning models used for person detection and pose estimation.

- **routes/**: Code for the API routes and calls. Each file under this directory handles a different route (e.g., uploading an image for person detection and pose estimation, or retrieving given a pose.)

- **schemas/**: Coming Soon

For more detailed documentation of the particular API calls, including input parameters and returned json forms, check the [official Swagger documentation](http://131.188.35.240:5000/apidocs/) (Coming Soon!).


## Adding New Routes

In this section, we show how to add new routes to establish API calls for your application.


## Contact

This project was developed by [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php)

In case of any questions or problems regarding the project or repository, do not hesitate to contact me at angelvillarcorrales@gmail.com.

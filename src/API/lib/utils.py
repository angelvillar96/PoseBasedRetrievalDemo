"""
Auxiliary methods for util purposes

@author: Angel Villar-Corrales
"""

import os
import json
import datetime


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



#

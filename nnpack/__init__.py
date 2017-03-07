
import os
import json
import shutil

import logging

logger = logging.getLogger("nnpack")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.DEBUG)
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)


def load_meta(path, filename):
    with open(path + filename, 'r') as f:
        return json.load(f)

def load_labels(path):
    labels_meta = load_meta(path, '/labels.json')
    index = {}
    for label in labels_meta['labels']:
        index[label['id']] = label
    return index

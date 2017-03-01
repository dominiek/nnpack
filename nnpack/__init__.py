
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


def load_meta(path):
    with open(path + '/index.json', 'r') as f:
        return json.load(f)

def load_labels(path):
    f = open(path + '/labels.jsons')
    index = {}
    for line in f:
        label = json.loads(line)
        index[label['id']] = label
    return index

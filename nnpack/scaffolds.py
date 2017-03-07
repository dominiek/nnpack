
import os
import json
import shutil
from . import load_meta, load_labels

import logging
logger = logging.getLogger("nnpack.scaffolds")

class InvalidScaffoldError(Exception):
    pass

def validate_scaffold(path):
    if not os.path.isdir(path):
        raise InvalidScaffoldError('Invalid Scaffold, should be a directory: {}'.format(path))
    if not os.path.isfile(path + '/nnscaffold.json'):
        raise InvalidScaffoldError('Invalid Scaffold, expected nnscaffold.json')
    try:
        meta = load_meta(path, '/nnscaffold.json')
    except Exception as e:
        raise InvalidScaffoldError('Could not load Scaffold meta data: {}'.format(str(e)))
    if not meta.has_key('id'):
        raise InvalidScaffoldError('Invalid Scaffold meta data, expected field `id` in nnscaffold.json')
    if not meta.has_key('name'):
        raise InvalidScaffoldError('Invalid Scaffold meta data, expected field `name` in nnscaffold.json')
    if not os.path.isfile(path + '/labels.json'):
        raise InvalidScaffoldError('Invalid Scaffold, expected labels.json')
    try:
        labels = load_labels(path)
    except Exception as e:
        raise InvalidScaffoldError('Could not load Scaffold labels: {}'.format(str(e)))
    for id in labels:
        if not labels[id].has_key('name'):
            raise InvalidScaffoldError('Missing name attribute for label {}'.format(id))
    if len(labels) < 1:
        raise InvalidScaffoldError('Expected at least 1 label in Scaffold')
    bounding_boxes_for_scaffold(path, validate=True)

def clear_scaffold_cache(scaffold_path):
    if os.path.isdir(scaffold_path + '/cache'):
        shutil.rmtree(scaffold_path + '/cache')

def bounding_boxes_for_scaffold(path, validate=False):
    if not os.path.isdir(path):
        raise Exception('Invalid model scaffold path: {}'.format(path))
    labels = load_labels(path)
    all_bounding_boxes = []
    for id in labels:
        bounding_boxes_path = path + '/images/' + id + '/bounding_boxes.json'
        if not os.path.isfile(bounding_boxes_path):
            if validate:
                continue
            else:
                raise Exception('No bounding boxes found for label {}: {}'.format(id, bounding_boxes_path))
        with open(bounding_boxes_path, 'r') as f:
            bounding_boxes = json.load(f)
            if not bounding_boxes.has_key('images'):
                raise Exception('Expected bounding_boxes.json to have an array of `images` defined')
            images = bounding_boxes['images']
            for image in images:
                image['image_path'] = path + '/images/' + id + '/' + image['image_path']
                image['label'] = labels[id]
            all_bounding_boxes = all_bounding_boxes + images
    return all_bounding_boxes

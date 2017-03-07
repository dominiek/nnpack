
import os
import json
import shutil
from . import load_meta, load_labels

import logging
logger = logging.getLogger("nnpack.models")

class InvalidModelError(Exception):
    pass

def validate_model(path):
    if not os.path.isdir(path):
        raise InvalidModelError('Invalid Model, should be a directory: {}'.format(path))
    #if not os.path.isdir(path + '/state'):
    #    raise InvalidModelError('Invalid Model, expected state directory: {}'.format(path + '/state'))
    if not os.path.isfile(path + '/nnpackage.json'):
        raise InvalidModelError('Invalid Model, expected nnpackage.json')
    try:
        meta = load_meta(path, '/nnpackage.json')
    except Exception as e:
        raise InvalidModelError('Could not load Model meta data: {}'.format(str(e)))
    if not meta.has_key('id'):
        raise InvalidModelError('Invalid Model meta data, expected field `id` in nnpackage.json')
    if not meta.has_key('name'):
        raise InvalidModelError('Invalid Model meta data, expected field `name` in nnpackage.json')
    if not os.path.isfile(path + '/labels.json'):
        raise InvalidModelError('Invalid Model, expected labels.json')
    try:
        labels = load_labels(path)
    except Exception as e:
        raise InvalidModelError('Could not load Model labels: {}'.format(str(e)))
    for id in labels:
        if not labels[id].has_key('name'):
            raise InvalidModelError('Missing name attribute for label {}'.format(id))
    if len(labels) < 1:
        raise InvalidModelError('Expected at least 1 label in Model')

def create_empty_model(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(path + '/state')

def save_model_benchmark_info(path, benchmark_info):
    state_dir = path + '/state'
    with open(state_dir + '/benchmark.json', 'w') as f:
        json.dump(benchmark_info, f)


import os
import json
import shutil
import time
import tensorflow as tf

import logging
logger = logging.getLogger("nnpack.engines.tensorflow")

def save_model_state(sess, path):
    state_dir = path + '/state'
    if os.path.isdir(state_dir):
        shutil.rmtree(state_dir)
    os.mkdir(state_dir)

    #sub_graph = graph_util.extract_sub_graph(sess.graph.as_graph_def(add_shapes=True), tensors)
    #tf.train.export_meta_graph(path + '/state/model.meta', graph_def=sub_graph)
    logger.info('Saving model to {} (num_tensors={}, tensor_namespaces={})'.format(path, len(get_tensors(sess)), ','.join(get_tensor_namespaces(sess))))
    saver = tf.train.Saver()
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    tf.train.export_meta_graph(path + '/state/model.meta', graph_def=graph_def, as_text=True)
    saver.save(sess, path + '/state/model', write_meta_graph=False)

def load_model_state(sess, path, namespace=None, exclude_meta=False):
    start_ts = time.time()
    if not os.path.isdir(path):
        raise Exception('No model dir found at {}'.format(path))
    if exclude_meta:
        saver = tf.train.Saver()
    else:
        saver = tf.train.import_meta_graph(path + '/state/model.meta')
    saver.restore(sess, path + '/state/model')
    logger.info('Loaded model from {} (took={}s, num_tensors={}, tensor_namespaces={})'.format(path, time.time()-start_ts, len(get_tensors(sess)), ','.join(get_tensor_namespaces(sess))))

def get_tensors(sess):
    layers = []
    for op in sess.graph.get_operations():
        layers.append(op.name)
    return layers

def get_tensor_namespaces(sess):
    namespaces = []
    for op in sess.graph.get_operations():
        path = op.name.split('/')
        if len(path) > 1 and path[0] not in namespaces:
            namespaces.append(path[0])
    return namespaces

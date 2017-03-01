
import os
import sys
import unittest
import shutil
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
import tensorflow as tf
from nnpack.models import *
from nnpack.engines._tensorflow import save_model_state, load_model_state, get_tensors
import logging
logger = logging.getLogger("nnpack")
logger.setLevel(logging.DEBUG)
import time

tmp_dir = test_dir + '/tmp'
examples_dir = test_dir + '/../examples'

if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)
os.mkdir(tmp_dir)

def create_graph():
    state = tf.Variable(0, name='state')
    increment = tf.Variable(1, name='increment')
    add = tf.add(state, increment)
    update = tf.assign(state, add, name='update')
    tf.Variable(99, name='unused')
    return state, update, increment

class ModelsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_validate(self):
        validate_model(examples_dir + '/models/scene_types')
        with self.assertRaises(InvalidModelError):
            validate_model(test_dir)

    def test_save_and_load_model_state_without_meta(self):
        tf.reset_default_graph()
        # Create TF graph
        state, update, increment = create_graph()

        # Create and init session
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        # Assert graph works as designed
        self.assertEqual(session.run(state), 0)
        session.run(update)
        self.assertEqual(session.run(state), 1)
        session.run(update, {increment: 2})
        self.assertEqual(session.run(state), 3)
        self.assertEqual(len(get_tensors(session)), 15)

        # Save state
        create_empty_model(tmp_dir + '/increment_without_meta')
        save_model_state(session, tmp_dir + '/increment_without_meta')

        # Clear graph
        tf.reset_default_graph()
        session = tf.Session()
        self.assertEqual(len(get_tensors(session)), 0)

        # Restore model fully from file
        #state, update, increment = create_graph()
        state, update, increment = create_graph()
        init = tf.global_variables_initializer()
        session.run(init)
        load_model_state(session, tmp_dir + '/increment_without_meta', exclude_meta=True)


        # Assert previous state
        self.assertEqual(session.run(state), 3)
        session.run(update, {increment: 3})
        self.assertEqual(session.run(state), 6)

    def test_save_and_load_model_state_with_meta(self):
        tf.reset_default_graph()
        # Create TF graph
        create_graph()

        # Create and init session
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)

        # Assert graph works as designed
        self.assertEqual(session.run('state:0'), 0)
        session.run('update')
        self.assertEqual(session.run('state:0'), 1)
        session.run('update', {'increment:0': 2})
        self.assertEqual(session.run('state:0'), 3)
        self.assertEqual(len(get_tensors(session)), 15)

        # Save state
        create_empty_model(tmp_dir + '/increment')
        save_model_state(session, tmp_dir + '/increment')

        # Clear graph
        tf.reset_default_graph()
        session = tf.Session()
        self.assertEqual(len(get_tensors(session)), 0)

        # Restore model fully from file
        load_model_state(session, tmp_dir + '/increment')

        # Assert previous state
        self.assertEqual(session.run('state:0'), 3)
        session.run('update', {'increment:0': 3})
        self.assertEqual(session.run('state:0'), 6)


if __name__ == "__main__":
    unittest.main()

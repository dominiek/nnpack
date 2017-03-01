
import os
import sys
import unittest
import time
test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')
examples_dir = test_dir + '/../examples'
import tensorflow as tf
from nnpack.scaffolds import *
import logging
logger = logging.getLogger("nnpack")
logger.setLevel(logging.DEBUG)
import time

class ScaffoldsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_validate(self):
        validate_scaffold(examples_dir + '/scaffolds/scene_types')
        with self.assertRaises(InvalidScaffoldError):
            validate_scaffold(test_dir)
        validate_scaffold(examples_dir + '/scaffolds/faces')


if __name__ == "__main__":
    unittest.main()

import tensorflow as tf
import numpy as np

from unittest import TestCase

from gaussian_detector.input.transform.transform_functions import slice_line_and_transform_to_float


class TestTransformFunctions(TestCase):
    def test_slice_line_and_transform_to_float(self):
        line = tf.constant("0.1,0.2,0.3,aaa", dtype=tf.string)
        line_to_float = slice_line_and_transform_to_float(line, [3])
        with tf.Session() as session:
            transformed_line = session.run(line_to_float)
        expected_line = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        self.assertTrue(np.array_equal(transformed_line, expected_line))

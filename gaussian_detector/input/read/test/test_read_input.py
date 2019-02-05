import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import tensorflow as tf

from gaussian_detector.input.read.read_input import read_input


class TestReadInput(TestCase):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    TRAIN_FILES_CNT = 10
    SAMPLES_PER_FILE = 1000
    FEATURES_CNT = 24

    @classmethod
    def setUpClass(cls):
        cls.setUpData()

    def test_read_input(self):
        batch_size = 10
        train_data_it = read_input(self.train_files[:1], batch_size, 1)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run([init, train_data_it.initializer])
            iterations = 0
            try:
                while True:
                    samples = session.run(train_data_it.get_next())
                    self.assertEqual(len(samples), batch_size)
                    iterations += 1
            except tf.errors.OutOfRangeError:
                pass
        self.assertEqual(iterations, self.SAMPLES_PER_FILE / batch_size)

    def test_read_input_with_map_function(self):
        batch_size = 10
        slice_cols = [self.FEATURES_CNT - 1]
        train_data_it = read_input(self.train_files, batch_size, 1,
                                   lambda line: self.input_line_slice_and_transform_to_float(line, slice_cols))
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run([init, train_data_it.initializer])
            iterations = 0
            try:
                while True:
                    samples = session.run(train_data_it.get_next())
                    self.assertEqual(samples.dtype, np.float32)
                    self.assertEqual(len(samples), batch_size)
                    self.assertEqual(samples.shape[1], self.FEATURES_CNT - 1)
                    iterations += 1
            except tf.errors.OutOfRangeError:
                pass
        self.assertEqual(iterations, self.TRAIN_FILES_CNT * self.SAMPLES_PER_FILE / batch_size)

    def test_read_with_multiple_epochs(self):
        batch_size = 10
        epochs = 3
        train_data_it = read_input(self.train_files[:1], batch_size, 3)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run([init, train_data_it.initializer])
            iterations = 0
            try:
                while True:
                    samples = session.run(train_data_it.get_next())
                    self.assertEqual(len(samples), batch_size)
                    iterations += 1
            except tf.errors.OutOfRangeError:
                pass
        self.assertEqual(iterations, epochs * self.SAMPLES_PER_FILE / batch_size)

    @staticmethod
    def input_line_slice_and_transform_to_float(line, slice_cols: int):
        line_splitted = tf.string_split([line], ",")
        str_data = tf.convert_to_tensor(line_splitted.values, dtype=tf.string)
        str_data = tf.slice(str_data, [0], slice_cols)
        float_data = tf.string_to_number(str_data, out_type=tf.float32)

        return float_data

    @classmethod
    def setUpData(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        cls.train_files = [cls.setUpTrainFile(i) for i in range(cls.TRAIN_FILES_CNT)]

    @classmethod
    def setUpTrainFile(cls, file_idx: int):
        data_array = np.random.normal(500, 100, cls.SAMPLES_PER_FILE * cls.FEATURES_CNT).reshape(
            (cls.SAMPLES_PER_FILE, cls.FEATURES_CNT))
        cols = ["feat{}".format(i) for i in range(data_array.shape[1])]
        df = pd.DataFrame(data_array, columns=cols)
        train_file_path = os.path.join(cls.DATA_DIR, "train-{}.csv".format(file_idx))
        df.to_csv(train_file_path, index=False)

        return train_file_path

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.DATA_DIR, ignore_errors=True)

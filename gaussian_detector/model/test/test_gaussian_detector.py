import os
from unittest import TestCase

import pandas as pd
import numpy as np

from gaussian_detector.graph.summary.writer.summary_writer import SummaryWriter
from gaussian_detector.model.gauss_detector import GaussDetector


class TestGaussianDetector(TestCase):
    PREPARED_TRAIN_FILES_DIR = os.path.join(os.path.dirname(__file__), "data/prepared")
    GENERATED_TRAIN_FILES_DIR = os.path.join(os.path.dirname(__file__), "data/generated")
    GRAPH_OUT_DIR = os.path.join(os.path.dirname(__file__), "summary")
    FEAT_CNT = 24
    SAMPLE_PER_FILE = 1440
    TRAIN_FILES_CNT = 5

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.GRAPH_OUT_DIR, exist_ok=True)
        cls.prepared_train_files = cls.list_files(cls.PREPARED_TRAIN_FILES_DIR)
        cls.setUpData()

    def test_train_model(self):
        feat_names = list(pd.read_csv(self.prepared_train_files[0]).columns.values)
        summary_writer = SummaryWriter(self.GRAPH_OUT_DIR)
        gauss_detector = GaussDetector(feat_names, 0.7, summary_writer, 1, 1)
        gauss_detector.train(self.prepared_train_files)
        print("Mean: {}, std: {}".format(gauss_detector.get_mean(), gauss_detector.get_std()))

    def test_train_model_random_files(self):
        expected_mean, expected_std = self.__calculate_train_files_mean_and_std()
        gauss_detector = self.__create_gauss_detector()
        mean = gauss_detector.get_mean()
        std = gauss_detector.get_std()
        mean_diff = np.linalg.norm(mean - expected_mean)
        self.assertLess(mean_diff, 1)
        std_diff = np.linalg.norm(std - expected_std)
        self.assertLess(std_diff, 1)

    def test_predict(self):
        gauss_detector = self.__create_gauss_detector()
        df = pd.read_csv(self.train_files[0])
        x = df.loc[:, df.columns != "_time"].as_matrix()
        predicted = gauss_detector.predict(x)
        self.assertEqual(len(predicted), len(x))

    def test_predict_proba(self):
        gauss_detector = self.__create_gauss_detector()
        df = pd.read_csv(self.train_files[0])
        x = df.loc[:, df.columns != "_time"].as_matrix()
        is_anomaly, total_proba, feat_proba = gauss_detector.predict_proba(x)
        self.assertEqual(len(is_anomaly), len(x))
        self.assertEqual(len(total_proba), len(x))
        self.assertEqual(feat_proba.shape, x.shape)

    def __create_gauss_detector(self):
        feat_names = self.__list_train_file_feat_names()
        summary_writer = SummaryWriter(self.GRAPH_OUT_DIR)
        gauss_detecctor = GaussDetector(feat_names, 0.7, summary_writer, 1, summary_step=200)
        test_files = self.train_files[:3]
        valid_files = self.train_files[3:]
        gauss_detecctor.train(self.train_files, test_files, valid_files)

        return gauss_detecctor

    def __calculate_train_files_mean_and_std(self):
        x_values = list()
        for train_file in self.train_files:
            df = pd.read_csv(train_file)
            x_values.append(df.loc[:, df.columns != "_time"].as_matrix())
        x = np.concatenate(x_values)

        return np.mean(x, axis=0), np.std(x, axis=0)

    @classmethod
    def setUpData(cls):
        if not os.path.isdir(cls.GENERATED_TRAIN_FILES_DIR):
            os.makedirs(cls.GENERATED_TRAIN_FILES_DIR)
        cls.train_files = [cls.setUpFile(i) for i in range(cls.TRAIN_FILES_CNT)]

    def __list_train_file_feat_names(self):
        return list(pd.read_csv(self.train_files[0]).columns.values)

    @classmethod
    def setUpFile(cls, file_idx: int):
        x = np.random.normal(300, 200, cls.FEAT_CNT * cls.SAMPLE_PER_FILE).reshape((cls.SAMPLE_PER_FILE, cls.FEAT_CNT))
        id = np.arange(len(x)).reshape((len(x), 1))
        x = np.concatenate((x, id), axis=1)
        cols = ["col%d" % i for i in range(x.shape[1] - 1)]
        cols.extend(["_time"])
        df = pd.DataFrame(x, columns=cols)
        train_file_path = os.path.join(cls.GENERATED_TRAIN_FILES_DIR, "train%d.csv" % file_idx)
        df.to_csv(train_file_path, index=False)

        return train_file_path

    @classmethod
    def list_files(self, dir_name):
        return [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]

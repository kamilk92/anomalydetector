from typing import List

import numpy as np
import tensorflow as tf
import logging

import time
import pandas as pd

from gaussian_detector.graph.summary.writer.summary_writer import SummaryWriter
from gaussian_detector.input.read.read_input import read_input
from gaussian_detector.input.transform import transform_functions


class GaussDetector(object):
    class ModelInput(object):
        def __init__(self, mean: tf.placeholder = None, std: tf.placeholder = None, input_data: tf.placeholder = None,
                     input_error_threshold: tf.placeholder = None):
            self.mean = mean
            self.std = std
            self.input_data = input_data
            self.input_error_threshold = input_error_threshold

    class ModelOutput(object):
        def __init__(self, is_anomaly=None, feat_proba=None, total_proba=None, anomaly_cnt=None):
            self.is_anomaly = is_anomaly
            self.feat_proba = feat_proba
            self.total_proba = total_proba
            self.anomaly_cnt = anomaly_cnt

    class Summarry(object):
        def __init__(self):
            self.means = list()
            self.stds = list()
            self.test_samples_anomalies_cnt = None
            self.test_anomalies_cnt_in = None  # type: tf.placeholder
            self.valid_samples_anomalies_cnt = None
            self.valid_anomalies_cnt_in = None  # type: tf.placeholder
            self.__merged_train_summaries = None
            self.__merged_test_summaries = None
            self.__merged_valid_summaries = None

        def get_train_summaries(self) -> list:
            return self.means + self.stds

        def get_test_summaries(self) -> list:
            return [self.test_samples_anomalies_cnt]

        def get_valid_summaries(self) -> list:
            return [self.valid_samples_anomalies_cnt]

        def merge_train_summaries(self):
            self.__merged_train_summaries = tf.summary.merge(self.get_train_summaries())

            return self.__merged_train_summaries

        def get_merged_train_summaries(self):
            return self.__merged_train_summaries

        def merge_test_summaries(self):
            self.__merged_test_summaries = tf.summary.merge(self.get_test_summaries())

            return self.__merged_test_summaries

        def get_merged_test_summaries(self):
            return self.__merged_test_summaries

        def merge_validation_summaries(self):
            self.__merged_valid_summaries = tf.summary.merge(self.get_valid_summaries())

            return self.__merged_valid_summaries

        def get_merged_valid_summaries(self):
            return self.__merged_valid_summaries

    def __init__(self, feature_names: List[str], error_threshold: float, summary_writer: SummaryWriter,
                 slice_cols=0, summary_step=100, save_model_path=None):
        self.__logger = logging.getLogger(__name__)
        self.__feature_names = feature_names
        self.__error_threshold = error_threshold
        self.__summary_writer = summary_writer
        self.__slice_cols = slice_cols
        self.__summary_step = summary_step
        self.__save_model_path = save_model_path
        self.__features_cnt = len(feature_names) - self.__slice_cols
        self.__mean = np.zeros((self.__features_cnt,), np.float32)
        self.__std = np.zeros((self.__features_cnt,), np.float32)
        self.__model_input = self.ModelInput()
        self.__model_out = self.ModelOutput()
        self.__summary = self.Summarry()
        self.__graph = None  # type: tf.Graph
        self.__global_train_step = 0

    def train(self, train_files: List[str], test_files: List[str] = None, valid_files: List[str] = None):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__train_with_custom_graph(train_files, test_files, valid_files)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out_is_anomaly, out_total_proba, out_feat_proba = self.__do_prediction(x, True)

        return out_is_anomaly, out_total_proba, out_feat_proba

    def predict(self, x: np.ndarray):
        out_is_anomaly = self.__do_prediction(x, False)

        return out_is_anomaly

    def get_mean(self) -> np.ndarray:
        return self.__mean

    def get_std(self) -> np.ndarray:
        return self.__std

    def evaluate(self, test_files: List[str], valid_files: List[str]):
        if test_files:
            self.__evaluate_test_files(test_files)
        if valid_files:
            self.__evaluate_valid_files(valid_files)

    def __evaluate_test_files(self, test_files: List[str]):
        self.__evaluate_multiple_files(test_files, self.__summary.get_merged_test_summaries(),
                                       self.__summary.test_anomalies_cnt_in)

    def __evaluate_valid_files(self, valid_files: List[str]):
        self.__evaluate_multiple_files(valid_files, self.__summary.get_merged_valid_summaries(),
                                       self.__summary.valid_anomalies_cnt_in)

    def __evaluate_multiple_files(self, files: List[str], merged_summaries, summaries_cnt_in: tf.placeholder):
        total_anomalies_cnt = 0
        for file in files:
            total_anomalies_cnt += self.__evaluate_file(file)
        session = tf.get_default_session()
        out_anomalies_cnt = session.run(merged_summaries, feed_dict={
            summaries_cnt_in: total_anomalies_cnt
        })
        self.__summary_writer.writer.add_summary(out_anomalies_cnt, self.__global_train_step)
        self.__summary_writer.writer.flush()

    def __evaluate_file(self, file: str):
        x = self.__collect_to_predict_x(file)
        session = tf.get_default_session()
        out_anomaly_cnt = session.run(self.__model_out.anomaly_cnt,
                                      feed_dict={
                                          self.__model_input.mean: self.__mean,
                                          self.__model_input.std: self.__std,
                                          self.__model_input.input_data: x,
                                          self.__model_input.input_error_threshold: self.__error_threshold
                                      })

        return out_anomaly_cnt

    def __train_with_custom_graph(self, train_files: List[str], test_files: List[str] = None,
                                  valid_files: List[str] = None):
        data_it = self.__read_input(train_files)
        next_data = data_it.get_next("input_line")
        std, mean = self.__train_model(next_data)
        in_ = self.__model_input
        out_ = self.__model_out
        out_.is_anomaly, out_.feat_proba, out_.total_proba, out_.anomaly_cnt, in_.mean, in_.std, in_.input_data, in_. \
            input_error_threshold = self.__predict_model()
        self.__create_summaries(mean, std)
        init_vars = tf.global_variables_initializer()
        train_summaries = self.__summary.merge_train_summaries()
        self.__summary.merge_test_summaries()
        self.__summary.merge_validation_summaries()
        with tf.Session() as session:
            self.__summary_writer.create(self.__graph)
            self.__logger.info("Training model...")
            train_start_time = self.__current_time_ms()
            session.run([init_vars, data_it.initializer])
            try:
                self.__global_train_step = 0
                while True:
                    self.__mean, self.__std = session.run([mean, std])
                    if (self.__global_train_step == 0) or (self.__global_train_step % self.__summary_step == 0):
                        self.evaluate(test_files, valid_files)
                        summaries_out = session.run(train_summaries)
                        self.__summary_writer.writer.add_summary(summaries_out, self.__global_train_step)
                        self.__summary_writer.writer.flush()
                    self.__global_train_step += 1
            except tf.errors.OutOfRangeError:
                train_end_time = self.__current_time_ms()
                self.__logger.info("Model trained in {} ms".format(train_end_time - train_start_time))
                self.__save_model()

    def __do_prediction(self, x: np.ndarray, with_proba=False):
        if not self.__graph:
            self.__read_model()

        with self.__graph.as_default():
            predicted = self.__do_prediction_with_custom_graph(x, with_proba)

        return predicted

    def __do_prediction_with_custom_graph(self, x: np.ndarray, with_proba=False):
        with tf.Session() as session:
            start_time = self.__current_time_ms()
            run_args = [self.__model_out.is_anomaly, self.__model_out.total_proba, self.__model_out.feat_proba] if \
                with_proba else self.__model_out.is_anomaly
            out = session.run(run_args, feed_dict={
                self.__model_input.mean: self.__mean,
                self.__model_input.std: self.__std,
                self.__model_input.input_data: x,
                self.__model_input.input_error_threshold: self.__error_threshold
            })
            end_time = self.__current_time_ms()
            self.__logger.debug("{} samples with proba={} predicted in {} ms.".format(len(x), with_proba,
                                                                                      end_time - start_time))

        return out

    def __read_input(self, train_files: List[str]):
        with tf.name_scope("read_input"):
            map_line_fn = lambda line: transform_functions.slice_line_and_transform_to_float(line,
                                                                                             [self.__features_cnt])
            data_it = read_input(train_files, 1, 1, map_line_fn)

        return data_it

    def __train_model(self, x):
        with tf.name_scope("calculate_mean_and_variance"):
            x = tf.reshape(x, (self.__features_cnt,))
            k = tf.get_variable("k", (), tf.float32, tf.zeros_initializer())  # type: tf.Variable
            k = k.assign_add(1)
            mean = tf.get_variable("mean", (self.__features_cnt,), tf.float32, tf.zeros_initializer())
            prev_mean = tf.get_variable("prev_mean", (self.__features_cnt,), tf.float32, tf.zeros_initializer())
            prev_mean = tf.assign(prev_mean, mean, name="assign_prev_mean")
            mean = tf.assign(mean, prev_mean + (x - prev_mean) / k, name="calculate_mean")
            variance = tf.get_variable("variance", (self.__features_cnt,), tf.float32, tf.zeros_initializer())
            variance = tf.assign(variance, variance + (x - prev_mean) * (x - mean), name="calculate_variance")
            std = tf.sqrt(variance / k, "calculate_std")

        return std, mean

    def __predict_model(self):
        with tf.name_scope("predict"):
            input_data = tf.placeholder(tf.float32, (None, self.__features_cnt), "x_to_predict")
            mean = tf.placeholder(tf.float32, (self.__features_cnt,), "final_mean")
            std = tf.placeholder(tf.float32, (self.__features_cnt,), "final_std")
            norm_dist = tf.distributions.Normal(mean, std, name="normal_distribution")
            proba = norm_dist.prob(input_data, "normal_prob")
            total_proba = tf.reduce_prod(proba, axis=1, name="total_prob")
            input_error_threshold = tf.placeholder(tf.float32, (), "error_threshold")
            is_anomaly = tf.less(total_proba, input_error_threshold, "is_anomaly")
            anomaly_cnt = tf.size(tf.where(is_anomaly), name="anomalies_cnt")

        return is_anomaly, proba, total_proba, anomaly_cnt, mean, std, input_data, input_error_threshold

    def __create_summaries(self, mean: tf.Tensor, std: tf.Tensor):
        for i in range(self.__features_cnt):
            feat_name = self.__feature_names[i]
            self.__summary.means.append(tf.summary.scalar("mean/{}".format(feat_name), mean[i]))
            self.__summary.stds.append(tf.summary.scalar("std/{}".format(feat_name), std[i]))

        self.__summary.test_anomalies_cnt_in = tf.placeholder(tf.float32, name="test_anomalies_cnt_in")
        self.__summary.test_samples_anomalies_cnt = tf.summary.scalar("evaluate/test",
                                                                      self.__summary.test_anomalies_cnt_in)
        self.__summary.valid_anomalies_cnt_in = tf.placeholder(tf.float32, name="valid_anomalies_cnt_in")
        self.__summary.valid_samples_anomalies_cnt = tf.summary.scalar("evaluate/validation",
                                                                       self.__summary.valid_anomalies_cnt_in)

    def __current_time_ms(self) -> int:
        return int(time.time() * 1000)

    def __read_model(self):
        if not self.__save_model_path:
            raise AttributeError("Model not trained and missing restore model path")

        raise NotImplementedError()

    def __save_model(self):
        self.__logger.error("Model not saved. Feature not implemented yet.")

    def __collect_to_predict_x(self, file: str):
        df = pd.read_csv(file)
        x = df.as_matrix()
        x = x[:, :self.__features_cnt]

        return x

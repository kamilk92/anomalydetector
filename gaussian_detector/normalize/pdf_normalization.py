from typing import Tuple

import tensorflow as tf
from dependency_injector.providers import Callable


class PdfNormalization(Callable):
    """
    https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
    """

    def __init__(self, features_cnt: int, precision=16):
        self.pdf_in, self.normalize_pdf = self.__build_model(features_cnt, precision)

    def __build_model(self, features_cnt, precision=16):
        pdf_in, x_out = self.__transform_input(features_cnt)
        threshold = self.__calculate_threshold(precision, features_cnt)
        x_out = self.__normalize(x_out, threshold, features_cnt)

        return pdf_in, x_out

    def __transform_input(self, features_cnt: Tuple[int, int]):
        pdf_in = tf.placeholder(tf.float64, (None, features_cnt), name="pdf")
        ln = tf.log(pdf_in, "log")
        max_ln = tf.reshape(tf.reduce_max(ln, 1, name="max_pdf"), (-1, 1))
        x_out = tf.subtract(ln, max_ln, "sub")

        return pdf_in, x_out

    def __calculate_threshold(self, precision: int, feats_cnt: int):
        epsilon = tf.constant(10 ^ -precision, dtype=tf.float64, name="epsilon")
        n = tf.constant(feats_cnt, tf.float64, name="n")
        threshold = tf.log(epsilon / n, "threshold")

        return threshold

    def __normalize(self, x_out, threshold, features_cnt: int):
        x_out = tf.exp(x_out, "exp")

        def replace_with_zero_values_below_threshold(x_row):
            less_than_threshold = tf.less(x_row, threshold)
            return tf.where(less_than_threshold, tf.zeros((features_cnt,), dtype=tf.float64), x_row)

        x_out = tf.map_fn(replace_with_zero_values_below_threshold, x_out)
        x_sum = tf.reshape(tf.reduce_sum(x_out, 1, name="sum"), (-1, 1))
        x_out = tf.divide(x_out, x_sum, name="divide")

        return x_out

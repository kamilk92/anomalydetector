import numpy as np
import tensorflow as tf

from unittest import TestCase

from gaussian_detector.normalize.pdf_normalization import PdfNormalization


class TestPdfNormalization(TestCase):
    def test_pdf_normalization(self):
        rows_cnt, features_cnt = 50, 24
        x = np.random.exponential(size=rows_cnt * features_cnt).reshape((rows_cnt, features_cnt)).astype(np.float64)
        x = np.divide(x, 1000)
        pdf_normalization = PdfNormalization(features_cnt)
        with tf.Session() as sess:
            x_out = sess.run(pdf_normalization.normalize_pdf, feed_dict={
                pdf_normalization.pdf_in: x
            })
        self.assertEqual(x_out.shape, x.shape)

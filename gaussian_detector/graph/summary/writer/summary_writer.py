import logging
import os
import tensorflow as tf
from datetime import datetime


class SummaryWriter(object):
    def __init__(self, graph_dir: str):
        self.__logger = logging.getLogger(__name__)
        self.graph_dir = graph_dir or os.getcwd()  # type: str
        self.graph_path = None  # type: str
        self.__writer = None

    def create(self, graph: tf.Graph = None) -> tf.summary.FileWriter:
        os.makedirs(self.graph_dir, exist_ok=True)
        self.graph_path = os.path.join(self.graph_dir, "run-{}".format(datetime.now().time().strftime("%Y%m%d%H%M%S")))
        self.__writer = tf.summary.FileWriter(self.graph_path, graph or tf.get_default_graph())
        self.__logger.info("Created summary writer inside directory: '{}'".format(self.graph_path))

        return self.__writer

    @property
    def writer(self) -> tf.summary.FileWriter:
        return self.__writer

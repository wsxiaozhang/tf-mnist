import os
import sys

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
ckpt_path='/training/tensorflow/logs/model.ckpt-0'
default_meta_graph_suffix='.meta'
meta_graph_file=ckpt_path + default_meta_graph_suffix
new_sess=tf.Session()
new_saver = tf.train.import_meta_graph(meta_graph_file,clear_devices=True)
new_saver.restore(new_sess, ckpt_path)
new_graph = tf.get_default_graph()
print("%d ops in the saved graph." % len(new_graph.as_graph_def().node))
print(new_graph.as_graph_def().node)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
for op in new_graph.get_operations():
  print(op.name, op.values())

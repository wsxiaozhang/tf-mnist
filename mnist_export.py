# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7
"""Train and export a simple Softmax Regression TensorFlow model.

The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.

Usage: mnist_export.py [--training_iteration=x] [--model_version=y] export_dir
"""

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print 'Please specify a positive value for training iteration.'
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print 'Please specify a positive value for version number.'
    sys.exit(-1)

  with tf.Session() as new_sess:
#   with new_sess.graph.as_default():
    print('++++++++++++++')
  #  tf.reset_default_graph()
  #  new_sess.run(tf.initialize_all_variables())
    new_saver = tf.train.import_meta_graph('/test/mnistoutput/ckpt.meta')
    print("before restore")
    new_saver.restore(new_sess, '/test/mnistoutput/ckpt')
    print("after restore")
  #  new_x = tf.get_collection('x')[0]
  #  print(new_x)
  #  new_y = tf.get_collection('y')[0]
  #  print(new_y)
    new_values = tf.get_collection('values')[0]
    print(new_values)
    new_graph = tf.get_default_graph()
    new_x = new_graph.get_tensor_by_name('x:0')
    print(new_x)
    new_y = new_graph.get_tensor_by_name('y:0')
    print(new_y)
  #  new_train_step = tf.get_collection('train_step')
  #  print(new_train_step)
    new_serialized_tf_example = new_graph.get_tensor_by_name('tf_example:0')
    print(new_serialized_tf_example)
    new_prediction_classes = tf.get_collection('prediction_classes')[0]
    print(new_prediction_classes)
    print('______________________')
    var = new_graph.get_tensor_by_name("Variable:0")
    print(var)

  # Export model
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
    print 'Exporting trained model to', export_path
    builder = saved_model_builder.SavedModelBuilder(export_path)
    print("builder--------")
    print(builder)

  # Build the signature_def_map.
    classification_inputs = utils.build_tensor_info(new_serialized_tf_example)
    classification_outputs_classes = utils.build_tensor_info(new_prediction_classes)
    classification_outputs_scores = utils.build_tensor_info(new_values)
    classification_signature = signature_def_utils.build_signature_def(
      inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
      outputs={
          signature_constants.CLASSIFY_OUTPUT_CLASSES:
              classification_outputs_classes,
          signature_constants.CLASSIFY_OUTPUT_SCORES:
              classification_outputs_scores
      },
      method_name=signature_constants.CLASSIFY_METHOD_NAME)
    print("classification_signature-------")
    print(classification_signature)

    tensor_info_x = utils.build_tensor_info(new_x)
    tensor_info_y = utils.build_tensor_info(new_y)

    prediction_signature = signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=signature_constants.PREDICT_METHOD_NAME)
    print("prediction_signature")
    print(prediction_signature)

    legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
    print("before add meta graph and variables to builder")
    print(new_sess.graph)

    builder.add_meta_graph_and_variables(
      new_sess, [tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
     #     signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
     #         classification_signature,
      },
      legacy_init_op=legacy_init_op)
    builder.save()

  print 'Done exporting!'


if __name__ == '__main__':
  tf.app.run()

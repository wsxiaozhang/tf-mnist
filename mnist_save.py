#!/usr/bin/env python2.7
"""Train and export a simple Softmax Regression TensorFlow model.

The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and save its checkpoints.

Usage: mnist_save.py [--training_iteration=x] [--work_dir=dataset_dir] [--log_dir=checkpoint_dir]
"""

import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_string('data_dir', '/tmp', 'Mnist dataset directory.')
tf.app.flags.DEFINE_string('log_dir', '/tmp/mnist_log', 'checkpoint and log directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if len(sys.argv) == 1:
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--data_dir=dataset_dir] [--log_dir=checkpoint_dir]')
  if FLAGS.training_iteration <= 0:
    print 'Please specify a positive value for training iteration.'
    sys.exit(-1)

  # Train model
  print 'Training model...'
  mnist = mnist_input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
 # sess = tf.InteractiveSession()
  with tf.Session() as sess:
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name

    with tf.name_scope('input_reshape'):
      image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
      tf.summary.image('input', image_shaped_input, 10)

    y_ = tf.placeholder('float', shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y), name='corss_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, name='train_step')
    values, indices = tf.nn.top_k(y, 10)
    prediction_classes = tf.contrib.lookup.index_to_string(
      tf.to_int64(indices), mapping=tf.constant([str(i) for i in xrange(10)]))

    tf.add_to_collection('values', values)
    tf.add_to_collection('prediction_classes', prediction_classes)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

    for step in range(FLAGS.training_iteration):
      batch = mnist.train.next_batch(50)
      _, summary = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1]})
      train_writer.add_summary(summary, step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print 'training accuracy %g' % sess.run(
      accuracy, feed_dict={x: mnist.test.images,
                           y_: mnist.test.labels})
    print 'Done training!'
    train_writer.close()
    checkpoint_dir =  FLAGS.log_dir
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    saver.save(sess, checkpoint_dir + "/ckpt")

  print 'Done saving!'


if __name__ == '__main__':
  tf.app.run()

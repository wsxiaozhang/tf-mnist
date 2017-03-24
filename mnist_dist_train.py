"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_string("log_dir", "/tmp/mnist-log",
                    "Directory for storing traing result data")
flags.DEFINE_boolean("log_device_placement", True,
                     "Enable log of training device placement information")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("num_gpus", 0,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 1000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")

FLAGS = flags.FLAGS


IMAGE_PIXELS = 28


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    if os.getenv("JOB_NAME") is not None:
      FLAGS.job_name = os.getenv("JOB_NAME")
    else:
      raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    if os.getenv("JOB_INDEX") is not None:
      FLAGS.task_index = int(os.getenv("JOB_INDEX"))
    else:
      raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  #Construct the cluster and start the server
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts =="":
    if os.getenv("PS_HOSTS") is not None:
      FLAGS.ps_hosts = os.getenv("PS_HOSTS")
    else:
      raise ValueError("Failed to find PS hosts info.")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts =="":
    if os.getenv("WORKER_HOSTS") is not None:
      FLAGS.worker_hosts = os.getenv("WORKER_HOSTS")
    else:
      raise ValueError("Failed to find Worker hosts info.")
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    if FLAGS.num_gpus < num_workers:
      raise ValueError("number of gpus is less than number of workers")
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu 
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)


  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU or CPU
  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Ops: located on the worker specified with FLAGS.task_index
    # Input placeholders
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name='x-input')
      y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # Reshape input image for summary show
    with tf.name_scope('input_reshape'):
      image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
      tf.summary.image('input', image_shaped_input, 10)

    # hidden layer - layer_linear
    with tf.name_scope('layer_linear'):
      with tf.name_scope('weights'):
        # Variables of the hidden layer
        hid_w = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                stddev=1.0 / IMAGE_PIXELS),
            name="hid_w")
        variable_summaries(hid_w)
      with tf.name_scope('bias'):
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
        variable_summaries(hid_b)
      with tf.name_scope('Wx_plus_b'):
        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        tf.summary.histogram('pre_activations', hid_lin)
      # activation - Relu
      hid = tf.nn.relu(hid_lin)
      tf.summary.histogram('activations', hid)

    # softmax layer
    with tf.name_scope('softmax_layer'):
      # Variables of the softmax layer
      sm_w = tf.Variable(
          tf.truncated_normal(
              [FLAGS.hidden_units, 10],
              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

    # lose function - cross_entropy
    with tf.name_scope('cross_entropy'): 
      cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Optimization
    with tf.name_scope('train'):
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

      if FLAGS.sync_replicas:
        if FLAGS.replicas_to_aggregate is None:
          replicas_to_aggregate = num_workers
        else:
          replicas_to_aggregate = FLAGS.replicas_to_aggregate

        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")

      train_step = opt.minimize(cross_entropy, global_step=global_step)

    # Accuracy
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist-data/log by defalut
    summary_op = tf.summary.merge_all()
        
    if FLAGS.sync_replicas:
      local_init_op = opt.local_step_init_op
      if is_chief:
        local_init_op = opt.chief_init_op

      ready_for_local_init_op = opt.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()

    init_op = tf.global_variables_initializer()

    # train_dir = tempfile.mkdtemp()
    log_dir = FLAGS.log_dir
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    if FLAGS.sync_replicas:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=FLAGS.log_dir,
          init_op=init_op,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    else:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=FLAGS.log_dir,
          init_op=init_op,
          recovery_wait_secs=1,
          global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url,
                                            config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    step = 0
    while True:
      # Test-set accuracy and record summary
      if local_step % 10 == 0:
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        acc, summary = sess.run([accuracy, summary_op], feed_dict=test_feed)
        test_writer.add_summary(summary, local_step)
        local_step += 1
        print("Accuracy at local step %s: %s" % (local_step, acc))
      else:
        # Training feed
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        _, summary, step = sess.run([train_step, summary_op, global_step], feed_dict=train_feed)
        local_step += 1
        train_writer.add_summary(summary, step)

        now = time.time()
        print("%f: Worker %d: training step %d done (global step: %d)" %
              (now, FLAGS.task_index, local_step, step))

      if step >= FLAGS.train_steps:
        break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
    train_writer.close()
    test_writer.close()
    # Validation feed
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))
    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict=val_feed))
    
    sv.stop()


if __name__ == "__main__":
  tf.app.run()

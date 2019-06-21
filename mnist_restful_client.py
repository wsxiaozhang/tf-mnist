#!/usr/bin/python

import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
  # test with a image file
  # image_file = 'img_4.jpg'
  # Convert arbitrary sized jpeg image to 28x28 b/w image.
  # data = np.array(Image.open(image_file).convert('L').resize((28, 28))).astype(np.float).reshape(-1, 28, 28, 1)
  # Dump jpeg image bytes as 28x28x1 tensor
  # np.set_printoptions(threshold=np.inf)

  # tset with MNIST test data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  x_val = mnist.test.images
  y_val = mnist.test.labels
  print(x_val[0].shape)
  print(y_val[0].shape)
  print(type(x_val[0])) #ndarray
  print(y_val[0])
  tensordata=tf.reshape(x_val[0], [-1, 784]) #ndarray to tensor
  print(type(tensordata))
  sess=tf.Session()
  with sess.as_default():
    data=tensordata.eval() # from tensor get ndarray
  np.set_printoptions(threshold=np.inf)
  json_request = '{{"signature_name":"predict_images", "instances" : {} }}'.format(np.array2string(data, separator=',', formatter={'float':lambda x: "%.5f" % x}))
  print(json_request)

  # import json
  # json_req=json.dumps({"signature_name": "predict_images", "instances": x_val[0].reshape(1, 784).tolist()})
  # print(json_req)

  resp = requests.post('http://localhost:8501/v1/models/mnist:predict', data=json_request)
  print('response.status_code: {}'.format(resp.status_code))
  print('response.content: {}'.format(resp.content))

main()

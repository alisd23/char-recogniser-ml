import tensorflow as tf
from classes import codeToClass, classToCode

CLASSES = len(codeToClass.keys())
IMAGE_SIZE = 32

# Create a new WEIGHT variable with random variation
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Create a new BIAS variable with random variation
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Create a CONV tensor
def conv2d(x, W):
  '''
  Creates a convolution tensor
  x - Input tensor
  W - weights
  '''
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  '''
  Creates a max pool tensor
  x - Input tensor
  '''
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

graph = tf.Graph()

with graph.as_default():
  # Examples and predictions (10 classes)
  x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE ** 2])
  y = tf.placeholder(tf.float32, shape=[None, CLASSES])

  # FIRST LAYER (CONV (ReLU), POOL)
  # Input size [32 x 32 x 1] | Output size [16 x 16 x 32]
  depth_conv1 = 32

  # [Patch width, Patch height, Input depth, Output depth (no of filters)]
  W_conv1 = weight_variable([5, 5, 1, depth_conv1])
  # Bias variable per filter
  b_conv1 = bias_variable([depth_conv1])

  # Reshape Image to [?, 32 width, 32 height, depth]
  # This is the input layer
  x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # SECOND LAYER (CONV (ReLU), POOL)
  # Input size [16 x 16 x 32] | Ouptut size [8 x 8 x 64]
  depth_conv2 = 64

  # [Patch width, Patch height, Input depth, Output depth (no of filters)]
  W_conv2 = weight_variable([5, 5, depth_conv1, depth_conv2])
  b_conv2 = bias_variable([depth_conv2])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # THIRD LAYER (Fully connected - 2048 neurons)
  # Input size [8 x 8 x 64] => [1 x 4096]| Ouptut size [1 x 2048]
  input_size = int((IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * depth_conv2)
  W_fc1 = weight_variable([input_size, 2048])
  b_fc1 = bias_variable([2048])

  h_pool2_flat = tf.reshape(h_pool2, [-1, input_size])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Apply dropout after 3rd layer
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # READOUT LAYER
  # Input size [1 x 2048] | Ouptut size [1 x 62]
  W_fc2 = weight_variable([2048, CLASSES])
  b_fc2 = bias_variable([CLASSES])

  # Predictions
  predictions = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

def getModel():
  return predictions, x, y, keep_prob

def getGraph():
  return graph

def getParameters():
  return W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2

def getActivations():
  return h_conv1

def exportParameters(session):
  return {
    'W_conv1': session.run(W_conv1).tolist(),
    'b_conv1': session.run(b_conv1).tolist(),
    'W_conv2': session.run(W_conv2).tolist(),
    'b_conv2': session.run(b_conv2).tolist(),
    'W_fc1': session.run(W_fc1).tolist(),
    'b_fc1': session.run(b_fc1).tolist(),
    'W_fc2': session.run(W_fc2).tolist(),
    'b_fc2': session.run(b_fc2).tolist(),
  }

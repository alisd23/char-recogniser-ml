import os
import math
import tensorflow as tf
from database import connect, getExamples
from classes import codeToClass, classToCode

CLASSES = len(codeToClass.keys())
BATCH_SIZE = 100
IMAGE_SIZE = 32

connect()

training_set, test_set = getExamples()

print('Training set size: {}'.format(len(training_set)))
print('Test set size: {}'.format(len(test_set)))

# Disable annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()

# Examples and predictions (10 classes)
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE ** 2])
y = tf.placeholder(tf.float32, shape=[None, CLASSES])

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
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# FIRST LAYER (CONV (ReLU), POOL)
# Input size [32 x 32 x 1] | Output size [16 x 16 x 32]

# [Patch width, Patch height, Input depth, Output depth (no of filters)]
W_conv1 = weight_variable([5, 5, 1, 32])
# Bias variable per filter
b_conv1 = bias_variable([32])

# Reshape Image to [?, 32 width, 32 height, depth]
# This is the input layer
x_image = tf.reshape(x, [-1, 32, 32, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND LAYER (CONV (ReLU), POOL)
# Input size [16 x 16 x 32] | Ouptut size [8 x 8 x 64]

# [Patch width, Patch height, Input depth, Output depth (no of filters)]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD LAYER (Fully connected - 1024 neurons)
# Input size [8 x 8 x 64] => [1 x 4096]| Ouptut size [1 x 1024]

input_size = int((IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * 64)
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

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# TRAIN
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
)

# Calculate gradients, and update parameters
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

def noOfBatches(set):
  '''
  Calculates the number of batches in a given set
  '''
  batches = math.floor(len(set) / BATCH_SIZE)
  remainder = len(set) % batches
  return batches if (remainder == 0) else batches + 1

def getBatch(set, batchNo):
  '''
  Get a batch of examples from the given set
  '''
  start = batchNo * BATCH_SIZE
  # If this batch will take us to the end of the set
  if (batchNo + 1) * BATCH_SIZE >= len(set):
    return set[start:]
  # Else this is not the last batch
  else:
    end = start + BATCH_SIZE
    return set[start:end]

# TRAINING
batches = noOfBatches(training_set)

print('Number of batches to run: {}'.format(batches))
for i in range(batches):
  batch = getBatch(training_set, i)
  # Every 50 batches - check training set accuracy
  if i % 50 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x: list(map(lambda ex: ex[0], batch)),
      y: list(map(lambda ex: ex[1], batch)),
      keep_prob: 1.0
    })
    print("Batch %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={
    x: list(map(lambda ex: ex[0], batch)),
    y: list(map(lambda ex: ex[1], batch)),
    keep_prob: 0.5
  })

# TEST
print("test accuracy %g" %accuracy.eval(feed_dict={
  x: list(map(lambda ex: ex[0], test_set)),
  y: list(map(lambda ex: ex[1], test_set)),
  keep_prob: 1.0
}))

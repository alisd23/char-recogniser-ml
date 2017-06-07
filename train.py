import os
import math
import tensorflow as tf
from model import getModel, exportParameters

# Disable annoying warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def noOfBatches(dataset, batch_size):
  '''
  Calculates the number of batches in a given set
  '''
  batches = math.floor(len(dataset) / batch_size)
  remainder = len(dataset) % batches
  return batches if (remainder == 0) else batches + 1

# Main training function
def train(
  training_set,
  test_set,
  batch_size,
  learning_rate,
  session
):
  print('Train model:')
  print('Batch size: {}'.format(batch_size))
  print('Learning Rate: {}'.format(learning_rate))
  print('Training set size: {}'.format(len(training_set)))
  print('Test set size: {}\n'.format(len(test_set)))

  predictions, x, y, keep_prob = getModel()

  # TRAIN
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions)
  )

  # Calculate gradients, and update parameters
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

  y_predictions = tf.argmax(predictions, 1)
  y_classes = tf.argmax(y, 1)
  correct_prediction_top1 = tf.cast(tf.equal(y_predictions, y_classes), tf.float32)

  in_top_1 = tf.nn.in_top_k(predictions, y_classes, 1)
  in_top_3 = tf.nn.in_top_k(predictions, y_classes, 3)

  acc_top_1 = tf.reduce_mean(tf.cast(in_top_1, tf.float32))
  acc_top_3 = tf.reduce_mean(tf.cast(in_top_3, tf.float32))

  session.run(tf.global_variables_initializer())

  # TRAINING
  batches = noOfBatches(training_set, batch_size)

  print('Number of training batches to run: {}'.format(batches))
  for i in range(batches):
    batch = training_set.nextBatch(batch_size)
    # Every 50 batches - check training set accuracy
    if i % 500 == 0:
      feed = {
        x: list(map(lambda ex: ex[0], batch)),
        y: list(map(lambda ex: ex[1], batch)),
        keep_prob: 1.0
      }
      train_acc_1 = session.run(acc_top_1, feed_dict=feed)
      train_acc_3 = session.run(acc_top_3, feed_dict=feed)
      print('[Batch {:3}] TRAIN Accuracy: Top 1: {:0.4f} | Top 3: {:0.4f}'.format(
        i,
        train_acc_1,
        train_acc_3
      ))

    session.run(train_step, feed_dict={
      x: list(map(lambda ex: ex[0], batch)),
      y: list(map(lambda ex: ex[1], batch)),
      keep_prob: 0.5
    })

  # TEST
  test_acc_1 = 0
  test_acc_3 = 0

  batches = noOfBatches(test_set, batch_size)

  print('Number of test batches to run: {}'.format(batches))
  for i in range(batches):
    batch = test_set.nextBatch(batch_size)
    feed = {
      x: list(map(lambda ex: ex[0], batch)),
      y: list(map(lambda ex: ex[1], batch)),
      keep_prob: 1.0
    }
    acc1 = session.run(acc_top_1, feed_dict=feed)
    acc3 = session.run(acc_top_3, feed_dict=feed)

    count = i * batch_size
    curr_batch_size = len(batch)
    test_acc_1 = ((test_acc_1 * count) + (acc1 * curr_batch_size)) / (count + curr_batch_size)
    test_acc_3 = ((test_acc_3 * count) + (acc3 * curr_batch_size)) / (count + curr_batch_size)

  parameters = exportParameters(session)

  return test_acc_1, test_acc_3, parameters

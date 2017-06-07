from database import connect, getRecentRun, getRandomExample, processExample
from model import getModel, getParameters
from classes import classToCode
import tensorflow as tf

connect()

session = tf.Session()

data = getRecentRun()

example = getRandomExample()
processed_example = processExample(example)

predictions, x, y, keep_prob = getModel()
W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2 = getParameters()

feed = {
  x: [processed_example[0]],
  y: [processed_example[1]],
  keep_prob: 1.0,
  W_conv1: data['parameters']['W_conv1'],
  b_conv1: data['parameters']['b_conv1'],
  W_conv2: data['parameters']['W_conv2'],
  b_conv2: data['parameters']['b_conv2'],
  W_fc1: data['parameters']['W_fc1'],
  b_fc1: data['parameters']['b_fc1'],
  W_fc2: data['parameters']['W_fc2'],
  b_fc2: data['parameters']['b_fc2'],
}

T_y_probs = tf.nn.softmax(predictions)
T_top_3_predictions = tf.nn.top_k(T_y_probs, 3)

y_probs, top_3 = session.run([T_y_probs, T_top_3_predictions], feed_dict=feed)
values, indices = top_3

print('Top 3 predictions\n')

print('[Actual value] - charcode: {} | character: {}\n'.format(
  example['charcode'],
  chr(example['charcode']),
))

for i in range(3):
  print('[{}] charcode: {:3} | character: {} | probability: {:0.2f}%'.format(
    i + 1,
    classToCode[indices[0][i]],
    chr(classToCode[indices[0][i]]),
    values[0][i] * 100
  ))

correct = classToCode[indices[0][0]] == example['charcode']

print('\n[Result] - {}'.format('CORRECT' if correct else 'WRONG'))

y_predictions = tf.argmax(y_probs, 1)
y_classes = tf.argmax(y, 1)

session.close()

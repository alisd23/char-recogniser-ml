from database import connect, getRecentRun, getRandomExample, processExample
from model import getModel, getParameters, getGraph
from classes import classToCode
import tensorflow as tf

def predict(image):
  data = getRecentRun()

  predictions, x, y, keep_prob = getModel()
  graph = getGraph()
  W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2 = getParameters()

  feed = {
    x: [image],
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

  # Get all predictions in order of probability
  T_top_predictions = tf.nn.top_k(T_y_probs, 62)

  with tf.Session(graph=graph) as session:
    y_probs, top = session.run([T_y_probs, T_top_predictions], feed_dict=feed)
    values, indices = top

    result = []
    for confidence, index in zip(values[0], indices[0]):
      result.append({
        'charcode': classToCode[index],
        'confidence': str(confidence)
      })

    session.close()
    return result

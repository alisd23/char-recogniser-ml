from database import connect, getExamples, saveRun
from train import train
import tensorflow as tf
import itertools

NO_OF_EXAMPLES = 731668

connect()

training_set, test_set = getExamples(NO_OF_EXAMPLES)

batch_size = 50
learning_rate = 1e-4

session = tf.Session()

acc_top_1, acc_top_3, parameters = train(
  training_set = training_set,
  test_set = test_set,
  batch_size = batch_size,
  learning_rate = learning_rate,
  session = session
)

print('Test accuracy:\nTop 1: {:0.4f}\nTop 3: {:0.4f}'.format(acc_top_1, acc_top_3))

print('Saving run to database...')

saveRun(acc_top_1, acc_top_3, parameters)

session.close()

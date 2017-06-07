from database import connect, getExamples
from train import train
import itertools
import tensorflow as tf

NO_OF_EXAMPLES = 200000

connect()

session = tf.Session()

training_set, test_set = getExamples(NO_OF_EXAMPLES)

batch_sizes = [40, 100, 150]
learning_rates = [1e-4, 3e-4]

results = []

for bs, lr in itertools.product(batch_sizes, learning_rates):
  acc_top_1, acc_top_3 = train(
    training_set = training_set,
    test_set = test_set,
    batch_size = bs,
    learning_rate = lr
  )
  results.append({
    'batch_size': bs,
    'learning_rate': lr,
    'accuracy_top_1': acc_top_1
    'accuracy_top_3': acc_top_3
  })

for result in results:
  print('Result [batch size, learning rate]: [{}, {}] - {:0.4f} (TOP 1) - {:0.4f} (TOP 3)'.format(
    result['batch_size'],
    result['learning_rate'],
    result['acc_top_1'],
    result['acc_top_3']
  ))

session.close()

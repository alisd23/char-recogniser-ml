from pymongo import MongoClient
from classes import codeToClass, classToCode
import numpy as np
import time
import json
import gridfs

CLASSES = len(codeToClass.keys())
SPLIT_VALUE = 0.8

HOST = 'localhost'
PORT = 27017

client = None
db = None

class Set:
  cursor = None

  def __init__(self, cursor):
    self.cursor = cursor

  def __len__(self):
    return self.cursor.count(with_limit_and_skip=True)

  def nextBatch(self, batch_size):
    dataset = []
    for i in range(batch_size):
      if self.cursor.alive:
        dataset.append(processExample(self.cursor.next()))
      else:
        break
    return dataset

def connect():
  global client, db
  client = MongoClient(HOST, PORT)
  db = client['char-recogniser']

def processExample(example):
  y = np.zeros(CLASSES)
  y[codeToClass[example['charcode']]] = 1
  return [example['data'], y]

def getRandomExample():
  return db.training_set.aggregate([{
    '$sample': { 'size': 1 }
  }]).next()

def getExamples(no_of_examples):
  print('Fetching {} examples'.format(no_of_examples))
  examples = db.training_set.find().limit(no_of_examples)

  # Split [training, test] - [70, 30]
  train_size = int(no_of_examples * SPLIT_VALUE)
  test_size = no_of_examples - train_size

  training_set = Set(examples.clone()[:train_size])
  test_set = Set(examples.clone()[train_size:no_of_examples])

  print('Splitting examples - {} train | {} test'.format(
    train_size,
    test_size
  ))

  return training_set, test_set

def saveRun(top_1_acc, top_3_acc, parameters):
  grid = gridfs.GridFS(db, 'training_runs')
  data = json.dumps({
    'top_1': top_1_acc,
    'top_3': top_3_acc,
    'parameters': parameters,
    'timestamp': time.ctime()
  })
  grid.put(data.encode('utf-8'))

def getRecentRun():
  grid = gridfs.GridFS(db, 'training_runs')
  grid_out = grid.find_one({ '$query': {}, '$orderby' : { '_id': -1 } })
  contents = grid_out.read()
  json_data = json.loads(contents.decode('utf-8'))
  return json_data

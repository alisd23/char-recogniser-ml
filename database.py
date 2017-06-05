from pymongo import MongoClient
from classes import codeToClass, classToCode
import random
import numpy as np

CLASSES = len(codeToClass.keys())
NO_OF_EXAMPLES = 200000

HOST = 'localhost'
PORT = 27017

client = None
db = None

def connect():
  global client, db
  client = MongoClient(HOST, PORT)
  db = client['char-recogniser']

def processExample(example):
  y = np.zeros(CLASSES)
  y[codeToClass[example["charcode"]]] = 1
  return [example["data"], y]

def getExamples():
  examples = list(db.training_set.find())
  random.shuffle(examples)
  examples = list(map(
    lambda ex: processExample(ex),
    examples[:NO_OF_EXAMPLES]
  ))
  # Split [training, test] - [70, 30]
  slicePoint = int(len(examples) * 0.7)

  training_set = examples[:slicePoint]
  test_set = examples[slicePoint:]

  return training_set, test_set

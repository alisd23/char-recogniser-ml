import numpy as np
from database import connect, getRecentRun
from utils import volumeToPixels

db = connect()

params = getRecentRun()

# Remove single dimensional entries (input has shape [5, 5, 1, 32])
conv1 = np.array(params["parameters"]["W_conv1"]).squeeze().tolist()

filters_1 = volumeToPixels(conv1)

db["values"].insert_one({
  "top1": params["top_1"],
  "top3": params["top_3"],
  "conv1": filters_1
})

print('Conv filters saved to DB')

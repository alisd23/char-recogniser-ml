from database import connect, getRecentRun

db = connect()

params = getRecentRun()
conv1 = params["parameters"]["W_conv1"]

noOfFilters = len(conv1[0][0][0])
spatialSize = len(conv1[0])
filters = []
allPixels = []

for i in range(noOfFilters):
  for row_i in range(spatialSize):
    for col_i in range(spatialSize):
      allPixels.append(conv1[col_i][row_i][0][i])

min_val = min(allPixels)
max_val = max(allPixels)

def normalise(value):
  v = (value - min_val) / (max_val - min_val)
  return v * 255

for i in range(noOfFilters):
  filter = []
  for row_i in range(spatialSize):
    for col_i in range(spatialSize):
      # store pixels left to right (by row)
      filter.append(normalise(conv1[col_i][row_i][0][i]))
  filters.append(filter)

db["values"].insert_one({
  "top1": params["top_1"],
  "top3": params["top_3"],
  "conv1": filters
})

print('Conv layer 1 - {} filters saved to DB'.format(len(filters)))

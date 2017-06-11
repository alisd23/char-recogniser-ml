def volumeToPixels(volume):
  depth = len(volume[0][0])
  spatialSize = len(volume)
  layers = []
  allPixels = []

  for i in range(depth):
    for col_i in range(spatialSize):
      for row_i in range(spatialSize):
        allPixels.append(volume[col_i][row_i][i])

  min_val = min(allPixels)
  max_val = max(allPixels)

  def normalise(value):
    v = (value - min_val) / (max_val - min_val)
    return v * 255

  for i in range(depth):
    filter = []
    for col_i in range(spatialSize):
      for row_i in range(spatialSize):
        # store pixels left to right (by row)
        filter.append(normalise(volume[col_i][row_i][i]))
    layers.append(filter)

  return layers

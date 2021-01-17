import visualizer
import numpy as np

data = np.zeros(shape=(100, 100), dtype=int)
data[5, 5] = 2
data[10, 10] = 1
# state = np.array([[2, 2],
#                   [3, 5],
#                   [6, 10]])
state = np.array([3, 5])
vis = visualizer.Visualizer(data)
vis.setup()
vis.visualize_racetrack(state)


import sys
argv = sys.argv

filepath = argv[1]

import numpy as np

arr = np.load(filepath)
print(arr.shape)
print(arr)
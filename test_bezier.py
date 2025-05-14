import os
import sys
import numpy as np


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


from bezier import *


if __name__ == "__main__":

    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    bezier(points)


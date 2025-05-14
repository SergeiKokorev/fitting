import os
import sys
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(threshold=sys.maxsize)


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


from mathtools import construct_matrixA3, construct_vectorB3
from solver import *



if __name__ == "__main__":

    points = np.array([
        [0.0, 0.0], [1.5, 0.5], [2.0, 0.75], [2.2, 1.5], [3.5, 1.75], [4.0, 1.5]
    ])

    A = construct_matrixA3(5)
    b = construct_vectorB3(points)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], label='Points', marker='*', color='r')
    ax.legend()
    ax.grid()

    plt.show()


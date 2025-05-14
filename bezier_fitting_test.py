import os
import sys
import numpy as np
import matplotlib.pyplot as plt


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


from bezier import bezier, bezier_fitting
from solver import gradient_descent, conjugate_gradient


if __name__ == "__main__":

    xpoints = np.linspace(0, 5 * np.pi / 3, 10)
    ypoints = np.cos(xpoints)
    points = np.column_stack((xpoints, ypoints))

    cpoints = bezier_fitting(points=points, parametrization='centripical', solver='bicgstab')
    
    if cpoints[1] is None:
        raise RuntimeError(f'Solution is {cpoints[1]} in number of iterations {cpoints[2]}')
        
    bpoints = bezier(cpoints, 100)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], marker='o', color='b', label='Q')
    ax.plot(cpoints[:, 0], cpoints[:, 1], marker='>', color='r', label='P')
    ax.plot(bpoints[:, 0], bpoints[:, 1], marker=None, color='k', label='Bezier')
    ax.grid()
    ax.legend()
    plt.show()


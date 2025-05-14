import os
import sys
import numpy as np


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


import solver


if __name__ == "__main__":

    A = np.array([
        [1.0, 3.2, 2.5],
        [3.2, 1.5, 0.5],
        [2.5, 0.5, 1.0]
    ])

    b = [0.5, 2.2, 1.5]


    xsol_gd = solver.gradient_descent(A, b)
    xsol_cg = solver.conjugate_gradient(A, b)

    print(f'Gradient descend method solution is {xsol_gd[1]}: {xsol_gd[0]}, Interation before convergent: {xsol_gd[2]}')
    print(f'Conjugate gradient method solution is {xsol_cg[1]}: {xsol_cg[0]}, Interation before convergent: {xsol_cg[2]}')


import numpy as np


from mathtools import bernstein
from solver import steepest_descend, conjugate_gradient


SOLVERS = {
    'steepest_descend': steepest_descend,
    'conjugate_gradient': conjugate_gradient
}


def bezier_fitting(points: np.ndarray, parametrization='uniform', solver='steepest_descend') -> np.ndarray:

    n, d = points.shape
    sol = np.zeros_like(points, dtype='float')
    
    def generate_t(points, parametrization):

        match parametrization:
            case 'uniform':
                t = np.linspace(0, 1, n)
            case 'chord_len':
                d = np.array([np.linalg.norm(points[i] - points[i - 1]) for i in range(1, n)])
                t = [0]
                for i in range(1, n):
                    t.append(t[i - 1] + d[i] / sum(d))
            case 'centripical':
                d = np.array([np.linalg.norm(points[i] - points[i - 1]) ** 0.5 for i in range(1, n)])
                t = [0]
                for i in range(1, n):
                    t.append(t[i - 1] + d[i - 1] / sum(d))
        return np.array(t)
    
    t = generate_t(points, parametrization)

    A = np.array([[bernstein(ti, n-1, i) for i in range(n)] for ti in t])

    for i in range(d):
        solution = SOLVERS[solver](A, points[:, i])
        if solution[1]:
            sol[:, i] = solution[0]
        else:
            raise RuntimeError('Bezier approximation failure. Solution not found')
    
    return sol


def bezier(points: np.ndarray, npoints: int) -> np.ndarray:

    n, d = points.shape
    t = np.linspace(0, 1, npoints - 1)
    sol = np.zeros(shape=(t.size, d))
    
    for i, ti in enumerate(t):
        sol[i] = sum([points[j] * bernstein(ti, n - 1, j) for j in range(n)])

    return sol


def bezier_spline_fitting(points: np.ndarray, parametrization='uniform', solver='steepest_descend') -> np.mdarray:

    pass

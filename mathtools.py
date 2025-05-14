import numpy as np


'''
Provides additional functions

:func factorial(n: int) -> int: computes factorial of n
:func binomial(n: int, i: int) -> float: computes binomial coefficient of n, i
:func bernstein(t: float, n: int, i: int) -> float: computes berstein basis polynomials of t, n, i

'''



def factorial(n: int) -> int:

    """
    Computes factorial of n

    Paramters:
        n : int
    
    Returns:
        res: int

    n! = n * (n - 1) * (n - 2) * ... * 1

    """

    if not isinstance(n, int):
        raise ValueError(f'Factorial error. n must be integer type. Given {type(n)}')
    elif n < 0:
        raise ValueError(f'Factorial error. n must be greater or eaqual 0. Given {n}')

    if n == 0 or n == 1:
        return 1
    else:
        return factorial(n - 1) * n


def binominal(n: int, i: int) -> float:
    
    '''
    Computes binomial coefficient n of i

    :param int n:
    :param int i:
    :returns res float:
    
    binomial(n, i) = n! / (i! * (n - i))!

    '''
    
    return factorial(n) / (factorial(i) * factorial(n - i))


def bernstein(t: float, n: int, i: int) -> float:

    '''
    Computes Bernsttein polynomial basis

    :param float t: Parameter
    :param int n:
    :param int i:
    :returns float res:

    berstein(t, n, i) = binomial(n, i) * (t ** i) * (1 - t) ** (n - i)

    '''

    return binominal(n ,i) * (t ** i) * (1 - t) ** (n - i)


def construct_matrixA3(npoints: int) -> np.ndarray:

    A = np.zeros(shape=(4 * (npoints - 1), 4 * (npoints - 1)))
    n = 3
    # First and second set of conditions (1) B_k(0) = Q_k; (2) B_k(1) = Q_k+1
    for i in range(npoints - 1):
        A[i, i * (n + 1)] = 1
        A[i + npoints - 1, (i + 1) * (n + 1) - 1] = 1

    # Third and forth set of conditoins (3) B'_k(1) = B'_k+1(0); (4) B''_k(1) = B''_k+1(0)
    for i in range(npoints - 2):
        row1 = (npoints - 1) * 2
        row2 = row1 + (npoints - 2)
        col1 = 2 + i * 4
        col2 = 1 + i * 3

        tmp1 = np.array([1, -1, 1, -1])
        tmp2 = np.array([1, -2, 1, -1, 2, -1])
        A[i + row1, col1 : col1 + 4] = tmp1
        A[i + row2, col2 : col2 + 6] = tmp2

    # Boundary conditions (normal spline) (1) B'_0(0) = 0; (2) B'_k(1) = 0
    A[4 * (npoints - 1) - 2, 0 : 2] = np.array([-1, 1])
    A[-1, -2 : ] = np.array([-1, 1])

    return A

def construct_vectorB3(points: np.ndarray) -> np.ndarray:
    npoints, dim = points.shape
    npoints -= 1
    b = np.zeros(shape=(4 * (npoints - 1), dim))

    b[:npoints] = points[:-1]
    b[npoints:npoints * 2] = points[1:]

    return b

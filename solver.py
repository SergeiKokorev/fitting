import numpy as np
from scipy.sparse import issparse


def gradient_descent(A: np.ndarray, b: np.ndarray, x0: np.ndarray=None, tol: float=1e-6, max_iter: int=800) -> np.ndarray:

    n = len(b)
    
    # Initialize
    if x0 is None:
        x = np.zeros_like(b, dtype='float')
    else:
        x = x0.copy()

    rk = b - A @ x
    residuals = [np.linalg.norm(rk)]

    for num_iter in range(1, max_iter + 1):
        rk = b - A @ x
        pk = rk.copy()
        alfak = sum(rk * rk) / (sum(rk * (A @ rk)))
        x += alfak * pk
        rk = b - A @ x
        residuals.append(np.linalg.norm(rk))
        if np.linalg.norm(rk) <= tol:
            return x, True, num_iter, residuals

    return x, False, max_iter, residuals


def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: np.ndarray=None, tol: float=1e-6, max_iter: int=None) -> np.ndarray:

    n = len(b)
    if not np.allclose(A, A.T, rtol=1e-05, atol=1e-08):
        raise RuntimeError('Conjugate gradient algorythm failed. Matrx A must be symmetric and positive defined')

    if max_iter is None:
        max_iter = n

    # Initialize
    if x0 is None:
        x = np.zeros_like(b, dtype='float')
    else:
        x = x0.copy()

    r = b - A @ x
    rsold = r.dot(r)

    if np.sqrt(rsold) < tol:
        return x, True, 1, np.sqrt(rsold)

    p = r.copy()
    residuals = [np.sqrt(r.dot(r))]

    for num_iter in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x += alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)

        residuals.append(np.sqrt(rsnew))
        
        if np.sqrt(rsnew) < tol:
            return x, True, num_iter, residuals
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, False, max_iter, residuals


def bicgstab(A: np.ndarray, b: np.ndarray, x0: np.ndarray=None, tol: float=1e-6, max_iter: int=1000, callback=None) -> np.ndarray:

    n = len(b)

    # Initialize
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r = b - (A @ x if not issparse(A) else A.dot(x))
    r0_hat = r.copy()
    residuals = [np.linalg.norm(r)]

    # Initialize values
    rho = alpha = omega = 1
    v = p = np.zeros_like(b)

    for num_iter in range(1, max_iter + 1):
        rho_prev = rho
        rho = np.dot(r0_hat, r)

        # Check for break down
        if rho == 0.0:
            break

        beta = (rho / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        # Matrix-vector product
        v = A @ p if not issparse(A) else A.dot(p)
        
        alpha = rho / np.dot(r0_hat, v)
        h = x + alpha * p
        
        # Early convergence check
        s = r - alpha * v
        t = A @ s if not issparse(A) else A.dot(s)
        
        omega = np.dot(t, s) / np.dot(t, t)
        x = h + omega * s
        r = s - omega * t
        
        residual_norm = np.linalg.norm(r)
        residuals.append(residual_norm)
        
        if callback is not None:
            callback(x)
        
        if residual_norm < tol:
            return x, True, num_iter, residuals
        
    return x, False, max_iter, residuals
        

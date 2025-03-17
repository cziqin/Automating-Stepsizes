import scipy
import numpy as np
import numpy.linalg as la
from sklearn.utils.extmath import safe_sparse_dot


def safe_sparse_add(a, b):
    """
    Adds two inputs (sparse matrices or dense arrays) safely.
    If both are sparse, they are added directly; otherwise, sparse inputs are converted to dense.
    """
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        return a + b
    else:
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


def logsig(x):
    """
    Numerically stable log-sigmoid function.
    Uses piecewise approximations to handle extreme input values.
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def logistic_smoothness(samples):
    return 0.25 * np.max(la.eigvalsh(samples.T @ samples / (samples.shape[0])))


def logistic_loss(params, samples, targets, l2):
    """
    Computes logistic loss with L2 regularization.
    """
    z = np.dot(samples, params)
    y = np.asarray(targets)
    return np.mean((1 - y) * z - logsig(z)) + (l2 / 2) * la.norm(params) ** 2



def logistic_gradient(params, samples, targets, l2, normalize=True):
    """
    Gradient of the logistic loss at point w with features X, labels y and l2 regularization.
    If labels are from {-1, 1}, they will be changed to {0, 1} internally.
    """
    y = (targets + 1) / 2 if -1 in targets else targets
    activation = scipy.special.expit(safe_sparse_dot(samples, params, dense_output=True).ravel())
    grad = safe_sparse_add(samples.T.dot(activation - y) / samples.shape[0], l2 * params)
    grad = np.asarray(grad).ravel()

    if normalize:
        return grad
    return grad * len(y)

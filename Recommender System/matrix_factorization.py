__author__ = 'Shubham'
#!/usr/bin/env python
import numpy as np


def PMF(X, rank=3, tol=1e-3, learning_rate=.1, regularization=.25,
        max_epoch=100, random_seed=0):
    X_mean = X.mean().mean()
    X = X - X_mean
    lr = learning_rate
    reg = regularization
    random_state = np.random.RandomState(random_seed)
    N, M = X.shape
    U = 0.1 * random_state.randn(N, rank)
    V = 0.1 * random_state.randn(M, rank)
    epoch = 0
    while epoch < max_epoch:
        X_pred = np.dot(U, V.T)
        mask = X == 0.
        X_pred[mask] = 0.
        ix = np.arange(N)
        jx = np.arange(M)
        random_state.shuffle(ix)
        random_state.shuffle(jx)
        for i in ix:
            for j in jx:
                e = X[i, j] - np.dot(U[i, :], V[j, :].T)
                U[i, :] += lr * (e * V[j, :] - reg * U[i, :])
                V[j, :] += lr * (e * U[i, :] - reg * V[j, :])
        sum_sq_err = np.sum((X - np.dot(U, V.T)) ** 2).sum()
        print sum_sq_err
        if sum_sq_err < tol:
            print "Tolerance %s reached" % tol
            break
        epoch += 1
    return np.dot(U, V.T) + X_mean


if __name__ == "__main__":
    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]], dtype=float)
    Q=PMF(R, rank=3)
    print Q


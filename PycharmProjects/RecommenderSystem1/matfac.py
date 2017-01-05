__author__ = 'rochak'

# Standard python modules
import sys
from math import *
# For scientific computing
import numpy
from numpy import *
import recommendations as rc
import random
import scipy.io, scipy.misc, scipy.optimize, scipy.cluster.vq

# For plotting
from matplotlib import pyplot, cm, colors, lines
from mpl_toolkits.mplot3d import Axes3D


movmap, ratmat, boolmat = rc.loadMovieLensCofi()

numpy.savetxt('ratings.csv', ratmat.T, delimiter=',')

def cofiGradFunc(X, theta, Y, R, num_users, num_movies, num_features, lamda):
    inner = (X.dot(theta.T) - Y) * R
    X_grad = inner.dot(theta) + lamda * X
    theta_grad = inner.T.dot(X) + lamda * theta
    return X_grad, theta_grad

def cofiCostFunc(X, theta, Y, R, num_users, num_movies, num_features, lamda):
    J = 0.5 * sum(((X.dot(theta.T) - Y) * R) ** 2)
    regularization = 0.5 * lamda * (sum(theta ** 2) + sum(X ** 2))
    return J + regularization

def matrix_factorization(R, I, N, M, K):
    num_users = N
    num_features = K
    num_movies = M
    U = numpy.random.random((num_users, num_features))
    Z = numpy.random.random((num_users, num_features))
    lamda = 0.1
    alpha = 0.002
    iterations = 10000

    J = cofiCostFunc(U, Z, R, I, num_users, num_movies, num_features, lamda)
    while J > 5:
        X_grad, theta_grad = cofiGradFunc(U, Z, R, I, num_users, num_movies, num_features, lamda)
        U -= alpha*X_grad
        Z -= alpha*theta_grad
        J = cofiCostFunc(U, Z, R, I, num_users, num_movies, num_features, lamda)
        # iterations -= 1

    return U, Z, J

def sim_asycos(ratings, p1, p2, m):

    c1 = 0
    c2 = 0
    ci = 0
    p1_mag = 0
    p2_mag = 0
    p1_p2 = 0

    for i in xrange(0, m):
        if ratings[p1][i] > 0 and ratings[p2][i] > 0:
            ci += 1
        if ratings[p1][i] > 0:
            c1 += 1
        if ratings[p2][i] > 0:
            c2 += 1
        p1_p2 += ratings[p1][i]*ratings[p2][i]
        p1_mag += ratings[p1][i]**2
        p2_mag += ratings[p2][i]**2

    if c1 == 0 or p1_mag == 0 or p2_mag == 0:
        return 0
    else:
        return (p1_p2/(sqrt(p1_mag)*sqrt(p2_mag)))*((2*ci*ci)/float((c1*(c1+c2))))


if __name__ == "__main__":

    # R = [[1.00, 0.33, 0.00, 0.16, 0.12, 0.08],
    #      [0.11, 1.00, 0.09, 0.11, 1.18, 0.68],
    #      [0.00, 0.28, 1.00, 0.15, 0.00, 0.37],
    #      [0.16, 0.33, 0.15, 1.00, 0.00, 0.40],
    #      [0.08, 0.37, 0.00, 0.00, 1.00, 0.13],
    #      [0.03, 0.82, 0.15, 0.16, 0.08, 1.00]]

    # I = [[1, 1, 0, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 1],
    #      [0, 1, 1, 1, 0, 1],
    #      [1, 1, 1, 1, 0, 1],
    #      [1, 1, 0, 0, 1, 1],
    #      [1, 1, 1, 1, 1, 1]]

    u = 6
    m = 943

    ratings = ratmat[0:u, 0:m]
    R = numpy.zeros((u, u))
    for i in xrange(0, u):
        for j in xrange(0, u):
            R[i][j] = sim_asycos(ratings, i, j, m)

    R *= 10

    print 'Similarity Matrix Complete'
    print R
    print ''

    t = range(u)
    I = numpy.zeros((u, u))
    for i in xrange(0, u):
        I[i][i] = 1
        random.shuffle(t)
        I[i, t[0:4]] = 1

    print 'Similarity Matrix Training Set'
    print I*R
    print ''

    N = len(R)
    M = len(R[0])
    err = sys.maxint
    mat = []
    k = 5
    for K in xrange(5, 6):
        nP, nQ, J = matrix_factorization(R, I, N, M, K)
        err_temp = nP.dot(nQ.T)
        err_t = J
        if err_t < err:
            err = err_t
            k = K
            mat = err_temp

    print J
    print 'Obtained at k' + str(k)
    print mat
    print err_t
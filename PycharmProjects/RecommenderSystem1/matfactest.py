__author__ = 'rochak'

# Standard python modules
import sys

# For scientific computing
import numpy
from numpy import *
import recommendations as rc
import random
import scipy.io, scipy.misc, scipy.optimize, scipy.cluster.vq
import pandas
# For plotting
from matplotlib import pyplot, cm, colors, lines
from mpl_toolkits.mplot3d import Axes3D


def normalizeRatings(Y, R):
    m = shape(Y)[0]
    Y_mean = zeros((m, 1))
    Y_norm = zeros(shape(Y))

    for i in range(0, m):
        idx = where(R[i] == 1)
        Y_mean[i] = mean(Y[i, idx])
        Y_norm[i, idx] = Y[i, idx] - Y_mean[i]

    return Y_norm, Y_mean


def unrollParams(params, num_users, num_movies, num_features):
    X = params[:num_movies * num_features]
    X = X.reshape(num_features, num_movies).transpose()
    theta = params[num_movies * num_features:]
    theta = theta.reshape(num_features, num_users).transpose()
    return X, theta


def cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lamda):
    X, theta = unrollParams(params, num_users, num_movies, num_features)
    inner = (X.dot(theta.T) - Y) * R
    X_grad = inner.dot(theta) + lamda * X
    theta_grad = inner.T.dot(X) + lamda * theta
    return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamda):
    X, theta = unrollParams(params, num_users, num_movies, num_features)
    J = 0.5 * sum(((X.dot(theta.T) - Y) * R) ** 2)
    regularization = 0.5 * lamda * (sum(theta ** 2) + sum(X ** 2))
    return J + regularization


def plot_it():
    movies, Y, R = rc.loadMovieLensCofi()
    print mean(extract(Y[0, :] * R[0, :] > 0, Y[0, :]))
    pyplot.imshow(Y)
    pyplot.ylabel('Movies')
    pyplot.xlabel('Users')
    pyplot.show()

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

def collaborativeFiltering():


    movmap, ratmat, boolmat = rc.loadMovieLensCofi()

    u = 10
    m = 943

    ratings = ratmat[0:u, 0:m]

    Y = numpy.ones((u, u))

    for i in xrange(0, u):
        for j in xrange(0, u):
            Y[i][j] = sim_asycos(ratings, i, j, m)

    print Y

    t = range(u)
    R = numpy.zeros((u, u))
    for i in xrange(0, u):
        R[i][i] = 1
        random.shuffle(t)
        R[i, t[0:5]] = 1

    # Y_norm, Y_mean = normalizeRatings(Y, R)
    print ''
    print Y*R
    print ''


    num_movies, num_users = shape(Y)
    num_features = 9

    X = numpy.random.random_sample(num_movies*num_features)
    theta = numpy.random.random_sample(num_users*num_features)
    initial_params = r_[X.T.flatten(), theta.T.flatten()]
    lamda = 10

    # result = scipy.optimize.fmin_cg(cofiCostFunc, fprime=cofiGradFunc, x0=initial_params,
    #                                 args=(Y, R, num_users, num_movies, num_features, lamda),
    #                                 maxiter=150, disp=True, full_output=True, method = 'Nelder-Mead')

    optsfmin = {'maxiter': None, 'disp': True, 'gtol': 1e-5, 'norm': numpy.inf,  'eps': 1.4901161193847656e-08}
    opts = {'maxiter': 150, 'disp': True}
    result = scipy.optimize.minimize(cofiCostFunc, x0=initial_params, jac=cofiGradFunc, args=(Y, R, num_users, num_movies, num_features, lamda), tol=1.5, method='CG', options=opts)
    # J, params = result[1], result[0]
    #
    params = result.x
    X, theta = unrollParams(params, num_users, num_movies, num_features)
    #
    # # return X, theta, movies
    #
    prediction = X.dot(theta.T)

    print prediction
    # print result

    # my_prediction = prediction[:, 0:1] + Y_mean
    #
    # idx = my_prediction.argsort(axis=0)[::-1]
    #
    # for i in range(0, 10):
    # j = idx[i, 0]
    #     print "Predicting rating %.1f for the movie %s" % (my_prediction[j], movies[j])

    # for k in range(1, 11):
    #     my_prediction = prediction[:, k - 1:k] + Y_mean
    #     idx = my_prediction.argsort(axis=0)[::-1]
    #     c = 0
    #     d = 0
    #     for i in range(0, 1682):
    #         j = idx[i, 0]
    #         if R[j][k - 1] == 1:
    #             d += 1
    #             # print "Predicting rating %.1f for the movie %s" % (my_prediction[j], movies[j])
    #             # print "Actual rating %.1f for the movie %s\n" % (Y[j][k-1], movies[j])
    #             val = my_prediction[j] - Y[j][k - 1]
    #             if val >= 1.0 or val <= -1.0:
    #                 c += 1
    #     print c
    #     print d
    #     print ''


def main():
    collaborativeFiltering()


if __name__ == '__main__':
    main()

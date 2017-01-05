__author__ = 'shubham'

# Standard python modules
import sys

# For scientific computing
import numpy
from numpy import *
import recommendations as rc
import random
import scipy.io, scipy.misc, scipy.optimize, scipy.cluster.vq

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


def collaborativeFiltering():
    movies, Y, R = rc.loadMovieLensCofi()
    num_movies = 1440
    num_users = 943
    li = range(0, 1682)
    temp = numpy.random.permutation(numpy.array(li))
    #print temp
    #print str(len(temp))
    ch = temp[0:1440]
    nch = temp[1440:1682]

    ch = sort(ch)
    #print ch
    #print str(len(ch))
    nch = sort(nch)
    Y = Y[ch, :]
    R = R[ch, :]
    numpy.savetxt('training.txt', ch, delimiter=',')
    numpy.savetxt('testing.txt', nch, delimiter=',')
    Y_norm, Y_mean = normalizeRatings(Y, R)
    num_features = 10

    X = numpy.random.randn(num_movies, num_features)
    theta = numpy.random.randn(num_users, num_features)
    initial_params = r_[X.T.flatten(), theta.T.flatten()]
    lamda = 5

    result = scipy.optimize.fmin_cg(cofiCostFunc, fprime=cofiGradFunc, x0=initial_params,
                                    args=(Y_norm, R, num_users, num_movies, num_features, lamda),
                                    maxiter=150, disp=True, full_output=True)
    J, params = result[1], result[0]

    X, theta = unrollParams(params, num_users, num_movies, num_features)

    return theta

    # prediction = X.dot(theta.T)

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

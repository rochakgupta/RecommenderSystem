__author__ = 'Rochak'

import numpy
import recommendations as rc
from math import *


def cos_similarity(u1, u2, R):
    m1 = 0
    m2 = 0
    m1_m2 = 0
    c1 = 0
    c2 = 0
    c1_c2 = 0
    index = R[u1, :]
    index = numpy.where(index > 0)[0]
    for i in index:
        c1 += 1
        m1 += R[u1][i] * R[u1][i]
        if R[u2][i] > 0:
            c2 += 1
            m2 += R[u2][i] * R[u2][i]
        if R[u1][i] > 0 and R[u2][i] > 0:
            c1_c2 += 1
            m1_m2 += R[u1][i] * R[u2][i]

    if c1_c2 == 0:
        return numpy.nan

    return (m1_m2 / float(sqrt(m1) * sqrt(m2)) * ((2 * (c1_c2)*(c1_c2)) / float(c1 * (c1 + c2))))

def msd_similarity(u1, u2, R):
    m1 = 0
    m2 = 0
    m1_m2 = 0
    c1 = 0
    c2 = 0
    c1_c2 = 0
    l = 16
    index = R[u1, :]
    index = numpy.where(index > 0)[0]
    for i in index:
        c1 += 1
        m1 += R[u1][i] * R[u1][i]
        if R[u2][i] > 0:
            c2 += 1
            m2 += R[u2][i] * R[u2][i]
        if R[u1][i] > 0 and R[u2][i] > 0:
            c1_c2 += 1
            m1_m2 += (R[u1][i] - R[u2][i]) ** 2

    if c1_c2 == 0:
        return numpy.nan

    return ((l-(m1_m2/float(c1_c2)))/float(l)) * ((2 * (c1_c2)*(c1_c2)) / float(c1 * (c1 + c2)))

def cosine_sim():
    movies, Y, R = rc.loadMovieLensCofi()
    Y = Y.transpose()
    ch = {}
    df = open('training.txt')
    c = 0
    for item in df:
        ch[c] = int(float(item))
        c += 1
    ch = ch.values()
    ch = numpy.array(ch)
    Y = Y[:, ch]
    u = 943
    pred = numpy.zeros((u, u))
    for i in xrange(0, u):
        for j in xrange(0, u):
            pred[i][j] = cos_similarity(i, j, Y)

    numpy.savetxt('matrixacos.csv', pred, delimiter=',')
    return Y,u

def msd_sim(Y, u):
    pred = numpy.zeros((u, u))
    for i in xrange(0, u):
        for j in xrange(0, u):
            pred[i][j] = msd_similarity(i, j, Y)

    numpy.savetxt('matrixamsd.csv', pred, delimiter=',')

def main():
    Y,u = cosine_sim()
    msd_sim(Y,u)


if __name__ == '__main__':
    main()
__author__ = 'Shubham'
import math
import numpy
from sklearn.preprocessing import Imputer
import recommendations
import itertools
imp = Imputer(strategy="m")
users={'User 1':{'Item 1':4,'Item 3':2},
         'User 2':{'Item 1':4,'Item 2':1,'Item 3':2,'Item 4':1,'Item 5':1,'Item 6':1},
         'User 3':{'Item 2':2,'Item 4':2},
         'User 4':{'Item 2':1,'Item 3':2},
         'User 5':{'Item 1':4,'Item 5':4,'Item 6':4},
         'User 6':{'Item 2':1,'Item 3':1,'Item 4':2,'Item 5':1,'Item 6':1}
         }

'''def matrix_factorization(R, P, Q, K, steps=5000, alpha=.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):

                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
                else:
                    eij = 0
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):

                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
                else:

                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

        if e < 0.01:
            break
    return P, Q.T
'''
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T
#a=numpy.zeros((6,6))
a=numpy.zeros((6,6))
def acosim(users,u1,u2):
    countint=0.000
    dotp=0.0000
    #users = users[:,arr]

    for item in users[u1]:
        if item in users[u2]:

            countint=countint+1
            dotp=dotp+users[u1][item]*users[u2][item]
    #print str(u1)+' '+str(u2)+' '+str(countint)+' '+str(dotp)
    if (countint==0):
        #a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=0
        a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=numpy.nan
        return 0

    u1mag=0.000000
    for item in users[u1]:
        u1mag=u1mag+pow(users[u1][item],2)
    u2mag=0.00000
    for item in users[u2]:
        u2mag=u2mag+pow(users[u2][item],2)

    u1mag=math.sqrt(u1mag)
    u2mag=math.sqrt(u2mag)
    #print str(u2mag)

    countu1=len(users[u1])
    #print countu1
    countu2=len(users[u2])

    sim=(2*countint*countint)/(countu1*(countu1+countu2))
    #print sim
    ans=dotp*sim/(u1mag*u2mag)
    #print sorted(users.keys()).index(u1)
    a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=ans
    #print str(ans)+' '


def amsdsim(users,u1,u2):
    countint=0.000
    msd=0.0000
    for item in users[u1]:
        if item in users[u2]:
            countint=countint+1
            msd=msd+pow(users[u1][item]-users[u2][item],2)
    #print str(u1)+' '+str(u2)+' '+str(countint)+' '+str(dotp)
    if (countint==0):
        a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=numpy.nan
        #a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=0
        return 0

    u1mag=0.000000
    msd=msd/countint
    L=16.0000


    countu1=len(users[u1])
    #print countu1
    countu2=len(users[u2])

    sim=(2*countint*countint)/(countu1*(countu1+countu2))
    #print sim
    ans=((L-msd)*sim)/L
    #print sorted(users.keys()).index(u1)
    a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=ans
    #print str(ans)+' '
def acosim1(users,u1,u2):
    countint=0.000
    dotp=0.0000
    #users = users[:,arr]
    print str(u1)+' '+str(u2)
    for item in users[u1]:
        if item in users[u2]:
            print item
            countint=countint+1
            dotp=dotp+users[u1][item]*users[u2][item]
    #print str(u1)+' '+str(u2)+' '+str(countint)+' '+str(dotp)
    if (countint==0):
        #a[sorted(users.keys()).index(u1)][sorted(users.keys()).index(u2)]=0
        a[(users.keys()).index(u1)][(users.keys()).index(u2)]=numpy.nan
        return 0

    u1mag=0.000000
    for item in users[u1]:
        u1mag=u1mag+pow(users[u1][item],2)
    u2mag=0.00000
    for item in users[u2]:
        u2mag=u2mag+pow(users[u2][item],2)

    u1mag=math.sqrt(u1mag)
    u2mag=math.sqrt(u2mag)
    #print str(u2mag)

    countu1=len(users[u1])
    #print countu1
    countu2=len(users[u2])

    sim=(2*countint*countint)/(countu1*(countu1+countu2))
    #print sim
    ans=dotp*sim/(u1mag*u2mag)
    #print sorted(users.keys()).index(u1)
    #print
    a[(users.keys()).index(u1)][(users.keys()).index(u2)]=ans
    #print str(ans)+' '



def calculatematrixacos(prefs):
    #pref2=sorted(prefs.keys())
    for u1 in sorted(prefs.keys()) :
        for u2 in sorted(prefs.keys()):
            if u1!=u2:
                #print
                print sorted(prefs.keys()).index(u1)
                acosim(prefs,u1,u2)
            else:
                 a[sorted(prefs.keys()).index(u1)][sorted(prefs.keys()).index(u2)]=1.0
    return 0
def calculatematrixacos1(prefs):
    #pref2=sorted(prefs.keys())
    for u1 in (prefs) :
        for u2 in (prefs):
            if u1!=u2:
                #print
                print prefs.values()[0]
                print str((prefs.keys()).index(u1))+' '+str((prefs.keys()).index(u2))
                acosim1(prefs,u1,u2)
            else:
                 a[(prefs.keys()).index(u1)][(prefs.keys()).index(u2)]=1.0
    return 0

def calculatematrixamsd(prefs):
    for u1 in sorted(prefs.keys()) :
        for u2 in sorted(prefs.keys()):
            if u1!=u2:
                #print
                amsdsim(prefs,u1,u2)
            else:
                a[sorted(prefs.keys()).index(u1)][sorted(prefs.keys()).index(u2)]=1
    return 0
def matrix_factorization1(R, P, Q, K, steps=10000, alpha=0.002, beta=0.02):
    Q = Q.T
    Indi = numpy.copy(R)
    Indi[Indi<>0] = 1
    for step in xrange(steps):
        Pred = P.dot(Q)
        _Pred = numpy.multiply(Indi, Pred)
        E = R -  _Pred
        P_tmp = numpy.copy(P)
        Q_tmp = numpy.copy(Q)
        P = P_tmp + alpha*(E.dot(Q_tmp.T) - beta*P_tmp)
        Q = Q_tmp + alpha*(P_tmp.T.dot(E) - beta*Q_tmp)
        rmse = numpy.sqrt(E.ravel().dot(E.flat) / len(Indi[Indi.nonzero()]))
        #print 'step:%s'%step
        #print "RMSE:", rmse
    return P, Q.T
#So you can see we ca
from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))
prefs=recommendations.loadMovieLens( )
'''
#print prefs['344']
print len(prefs['87'])
#top5 =itertools.islice(prefs.iteritems(),0, 5)

#print users
#print n_items
'''
n_items = take(100, prefs.iteritems())
calculatematrixacos1(users)


#b=open('training.txt','r')
#arr=b.read()
#with open('training.txt') as f:
 #   lines = f.readlines()

#for item in prefs:
#    strarr=
#print arr[]

print "Acos"
print a



numpy.savetxt('matrixacos5.csv', a, delimiter=',')
'''
import pymf
nmf_mdl = pymf.NMF(a,num_bases=7)
#nmf_mdl.initialization()
nmf_mdl.factorize(niter=100)
V_approx=numpy.dot(nmf_mdl.W,nmf_mdl.H)
print "hello" +str(V_approx)
'''
'''
import probabilistic
import matplotlib.pyplot as plt

U, V, m = probabilistic.PMF(a, learning_rate=0.001, momentum=0.95,
                  minibatch_size=2, rank=7, max_epoch=250, random_state=1999)
R2 = numpy.dot(U, V.T) + m
#plt.matshow(R * (R > 0))
#    plt.title("Ground truth ratings")
#    plt.matshow(R2 * (R > 0))
print R2*(R2>=0)
'''
'''
from completethat import MatrixCompletionBD
problem = MatrixCompletionBD(a)
problem.train_sgd(dimension=6,init_step_size=.01,min_step=.000001, reltol=.001, maxiter=1000,batch_size_sgd=50000,shuffle=True)
problem.validate_sgd(a)
problem.save_model()
'''

'''
N = len(a)
M = len(a[0])
K = 9

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
#print P
nP, nQ = matrix_factorization(a, P, Q, K)
print "Matrix Factorization"
print numpy.dot(nP,numpy.transpose(nQ))
calculatematrixamsd(n_items)
print "Amsd"
print a

'''
'''
N = len(a)
M = len(a[0])
K = 2

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
#print P
nP, nQ = matrix_factorization(a, P, Q, K)
print "Matrix Factorization"
print numpy.dot(nP,numpy.transpose(nQ))
'''

'''import nimfa


V = numpy.array([[1,numpy.nan , 3], [4, 5, 6], [6, 7, 8]])
print('Target:\n%s' % a)

lsnmf = nimfa.Pmf(a, max_iter=100, rank=5)
lsnmf_fit = lsnmf()

W = lsnmf_fit.basis()
#print('Basis matrix:\n%s' % W)

H = lsnmf_fit.coef()
#print('Mixture matrix:\n%s' % H)

#print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))

#print('Rss: %5.3f' % lsnmf_fit.fit.rss())
#print('Evar: %5.3f' % lsnmf_fit.fit.evar())
#print('Iterations: %d' % lsnmf_fit.n_iter)
print('Target estimate:\n%s' % numpy.dot(W, H))'''
'''
a = imp.fit_transform(a)
print a
'''
'''
from sklearn.decomposition import NMF
nmf = NMF()
W = nmf.fit_transform(a)
H = nmf.components_
nR = numpy.dot(W,H)
print nR
'''


'''
from pandas import DataFrame
import numpy as np

from sklearn.decomposition import ProjectedGradientNMF
users1=range(0,6);
users2=range(0,6);
X = DataFrame(a, index=users1, columns=users2)

X_imputed = X.copy()
msk = (X.values + np.random.randn(*X.shape) - X.values) < 1.0
X_imputed.values[~msk] = 0
nmf_model = ProjectedGradientNMF(n_components = 2)
W = nmf_model.fit_transform(X_imputed.values)
H = nmf_model.components_


while nmf_model.reconstruction_err_**2 > 10:
   nmf_model.fit_transform(X_imputed.values)
   W = nmf_model.fit_transform(X_imputed.values)
   H = nmf_model.components_
   X_imputed.values[~msk] = W.dot(H)[~msk]

print np.dot(W,H)
'''
'''
import nimfa
#specify nmf model for X

model = nimfa.Lsnmf(a, max_iter=100000,rank =4)

#fit the model
fit = model()
#get U and V matrices from fit
U = fit.basis()
V = fit.coef()

#let's have a look at them
#print "X=\n"  + str(X)
#print "U=\n" + str(U)
#print "V=\n" + str(V.round(1))
print numpy.dot(U,V)

'''

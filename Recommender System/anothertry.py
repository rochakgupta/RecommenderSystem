__author__ = 'Shubham'
import numpy as np
import scipy
from scipy import linalg
from numpy import dot
import scipy.sparse as sps

def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print 'fit residual', np.round(fit_residual, 4),
            print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y
shape = (1000, 2000)
rows, cols = 4, 5
a = sps.coo_matrix((rows, cols))

print a
#a=np.zeros((200,300))
n1,n2=nmf(a,100,1000,1e-6,1e-6)

print np.dot(n1,n2)
from sklearn.decomposition import NMF , ProjectedGradientNMF
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
R = np.array(R)
nmf = NMF()
W = nmf.fit_transform(R);
H = nmf.components_;
nR = np.dot(W,H)
print nR
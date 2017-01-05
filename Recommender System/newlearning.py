from sklearn.preprocessing import Imputer
import numpy as np
imp = Imputer(strategy="mean")
a = np.random.random((5,5))
a[(1,4,0,3),(2,4,2,0)] = np.nan
print a
'''array([[ 0.77473361,  0.62987193,         nan,  0.11367791,  0.17633671],
       [ 0.68555944,  0.54680378,         nan,  0.64186838,  0.15563309],
       [ 0.37784422,  0.59678177,  0.08103329,  0.60760487,  0.65288022],
       [        nan,  0.54097945,  0.30680838,  0.82303869,  0.22784574],
       [ 0.21223024,  0.06426663,  0.34254093,  0.22115931,         nan]])'''
a = imp.fit_transform(a)
print a
'''array([[ 0.77473361,  0.62987193,  0.24346087,  0.11367791,  0.17633671],
       [ 0.68555944,  0.54680378,  0.24346087,  0.64186838,  0.15563309],
       [ 0.37784422,  0.59678177,  0.08103329,  0.60760487,  0.65288022],
       [ 0.51259188,  0.54097945,  0.30680838,  0.82303869,  0.22784574],
       [ 0.21223024,  0.06426663,  0.34254093,  0.22115931,  0.30317394]])'''
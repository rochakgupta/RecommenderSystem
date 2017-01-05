
__author__ = 'Shubham'
import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.decomposition import ProjectedGradientNMF

# toy example data, actual data is ~500 by ~ 250
customers = range(20)
features = range(15)

toy_vals = np.random.random(20*15).reshape((20,15))
toy_mask = toy_vals < 0.9
toy_vals[toy_mask] = np.nan

X = DataFrame(toy_vals, index=customers, columns=features)
# end toy example data gen.
print X
# imputation w/ nmf loops
X_imputed = X.copy()
msk = (X.values + np.random.randn(*X.shape) - X.values) < 0.8
X_imputed.values[~msk] = 0

print X_imputed
nmf_model = ProjectedGradientNMF(n_components = 5)
W = nmf_model.fit_transform(X_imputed.values)
H = nmf_model.components_


while nmf_model.reconstruction_err_**2 > 1:
   nmf_model.fit_transform(X_imputed.values)
   W = nmf_model.fit_transform(X_imputed.values)
   H = nmf_model.components_
   X_imputed.values[~msk] = W.dot(H)[~msk]

print np.dot(W,H)

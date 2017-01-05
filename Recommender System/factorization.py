__author__ = 'Shubham'
import pymf
import numpy as np
data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
nmf_mdl = pymf.NMF(data, num_bases=2, niter=10)
nmf_mdl.initialization()
nmf_mdl.factorize()
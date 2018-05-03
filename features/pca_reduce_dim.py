# -*- coding: utf-8 -*-

import numpy as np
import pylab as Plot


#%%
def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print ("Preprocessing the data using PCA...")
	(n, d) = X.shape
	X = X - np.tile(np.mean(X, 0), (n, 1))
	(l, M) = np.linalg.eig(np.dot(X.T, X))
	Y = np.dot(X, M[:,0:no_dims])
	return Y
 
if __name__ == '__main__':
    X = np.loadtxt('fc7_features.txt')
    X = pca(X, 200).real
    np.savetxt('fc7_features.txt',X)

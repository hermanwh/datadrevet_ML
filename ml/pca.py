import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler

# read data
df = pd.read_csv("../datasets/iris_mod.csv")
# get values instead of df
x = df.values
print(x[:5,:])
# a standardScaler is used to standardize the input, aka mean = 0 and variance = 1
standardScaler = StandardScaler()
# fit_transform transforms the input x to the standardized domain and fits the standardScaler
x = standardScaler.fit_transform(x)
# define the principal componenet analysis model. n_componenets = dataset_size ==> 100% accuracy, obviously
pca = decomposition.PCA(n_components=2)
# train model
pca.fit(x)
# get the PCA transformation of the dataset, aka with n_componenets number of columns
transformed = pca.transform(x)
# inverse the transformation, aka try to re-construct the original, post-standardized data
inv_transformed = pca.inverse_transform(transformed)
# inverse the standardization, aka try to re-construct the original, pre-standardized data 
inv_standardized = standardScaler.inverse_transform(inv_transformed)

print(inv_standardized[:5,:])

# explains how well each feature represents the dataset. If n_componenets == dataset_size, it will sum to 1.0
# any lower number of components will sum to a number smaller than 1.0 
print(pca.explained_variance_ratio_)
#
print(pca.components_)

# this is the cov-matrix of the data set, regardless of PCA model (I think)
print(pca.get_covariance())
# this is the same cov-matrix using numpy, which should be equal to the one above
print(np.cov(x.T))
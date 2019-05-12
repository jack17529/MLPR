import numpy as np

from skfuzzy.cluster import cmeans

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
data = scale(iris.data)

n_samples, n_features = data.shape
n_iris = len(np.unique(iris.target))
target = iris.target

estimator = KMeans(n_clusters=3)

labels = estimator.fit_predict(data)
print ('K-Means Algorithm Accuracy:', accuracy_score(target, labels))

centr, u_origin, _, _, _, _, fpc = cmeans(data, c=10, m=2, 
                                          error=0.005, 
                                          maxiter=1000)

print('Fuzzy C-Means Accuracy:', fpc)
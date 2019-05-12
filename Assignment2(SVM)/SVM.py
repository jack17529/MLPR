# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
#from sklearn import cross_validation


# get the data
iris = load_iris()
data0 = iris.data
labels0 = iris.target
(n,p) = data0.shape

#train test split.
xtrain,xtest,ytrain,ytest = train_test_split(data0,labels0,test_size=0.3, random_state=21)


# rbf kernel.
rbfn = svm.SVC(C=10,kernel='rbf',gamma=0.001)
rbfn.fit(xtrain,ytrain)
pred_rbf=rbfn.predict(xtest)
acc_rbf = accuracy_score(ytest, pred_rbf)
print("Accuracy Score of Kernel=rbf is ",acc_rbf)

# poly kernel
quad = svm.SVC(C=10.0,kernel='poly',degree=2,gamma='auto')
quad.fit(xtrain,ytrain)
pred_quad=quad.predict(xtest)
acc_quad=accuracy_score(ytest,pred_quad)
print("Accuracy Score of Kernel=poly is ",acc_quad)

# sigmoid kernel
sigm = svm.SVC(C=10.0, kernel='sigmoid', gamma=0.001)
sigm.fit(xtrain, ytrain)
pred_sigmoid = sigm.predict(xtest)
acc_sigmoid = accuracy_score(ytest, pred_sigmoid)
print("Accuracy Score of Kernel=sigmoid is ",acc_sigmoid)

#plotting graph
results = []

results.append(accuracy_score(ytest, pred_rbf))
results.append(accuracy_score(ytest, pred_quad))
results.append(accuracy_score(ytest, pred_sigmoid))

label = ['rbf','quad','sigmoid']
index = np.arange(len(label))
plt.bar(index, results)
plt.xticks(index, label, fontsize=20, rotation=30)
plt.show()
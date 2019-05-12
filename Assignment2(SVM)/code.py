import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

data = pd.read_csv('training.csv')
labels = data['signal']
data = data.drop(['signal', 'id'], 1)
data = data[data.keys()[:5]]

_, data, _, labels = train_test_split(data, labels, test_size=0.01, random_state=21)
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.3, random_state=21)

print('Gaussian SVM.....')
clf = SVC(C=10.0, kernel='rbf', gamma='scale')
clf.fit(xtrain, ytrain)
pred_rbf = clf.predict(xtest)
acc_rbf = accuracy_score(ytest, pred_rbf)
print(acc_rbf)

print('Sigmoid SVM.....')
clf = SVC(C=10.0, kernel='sigmoid', gamma='scale')
clf.fit(xtrain, ytrain)
pred_sigmoid = clf.predict(xtest)
acc_sigmoid = accuracy_score(ytest, pred_sigmoid)
print(acc_sigmoid)

print('Quadratic SVM.....')
clf = SVC(C=10.0, kernel='poly',degree=2, gamma='auto')
clf.fit(xtrain, ytrain)
pred_quad = clf.predict(xtest)
acc_quad = accuracy_score(ytest, pred_quad)
print(acc_quad)

print('Showing Graph.....')
results = []

results.append(accuracy_score(ytest, pred_rbf))
results.append(accuracy_score(ytest, pred_sigmoid))
results.append(accuracy_score(ytest, pred_quad))

plt.plot(['rbf', 'sigmoid', 'quadratic'], results)
plt.show()
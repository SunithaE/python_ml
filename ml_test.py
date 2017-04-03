# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:34:02 2017

@author: seswaraiah
"""

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

'''Load the iris dataset'''
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pd.read_csv(url, names=names)
'''Dimension of the data'''
print(dataset.shape)

'''sample data'''
print(dataset.head(10))

'''Dataset description, this will  give metrics like min, max, count, mean, std'''
print(dataset.describe())


'''How the class field is classified'''
print(dataset.groupby('class').size())

'''Univariate Plot the dataset'''
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False,sharey=False)
plt.show()

dataset.hist()
plt.show()


'''Multivariate plot'''
scatter_matrix(dataset)
plt.show()

'''Split the data (training and test)'''
array = dataset.values
print(array)
X = array[:,0:4]
Y = array[:,4]
print(X)
print(Y)
validation_size = 0.20
seed = 7
scoring = 'accuracy'
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X, Y, 
                                                                 test_size=validation_size,
                                                                 random_state=seed)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

'''Build the model, Simple linear (LDA and LR) and nonlinear(KNN,CART,NB,SVM)'''
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM',SVC()))

'''Evaluate the model'''
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,
                                                 cv = kfold, scoring=scoring)
    print("results", cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)


'''compare the algorithms'''
fig = plt.figure()
fig.suptitle('Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

'''Run the test set'''
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print("Predictions ", predictions)
accuracy = accuracy_score(Y_test, predictions)
conf_marix = confusion_matrix(Y_test, predictions)
classification_report = classification_report(Y_test, predictions)
print("Accuracy is ", accuracy)
print("Confusion matrix is ", conf_marix)
print("Classification report is ", classification_report)

#https://en.wikipedia.org/wiki/Machine_learning
















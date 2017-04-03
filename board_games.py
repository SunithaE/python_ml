# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:00:45 2017

@author: seswaraiah

This is to predict the reviews of a game with the existing average reviews and the user reviews
Clustering is done based on the features in it and we arent sure about the labels in this. 
So kmeans is used. 
PCA is used here inoder to conflate the features
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('C:/python_workspace/board_games.csv')
print(dataset.head(5))
print(dataset.columns)
print(dataset.shape)

plt.hist(dataset["average_rating"])
plt.show()

#print(dataset[dataset["average_rating"]==0])

print(dataset[dataset["average_rating"]==0].iloc[0])
print(dataset[dataset["average_rating"]>0].iloc[0])

# remove the data that has user reviews 0 which doesnt make sense to do the prediction.
dataset = dataset[dataset["users_rated"]>0]

dataset = dataset.dropna(axis=0)

print(dataset.shape)

# Lets do the clustering to see how the games have been reviews on what basis
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = dataset._get_numeric_data()
print(good_columns.columns)
kmeans_model.fit(good_columns)


labels = kmeans_model.labels_
print("labels", type(labels))

# create a PCA model

pca = PCA(2)
# fit the columns 
plot_columns = pca.fit_transform(good_columns)
print("pca columns ", plot_columns)

#plot it 
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

print(dataset.corr()["average_rating"])

columns = dataset.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating ",
                                         "type", "name"]]
# These will be the features
print(columns)
# this will be the label. So to segregate between the label and the features
target ="average_rating"

train = dataset.sample(frac=0.8, random_state=1)
test = dataset.loc[~dataset.index.isin(train.index)]

print(train.shape)
print(test.shape)
#Linear regression
model = LinearRegression()
model.fit(train[columns], train[target])

prediction = model.predict(test[columns])
print(prediction)
print(mean_squared_error(prediction, test[target]))

#Randomforest
model2 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
model2.fit(train[columns], train[target])
prediction2 = model2.predict(test[columns])
print(mean_squared_error(prediction2, test[target]))






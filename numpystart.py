# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:25:16 2017

@author: seswaraiah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
#import tensorflow as tf
from tensorflow import tensorflow

# A list
lst = [1,2,3]

# numpy list
a = np.array([1,2,3])

#iterate through the list
for i in a:
    print(i)


#Cant append elements to the numpy list
#a = a.append([5]) # This will throw error. But default list will do append. 

''' 2*lst will give [1,2,3,1,2,3]
 2*a (numpy list) will give [2,4,6]

 2**lst -> error 
 a**2 -> will do the squaring. (numpy does the squaring whereas the list doesnt)
 '''

#squareroot and log
np_log = np.log(a)
np_sqrt = np.sqrt(a)
np_exp = np.exp(a)
print(np_log)
print(np_sqrt)
print(np_exp)

#dot product of two array
x = np.array([1,2])
y = np.array([2,1])
print(x*y)
print("Numpy multiplication is ", np.sum(x*y))
dot_prod = np.dot(x,y)
print("dot prod is ", dot_prod)

#magnitude 
print(np.sqrt((x*x).sum()))

#linearalgebra
print(np.linalg.norm(x))

cosangle = np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))
print(cosangle)
angle = np.arccos(cosangle)
print(angle)

mat = np.matrix([[1,2],[2,2]])
print(mat)

print(np.array([[1,2],[2,2]]))

#matrix with zeros
zero_mat = np.zeros([5])
print(zero_mat)

#matrix with ones 
one_mat = np.ones([5])
print(one_mat)

# random numbers
rand_mat = np.random.random([5])
print(rand_mat)
    
gaussian_mat = np.random.randn(5)
print(gaussian_mat)

print("gaussion mean is ", gaussian_mat.mean())
print("gaussian variance is ", gaussian_mat.var())


''' problem solve: a event fair conducted where entry fee for childern per head
is  $1.5 and for adults it is $4. $5050 was collected  from the entire event.
2200 people attended. Find how many children and adult attended. 

1.5x1 + 4x2 = 5050
x1 + x2 = 2200 '''

x1 = np.array([[1,1],[1.5,4]])
x2 = np.array([2200,5050])
lin = np.linalg.solve(x1, x2)
print(lin)

df_list = []

for line in open('data_2d.csv'):
    row = line.split(',')
    #print(row)
    #sample = map(float,row)
    df_list.append(row)
#print(df_list)

df_np = np.array(df_list)
print(df_np.shape)


# Pandas
pd_input = pd.read_csv("data_2d.csv", header=None)
print(type(pd_input))
print(pd_input.info())

print(pd_input.head(10))

pd_mat = pd_input.as_matrix()
print(type(pd_mat))
print(pd_input[0])

#matplot
x_axis = np.linspace(0,10,50)
y_axis = np.sin(x_axis)
plt.plot(x_axis,y_axis)
plt.show()


data_1d = pd.read_csv('data_1d.csv', header=None).as_matrix()

x_a = data_1d[:,0]
y_a = data_1d[:,1]

plt.scatter(x_a,y_a)
plt.show()


# Scipy

print(norm.pdf(0))

print(norm.pdf(0, loc=5, scale=10))

r = np.random.randn(10)
print(norm.pdf(r))
print(norm.logpdf(r))


r1 = np.random.randn(1000)
plt.hist(r1, bins=100)
plt.show()


r2 = np.random.randn(1000 , 2)
plt.scatter(r2[:,0], r2[:,1])
plt.show()


l1 = np.linspace(0,10,100)
print(l1)
s1 = np.sin(l1) + np.sin(3*l1)
plt.plot(s1)
plt.show()

#tensorflow
x = tf.constants(35, name='x')
y = tf.Variable(x+5, name='y')












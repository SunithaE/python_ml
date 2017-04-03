# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:14:19 2017

@author: seswaraiah
"""

import os
import scipy as sp
import random
from scipy.stats import gamma
'''Create necessary dir'''

#DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
#CHART_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "charts")
#
#for d in [DATA_DIR, CHART_DIR]:
#    if not os.path.exists(d):
#        os.mkdir(d)
      
print(random.seed(10))
print(random.random())
print(random.random())

print(sp.random.seed(3))

x = sp.arange(1, 31*24)
print("x is ", x)
y = sp.array(200*(sp.sin(2*sp.pi*x/(7*24))))
print("y is ", y)
y += gamma.rvs(15, loc=0, scale = 100, size=len(x))
y += 2*sp.exp(x/100.0)
print("y +", y)












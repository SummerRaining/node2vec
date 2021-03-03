# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:46:22 2021

@author: shujie.wang
"""

from joblib import Parallel, delayed
import time, math

def my_fun(i):
    """ We define a simple function here.
    """
    time.sleep(1)
    return math.sqrt(i**2)

num = 10
start = time.time()
for i in range(num):
    my_fun(i)
    
end = time.time()

print('{:.4f} s'.format(end-start))

start = time.time()
# n_jobs is the number of parallel jobs
Parallel(n_jobs=3)(delayed(my_fun)(i) for i in range(num))
end = time.time()
print('{:.4f} s'.format(end-start))
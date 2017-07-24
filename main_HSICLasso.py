'''
Created on 2017/07/24

@author: myamada
'''
import numpy as np
from HSICLasso import *
#from kernel_Gaussian import *
from pylab import *
import scipy.io as spio

#Reading Matlab file
data = spio.loadmat('feat_select_data.mat')

dataset = 1 #1 for regression, 2 for classification

Xin     = data['X']
Yin     = data['Y']
beta0   = data['beta']
path0 = data['path']

if dataset == 1:
    path, beta,A,lam = hsiclasso(Xin,Yin,numFeat=5)
else:
    #Generate label data
    Yin = (np.sign(Yin) + 1) / 2 + 1

    path, beta, A, lam = hsiclasso(Xin, Yin, numFeat=5,ykernel='Delta')


nonzero_ind = beta.nonzero()[0]
t = path.sum(0)
figure()
hold(True)
for ind in range(0,len(A)):
    plot(t,path[A[ind],:])

show()



#!/usr/bin/env python
# coding: utf-8

# # Ergodicity breaking test
# 
# 
# 
# Magdziarz and Weron proposed *ergodicity and mixing estimators* based on a single particle trajectory.
# In order to test ergodicity for process $x(n)$ one needs to calculate the functional $y(n)$ for trajectory $y(n) = x(n+1)-x(n)$:
# $$ E(n) = <\exp(i[y(n)-y(0)])> - |<\exp(iy(0))>|^2,$$
# where $<>$ is ensemble average (so, average over different number of trajectory).
# 
# This can be applied for stationary infitely divisible processes. For practical applications we substitute ensemble average by time-average. 
# These processes are ergodic if and only if $E(n)\rightarrow0$ for large $n$. 
# For finite trajectory the ergodicity estimate is:
# $$ 
# E(n) = \frac{1}{N-n+1} \sum_{k=0}^{N-n} e^{i[y(k+n) - y(k)]} - |\sum_{k=0}^{N} \frac{e^{iy(k)}}{N+1}|.
# $$

# In[1]:


import andi
import numpy as np
import andi
from matplotlib import pyplot as plt
# import fbm module # !pip install fbm

AD = andi.andi_datasets()


##################################################
# set common paramter for all trajectories
Time = 500

##################################################


#X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N = 10)
#X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N = 10, tasks = 1, dimensions = 3)



def plot_trajectory(X,Y):
    'plotting trajectories'
    plt.plot(X,Y)#dataset[0,2:12], dataset[0, 12:], '-')
    plt.scatter(X,Y)#dataset[0,2:12], dataset[0,12:], c=np.arange(10), s = 250)
    plt.colorbar().set_label('Time'); plt.xlabel('X'); plt.ylabel('Y');



    
# start to generate AD dataset
AD = andi.andi_datasets()
AD.avail_models_name

datasetATTM = AD.create_dataset(T = Time , N = 1, exponents = [0.2], models = [0], dimension = 2)
#print(np.round(dataset[0], 2))
#print(type(X1),np.shape(Y1))
print(np.shape(datasetATTM))
print('trajectory loaded')
plot_trajectory(datasetATTM[0,2:Time+2],datasetATTM[0,Time+2:]) # 2D trajectory with first X coordinates, then Y coordinates 


# In[2]:


import andi
import numpy as np

#N_given = 10 #steps
#X1, Y1 = andi_dataset(N_given)

AD = andi.andi_datasets()
datasetCTRW = AD.create_dataset(T = Time , N = 1, exponents = [0.7], models = [1], dimension = 2)
#print(np.round(dataset[0], 2))
#print(type(X1),np.shape(Y1))
print(np.shape(datasetCTRW))
print('trajectory loaded')
plot_trajectory(datasetCTRW[0,2:Time+2],datasetCTRW[0,Time+2:]) # 2D trajectory with first X coordinates, then Y coordinates 
print(np.shape((datasetCTRW[0, 2:Time+2])))


# # Function for ergodicity breaking

# In[35]:


#function [E,F] = EFestimator(X,omega); 
#% This function implements the improved mixing and ergodicity estimators
# See article by Y. Lanoiselee and D. Grebenkov 
#INPUT: X - vector containing positions of the analyzed trajectory %
#omega - (optional) frequency (the default value is 2)
#% OUTPUT: E - the real part of the mixing estimator as a function of n %
#F - the real part of the ergodicity estimator as a function of n
import numpy as np
import math
import cmath

def erg_break(X, omega, n, N):
    '''
    X - vector containing positions of the analyzed trajectory 
    test for 1D array
    E - the real part of the mixing estimator as a function of n %
    F - the real part of the ergodicity estimator as a function of n
    
    for each n we get one number
    '''
    #if (nargin < 2):
    #omega = 2 # Default value for omega 
    N = np.size(X)-1 #Trajectory points are enumerated as X(0), ..., X(N) 
    X = X/np.std(np.diff(X)); # Normalization by the empirical standard deviation of displacements 

    print(np.shape(X))
    
    #initialisation of arrays E and D
    D = 0 # initial value # D=[]#np.zeros(N)
    E = 0 # initial value # E=[]#np.zeros(N)
    
    for k in range(0,N-n):
        #print(X[n+k])
        #D(n+1) = sum( exp( (1i)*omega*(X(n+1:end) - X(1:end-n)) ) )/(N-n+1);
        D = D+((cmath.exp( (1j)*omega*(X[n+k] - X[n]) ) ))   #for n in np.arange(N)
        
    D= D* 1./(N-n+1) #normalisation
    #E = D- abs(sum( cmath.exp((1j)*omega*X) ))^2/N/(N+1) + 1/N #difference of arrays D and E 
    #print(D)
    return E, D
    
    


# In[36]:


import matplotlib.pyplot as plt

X = np.random.rand(100)

omega = 2
Time = 10
E,D = erg_break(X, omega, n, Time)

# plot on complex plane
#XD = [x.real for x in D]
#YD = [x.imag for x in D]
#plt.scatter(XD,YD, color='red')
#plt.show()

print('ergodicity parameter for X random', D)


# # Erogidicity test for 1D

# In[82]:


Time = 100

AD = andi.andi_datasets()
datasetCTRW1D1 = AD.create_dataset(T = Time , N = 1, exponents = [0.7], models = [1], dimension = 1)
datasetCTRW1D2 = AD.create_dataset(T = Time , N = 1, exponents = [0.7], models = [1], dimension = 1)

print(np.shape(datasetCTRW1D))
print(np.shape(np.reshape(datasetCTRW1D, Time+2)))

omega = 2 #n=5 # average time parameter
delta=5 # average window size

'''
data from ANDI should be reshaped to (Dim * Time + 2)
np.reshape(datasetCTRW1D1, Dim * Time + 2)
'''

print(E,D)
D_list = []
D_re = np.zeros(Time-delta)
D_im = np.zeros(Time-delta)
for n in range(0, Time-delta):
    E,D = erg_break(np.reshape(datasetCTRW1D1, Time+2), omega, n, Time)
    D_re[n] = D.real
    D_im[n] = D.imag
    D_list.append(D)
    
    
def array_D_ergodicity(dataset, omega, n, Time):
    D_list = []
    D_re = np.zeros(Time-delta)
    D_im = np.zeros(Time-delta)
    for n in range(0, Time-delta):
        E,D = erg_break(np.reshape(dataset, Time+2), omega, n, Time)
        D_re[n] = D.real
        D_im[n] = D.imag
        D_list.append(D)
    return D_re #, D_im
    


D1 = array_D_ergodicity(datasetCTRW1D1, omega, n, Time)
D2 = array_D_ergodicity(datasetCTRW1D2, omega, n, Time)

plt.plot(D1)
plt.plot(D2)
#plt.plot(D_im)
#xlabel('window size')


# # Ergodicity test for 2D
# 
# We define it for L1 or L2 measures. 
# 

# In[46]:


import math
import cmath

def erg_break_2D(X2D, omega, n, Time):
    '''
    X2D - vector containing positions of the analyzed trajectory 
    for 2D: X = X2D[2:N], Y = X2D[N+1, 2*N] 
    
    E - the real part of the mixing estimator as a function of n %
    F - the real part of the ergodicity estimator as a function of n
    
    n is parameter for 
    for each n we get one number
    '''
    #if (nargin < 2):
    #omega = 2 # Default value for omega 
    #Time = np.size(X)-1 #Trajectory points are enumerated as X(0), ..., X(N) 
    
    X_ar = X2D[0,2:Time+2]
    Y_ar = X2D[0,Time + 2: 2* Time +2]
    #X = X/np.std(np.diff(X)); # Normalization by the empirical standard deviation of displacements 
    print('array of X' , np.shape(X_ar))
    
    X = np.reshape(X_ar, Time)
    Y = np.reshape(Y_ar, Time)
    print(np.shape(X))
    print(np.shape(Y))
    
    #initialisation of arrays E and D
    D = 0 # initial value # D=[]#np.zeros(N)
    E = 0 # initial value # E=[]#np.zeros(N)
    
    for k in range(0,Time-n):
        print(k)
        print(X[n+k+2])
        #D(n+1) = sum( exp( (1i)*omega*(X(n+1:end) - X(1:end-n)) ) )/(N-n+1);
        D = D+((cmath.exp( (1j)*omega*(np.sqrt((X[n+k+2] - X[n+2])**2 + (Y[n+k+2] - Y[n+2])**2) )  ) ))   #for n in np.arange(N)
        
        #D = D+((cmath.exp( (1j)*omega*(X[n+k] - X[n]) ) ))   #for n in np.arange(N)
        
        
    D= D* 1./(Time-n+1) #normalisation
    #E = D- abs(sum( cmath.exp((1j)*omega*X) ))^2/N/(N+1) + 1/N #difference of arrays D and E 
    #print(D)
    return E, D


# In[47]:



omega = 2 #n=5 # average time parameter
delta=5 # average window size
Time = 100 
n = 10 # parameter for ergodicity breaking test

AD = andi.andi_datasets()
datasetCTRW2D1 = AD.create_dataset(T = Time , N = 1, exponents = [0.7], models = [1], dimension = 2)
datasetCTRW2D2 = AD.create_dataset(T = Time , N = 1, exponents = [0.7], models = [1], dimension = 2)

print('model size', np.size(datasetCTRW2D1))

E,D = erg_break_2D(datasetCTRW2D1, omega, n, Time)


# In[ ]:





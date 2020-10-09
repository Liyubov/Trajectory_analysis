#!/usr/bin/env python
# coding: utf-8

# In[1]:


import andi
import numpy as np
from matplotlib import pyplot as plt
# import fbm module # !pip install fbm

AD = andi.andi_datasets()

# general common paramter for all trajectories
Time = 500

N_given = 10 #steps
#X1, Y1 = andi_dataset(N_given)

def plot_trajectory(X,Y):
    'plotting trajectories'
    plt.plot(X,Y)#dataset[0,2:12], dataset[0, 12:], '-')
    plt.scatter(X,Y)#dataset[0,2:12], dataset[0,12:], c=np.arange(10), s = 250)
    plt.colorbar().set_label('Time'); plt.xlabel('X'); plt.ylabel('Y');




AD = andi.andi_datasets()
AD.avail_models_name

datasetATTM = AD.create_dataset(T = Time , N = 1, exponents = [0.2], models = [0], dimension = 2)
#print(np.round(dataset[0], 2))
#print(type(X1),np.shape(Y1))
print(np.shape(datasetATTM))
print(datasetATTM)
print('trajectory loaded')
plot_trajectory(datasetATTM[0,2:Time+2],datasetATTM[0,Time+2:]) # 2D trajectory with first X coordinates, then Y coordinates 



# # Test of alpha calculation 
# We use the module **powerlaw** for calculation of exponent. 
# 
# 
# #Calculating best minimal value for power law fit 
# 
#     fit = powerlaw.Fit(data) 
# 
#     fit.power_law.alpha 
# 
#     #e.g.2.273
#     fit.power_law.sigma 
#     #0.167
#     fit.distribution_compare(’power_law’, ’exponential’) 
#     #(12.755, 0.152)
# 
# https://arxiv.org/abs/1305.0215
# 
# # Temporal distribution calculation 
# 

# In[3]:


import powerlaw
import numpy as np
from scipy.optimize import curve_fit


def Time_distribution(x_data, y_data):
    '''
    This function calculates the distribution of times between consequent jumps
    '''
    # 1. calculate distribution between points
    step_dist = [np.sqrt((x[:,:-1]-x[:,1:])**2 + (y[:,:-1]-y[:,1:])**2) for x,y in zip(x_data, y_data)]

    # 2. if distance between points is 0 then RW does not move 
    #print(step_dist)
    
    #print(np.nonzero(step_dist))
    nnz_steps = np.nonzero(step_dist)
    
    # 3. then time distribution is difference between each two non-zero arrays
    time_dist = np.diff(nnz_steps)[1]
    
    print('shape of temporal distribution' , time_dist.shape)
    return time_dist 





def Dist_distribution(x_data, y_data):
    '''
    This function calculates the distribution of distance travelled per unit of time and retuns an histogram of this distribution. Note that we fixed the range and number of bins for simplicity, but this might need to be 
    '''
    step_dist = [np.sqrt((x[:,:-1]-x[:,1:])**2 + (y[:,:-1]-y[:,1:])**2) for x,y in zip(x_data, y_data)]
    max_value = [np.max(i) for i in step_dist]
    distributions = [np.histogram(sd,range=[0,ix],bins=20)[0] for sd,ix in zip(step_dist,max_value)]
    l_dist = [distributions[i].shape[0] for i in np.arange(len(distributions))]
    return distributions, l_dist

lags = range(2,100)

def hurst_exponen_data(p):
    '''
    given series array of x, y inside p(t), where t is time 
    p(t) is format of zip(list) of arrays from X and Y
    calculate hurst exponent
    
    main idea of hurst exponent is that it can show how persisent behaviour is.
    e.g. if the Hurst value is more than 0.5 then it would indicate a persistent time series
    '''    
    variancetau = []; 
    tau = []
    lags = range(2,100)

    for lag in lags: 
        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)
        #print(variancetau) # nan problem 
        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = np.subtract(p[lag:], p[:-lag])
        print(np.var(pp))
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = np.polyfit(np.log10(tau),np.log10(variancetau),1)
    print(m)
    hurst = m[0] / 2

    return hurst

##################################################
# calculate hurst exponent using built-in function 
import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc, random_walk

# Use random_walk() function or generate a random walk series manually:
# series = random_walk(99999, cumprod=True)
np.random.seed(42)
random_changes = 1. + np.random.randn(99999) / 1000.
series = np.cumprod(random_changes)  # create a random walk from random changes

print(series.shape)

# Evaluate Hurst equation
H, c, data = compute_Hc(series, kind='price', simplified=True)

# Plot
f, ax = plt.subplots()
ax.plot(data[0], c*data[0]**H, color="deepskyblue")
ax.scatter(data[0], data[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

print("H={:.4f}, c={:.4f}".format(H,c))





# In[29]:



def alpha_calculation(x, y, cat):

    ''' 
    For ATTM alpha = sigma * 1./gamma, but we use the simplest version for calculation of alpha, 
    For CTRW alpha is calculated simplest way, using distribution of distances
    For FBM we use special function https://pypi.org/project/fbm/ hurst exponent 
    since FBM has two regimes: one where the noise is persistent and positively correlated (1 < α < 2) 
    and where the is noise is antipersistent and negatively correlated (0 < α < 1)
    
    For LW alpha = 2 if (sigma >0)and(sigma<1) and fitted exponent otherwise
    For SBM alpha is calculated using powerlaw function
    '''

    if cat == 0: # ATTM
        
        print('WARNING: for ATTM type of trajectory we detect changes in diffusion coefficient. Then we estimate relationship between the transition and diffusion coefficient from Leweinstein article')
        # msd = (x-x[:,0])**2 + (y-y[:,0])**2
        
        # Difdist is distribution of diffusion coefficients for trajectory
        #delta = 5
        #Difdist = Convex_hull_regime(x_data,y_data, delta)
        #gamma  =  powerlaw.Fit(Difdist.reshape(-1)).power_law.alpha
        #psi = Dist_distribution(x,y)
        #sigma = powerlaw.Fit(psi.reshape(-1)).power_law.alpha
        #alpha = sigma * 1./gamma #powerlaw.Fit(msd.reshape(-1)).power_law.alpha
        
        psi = Dist_distribution(x,y)
        # we need to use list as an object, and not the array: np.array(Dist_distribution(x,y)[0])[0]
        psi_distribution = np.array(Dist_distribution(x,y)[0])[0]
        #print('size of distribution', np.array(Dist_distribution(x,y)[0])[0].shape)
        alpha1 = powerlaw.Fit(psi_distribution.reshape(-1)).power_law.alpha - 1 

        
    if cat == 1: # CTRW
        # estimate waiting time steps distribution psi(t)
        #msd = (x-x[:,0])**2 + (y-y[:,0])**2
        #powerlaw.Fit(msd.reshape(-1)).power_law.alpha
        psi = Dist_distribution(x,y)
        psi_distribution = np.array(Dist_distribution(x,y)[0])[0]
        alpha1 = powerlaw.Fit(psi_distribution.reshape(-1)).power_law.alpha - 1 # alpha = sigma -1 

        fit = powerlaw.Fit(psi_distribution.reshape(-1)) 
        sigma = fit.power_law.sigma #0.167
        print('sigma', sigma)
        print('fitting comparison', fit.distribution_compare("power_law", "exponential") )
        
    if cat == 2: # FBM
        print('WARNING: For this trajectory estimated as fractional Brownian motion trajectory short, the probability of correct estimate the exponent correctly from tMSD is low. ')
        listxy = list(zip(x,y))
        #hurst_exp = hurst_exponen_data(listxy)        # Evaluate Hurst equation
        #hurst_exp, c, data = compute_Hc(listxy, kind='price', simplified=True) # works for 1D trajectory
        psi = Dist_distribution(x,y)
        psi_distribution = np.array(Dist_distribution(x,y)[0])[0]

        #msd = (x-x[:,0])**2 + (y-y[:,0])**2
        alpha1 = powerlaw.Fit(psi_distribution.reshape(-1)).power_law.alpha
        fit = powerlaw.Fit(psi_distribution.reshape(-1)) 
        sigma = fit.power_law.sigma #0.167
        print('sigma', sigma)
        print('fitting comparison',   fit.distribution_compare("power_law", "exponential") )
        
        if (alpha1 == 1):
            print('Brownian motion')
        
    if cat == 3: # LW
        #msd = (x-x[:,0])**2 + (y-y[:,0])**2
        Time_dist = np.array(Time_distribution(x, y))
        print(Time_dist)
        sigma = powerlaw.Fit(Time_dist.reshape(-1)).alpha
        print('sigma', sigma)
        
        psi = Dist_distribution(x,y)
        psi_distribution = np.array(Dist_distribution(x,y)[0])[0]
        if (sigma >0)and(sigma<1): 
            print('0<sigma<1')
            alpha1 = 2
        if (sigma>1)and(sigma<2):
            print('0<sigma<1')
        # default alpha value when sigma non estimated
            alpha1 = powerlaw.Fit(psi_distribution.reshape(-1)).power_law.alpha
        print('sigma', sigma)
        alpha1 = 0
        
    if cat == 4: # SBM
        #msd = (x-x[:,0])**2 + (y-y[:,0])**2
        psi = Dist_distribution(x,y)
        psi_distribution = np.array(Dist_distribution(x,y)[0])[0]

        alpha1 = powerlaw.Fit(psi_distribution.reshape(-1)).power_law.alpha
        
    return alpha1 


# In[22]:


# test functions above
import andi
import numpy as np
from matplotlib import pyplot as plt
from andi import diffusion_models as DF
# import fbm module # !pip install fbm




###################################################################
#generat trajectories generation with paramter for all trajectories
Time = 500
N_given = 10 #steps
#X1, Y1 = andi_dataset(N_given)

AD = andi.andi_datasets()
print(AD.avail_models_name)
datasetATTM = AD.create_dataset(T = Time , N = 1, exponents = [0.2], models = [0], dimension = 2)

# We create one ATTM and one FBM trajectory with alpha = 0.2
attm = DF.twoD().attm(T = 1000, alpha = 0.2)
fbm = DF.twoD().fbm(T = 1000, alpha = 0.2)
print(attm.shape)

####################################################################
x = [datasetATTM[0,2:Time+2].reshape(1,-1)]
y = [datasetATTM[0,Time+2:].reshape(1,-1)]
Time_dist = Time_distribution(x,y)
print(Time_dist)

# testing on data from attm ANDI data
x_attm = [attm[0:Time].reshape(1,-1)]
y_attm = [attm[Time+1:].reshape(1,-1)]
#Time_distribution(x_attm,y_attm)

# To test if alpha given and estimated are the same 


# In[30]:


# Testing alpha function estimation

# parameters of trajectory 
exp_test = 0.5
print('for category 0 ATTM, ATTM only allows for anomalous exponents <= 1, exponent =', exp_test)
cat = 0
AD = andi.andi_datasets()
print(AD.avail_models_name)
# generate trajectory
Time = 500
datasetATTM = AD.create_dataset(T = Time , N = 1, exponents = [exp_test], models = [cat], dimension = 2)
x = [datasetATTM[0,2:Time+2].reshape(1,-1)]
y = [datasetATTM[0,Time+2:].reshape(1,-1)]
# test and calculate alpha
alpha = alpha_calculation(x, y, cat)
print('alpha', alpha)


exp_test = 0.6
Time =1000
print('for category 1 CTRW, Continuous random walks with anomalous exponents <= 1, exponent=', exp_test)
cat = 1
# generate trajectory
dataset = AD.create_dataset(T = Time , N = 1, exponents = [exp_test], models = [cat], dimension = 2)
x = [dataset[0,2:Time+2].reshape(1,-1)]
y = [dataset[0,Time+2:].reshape(1,-1)]
# test and calculate alpha
alpha = alpha_calculation(x, y, cat)
print('alpha', alpha)


exp_test_FBM = 1.5 
print('for category 2 FBM, exponent should be more than 1, exponent = ', exp_test_FBM)
cat = 2
# generate trajectory
dataset = AD.create_dataset(T = Time , N = 1, exponents = [exp_test_FBM], models = [cat], dimension = 2)
x = [dataset[0,2:Time+2].reshape(1,-1)]
y = [dataset[0,Time+2:].reshape(1,-1)]
# test and calculate alpha
alpha = alpha_calculation(x, y, cat)
print('alpha', alpha)

print('for category 3, LW')
cat = 3
# generate trajectory
dataset = AD.create_dataset(T = Time , N = 1, exponents = [1.2], models = [cat], dimension = 2)
x = [dataset[0,2:Time+2].reshape(1,-1)]
y = [dataset[0,Time+2:].reshape(1,-1)]
# test and calculate alpha
alpha = alpha_calculation(x, y, cat)
print('alpha', alpha)

print('for category 4, SBM')
cat = 4
# generate trajectory
dataset = AD.create_dataset(T = Time , N = 1, exponents = [1.2], models = [cat], dimension = 2)
x = [dataset[0,2:Time+2].reshape(1,-1)]
y = [dataset[0,Time+2:].reshape(1,-1)]
# test and calculate alpha
alpha = alpha_calculation(x, y, cat)
print('alpha', alpha)


# # tMSD method from ANDI tutorial 
# 
# One way to extract the anomalous exponent is by fitting the tMSD:
# $$
# \mbox{tMSD}(\Delta) = \frac{1}{T-\Delta} \sum_{i=1}^{T-\Delta}(x(t_i + \Delta)-x(t_i))^2,
# $$
# where $\Delta$ is defined as the time lag and $T$ is length of the trajectory.
# 
# **Problem**: tMSD works very well for ergodic processes, but fails horribly for non-ergodic, for which we usually have that 𝑡𝑀𝑆𝐷∼Δ. Nevertheless, let's use it to fit the exponent of the 1D training dataset:

# In[21]:


import numpy as np
import matplotlib.pyplot as plt

def TMSD(traj, t_lags):
    ttt = np.zeros_like(t_lags, dtype= float)
    for idx, t in enumerate(t_lags): 
        for p in range(len(traj)-t):
            ttt[idx] += (traj[p]-traj[p+t])**2            
        ttt[idx] /= len(traj)-t    
    return ttt


# In[24]:


from andi import diffusion_models as DF
import powerlaw

# We create one ATTM and one FBM trajectory with alpha = 0.2
Time = 100
attm = DF.oneD().attm(T = Time, alpha = 0.2)
fbm = DF.oneD().fbm(T = Time, alpha = 0.2)
ctrw =  DF.oneD().ctrw(T = Time, alpha = 0.2)

print('attm data', attm.shape)


# In[25]:



# We calculate their tMSD
t_lags_num = 200
t_lags = np.arange(2, t_lags_num)
attm_tmsd = TMSD(attm, t_lags = t_lags)
fbm_tmsd = TMSD(fbm, t_lags = t_lags)
ctrw_tmsd = TMSD(ctrw, t_lags = t_lags)


#print('fit of attm', attm_tmsd)
#fit = powerlaw.Fit(attm_tmsd) 
#print('alpha attm', fit.power_law.alpha )
#fit.power_law.sigma


#fit = powerlaw.Fit(fbm_tmsd) 
#print('alpha fbm', fit.power_law.alpha )
#fit.power_law.sigma


#fit = powerlaw.Fit(ctrw_tmsd) 
#print('alpha ctrw', fit.power_law.alpha )
#fit.power_law.sigma


# In[6]:


fig, ax = plt.subplots(1,2, figsize = (10, 4))

ax[0].loglog(t_lags, fbm_tmsd, '-o', lw = 1)
ax[0].loglog(t_lags, t_lags**0.2/(t_lags[0]**0.2)*fbm_tmsd[0], ls = '--')
ax[0].loglog(t_lags, t_lags/(t_lags[0])*fbm_tmsd[0], ls = '--')
ax[0].set_title(r'FBM $\rightarrow$ Ergodic process')

ax[1].loglog(t_lags, attm_tmsd, '-o', lw = 1,label = 'tMSD')
ax[1].loglog(t_lags, t_lags**0.2/(t_lags[0]**0.2)*attm_tmsd[0], ls = '--', label = r'$\sim \Delta^{0.2}$')
ax[1].loglog(t_lags, t_lags/(t_lags[0])*attm_tmsd[0], ls = '--', label = r'$\sim \Delta$')
ax[1].set_title(r'ATTM $\rightarrow$ Non-ergodic process')
ax[1].legend(fontsize = 16)

plt.setp(ax, xlabel = r'$\Delta$', ylabel = 'tMSD');
fig.tight_layout()


# # Fitting 
# 
# We need to mkae sure that 

# In[52]:


# fitting log log plots 

t_lags = np.arange(2,10)
predictions = []
t_lags_num = 20
t_lags = np.arange(2, t_lags_num) # np.arange(1, t_lags_num) 
attm_tmsd = TMSD(attm, t_lags = t_lags)
ctrw_tmsd =  TMSD(ctrw, t_lags = t_lags)

print('tmsd should not contain nans!!!',  attm_tmsd)
print('shape of files', t_lags.shape)

print('polyfit', np.polyfit(np.log(t_lags),np.log(ctrw_tmsd),1)[0])
#for traj in attm: # loop through elements of attm X-coordinates
#    #tmsd = TMSD(traj, t_lags)
#    predictions.append(np.polyfit(np.log(t_lags), np.log(tmsd),1)[0])
    
#print('MAE = '+str(np.round(np.mean(np.abs(np.array(predictions)-Y1[0])), 4)))

print('log log', np.log(t_lags),np.log(attm_tmsd))
plt.plot(np.log(t_lags),np.log(attm_tmsd),'*') 
plt.show()


# In[30]:



# Fake Data 
x = range(1,101)
y = 5 * np.log(x) + np.random.rand(len(x))

# Fit 
coefficients = np.polyfit(np.log(x),y,1) # Use log(x) as the input to polyfit.
fit = np.poly1d(coefficients) 

plt.plot(x,y,"o",label="data")
plt.plot(x,fit(np.log(x)),"--", label="fit")
plt.legend()
plt.show()






#!/usr/bin/env python
'''
This contains example of how to train a neural network to predict the posterior marginal distribution of of resistivities, as described in the paper:

Use of machine learning to estimate statistics of the posterior distribution in probabilistic inverse problems - an application to airborne EM data.
T. M. Hansen,C. C. Finlay
Journal of Geophysical Research: Solid Earth, 2022.
First published: 20 October 2022 https://doi.org/10.1029/2022JB024703

'''


# Estimate statistical properties of the posterior distribution using a neural network.
#%% Imports
import numpy as np

from ip_and_ml import ml_regression
#%% m: restivity 
useM='M1'

# FULL TEST
usePrior_arr  = ['A','B','C']
tfp_dist_arr = ['Normal','MixtureNormal2','MixtureNormal3','GeneralizedNormal']
N_use_arr = np.array([1000,5000,10000,50000,100000,500000,1000000,5000000]) #  Max size of training data set (multiple runs)
nhidden_arr = [8]
nunits=40
useRef_arr = [0]
force_train = False
nepochs=2000
useLearningSchedule=True;learning_rate=0.01

act='selu'
act='relu'
act='elu'

#useLearningSchedule=False;learning_rate=0.001

# SMALL TEST
doSmallTest=1
if (doSmallTest==1):
    #%% MULTI M1

    usePrior_arr  = ['C']
    tfp_dist_arr = ['Normal']
    N_use_arr = np.array([5000000]) #  Max size of training data set (multiple runs)
    N_use_arr = np.array([25000])
    nhidden_arr = [8]
    ls_arr=[0] # Usew learning schedule
    useHTX=1
    useData_arr = [2]
    useHTX_data_arr = [1]
    force_train = False
    nunits_arr=[40]
    act_arr=['relu']
    useRef=0

# for useBatchNormalization in [True,False]:
for useBatchNormalization in [True]:
    for useData in useData_arr:
        for useHTX_data in useHTX_data_arr:
            for act in act_arr:
                for useLearningSchedule in ls_arr:
                    for usePrior in usePrior_arr:
                        for N_use in N_use_arr:
                            for nhidden in nhidden_arr:
                                for nunits in nunits_arr:
                                    for tfp_dist in tfp_dist_arr:
                                        ml_regression(useM=useM,
                                                    usePrior=usePrior, 
                                                    N_use=N_use, 
                                                    nunits=nunits,
                                                    nhidden=nhidden,
                                                    nepochs=nepochs,
                                                    tfp_dist=tfp_dist, 
                                                    force_train=force_train,
                                                    useRef=useRef,
                                                    act=act,
                                                    useData=useData,useHTX=useHTX,useHTX_data=useHTX_data,
                                                    useBatchNormalization=useBatchNormalization,
                                                    learning_rate=learning_rate,useLearningSchedule=useLearningSchedule)    
#%%
#import sys
#sys.exit("Stopping ,,,,,,")        
    
    
#%% n2: Thickness of layers with resistivity>225 ohm-m
useHTX=1
useHTX_data = 1
useData = 2
force_train = False
usePrior_arr  = ['C']

N_use_arr = np.array([1000,10000,100000,1000000,5000000]) #  Max size of training data set (multiple runs)
useM='M5'
nhidden_arr = [4]
act='selu'
tfp_dist_arr = ['Normal','MixtureNormal2','MixtureNormal3','GeneralizedNormal']


# SMALL TEST
doSmallTest=1
if (doSmallTest==1):
    tfp_dist_arr = ['Normal']
    N_use_arr = np.array([10000]) #  Max size of training data set (multiple runs)

for act in act_arr:
    for useLearningSchedule in [True]:
        for usePrior in usePrior_arr:
            for N_use in N_use_arr:
                for nhidden in nhidden_arr:
                    for tfp_dist in tfp_dist_arr:
                        for useRef in useRef_arr:
                            ml_regression(useM=useM, 
                                        usePrior=usePrior, 
                                        N_use=N_use, 
                                        nunits=nunits,
                                        nhidden=nhidden,
                                        nepochs=nepochs,
                                        tfp_dist=tfp_dist, 
                                        force_train=force_train,
                                        useRef=useRef,
                                        act=act,
                                        useHTX_data=useHTX_data,
                                        learning_rate=learning_rate,useLearningSchedule=useLearningSchedule)        
    
    

    
#%%
import sys
sys.exit("Stopping ,,,,,,")
#%%

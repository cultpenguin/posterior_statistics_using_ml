#!/usr/bin/env python
# coding: utf-8

# # Estimating posterior probability of feature (P(LayerInterface))
'''
Below is an example of how to construct a neural network that allow estimating the posterior probability
of categorical model parameters.

* Case A ('M3')
P(LayerInterface|\sigma)


* Case B ('M2'/'M4')
P(Lithology|\sigma)


'''

# In[1]:
import os
import numpy as np
import matplotlib.pyplot as plt

from ip_and_ml import ml_classification
#%%


useM = 'M3' # P(Layer)
useM = 'M4' # P(Lithology) % Label '0','1', and '2'
useM = 'M2' # P(Lithology) % One-Hot representation (probability of class)

usePrior_arr  = ['A','B','C']

N_use_arr = [1000,10000,100000,1000000,5000000] #  Max size of training data set (multiple runs)
#N_use_arr = 1000000000 # Max size of training data set

# Network setup
nhidden_arr = [4]  # Number of hidden layers
nunits=40  # Number of units in hidden layers
pdropout=0;# Percentage of dropout at each hidden layer

act='elu'
#act='relu'

                
# Network training 
learning_rate=1e-3
useLearningSchedule = True
patience = 50   # Early stopping when validation loss does not drop after 'patience' iterations
force_train=0 # Do not train if model model has allready been trained (stored in h5 file)
#force_train=1 # Always train the model
nepochs=2000
verbose=1

#%% Specific examples of settings

useM_arr=['M3','M4']
N_use_arr = np.array([1000,5000,10000,50000,100000,500000,1000000,2000000,5000000]) #  Max size of training data set (multiple runs)
N_use_arr = np.array([1000,5000,10000,50000,100000,500000,1000000,2000000]) #  Max size of training data set (multiple runs)
N_use_arr = np.array([1000,5000,10000,50000,100000,500000,1000000]) #  Max size of training data set (multiple runs)
N_use_arr = np.array([1000,10000,100000,1000000]) #  Max size of training data set (multiple runs)
N_use_arr = np.array([1000,10000,100000,1000000,5000000]) #  Max size of training data set (multiple runs)
#N_use_arr = np.array([5000000]) #  Max size of training data set (multiple runs)
nunits=40
#nhidden_arr = [2,4]
nhidden_arr = [4]
#usePrior_arr  = ['A','B','C']
pdropout=0;# Percentage of dropout at each hidden layer
useLearningSchedule =True
force_train=False

        
useHTX=1
useHTX_data_arr = [1]
useData_arr = [2]
act_arr=['relu','selu','elu']

force_train=False

doSmallTest=1
if (doSmallTest==1):
    usePrior_arr  = ['C']
    N_use_arr = np.array([1000])

#%% Run all
#for useBatchNormalization in [True,False]:
for useBatchNormalization in [True]:
    for useData in useData_arr:
        for useHTX_data in useHTX_data_arr:
            for act in act_arr:
                for useM in useM_arr:
                    if (useM=='M2')|(useM=='M4'):
                        usePrior_arr  = ['A','B','C']
                        #act='elu'
                        #nhidden_arr=[4]  # Number of hidden layers
                    else:
                        usePrior_arr  = ['B','C']
                        #act='elu'
                        #nhidden_arr=[2,3]  # Number of hidden layers
                    for usePrior in usePrior_arr:
                        for nhidden in nhidden_arr:
                            for N_use in N_use_arr:
                                ml_classification(usePrior=usePrior,useM=useM,nhidden=nhidden,
                                                nunits=nunits,N_use=N_use,nepochs=nepochs,
                                                force_train=force_train,useRef=0,
                                                verbose=verbose,pdropout=pdropout,
                                                learning_rate=learning_rate,useLearningSchedule=useLearningSchedule,
                                                act=act,
                                                useData=useData,useHTX=useHTX,useHTX_data=useHTX_data,
                                                useBatchNormalization=useBatchNormalization)
                                #ml_classification(usePrior=usePrior,useM=useM,nhidden=nhidden,
                                #                nunits=nunits,N_use=N_use,nepochs=nepochs,
                                #                force_train=0,useRef=7,
                                #                verbose=verbose,pdropout=pdropout,
                                #                learning_rate=learning_rate,useLearningSchedule=useLearningSchedule,
                                #                act=act)

import sys
sys.exit("Stopping ,,,,,,")


#%% SMALL TEST
#ml_classification(usePrior='C',useM='M4',nhidden=1,nunits=40,N_use=50001,force_train=True,learning_rate=0.01,patience=500)
#ml_classification(usePrior='C',useM='M4',nhidden=1,nunits=40,N_use=50001,force_train=True,learning_rate=0.01,patience=500,pdropout=0.1)
ml_classification(usePrior='C',useM='M3',nhidden=4,nunits=40,N_use=200001,
                  useLearningSchedule = True,
                  force_train=True,
                  learning_rate=0.01,
                  patience=50)
                  


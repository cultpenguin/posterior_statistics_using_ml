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
#
# First a simple test, using a training set of size N_use=100000, 
# using MLP-NN with 3 hidden layers with 40 units.
# rho_A, rho_B, and rho_C are used, and the posterior mean and variance from sigma(m) is computed
#
# Tensorboard 
#   tensorboard --logdir=logs_A_M1 
#   tensorboard --logdir=logs_B_M1 
#   tensorboard --logdir=logs_C_M1 
useSimpleTest=True
if (useSimpleTest):
    useM='M1' # This refer to choosing the resistivity model paramerers, p=[m]; See paper
    N_use = 100000

    ml_regression(useM=useM,
                usePrior='A', 
                N_use=N_use, 
                nunits=40,
                nhidden=3)           
    ml_regression(useM=useM,
                usePrior='B', 
                N_use=N_use, 
                nunits=40,
                nhidden=3)           
    ml_regression(useM=useM,
                usePrior='C', 
                N_use=N_use, 
                nunits=40,
                nhidden=3)           
            
    import sys
    sys.exit("Stopping ,,,,,,")


#%% The example below trains a number for different networks, 
# using specific chhoices of prior A, B, and C, 
# and using different target distributions

useM='M1' # This refer to choosing the resistivity model paramerers, p=[m]; See paper
usePrior_arr  = ['A','B','C']
tfp_dist_arr = ['Normal','MixtureNormal2','MixtureNormal3','GeneralizedNormal']
N_use_arr = np.array([1000,5000,10000,50000,100000,500000,1000000,5000000]) #  Max size of training data set (multiple runs)
nhidden_arr = [8]
nunits_arr=[40]
useRef_arr = [0]
act_arr=['selu','relu','elu']
ls_arr=[0,1]; # Use learning schedule

force_train = False
learning_rate=0.01
nepochs=2000
useBatchNormalization = True
# for useBatchNormalization in [True,False]:
for usePrior in usePrior_arr:
    for tfp_dist in tfp_dist_arr:
        for usePrior in usePrior_arr:
            for N_use in N_use_arr:
                for act in act_arr:
                    for useLearningSchedule in ls_arr:
                        for nhidden in nhidden_arr:
                            for nunits in nunits_arr:
                                ml_regression(useM=useM,
                                            usePrior=usePrior, 
                                            N_use=N_use, 
                                            nunits=nunits,
                                            nhidden=nhidden,
                                            nepochs=nepochs,
                                            tfp_dist=tfp_dist, 
                                            force_train=force_train,
                                            act=act,
                                            useBatchNormalization=useBatchNormalization,
                                            learning_rate=learning_rate,useLearningSchedule=useLearningSchedule)    
    
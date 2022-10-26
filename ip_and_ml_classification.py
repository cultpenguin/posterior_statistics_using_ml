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
avoidGPU=0
if avoidGPU==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import time
import h5py
from urllib.request import urlretrieve as urlretrieve
import matplotlib.pyplot as plt


#%%

# Define learning-rate scheduler
def scheduler(epoch,lr):
    epoch_start = 10
    if epoch < epoch_start:
        return lr
    else:
        return lr * tf.math.exp(-0.005)


#%%

def ml_classification(usePrior='A',useM='M4',nhidden=4,nunits=40,
                      N_use=1000,nepochs=2000,pdropout=0,learning_rate=0.001,patience=50,
                      useLearningSchedule=False,force_train=0,verbose=1,useRef=0,
                      act='elu',
                      useBatchNormalization=True,
                      useHTX=1,useHTX_data=1,useData=2):
    txt=''
    N = 1000000
    N = 2000000
    N = 5000000

    if (usePrior=='A'):
        file_training = '1D_P23_NO500_451_ABC%d_0000_D%d_HTX%d_%d.h5' % (N,useData,useHTX,useHTX_data)
        file_sampling = '1D_P23_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5' % (N,useData,useHTX,useHTX_data)
        if useRef==7:
            file_sampling = '1D_P23_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref7.h5' % (N);txt='ref7' # !!
    elif (usePrior=='B'):
        file_training = '1D_P51_NO500_451_ABC%d_0000_D%d_HTX%d_%d.h5' % (N,useData,useHTX,useHTX_data)
        file_sampling = '1D_P51_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5' % (N,useData,useHTX,useHTX_data)        
        if useRef==7:
            file_sampling = '1D_P51_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref7.h5' % (N);txt='ref7'
    elif (usePrior=='C'):
        file_training = '1D_P22_NO500_451_ABC%d_0000_D%d_HTX%d_%d.h5' % (N,useData,useHTX,useHTX_data)
        file_sampling = '1D_P22_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5' % (N,useData,useHTX,useHTX_data)        
        if useRef==7:
            file_sampling = '1D_P22_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref7.h5' % (N);txt='ref7'
        
    # Download if h5 files does not exist    
    if not os.path.exists(file_training):
      print("File (training) '%s' does not exist" % (file_training))
      print("Downloading '%s'" % (file_training))
      urlretrieve('https://zenodo.org/record/7253825/files/1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1.h5?download=1','1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1.h5')
      
    if not os.path.exists(file_sampling):
      print("File '%s' does not exist" % (file_sampling))
      print("Downloading %s" % (file_sampling))
      urlretrieve('https://zenodo.org/record/7253825/files/1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5?download=1','1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5')
            
            
    f_training = h5py.File(file_training, 'r')
    f_sampling = h5py.File(file_sampling, 'r')
    
    N = f_training[useM].shape[0]
    if N_use > N:
        print('Requesting larger training data set than available (%d>%d)' % (N_use,N))
        return 1

    N_use = np.min([N,N_use])
    print('Loading %d training data (models) from %s' % (N_use,file_training))        
    
    
    M = f_training[useM][0:N_use,:]
    if useM=='M4':    
        # Reshape to match categories
        nm=M.shape[1]
        print(' Reshape, nm= %d' % nm) 
        Ms=M.shape
        M = np.reshape(M,(Ms[0],125,3), order='F')
            
    print('Loading %d training data (data) from %s' % (N_use,file_training))        
    D = f_training['D2'][0:N_use,:]
    
    nd=D.shape[1]
    nm=M.shape[1]
    print(' nm= %d' % nm) 
    print(' nd= %d' % nd)
    
    # split data into training and validation data
    m_train, m_test, d_train, d_test = train_test_split(M,D, test_size=0.33, random_state=42)
    print(' M train shape =', (m_train.shape))
    print(' D train shape =', (d_train.shape))
    print(' M test shape =', (m_test.shape))
    print(' D test shape =', (d_test.shape))
    
    # plot some data
    Nshow=40;
    plt.figure(figsize=(20,12))
    plt.subplot(1,2,1)
    if useM=='M3':
        plt.imshow(M[0:Nshow,:].T)
    elif (useM=='M4')|(useM=='M2'):
        plt.imshow(M[0:Nshow:,:])
        plt.xlabel('iz')
        plt.ylabel('imodel')
        
    plt.title('model #1')
    plt.title('data #1')
    plt.plot(D[0:Nshow,:].T)
    plt.imshow(D[0:Nshow,:].T)
    plt.xlabel('imodel')
    plt.ylabel('id')
    plt.suptitle('Using prior %s, %s' % (usePrior,useM))
            
    #% Setup NN
    #act='relu'
    #act='elu'
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nunits, input_dim=nd, kernel_initializer='normal', activation=act))
    if (useBatchNormalization):
        model.add(tf.keras.layers.BatchNormalization())
    for i in range(nhidden):
        model.add(tf.keras.layers.Dense(nunits, activation=act))
    
    if useM=='M3': 
        # P layer
        model.add(tf.keras.layers.Dense(nm, activation='sigmoid'))
    if (useM=='M4')|(useM=='M2'):
        # P class - One Hot Enconding
        model.add(tf.keras.layers.Dense(3*nm, activation='relu'))
        model.add(tf.keras.layers.Reshape((nm,3)))
        model.add(tf.keras.layers.Activation(tf.keras.activations.softmax))
    
    
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    if useM=='M3':
        model.compile(optimizer=optim,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'])
    elif useM=='M4':
        model.compile(optimizer=optim,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
    elif useM=='M2':
        model.compile(optimizer=optim,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        
    # Print model info
    model.summary()


    #% Train the network
    batch_size = np.ceil(np.power(2.1,4+np.log10(N_use))).astype(int)        
    
    # Network complexity follows training data sets size
    #nunits = np.ceil(20*(np.log10(N_use)-2)).astype(int)
           
    fout='Prior%s_%s_N%d_class_bs%d_ep%d_nh%d_nu%d_do%02.0f_ls%d_%s_BN%d_HTX%d_%d' % (usePrior,useM,N_use,batch_size,nepochs,nhidden,nunits,pdropout*100,useLearningSchedule,act,useBatchNormalization,useHTX,useHTX_data)
    #fout='Prior%s_%s_N%d_meanstd_bs%d_ep%d_nh%d_nu%d_do%02.0f_ls%d_%s' % (usePrior,useM,N_use,batch_size,nepochs,nhidden,nunits,pdropout*100,useLearningSchedule,act)
    
    model_h5 = fout + '.h5'
    
    if (os.path.isfile(model_h5))&(force_train==0):
        # Neural net has allready been trained
        print('Neural network has allready been trained - loading weights from %s' % (model_h5))
        model.load_weights(model_h5)
    
    else:
        # Neural net has not been trained
        print('=======================')
        print('Training neural network')
        
        # Tensorboard
        logdir = os.path.join("logs_"+usePrior+'_'+useM , fout )
        print("Logs in %s" % (logdir) )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)
        
        # Early stopping
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=3)
        
        # Learningrate shcedule
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        if useLearningSchedule:
            print("Using LearningSchedule")
            callbacks = [tensorboard_callback,earlystopping_callback,lr_callback]
        else:
            callbacks = [tensorboard_callback,earlystopping_callback]
        
        
        print("Training: %s" % (fout))
        t0=time.time()    
        out = model.fit(d_train, m_train, 
                epochs=nepochs, 
                batch_size=batch_size,
                verbose=verbose,
                validation_data=(d_test,m_test),
                callbacks=callbacks)
    
        t1=time.time()
        t_train=t1-t0
        print("Elapsed time for training = %3.1fs" % (t_train))
    
        # Save h5 model
        print('%s: Save model weights' % (fout))
        #tf.keras.models.save_model(model, model_h5, save_format='h5')    
        model.save_weights(model_h5)
        hf = h5py.File(model_h5, 'a')
        hf.create_dataset('loss', data=out.history['loss'])
        hf.create_dataset('val_loss', data=out.history['val_loss'])
        hf.create_dataset('t_train', data=t_train)
        hf.close()
    
    
        # Plot loss
        plt.figure(figsize=(20,12))
        plt.plot(out.history['loss'][30:-1], label='Train')
        plt.plot(out.history['val_loss'][30:-1], label='Validation')
        plt.xlabel('Iteration #')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
    
        
    
    
    #% PREDICT
    
    # load observed data
    print(list(f_sampling.keys()))
    
    P_lith1_sampling = f_sampling['M_lith1'][:].T
    P_lith2_sampling = f_sampling['M_lith2'][:].T
    P_lith3_sampling = f_sampling['M_lith3'][:].T
    
    D_obs= f_sampling['D_obs'][:]
    n_est = D_obs.shape[0]
    
    t0=time.time()
    POST = model(D_obs)
    t1=time.time()
    t_pred=t1-t0
    print("t_pred = %gms'" % (t_pred*1000))    
    print('prediction time: %5.1fms / %d models' % (1000*t_pred, n_est))
    print('prediction time: %4.0f modsels/s' % (1/(t_pred/n_est)))
        
        
    
    fout2='%s%s_D%d' % (fout,txt,useData)
    fout_h5 = '%s_est.h5' % (fout2)
    
    if useM=='M3':
    
        #t0=time.time()
        #POST = model(D_obs)
        P_layer=POST.numpy().T
        #t1=time.time()
        #t_pred=t1-t0
        
        print("Writing to %s" % (fout_h5) )
        hf = h5py.File(fout_h5, 'w')
        hf.create_dataset('P_layer', data=P_layer)
        hf.create_dataset('t_pred', data=t_pred)
        hf.close()
        
        # load mean and sdt obtained using sampling
        P_layer_sampling = f_sampling['M_prob_layer'][:].T
        
        #%
        plt.figure(figsize=(32,12))
        plt.subplot(211)
        #plt.imshow(P_layer, vmin=0, vmax=0.2, cmap=plt.cm.get_cmap('jet').reversed())
        plt.imshow(P_layer, vmin=0, vmax=0.3, cmap=plt.cm.get_cmap('gray').reversed())
        plt.colorbar()
        plt.title('P(Layer) ML  t = %4.2f ms' % (1000*t_pred))
        
        plt.subplot(212)
        #plt.imshow(P_layer_sampling, vmin=0, vmax=0.2, cmap=plt.cm.get_cmap('jet').reversed())
        plt.imshow(P_layer_sampling, vmin=0, vmax=0.3, cmap=plt.cm.get_cmap('gray').reversed())
        plt.title('P(Layer) sampling')
        plt.colorbar()
        plt.suptitle(fout2+' - P(layer)')
        plt.savefig(fout2 + '_Player')    
        #plt.show()
        #%
    
    elif (useM=='M2')|(useM=='M4'):
            
        P_lith1 =  POST[:,:,0].numpy().T
        P_lith2 =  POST[:,:,1].numpy().T
        P_lith3 =  POST[:,:,2].numpy().T
        
        print("Writing to %s" % (fout_h5) )
        hf = h5py.File(fout_h5, 'w')
        hf.create_dataset('P_lith1', data=P_lith1)
        hf.create_dataset('P_lith2', data=P_lith2)
        hf.create_dataset('P_lith3', data=P_lith3)
        hf.create_dataset('t_pred', data=t_pred)
        hf.close()
                        
        plt.figure(figsize=(20,12))
        
        plt.subplot(321)
        plt.imshow(P_lith1_sampling, cmap='hot_r', vmin=0.0, vmax=1)
        plt.colorbar()
        plt.title('P(L1) FROM INVERSION')
        
        plt.subplot(323)
        plt.imshow(P_lith2_sampling, cmap='hot_r', vmin=0.0, vmax=1)
        plt.colorbar()
        plt.title('P(L2) FROM INVERSION')
        
        plt.subplot(325)
        plt.imshow(P_lith3_sampling, cmap='hot_r', vmin=0.0, vmax=1)
        plt.colorbar()
        plt.title('P(L3) FROM INVERSION')
        
        plt.subplot(322)
        plt.imshow(P_lith1, cmap='hot_r', vmin=0.0, vmax=1)
        plt.colorbar()
        plt.title('P(L1) FROM ML')
        
        plt.subplot(324)
        plt.imshow(P_lith2, cmap='hot_r', vmin=0.0, vmax=1)
        plt.colorbar()
        plt.title('P(L2) FROM ML')
        
        plt.subplot(326)
        plt.imshow(P_lith3, cmap='hot_r', vmin=0.0, vmax=1)
        plt.colorbar()
        plt.title('P(L3) FROM ML')
        
        plt.suptitle(fout2+' - P(lithology)')
        
        plt.savefig(fout2 + '_Plith')      
        #plt.show()


  
#%%
#import sys
#sys.exit("Stopping ,,,,,,")


#%% SETTINGS


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
                  


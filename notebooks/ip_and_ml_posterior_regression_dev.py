#!/usr/bin/env python
# coding: utf-8

# Estimate statistical properties of the posterior distribution using a neural network.

# In[1]:

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.model_selection import train_test_split

import numpy as np
import os
import time
import h5py
from urllib.request import urlretrieve as urlretrieve

import matplotlib.pyplot as plt


# ## (Down)load data

# In[2]:

# Define negative log-likelihood (used as loss function)
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

# Define learning-rate scheduler
def scheduler(epoch,lr):
    epoch_start = 10
    if epoch < epoch_start:
        return lr
    else:
        return lr * tf.math.exp(-0.005)

usePrior = 'B'
txt=''
if (usePrior=='A'):
    file_training = '1D_P23_NO500_451_ABC1000000_0000.h5'
    file_sampling = '1D_P23_NO500_451_ABC1000000_0000_ME0_aT1_CN1.h5'
    #file_sampling = '1D_P23_NO500_451_ABC5000000_0000_ME0_aT1_CN1_ref7.h5';txt='ref7' # !!
    #file_training = '1D_P23_NO500_451_ABC500000_0000.h5'
    #file_sampling = '1D_P23_NO500_451_ABC500000_0000_ME0_aT1_CN1.h5'
elif (usePrior=='B'):
    file_training = '1D_P51_NO500_451_ABC1000000_0000.h5'
    file_sampling = '1D_P51_NO500_451_ABC100000_0000_ME0_aT1_CN1.h5'
    #file_sampling = '1D_P51_NO500_451_ABC1000000_0000_ME0_aT1_CN1_ref7.h5';txt='ref7'
    #file_training = '1D_P51_NO500_451_ABC500000_0000.h5'
    #file_sampling = '1D_P51_NO500_451_ABC500000_0000_ME0_aT1_CN1.h5'
elif (usePrior=='C'):
    file_training = '1D_P22_NO500_451_ABC1000000_0000.h5'
    file_sampling = '1D_P22_NO500_451_ABC1000000_0000_ME0_aT1_CN1.h5'
    file_sampling = '1D_P22_NO500_451_ABC1000000_0000_ME0_aT1_CN1_ref7.h5';txt='ref7'
    #file_training = '1D_P22_NO500_451_ABC500000_0000.h5'
    #file_sampling = '1D_P22_NO500_451_ABC500000_0000_ME0_aT1_CN1.h5'

# Download if h5 files does not exist    
if not os.path.exists(file_training):
  print("Downloading %s" % (file_training))
  urlretrieve("https://github.com/cultpenguin/probabilistic-inverse-problems_and_ml/raw/main/%s" % (file_training),file_training)
if not os.path.exists(file_sampling):
  print("Downloading %s" % (file_sampling))
  urlretrieve("https://github.com/cultpenguin/probabilistic-inverse-problems_and_ml/raw/main/%s" % (file_sampling),file_sampling)    
    
    
f_training = h5py.File(file_training, 'r')
f_sampling = h5py.File(file_sampling, 'r')

#M = f_training['M1'][:]
#D = f_training['D2'][:]
   
#tfp_dist = 'Normal'
#tfp_dist = 'LogNormal'
#tfp_dist = 'GeneralizedNormal'
#tfp_dist = 'Uniform' # does not work
#tfp_dist = 'GeneralizedUniform'
#tfp_dist = 'MixtureNormalC1' # Similar to 'Normal'
#tfp_dist = 'MixtureNormalC2' # Gaussian Mixture (with 2 classes), constant uniform class probability
#tfp_dist = 'MixtureNormalC3' # Gaussian Mixture (with 3 classes), constant uniform class probability
#tfp_dist = 'MixtureNormal1' # Gaussian Mixture (with 1 class)
#tfp_dist = 'MixtureNormal2' # Gaussian Mixture (with 2 classes)
#tfp_dist = 'MixtureNormal3' # Gaussian Mixture (with 3 classes)

useM = 'M1' # Resistivity
#useM = 'M5' # Thickness
    
#for N_use in [1000, 10000, 100000, 1000000, 3000000, 5000000]:
#for N_use in [1000000,500000,100000, 50000,10000,1000,5000000]:
for N_use in [100013]:
    #for tfp_dist in ['GeneralizedNormal']:
    for tfp_dist in ['MixtureNormal3']:
    #or tfp_dist in ['Normal','LogNormal','GeneralizedNormal','MixtureNormal2','MixtureNormal3']:
    #for tfp_dist in ['Normal','LogNormal','GeneralizedNormal','MixtureNormal1','MixtureNormal2','MixtureNormal3','MixtureNormalC2','MixtureNormalC3']:
        
        N = f_training[useM].shape[0]
        N_use = np.min([N,N_use])
        
        
        M = f_training[useM][0:N_use,:]
        D = f_training['D2'][0:N_use,:]
        
        nd=D.shape[1]
        nm=M.shape[1]
        print('nm= %d' % nm)
        print('nd= %d' % nd)
        plt.subplot(1,2,1)
        plt.plot(M[0:10,:].T)
        plt.title('model #1')
        plt.subplot(1,2,2)
        plt.title('data #1')
        plt.plot(D[0:10,:].T)
        plt.suptitle('Using prior %s' % (usePrior))
        
        # split data into training and validation data
        m_train, m_test, d_train, d_test = train_test_split(M,D, test_size=0.33, random_state=42)
        print(m_train.shape)
        print(d_train.shape)
        print(m_test.shape)
        print(d_test.shape)
        
        
        ## SETUP NETWORK
        ## Network settings
        
        if 'M1' in useM:
            nhidden=8 #
            nunits=80
            pdropout=0;
        else:
            nhidden=3 #
            nunits=20
            pdropout=0;
                
        
        d_floor = 0.1#1e-3 
        d_scale=1
            
        ## Training settings
        learning_rate = 1e-3
        patience = 25   # was 50
        batch_size = 64;
        n_epochs = 2000;
        if N_use >= 10000:
            batch_size = 128;n_epochs = 2000
        if N_use >= 50000:
            batch_size = 256;n_epochs = 2000
        if N_use >= 100000:
            batch_size = 512;n_epochs = 2000
        if N_use >= 800000:
            batch_size = 2*1024;n_epochs = 2000
        
        fout='Prior%s_%s_N%d_meanstd_bs%d_ep%d_nh%d_nu%d_do%d_%s' % (usePrior,useM,N_use,batch_size,n_epochs,nhidden,nunits,pdropout*100,tfp_dist)
        
        # The network
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nunits, input_dim=nd, kernel_initializer='normal', activation='relu'))
        for i in range(nhidden):
            model.add(tf.keras.layers.Dense(nunits, activation='relu'))
            if (pdropout>0):
                model.add(tf.keras.layers.Dropout(pdropout))
        
        if tfp_dist.lower()=="normal":
            # 1D normal distribution
            model.add(tf.keras.layers.Dense(nm+nm))
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[..., :nm],
                                                                                       scale=d_floor + tf.math.softplus(d_scale * t[..., nm:])))),
        elif tfp_dist.lower()=="lognormal":
            # 1D log-normal distribution
            model.add(tf.keras.layers.Dense(nm+nm))
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.logNormal(loc=t[..., :nm],
                                                                                       scale=d_floor + tf.math.softplus(d_scale * t[..., nm:])))),
        elif tfp_dist.lower()=="generalizednormal":
            # 1D generalized-normal distribution
            model.add(tf.keras.layers.Dense(nm+nm+nm))
            p_min = 1.5
            p_max = 6
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.GeneralizedNormal(loc=t[..., :nm],
                                                                                                  scale=d_floor + tf.math.softplus(d_scale * t[..., (nm):(2*nm)]),
                                                                                                  power=p_min + (p_max-p_min)*tf.math.sigmoid(d_scale * t[..., (2*nm):]),
                                                                                                 ))),
        elif tfp_dist.lower()=="generalizeduniform":
            # 1D generalized-normal/uniform distribution. More sttable than using  tfp.distributions.Uniform
            # For some reason, estimation the pdf of the traiuned model only works for CPU?
            model.add(tf.keras.layers.Dense(nm+nm))
            p = 6
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.GeneralizedNormal(loc=t[..., :nm],
                                                                                                  scale=d_floor + tf.math.softplus(d_scale * t[..., (nm):(2*nm)]),
                                                                                                  power=p*np.ones(nm, dtype='float32'),
                                                                                                 ))),
        elif tfp_dist.lower()=="uniform":
            # 1D Uniform distribution
            # it is typically difficult to train a network using uniform as output distribtion.
            model.add(tf.keras.layers.Dense(nm+nm))
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Uniform(low=t[..., :nm],  
                                                                                       high=t[..., :nm]+t[..., nm:])))
        elif "mixturenormalc" in tfp_dist.lower():
            # 1D Gaussian mixture model, with constant (uniform) class probability

            # Get number of classes
            try:
                nc=np.fromstring(tfp_dist[-1], dtype=int, sep=" ");
                nc=nc[0]
            except:
                nc = 2
            
            model.add(tf.keras.layers.Dense(2*nc*nm))
            model.add(tf.keras.layers.Reshape((nm,2*nc)))
            mix = np.ones(nc, dtype='float32')/nc
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.MixtureSameFamily(
                                                        mixture_distribution=tfd.Categorical(
                                                            probs=mix),
                                                        components_distribution=tfd.Normal(
                                                          loc = t[..., 0:nc],       # One for each component.
                                                          scale= 0.01 + tf.math.softplus(t[..., nc:]))
                                                        )
                                                    ))    
        elif "mixturenormal" in tfp_dist.lower():
            # 1D Gaussian mixture model with up to 3 classes
            # Get number of classes

            try:
                nc=np.fromstring(tfp_dist[-1], dtype=int, sep=" ");
                nc=nc[0]
            except:
                nc = 2
        
            model.add(tf.keras.layers.Dense(3*nc*nm))
            model.add(tf.keras.layers.Reshape((nm,3*nc)))
            model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.MixtureSameFamily(
                                                        mixture_distribution=tfd.Categorical(
                                                            logits=t[..., 0:nc]),
                                                        components_distribution=tfd.Normal(
                                                          loc = t[..., nc:(2*nc)],       # One for each component.
                                                          scale= 0.01 + tf.math.softplus(t[..., (2*nc):]))
                                                        )
                                                    ))
        else:
            print("Distribution '%s'is not known" % (tfp_dist))
                      
        #print("tfp distrbution: %s" % tfp_dist)
                
        
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optim, loss=negloglik);
        
        # Print model info
        #model.summary()
        
        ## Network train
        model_h5 = fout + '.h5'
        force_train=0;
        
        if (os.path.isfile(model_h5))&(force_train==0):
            # Neural net has allready been trained
            print('Neural network has allready been trained - loading weights from %s' % (model_h5))
            model.load_weights(model_h5)
        
        else:
            # Neural net has not been trained
            print('Training neural network')
            
            # Tensorboard
            logdir = os.path.join("logs_"+usePrior+'_'+useM , fout )
            print("Logs in %s" % (logdir) )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)
            
            # Early stopping
            earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=3)
            
            # Learningrate shcedule
            callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
            
            t0=time.time()
        
            out = model.fit(d_train, m_train, 
                    epochs=n_epochs, 
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(d_test,m_test),
                    callbacks=[tensorboard_callback,earlystopping_callback,callback_lr],
                    #callbacks=[tensorboard_callback,earlystopping_callback],
                    )
        
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
            plt.figure(1)
            plt.semilogy(out.history['loss'], label='Train')
            plt.semilogy(out.history['val_loss'], label='Validation')
            plt.xlabel('Iteration #')
            plt.ylabel('Loss')
            plt.grid()
            plt.legend()
            plt.savefig(fout + '_history')   
            
            
        #%%## Predict posterior mean and standard deviation
        
        # load observed data
        D_obs= f_sampling['D_obs'][:]
        n_est = D_obs.shape[0]
        
        t0=time.time()
        POST = model(D_obs)
        M_mean = POST.mean().numpy().T
        M_std = POST.stddev().numpy().T
        t1=time.time()
        t_pred = t1-t0;
        print('prediction time: %5.1fms / %d models' % (1000*t_pred, n_est))
        print('prediction time: %4.0f models/s' % (1/(t_pred/n_est)))
        print('prediction time: %8.4f ms/model' % (1000*(t_pred/n_est)))
        
        fout2='%s%s' % (fout,txt)
        fout_h5 = '%s_est.h5' % (fout2)
        print("Writing to %s" % (fout_h5) )
        hf = h5py.File(fout_h5, 'w')
        hf.create_dataset('M_mean', data=M_mean)
        hf.create_dataset('M_std', data=M_std)
        
        # ## Plot some figures

        if nm==1:
            plt.figure(figsize=(10,8))
            plt.plot(M_mean.flatten())
            plt.plot(M_mean.flatten()-2*M_std.flatten())
            plt.plot(M_mean.flatten()+2*M_std.flatten())
            plt.ylabel('Thickness')
            plt.savefig(fout2 + '_thick')    
            
        else:                
            # load mean and sdt obtained using sampling
            M_mean_sampling = f_sampling['M_est'][:].T
            M_std_sampling = f_sampling['M_std'][:].T
            
            # MEAN
            plt.figure(figsize=(30,22))
            plt.subplot(211)
            A = M_std/1.4;
            indices = A > 1
            A[indices] = 1
            A=1-A
            print(np.max(A))
            #plt.imshow(M_mean, vmin=1, vmax=2.5, cmap='jet', alpha=A)
            plt.imshow(M_mean, vmin=1, vmax=2.5, cmap='jet')
            plt.colorbar()
            plt.title('M mean ML  t = %4.2f ms' % (1000*t_pred))
            
            plt.subplot(212)
            A = M_std_sampling/1.4;
            indices = A > 1
            A[indices] = 1
            A=1-A
            #plt.imshow(M_mean_sampling, vmin=1, vmax=2.5, cmap='jet', alpha=A)
            plt.imshow(M_mean_sampling, vmin=1, vmax=2.5, cmap='jet')
            plt.title('M mean sampling')
            plt.colorbar()
            plt.savefig(fout2 + '_mean')    
            
            # STD
            plt.figure(figsize=(30,22))
            plt.subplot(211)
            plt.imshow(M_std, vmin=0, vmax=1, cmap='gray_r')
            plt.title('M std ML  t = %4.2f ms' % (1000*t_pred))
            plt.colorbar()
            plt.subplot(212)
            plt.imshow(M_std_sampling, vmin=0, vmax=1, cmap='gray_r')
            plt.title('M std sampling')
            plt.colorbar()
            plt.savefig(fout2 + '_std')    
            
            
            plt.figure(figsize=(30,22))
            plt.clf()
            # MEAN 
            if "normal" in tfp_dist.lower():
                plt.subplot(311)
                plt.imshow(M_mean, vmin=1, vmax=2.5, cmap='jet')
                plt.colorbar()
                plt.title('Mean,  t = %4.2f ms' % (1000*t_pred))
            
            # std 
            if "normal" in tfp_dist.lower():
                plt.subplot(312)
                plt.imshow(M_std, vmin=0, vmax=1, cmap='gray_r')
                plt.colorbar()
                plt.title('Standard deviation')    
                
            # POWER
            if tfp_dist.lower()=="generalizednormal":
                M_power = POST.power.numpy().T
                hf.create_dataset('M_power', data=M_power)
                plt.subplot(313)
                plt.imshow(M_power, vmin=1.0, vmax=6.0, cmap='jet')
                plt.colorbar()
                plt.title('Power of Generalized Gaussian')
                
            plt.suptitle("Distribution: %s" % tfp_dist)
            plt.savefig(fout2 + "_mul")   
            
        exportPDF=1;
        if exportPDF==1:
            #% PDF extract for all points along line!!
            print("Exporting full PDF")
            n_test = 101
            n_soundings = D_obs.shape[0]
            pdf_est = np.zeros((n_soundings,nm,n_test), dtype='float32')
            if nm==1:
                # Thickness
                m_test = np.linspace(0.0, 100, num=n_test, dtype='float32')
                t0=time.time()    
                P1D = model(D_obs)
                i=-1
                for m in m_test:
                    i=i+1
                    p = np.exp(P1D.log_prob(m).numpy()).T
                    pdf_est[:,:,i]=p.T
                t1=time.time()    
            
                plt.figure(figsize=(12,8))
                plt.clf()
                plt.imshow(np.flipud(pdf_est[:,0,:].T), extent=[1,n_soundings,m_test[0],m_test[-1]])
                plt.plot(M_mean.T,'k')
                plt.savefig(fout2 + "_pdf")   
            
            else:
                # Resistivity
                m_test = np.linspace(0.0, 4, num=n_test, dtype='float32')
          
                P1D = model(D_obs)
                i=-1
                for m in m_test:
                    i=i+1
                    p = np.exp(P1D.log_prob(m).numpy())
                    pdf_est[:,:,i]=p
                t1=time.time()    
            
                i_data = 0
                plt.figure(figsize=(12,8))
                plt.clf()
                plt.subplot(121)
                plt.pcolor(m_test,np.arange(nm),pdf_est[i_data,:,:], shading='nearest', cmap='gray_r');
                plt.colorbar()
                plt.gca().invert_yaxis()
                plt.subplot(122)
                try:
                    i_m = 45;
                    plt.plot(m_test,pdf_est[i_data,i_m,:])
                    plt.title('1D pdf at [i_d,i_m]=[%d,%d]' % (i_data,i_m))
                    plt.suptitle('i_d = %d' % (i_data))
                    plt.savefig(fout2 + "_pdf") 
                except:
                    print('Trouble')
                
    
    
            hf.create_dataset('m_test', data=m_test)
            hf.create_dataset('pdf_est', data=pdf_est)
            #hf.create_dataset('i_data', data=i_data)
        
                
        hf.close()
        #plt.show()
    
        
        
    
# In[ ]:
import sys
sys.exit("Stopping ,,,,,,")

#%%SMALL TEST - THIS WORKS GREAT FOR A SINGLE MODEL PARAMETER


negloglik = lambda y, rv_y: -rv_y.log_prob(y)

nm=1
nd=12
nhidden=8;nunits=80

d_floor = 0.1#1e-3 
d_scale=1

nc=1

use=1;

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(nunits, input_dim=nd, kernel_initializer='normal', activation='relu'))
for i in range(nhidden):
    model.add(tf.keras.layers.Dense(nunits, activation='relu'))
if use == 0:
    model.add(tf.keras.layers.Dense(2*nc*nm))
    mix = np.ones(nc, dtype='float32')/nc
    model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.MixtureSameFamily(
                                                mixture_distribution=tfd.Categorical(
                                                    probs=mix),
                                                components_distribution=tfd.Normal(
                                                  loc = t[..., 0:nc],       # One for each component.
                                                  scale= 0.01 + tf.math.softplus(t[..., nc:]))
                                                )
                                            ))

                                            
        
else:
    model.add(tf.keras.layers.Dense(nm+nm))
    model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[..., :nm],scale=d_floor + tf.math.softplus(d_scale * t[..., nm:])))),


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=negloglik);


#N=20000;
iz=11;
m_train, m_test, d_train, d_test = train_test_split(M,D, test_size=0.33, random_state=42)
nm=1
m_train = m_train[:N,iz]
m_test = m_test[:N,iz]
d_train = d_train[:N,:]
d_test = d_test[:N,:]
#d_train = np.random.randn(N,nd)
#m_train = np.random.randn(N,nm)
out = model.fit(d_train, m_train, 
          epochs=40, 
          batch_size=124,
          verbose=2,
          validation_data=(d_test,m_test),
          )

P = model(D_obs)
P_std = P.stddev().numpy()
P_m0 =  P.mean().numpy()
S=P.sample(1111).numpy()

plt.subplot(131)
plt.plot(out.history['loss'])
plt.subplot(132)
plt.hist(S[:,31])
plt.subplot(133)
plt.plot(M_mean[iz,:])
plt.plot(P_m0)



#%% logNormal TEST
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

# CREATE SMAPLE OF LOGNORMAL
dis=tfp.distributions.LogNormal(loc=np.log(np.exp(3.8)),scale=.1)

N=100000
D=dis.sample(N).numpy()
print(np.mean(D))
plt.hist(D,101)

# CRETAE TRAINING DATA SET
# ESTIMATE PARAMETERS OF TRAINING DATA D SET


#%%



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
np.random.seed(42)
dataframe = pd.read_csv('Apple_Data_300.csv').ix[0:800,:]
dataframe.head()



plt.plot(range(0,dataframe.shape[0]),dataframe.iloc[:,1])

x1=np.array(dataframe.iloc[:,1]+np.random.randn(dataframe.shape[0])).astype(np.float32).reshape(-1,1)

y=np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)

tfd = tfp.distributions



#%%SMALL TEST - SAME AS ABOVE BUT FOR TWO MODEL PARAMETERS


negloglik = lambda y, rv_y: -rv_y.log_prob(y)


nm=35
nd=12
nhidden=8;nunits=80

d_floor = 0.1#1e-3 
d_scale=1

nc=2

use=0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(nunits, input_dim=nd, kernel_initializer='normal', activation='relu'))

for i in range(nhidden):
    model.add(tf.keras.layers.Dense(nunits, activation='relu'))

if use == 0:
    model.add(tf.keras.layers.Dense(2*nc*nm))
    model.add(tf.keras.layers.Reshape((nm,2*nc)))
    mix = np.ones(nc, dtype='float32')/nc
    model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.MixtureSameFamily(
                                                mixture_distribution=tfd.Categorical(
                                                    probs=mix),
                                                components_distribution=tfd.Normal(
                                                  loc = t[..., 0:nc],       # One for each component.
                                                  scale= 0.01 + tf.math.softplus(t[..., nc:]))
                                                )
                                            ))
        
else:
    model.add(tf.keras.layers.Dense(nm+nm))
    model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[..., :nm],scale=d_floor + tf.math.softplus(d_scale * t[..., nm:])))),


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=negloglik);


#N=20000;
iz=0+np.arange(nm)
m_train, m_test, d_train, d_test = train_test_split(M,D, test_size=0.33, random_state=42)

m_train = m_train[:N,iz]
m_test = m_test[:N,iz]
d_train = d_train[:N,:]
d_test = d_test[:N,:]
#d_train = np.random.randn(N,nd)
#m_train = np.random.randn(N,nm)
out = model.fit(d_train, m_train, 
          epochs=40, 
          batch_size=124,
          verbose=2,
          validation_data=(d_test,m_test),
          )

P = model(D_obs)
P_std = P.stddev().numpy()
P_m0 =  P.mean().numpy()
S=P.sample(1111).numpy()

plt.subplot(131)
plt.plot(out.history['loss'])
plt.subplot(132)
plt.hist(S[:,31])
plt.subplot(133)
plt.plot(M_mean[iz,:].T)
plt.plot(P_m0)


#%% 2 'batches' of a mixture dist with 2 gausian
# https://stackoverflow.com/questions/58266812/probability-of-batched-mixture-distribution-in-tensorflow-probability
tfd = tfp.distributions
mix = np.float32(np.array([[0.6, 0.4],[0.3, 0.7]] ))
mix = [[0.6, 0.4],[0.3, 0.7]] 
#mix = [0.6, 0.4, 0.3, 0.7] 
bimix_gauss = tfd.Mixture(
  cat=tfd.Categorical(probs=mix),
  components=[
    tfd.Normal(loc=[-1.0, -2.0], scale=[0.1, 0.1]),
    tfd.Normal(loc=[+1.0, +2.0], scale=[0.5, 0.5]),
])

print(bimix_gauss.sample())
print(bimix_gauss.prob(0.0))


#%% 2 'batches' of a mixture dist with 3 gausian
# https://stackoverflow.com/questions/58266812/probability-of-batched-mixture-distribution-in-tensorflow-probability
tfd = tfp.distributions
mix = np.float32(np.array([[0.6, 0.2, 0.1],[0.2, 0.7, 0.1]] ))
mix = [[0.6, 0.3, 0.1],[0.4, 0.3, 0.3]] 
bimix_gauss = tfd.Mixture(
  cat=tfd.Categorical(probs=mix),
  components=[
    tfd.Normal(loc=[-3.0, -2.0], scale=[0.1, .3]),
    tfd.Normal(loc=[-1.0, +0.0], scale=[0.5, .6]),
    tfd.Normal(loc=[+1.2, +2.2], scale=[.1, .3]),
])

print(bimix_gauss.sample())
print(bimix_gauss.prob(0.0))

x=2*np.random.randn(10000,1)
plt.plot(x,bimix_gauss.prob(x).numpy(),'.')


#%%
#%%
#%%
tfd = tfp.distributions

### Create a mixture of two scalar Gaussians:

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.2, .8]),
    components_distribution=tfd.Normal(
      loc=[-1., 1],       # One for each component.
      scale=[0.1, 0.5]))  # And same here.

gm.mean()
# ==> 0.4

gm.variance()
# ==> 1.018
# Plot PDF.
x = np.linspace(-2., 3., int(1e4), dtype=np.float32)
import matplotlib.pyplot as plt
plt.plot(x, gm.prob(x));

### Create a mixture of two Bivariate Gaussians:

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.6, 0.4]),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-.5, .5],  # component 1
             [1, -1]],  # component 2
        scale_identity_multiplier=[.9,0.9]))

gm.mean()
# ==> array([ 0.4, -0.4], dtype=float32)

gm.covariance()
# ==> array([[ 1.119, -0.84],
#            [-0.84,  1.119]], dtype=float32)

# Plot PDF contours.
def meshgrid(x, y=x):
  [gx, gy] = np.meshgrid(x, y, indexing='ij')
  gx, gy = np.float32(gx), np.float32(gy)
  grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
  return grid.T.reshape(x.size, y.size, 2)
grid = meshgrid(np.linspace(-2, 2, 100, dtype=np.float32))
plt.contour(grid[..., 0], grid[..., 1], gm.prob(grid));

#%% Compare Normal to Mixture Gaussian
tfd = tfp.distributions
gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[1, .00011]),
    components_distribution=tfd.Normal(
      loc=[0.0, 1.0],       # One for each component.
      scale=[1, 1]))  # And same here.

g = tfd.Normal(loc=0, scale=1);


print('P(0)=%5.4f (MIXTURE NORMAL)' % (gm.log_prob(0).numpy() ))
print('P(0)=%5.4f (NORMAL)' % (g.log_prob(0).numpy() ))



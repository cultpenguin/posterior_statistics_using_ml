# ip_and_ml.py


import os
avoidGPU=1
if avoidGPU==1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    

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

    #%% FUNCTIONS

# Define negative log-likelihood (used as loss function)
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

# Define learning-rate scheduler
def scheduler(epoch,lr):
    epoch_start = 10
    if epoch < epoch_start:
        return lr
    else:
        return lr * tf.math.exp(-0.005)

#%%

def ml_regression(usePrior='A',useM='M4',nhidden=4,nunits=40,
                      N_use=1000,nepochs=2000,pdropout=0,learning_rate=0.001,patience=50,
                      useLearningSchedule=False,force_train=0,verbose=1,useRef=0,
                      tfp_dist='Normal',
                      act='relu',
                      useBatchNormalization=True,
                      useHTX=1,useHTX_data=1,useData=2):

    txt=''
    N = 250000 # CHANGE THIS TO THE MAX LOOKUP SIZE FOR LOCALIZED REJECTION!!!
    N = 5000000
    if (usePrior=='A'):
        file_training = '1D_P23_NO500_451_ABC%d_0000_D%d_HTX%d_%d.h5' % (N,useData,useHTX,useHTX_data)
        file_sampling = '1D_P23_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5' % (N,useData,useHTX,useHTX_data)
        if useRef==7:
            file_sampling = '1D_P23_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref7.h5' % (N);txt='ref7' # !!
        # Download if h5 files does not exist    
        if not os.path.exists(file_training):
            print("File (training) '%s' does not exist" % (file_training))
            print("Downloading '%s'" % (file_training))
            urlretrieve('https://zenodo.org/record/7254008/files/1D_P23_NO500_451_ABC5000000_0000_D2_HTX1_1.h5?download=1','1D_P23_NO500_451_ABC5000000_0000_D2_HTX1_1.h5')
        if not os.path.exists(file_sampling):
            print("File '%s' does not exist" % (file_sampling))
            print("Downloading %s" % (file_sampling))
            urlretrieve('https://zenodo.org/record/7254008/files/1D_P23_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5?download=1','1D_P23_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5')

    elif (usePrior=='B'):
        file_training = '1D_P51_NO500_451_ABC%d_0000_D%d_HTX%d_%d.h5' % (N,useData,useHTX,useHTX_data)
        file_sampling = '1D_P51_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5' % (N,useData,useHTX,useHTX_data)
        if useRef==7:
            file_sampling = '1D_P51_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref7.h5' % (N);txt='ref7'
        # Download if h5 files does not exist    
        if not os.path.exists(file_training):
            print("File (training) '%s' does not exist" % (file_training))
            print("Downloading '%s'" % (file_training))
            urlretrieve('https://zenodo.org/record/7254030/files/1D_P51_NO500_451_ABC5000000_0000_D2_HTX1_1.h5?download=1','1D_P51_NO500_451_ABC5000000_0000_D2_HTX1_1.h5')
        if not os.path.exists(file_sampling):
            print("File '%s' does not exist" % (file_sampling))
            print("Downloading %s" % (file_sampling))
            urlretrieve('https://zenodo.org/record/7254030/files/1D_P51_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5?download=1','1D_P51_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5')
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
    
    print("Reading training data from %s" % (file_training))    
    print("Reading Sampling data from %s" % (file_sampling))    
    
    N = f_training[useM].shape[0]
    if N_use > N:
        print('Requesting larger training data set than available (%d>%d)' % (N_use,N))
        return 1

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
    
    
    # Batch size
    ## Training settings
    batch_size = 64;
    if N_use >= 10000:
        batch_size = 128;
    if N_use >= 50000:
        batch_size = 256;
    if N_use >= 100000:
        batch_size = 512;
    if N_use >= 800000:
        batch_size = 2*1024;
    d_floor = 0.1#1e-3 
    d_scale=1
    #act='relu'
    #act='elu'
    
    fout='Prior%s_%s_N%d_meanstd_bs%d_ep%d_nh%d_nu%d_do%02.0f_ls%d_%s_%s_BN%d_HTX%d_%d' % (usePrior,useM,N_use,batch_size,nepochs,nhidden,nunits,pdropout*100,useLearningSchedule,tfp_dist,act,useBatchNormalization,useHTX,useHTX_data)
    print(fout)    
    
    # Network setup
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nunits, input_dim=nd, kernel_initializer='normal', activation=act))
    if (useBatchNormalization):
        model.add(tf.keras.layers.BatchNormalization())    
    for i in range(nhidden):
        model.add(tf.keras.layers.Dense(nunits, activation=act))
        if (pdropout>0):
            model.add(tf.keras.layers.Dropout(pdropout))
    
    if tfp_dist.lower()=="normal":
        # 1D normal distribution
        model.add(tf.keras.layers.Dense(nm+nm))
        model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[..., :nm],
                                                                                   scale=d_floor + tf.math.softplus(d_scale * t[..., nm:])))),
    elif tfp_dist.lower()=="lognormal":
        # 1D log-normal distribution
        model.add(tf.keras.layers.Dense(nm, activation='relu'))
        model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.LogNormal(loc=tf.math.log(0.001 +tf.math.softplus(t[..., :nm])),
                                                                                   scale=0.1,
                                                                                   ))),
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
    model.summary()
    
    ## Network train
    model_h5 = fout + '.h5'
    
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
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        if useLearningSchedule:
            callbacks = [tensorboard_callback,earlystopping_callback,lr_callback]
        else:
            callbacks = [tensorboard_callback,earlystopping_callback]
            
            
        t0=time.time()
    
        out = model.fit(d_train, m_train, 
                epochs=nepochs, 
                batch_size=batch_size,
                verbose=1,
                validation_data=(d_test,m_test),
                callbacks=callbacks,
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
        
    ## Predict posterior mean and standard deviation
    
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
    
    fout2='%s%s_D%d' % (fout,txt,useData)
    fout_h5 = '%s_est.h5' % (fout2)
    print("Writing to %s" % (fout_h5) )
    hf = h5py.File(fout_h5, 'w')
    hf.create_dataset('M_mean', data=M_mean)
    hf.create_dataset('M_std', data=M_std)
    
    # ## Plot some figures
    if nm==1:
        T_mean_sampling = f_sampling['T_est'][:].T
        T_std_sampling = f_sampling['T_std'][:].T
        
        plt.figure(figsize=(10,8))
        plt.plot(M_mean.flatten(),'k-')
        plt.plot(M_mean.flatten()-2*M_std.flatten(),'k:')
        plt.plot(M_mean.flatten()+2*M_std.flatten(),'k:')
        plt.plot(T_mean_sampling.flatten(),'r-')
        plt.plot(T_mean_sampling.flatten()-2*T_std_sampling.flatten(),'r:')
        plt.plot(T_mean_sampling.flatten()+2*T_std_sampling.flatten(),'r:')
        plt.ylabel('Thickness')
        plt.savefig(fout2 + '_thick')    
        
    else:               
        #%
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
            
        plt.suptitle("Distribution: %s, N=%d" % (tfp_dist,N_use) )
        plt.savefig(fout2 + "_mul")   
        
    exportPDF=1;
    if exportPDF==1:
        #% PDF extract for all points along line!!
        print("Exporting full PDF")
        n_test = 151
        n_soundings = D_obs.shape[0]
        pdf_est = np.zeros((n_soundings,nm,n_test), dtype='float32')
        if nm==1:
            # Thickness
            m_test = np.linspace(0.0, 150, num=n_test, dtype='float32')
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
            #plt.imshow(np.flipud(pdf_est[:,0,:].T), extent=[1,n_soundings,m_test[0],m_test[-1]], cmap='gray_r', vmin=0, vmax = .2)
            plt.imshow(np.flipud(pdf_est[:,0,:].T), extent=[1,n_soundings,m_test[0],m_test[-1]], cmap='gray_r', vmin=0)
            plt.plot(M_mean.T,'b-')
            plt.plot(M_mean.T+2*M_std.T,'b:')
            plt.plot(M_mean.T-2*M_std.T,'b:')
            plt.plot(T_mean_sampling.flatten(),'r-')
            plt.savefig(fout2 + "_pdf")   
            plt.title(tfp_dist)
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
                hf.close()
            


        hf.create_dataset('m_test', data=m_test)
        hf.create_dataset('pdf_est', data=pdf_est)
        #hf.create_dataset('i_data', data=i_data)
    
            
    hf.close()

    
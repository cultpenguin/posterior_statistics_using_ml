if ~exist('i_prior');i_prior=1;end
if ~exist('useRef');useRef=0;end

clear all;close all
load morrill_model
x=model.X(1,:)/1000;
plot_mov=0;
cmap_rho=jet;
cmap_kl=flipud(hot);
cmap_kl=cmap_geosoft;
cmap_kl=cmap_linear(flipud([0 0 0; 1 0 0;  0 1 0 ; 1 1 1]));
cmap_std=([flipud(gray);hot]);

cax=log10([10 320]);
cax_ytick=log10([10 20 40 80 160 320]);

cax_std=[0 1];
cax_std_ytick=[0,.5,1];

pdf_types={'Normal','GeneralizedNormal','MixtureNormal2','MixtureNormal3'};

use_prior_types=[23,51,22];
prior_types={'A','B','C'};

%use_prior=23;prior_type='A';
%use_prior=51;prior_type='B';
%use_prior=22;prior_type='C';

N=1000000;

i_prior = 1;
use_prior = use_prior_types(i_prior);
prior_type = prior_types{i_prior};
%% GET REFERENCE SAMPLING RESULTS
N_sampling=2000000;

useRef=0;
%useRef=7;
if useRef>1
    reftxt=sprintf('ref%d',useRef);
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref%d.h5',use_prior,N_sampling,useRef);
else
    reftxt='';
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_ME0_aT1_CN1.h5',use_prior,N_sampling);
end
%h=h5info(h5_MCMC);h.Datasets.Name
MCMC_mean = h5read(h5_MCMC,'/M_est');
MCMC_std = h5read(h5_MCMC,'/M_std');


%% M1: Regression estimation
clear H5 ML_*
pdf_type=pdf_types{1};
act='relu';
ls=1;
for nh=[8]
    i=0;
    %i=i+1;H5{i}=sprintf('Prior%s_M1_N1000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);
    %i=i+1;H5{i}=sprintf('Prior%s_M1_N5000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);

    i=i+1;H5{i}=sprintf('Prior%s_M1_N10000_meanstd_bs128_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N50000_meanstd_bs256_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N100000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N500000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N1000000_meanstd_bs2048_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s',prior_type,nh,ls,pdf_type,reftxt,act);
    
    
    
    for i=1:length(H5);
        %h=h5info(h5_ML);h.Datasets.Name;
        ML_mean{i} = h5read([H5{i},'_est.h5'],'/M_mean')';
        ML_std{i} = h5read([H5{i},'_est.h5'],'/M_std')';
        ML_pdf_est{i} = h5read([H5{i},'_est.h5'],'/pdf_est');
        t_train(i) =  h5read([H5{i},'.h5'],'/t_train');
        loss{i} =  h5read([H5{i},'.h5'],'/loss');
        val_loss{i} =  h5read([H5{i},'.h5'],'/val_loss');
    end
    
    figure(30+nh);clf;set_paper('portrait')
    subplot(6,1,1);
    morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
    colormap(gca,cmap_rho)
    text(0.05, 0.9, sprintf('a) Sampling'),'Units','Normalized')
    for i=1:length(H5)
        subplot(6,1,i+1);
        morill_2d_plot(ML_mean{i},cax,cax_ytick,1);
        colormap(gca,cmap_rho)
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        text(0.05, 0.9, sprintf('%s) N=%s',96+i+1,Nstr),'Units','Normalized')
    end
    print_mul(sprintf('Regression_M1_compare_mean_%s_%s_nh%d_%s',reftxt,prior_type,nh,pdf_type))
    
        
    figure(40+nh);clf;set_paper('portrait')
    subplot(6,1,1);
    morill_2d_plot(MCMC_std,cax_std,cax_std_ytick,0);
    colormap(gca,cmap_std)
    text(0.05, 0.9, sprintf('a) Sampling'),'Units','Normalized')
    for i=1:length(H5)
        subplot(6,1,i+1);
        morill_2d_plot(ML_std{i},cax_std,cax_std_ytick,0);
        colormap(gca,cmap_std)
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        text(0.05, 0.9, sprintf('%s) N=%s',96+i+1,Nstr),'Units','Normalized')
    end
    print_mul(sprintf('Regression_M1_compare_std_%s_%s_nh%d_%s',reftxt,prior_type,nh,pdf_type))

    
end

%% MEAN OF PDF TYPES
figure(11);clf;
figure(12);clf;

figure(11);clf;set_paper('portrait')
subplot(6,1,1);
morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
colormap(gca,cmap_rho)
text(0.05, 0.9, sprintf('a) Sampling'),'Units','Normalized')

figure(12);clf;set_paper('portrait')
subplot(6,1,1);
morill_2d_plot(MCMC_std,cax_std,cax_std_ytick,0);
text(0.05, 0.9, sprintf('a) Sampling'),'Units','Normalized')
colormap(gca,cmap_std)


for i=1:length(pdf_types)
    pdf_type=pdf_types{i};
    h5_ML=sprintf('Prior%s_M1_N10000_meanstd_bs128_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s_est.h5',prior_type,nh,ls,pdf_type,reftxt,act);
    try
        h=h5info(h5_ML);h.Datasets.Name;
        ML_mean = h5read(h5_ML,'/M_mean')';
        ML_std = h5read(h5_ML,'/M_std')';
        ML_pdf_est = h5read(h5_ML,'/pdf_est');
        
        figure(11);
        subplot(6,1,i+1);
        morill_2d_plot(ML_mean,cax,cax_ytick,1);
        colormap(gca,cmap_rho)
        text(0.05, 0.9, sprintf('%s) %s',96+i+1,pdf_type),'Units','Normalized')
        
        figure(12);
        subplot(6,1,i+1);
        morill_2d_plot(ML_std,cax_std,cax_std_ytick,0,1:length(x),'Std');
        colormap(gca,cmap_std)
        text(0.05, 0.9, sprintf('%s) %s',96+i+1,pdf_type),'Units','Normalized')
        %title(sprintf('%s) Std, ML %s',96+i+1,pdf_type))
    catch
        disp(sprintf('Could not load %s',h5_ML))
    end
end
figure(11)
print_mul(sprintf('Prior%s%s_Mean_N%d',prior_type,reftxt,N))
figure(12)
print_mul(sprintf('Prior%s%s_Std_N%d',prior_type,reftxt,N))


%% Print Mean/STD as a function of training data set size

target='Normal';
useAlpha=1;
j=1;
prior_type_arr={'A','B','C'};
for j=1:length(prior_type_arr);
    prior_type=prior_type_arr{j};
    clear h5arr
    ih=0;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N1000_meanstd_bs64_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);     N_arr(ih)=1000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N10000_meanstd_bs128_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);   N_arr(ih)=10000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N10000_meanstd_bs128_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s_est.h5',prior_type,nh,ls,target,reftxt,act);     N_arr(ih)=10000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N50000_meanstd_bs128_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);   N_arr(ih)=50000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N100000_meanstd_bs512_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);  N_arr(ih)=100000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N500000_meanstd_bs512_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);  N_arr(ih)=500000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N1000000_meanstd_bs2048_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);N_arr(ih)=1000000;
    %ih=ih+1;h5arr{ih}=sprintf('Prior%s_N5000000_meanstd_bs2048_ep2000_nh8_nu80_do0_%s_est.h5',prior_type,target);N_arr(ih)=5000000;
    

    ih=ih+1;h5arr{ih}=sprintf('Prior%s_M1_N1000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s_est.h5',prior_type,nh,ls,target,reftxt,act);     N_arr(ih)=1000;
    ih=ih+1;h5arr{ih}=sprintf('Prior%s_M1_N10000_meanstd_bs128_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s_est.h5',prior_type,nh,ls,target,reftxt,act);     N_arr(ih)=10000;
    ih=ih+1;h5arr{ih}=sprintf('Prior%s_M1_N100000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s_est.h5',prior_type,nh,ls,target,reftxt,act);     N_arr(ih)=100000;
    ih=ih+1;h5arr{ih}=sprintf('Prior%s_M1_N1000000_meanstd_bs2048_ep2000_nh%d_nu40_do00_ls%d_%s%s_%s_est.h5',prior_type,nh,ls,target,reftxt,act);     N_arr(ih)=1000000;



    figure(30+j);clf;set_paper('portrait')
    i=0;
    for i=1:length(h5arr);
        h5_ML=h5arr{i};
        try
            h=h5info(h5_ML);h.Datasets.Name;
            ML_mean = h5read(h5_ML,'/M_mean')';
            ML_std = h5read(h5_ML,'/M_std')';
            ML_pdf_est = h5read(h5_ML,'/pdf_est');
            max_std = 1;min_std = 0.4;
            max_std = 0.7;min_std = 0.2;
            ML_A = ML_std- min_std;
            ML_A(ML_A<min_std)=0;
            ML_A(ML_A>(max_std-min_std))=(max_std-min_std);
            ML_A=1-ML_A./(max_std-min_std);
            subplot(6,1,i);
            morill_2d_plot(ML_mean,cax,cax_ytick,1);
            if useAlpha==1;alpha(ML_A);end
            colormap(gca,cmap_rho)
            text(0.05, 0.9, sprintf('%s) N=%d',96+i,N_arr(i)),'Units','Normalized')
            drawnow;
        end
        
    end
    print_mul(sprintf('Prior%s_%s_compare_N_A%d',prior_type,target,useAlpha))
end


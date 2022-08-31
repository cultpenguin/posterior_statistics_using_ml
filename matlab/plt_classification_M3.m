% plt_classification_M3 : Player
%clear all;
close all

% if ~exist('i_prior');i_prior=3;end
% if ~exist('useRef');useRef=0;end
% 
% if ~exist('useHTX');useHTX=1;end
% if ~exist('useHTX_data');useHTX_data=1;end
% if ~exist('useData');
%     useData=1; % 'Original' forward ref
%     useData=2; % Morrill data
% end
% 
plt_defs;

use_prior = use_prior_types(i_prior);
prior_type = prior_types{i_prior};
%% GET REFERENCE SAMPLING RESULTS
%if ~exist('N_sampling')
%    N_sampling=1000000;
%    %N_sampling=5000000;
%end
REF=[];

if useRef>1
    reftxt=sprintf('ref%d',useRef);
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref%d.h5',use_prior,N_sampling,useRef);
    
    try
        REF = load('Morill_data_ml_IP7.mat');
    end
        
else
    reftxt='';
    %h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_ME0_aT1_CN1.h5',use_prior,N_sampling);
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5',use_prior,N_sampling,useData,useHTX,useHTX_data);
end
%h=h5info(h5_MCMC);h.Datasets.Name
MCMC_est = h5read(h5_MCMC,'/M_est');
MCMC_std = h5read(h5_MCMC,'/M_std');
MCMC_Test = h5read(h5_MCMC,'/T_est');
MCMC_Tstd = h5read(h5_MCMC,'/T_std');
MCMC_Player = h5read(h5_MCMC,'/M_prob_layer');
MCMC_P1 = h5read(h5_MCMC,'/M_lith1');
MCMC_P2 = h5read(h5_MCMC,'/M_lith2');
MCMC_P3 = h5read(h5_MCMC,'/M_lith3');
try
    MCMC_Dobs = h5read(h5_MCMC,'/D_obs');
    MCMC_Dstd = h5read(h5_MCMC,'/D_std');
end

%% M1: Regression estimation
clear H5 ML_*

%for nh=[2,4,8]
for nh=[4]
%     i=0;
%     i=i+1;H5{i}=sprintf('Prior%s_M3_N1000_meanstd_bs181_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M3_N5000_meanstd_bs303_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M3_N10000_meanstd_bs379_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M3_N50000_meanstd_bs636_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M3_N100000_meanstd_bs795_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M3_N500000_meanstd_bs1335_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M3_N1000000_meanstd_bs1668_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
    
    i=0;
    i=i+1;H5{i}=sprintf('Prior%s_M3_N1000_class_bs181_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M3_N5000_class_bs303_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M3_N10000_class_bs379_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M3_N50000_class_bs636_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M3_N100000_class_bs795_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M3_N500000_class_bs1335_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M3_N1000000_class_bs1668_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M3_N5000000_class_bs2802_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
           
    for i=1:length(H5);
        %h=h5info(h5_ML);h.Datasets.Name;
        h5_file = sprintf('%s.h5',H5{i});
        
        %h5_est_file = sprintf('%s%s_est.h5',H5{i},reftxt);
        % USE NEXT LINE
        h5_est_file = sprintf('%s%s_D%d_est.h5',H5{i},reftxt,useData); %        

        ML_Player{i} = h5read(h5_est_file,'/P_layer')';
        t_pred(i) =  h5read(h5_est_file,'/t_pred');
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        N_arr(i)=str2num(Nstr);
    end
    
    
    %% Player combined
    figure(30+nh);clf;set_paper('portrait');clf
    subplot(6,1,1);
    cax_p=[0 .25];
    morill_2d_plot(MCMC_Player,cax_p,[0:.1:.3],0,1:451,'P(interface)');
    colormap(gca,cmap_prob);
    text(0.05, 0.9, sprintf('a) Sampling'),'Units','Normalized')
    for i=1:length(H5)
        subplot(6,1,i+1);
        morill_2d_plot(ML_Player{i},cax_p,[0:.1:.3],0,1:451,'P(interface)');
        colormap(gca,cmap_prob);
        text(0.05, 0.9, sprintf('%s) N=%d',96+i+1,N_arr(i)),'Units','Normalized')
    end
    if ~isempty(REF)
        thres=log10(1.6);
        i=i+1;
        P =(abs(diff(REF.M)))>thres;
        REF_Player = 0.*MCMC_Player;
        REF_Player(1:end-1,:)=P;
        
        subplot(6,1,i+1);
        morill_2d_plot(REF_Player,cax_p,[0:.1:.3],0,1:451,'P(interface)');
        colormap(gca,cmap_prob);
        text(0.05, 0.9, sprintf('%s) Reference model',96+i+1),'Units','Normalized')
    end
    print_mul(sprintf('Regression_M3_%s_Pinterface_%s_nh%d_%s',prior_type,reftxt,nh,plt_txt))
    
    %% Player compare
    for i=1:length(H5)
        figure(100+i);clf;set_paper('landscape');clf
        
        ia=97;
        if i_prior==3;
            ia=ia+2;
        end
        ib=ia+1;

        subplot(3,1,1);
        morill_2d_plot(MCMC_Player,cax_p,[0:.1:.3],0,1:451,'P(interface)');
        colormap(gca,cmap_prob);
        text(0.05, 0.9, sprintf('%s) P(interface) Sampling',ia),'Units','Normalized')
    
        subplot(3,1,2);
        morill_2d_plot(ML_Player{i},cax_p,[0:.1:.3],0,1:451,'P(interface)');
        colormap(gca,cmap_prob);
        text(0.05, 0.9, sprintf('%s) P(interface) ML',ib),'Units','Normalized')
        drawnow;
        
        print_mul(sprintf('Regression_M3_%s_Pinterface_%s_nh%d_N%d_%s',prior_type,reftxt,nh,N_arr(i),plt_txt))
        
        if ~isempty(REF)
            thres=log10(1.6);
            P =(abs(diff(REF.M)))>thres;
            REF_Player = 0.*MCMC_Player;
            REF_Player(1:end-1,:)=P;
            
            subplot(3,1,3);
            morill_2d_plot(REF_Player,cax_p,[0:.1:.3],0,1:451,'P(interface)');
            colormap(gca,cmap_prob);
            text(0.05, 0.9, sprintf('c) Reference model'),'Units','Normalized')
            print_mul(sprintf('Regression_M3_%s_Pinterface_%s_nh%d_N%d_ref_%s',prior_type,reftxt,nh,N_arr(i),plt_txt))
        end
    end
    
end

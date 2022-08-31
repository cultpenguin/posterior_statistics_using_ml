% plt_regression_M4
%clear all;close all

%if ~exist('i_prior');i_prior=3;end
%if ~exist('useRef');useRef=0;end

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
        thres=[-10 1.5 2.35 100];
        for i=1:3
            REF_P{i}=REF.M.*0;            
            ii=find( (REF.M>thres(i)) & (REF.M<thres(i+1)) );
            REF_P{i}(ii)=1;
        end
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
MCMC_P{1} = h5read(h5_MCMC,'/M_lith1');
MCMC_P{2} = h5read(h5_MCMC,'/M_lith2');
MCMC_P{3}= h5read(h5_MCMC,'/M_lith3');
try
    MCMC_Dobs = h5read(h5_MCMC,'/D_obs');
    MCMC_Dstd = h5read(h5_MCMC,'/D_std');
end

MCMC_P_mode = zeros(size(MCMC_P{1}));
MCMC_H_mode = 0.*MCMC_P_mode;
for k=1:prod(size(MCMC_P_mode))
    for j=1:3;
        p(j)=MCMC_P{j}(k);
    end
    i_mode=find(p==max(p));
    MCMC_P_mode(k)=i_mode(1);
    MCMC_H_mode(k)=entropy(p,3);
end
  




%% M1: Classificatoin/lithology estimation
clear H5 ML_*

%for nh=[2,4,8]
for nh=[4]
%    i=0;
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N1000_meanstd_bs181_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     %i=i+1;H5{i}=sprintf('Prior%s_M4_N5000_meanstd_bs303_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N10000_meanstd_bs379_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     %i=i+1;H5{i}=sprintf('Prior%s_M4_N50000_meanstd_bs636_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N100000_meanstd_bs795_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     %i=i+1;H5{i}=sprintf('Prior%s_M4_N500000_meanstd_bs1335_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N1000000_meanstd_bs1668_ep2000_nh%d_nu40_do00_ls0',prior_type,nh);
%     i=0;
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N1000_meanstd_bs181_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M4_N5000_meanstd_bs303_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N10000_meanstd_bs379_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M4_N50000_meanstd_bs636_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N100000_meanstd_bs795_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M4_N500000_meanstd_bs1335_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M4_N1000000_meanstd_bs1668_ep2000_nh%d_nu40_do00_ls%d_%s',prior_type,nh,ls,act);

    i=0;
    i=i+1;H5{i}=sprintf('Prior%s_M4_N1000_class_bs181_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M4_N5000_class_bs303_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M4_N10000_class_bs379_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M4_N50000_class_bs636_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M4_N100000_class_bs795_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M4_N500000_class_bs1335_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M4_N1000000_class_bs1668_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M4_N5000000_class_bs2802_ep2000_nh%d_nu40_do00_ls%d_%s_BN%d_HTX%d_%d',prior_type,nh,ls,act,useBatchNormalization,useHTX,useHTX_data);



    %%
    for i=1:length(H5);
        %h=h5info(h5_ML);h.Datasets.Name;
        h5_file = sprintf('%s.h5',H5{i});

        %h5_est_file = sprintf('%s%s_est.h5',H5{i},reftxt);
        % USE NEXT LINE
        h5_est_file = sprintf('%s%s_D%d_est.h5',H5{i},reftxt,useData); %        

        %ML_mean{i} = h5read(h5_est_file,'/M_mean')';
        %ML_std{i} = h5read(h5_est_file,'/M_std')';
        ML_P{1}{i} = h5read(h5_est_file,'/P_lith1')';
        ML_P{2}{i} = h5read(h5_est_file,'/P_lith2')';
        ML_P{3}{i} = h5read(h5_est_file,'/P_lith3')';
        t_pred(i) =  h5read(h5_est_file,'/t_pred');
        %t_train(i) =  h5read(h5_file,'/t_train');
        %loss{i} =  h5read(h5_file,'/loss');
        %val_loss{i} =  h5read(h5_file,'/val_loss');
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        N_arr(i)=str2num(Nstr);

        P_mode{i} = zeros(size(ML_P{1}{1}));
        H_mode{i} = 0.*P_mode{i};
        for k=1:prod(size(P_mode{i}))
            for j=1:3;
                p(j)=ML_P{j}{i}(k);
            end
            i_mode=find(p==max(p));
            P_mode{i}(k)=i_mode(1);
            H_mode{i}(k)=entropy(p,3);
        end
    
    end
    
    
    %% PLith combined
    for i_lith=[1,2,3]
        figure(30+2*nh+i_lith);clf;set_paper('portrait');clf
        subplot(6,1,1);
        cax_p=[0 1];
        cax_tick = [0:.5:1];
        morill_2d_plot(MCMC_P{i_lith},cax_p,cax_tick,0,1:451,'Probability');
        colormap(gca,cmap_prob);
        text(0.05, 0.9, sprintf('a) Sampling'),'Units','Normalized')
        for i=1:length(H5)
            subplot(6,1,i+1);
            morill_2d_plot(ML_P{i_lith}{i},cax_p,cax_tick,0,1:451,'Probability');
            colormap(gca,cmap_prob);
            text(0.05, 0.9, sprintf('%s) N=%d',96+i+1,N_arr(i)),'Units','Normalized')
        end
        print_mul(sprintf('Regression_M4_%s_Plith%d_%s_nh%d_%s',prior_type,i_lith,reftxt,nh,plt_txt))
        if ~isempty(REF)
            thres=log10(1.6);
            i=i+1;
            subplot(6,1,i+1);
            morill_2d_plot(REF_P{i_lith},cax_p,cax_tick,0,1:451,'Probability');
            colormap(gca,cmap_prob);
            text(0.05, 0.9, sprintf('%s) Reference model',96+i+1),'Units','Normalized')
            print_mul(sprintf('Regression_M4_%s_Plith%d_%s_nh%d_ref_%s',prior_type,i_lith,reftxt,nh,plt_txt))
        end
    end
    
    
     %% Pith compare
     figure(nh);clf;set_paper('landscape');clf
     for i=1:3
         is=2*(i-1)+1;
         subplot(4,2,is);
         cax_tick = [0:.5:1];
         morill_2d_plot(MCMC_P{i},cax_p,cax_tick,0,1:451,'Probability');
         colormap(gca,cmap_prob);
         text(0.05, 0.9, sprintf('%s) P(lith=%d) Sampling',96+is,i),'Units','Normalized')
     end
     for i=1:3
         is=2*(i-1)+2;
         subplot(4,2,is);
         cax_tick = [0:.5:1];
         morill_2d_plot(ML_P{i}{end},cax_p,cax_tick,0,1:451,'Probability');
         colormap(gca,cmap_prob);
         text(0.05, 0.9, sprintf('%s) P(lith=%d) ML',96+is,i),'Units','Normalized')
     end
     print_mul(sprintf('Regression_M4_%s_Pliths_%s_nh%d_%s',prior_type,reftxt,nh,plt_txt))   

     %% MODE
     figure(111);set_paper('portrait');clf         
     subfigure(1,2,1)
     for i=1:length(H5)
         subplot(6,1,i);
         cmap=cmap_linear([0 0 0; 1 0 0; 0 0 1]);
         cmap=hsv(3);
         morill_2d_plot(P_mode{i},[.5 3.5],[1,2,3],0,1:451,'Lithology');
         alpha(1-H_mode{i})
         colormap(cmap)
         text(0.05, 0.9, sprintf('%s) N=%d',96+i+1,N_arr(i)),'Units','Normalized')
     end
     subplot(6,1,i+1);
     morill_2d_plot(MCMC_P_mode,[.5 3.5],[1,2,3],0,1:451,'Lithology');
     alpha(1-MCMC_H_mode)
     colormap(cmap)
     text(0.05, 0.9, sprintf('%s) Sampling',96+i+1),'Units','Normalized')
     print_mul(sprintf('Regression_M4_%s_Pliths_multimode_%s_nh%d_%s',prior_type,reftxt,nh,plt_txt))   
     

     %% MODE BEST COMPARE
     figure(111+nh);clf;set_paper('landscape');clf
             
     subplot(3,1,1);
     morill_2d_plot(MCMC_P_mode,[.5 3.5],[1,2,3],0,1:451,'Lithology');
     alpha(1-MCMC_H_mode)
     colormap(cmap)
     text(0.05, 0.9, sprintf('a) P(interface) Sampling'),'Units','Normalized')
     
     subplot(3,1,2);
     morill_2d_plot(P_mode{end},[.5 3.5],[1,2,3],0,1:451,'Lithology');
     alpha(1-H_mode{i})
     colormap(cmap)
     text(0.05, 0.9, sprintf('b) P(interface) ML'),'Units','Normalized')
     drawnow;
     
     print_mul(sprintf('Regression_M4_%s_Pliths_multimode_%s_nh%d_N%d_%s',prior_type,reftxt,nh,N_arr(end),plt_txt))   
     
     
end

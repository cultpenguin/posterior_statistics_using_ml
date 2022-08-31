%plt_regression_M5
%clear all;
close all

plt_defs;

use_prior = use_prior_types(i_prior);
prior_type = prior_types{i_prior};
%% GET REFERENCE SAMPLING RESULTS
if ~exist('N_sampling')
    N_sampling=1000000;
    N_sampling=5000000;
end
REF=[];

if useRef>1
    reftxt=sprintf('ref%d',useRef);
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_ME0_aT1_CN1_ref%d.h5',use_prior,N_sampling,useRef);
    
    try
        REF = load('Morill_data_ml_IP7.mat');
    end
        
else
    reftxt='';
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_ME0_aT1_CN1.h5',use_prior,N_sampling);
    h5_MCMC = sprintf('1D_P%d_NO500_451_ABC%d_0000_D%d_HTX%d_%d_ME0_aT1_CN1.h5',use_prior,N_sampling,useData,useHTX,useHTX_data);
end
%h=h5info(h5_MCMC);h.Datasets.Name
MCMC_mean = h5read(h5_MCMC,'/T_est');
MCMC_std = h5read(h5_MCMC,'/T_std');


%% M1: Regression estimation
clear H5 ML_*
pdf_type=pdf_types{1};
%pdf_type=pdf_types{end};

%for nh=[2,4,8]
for nh=[4]
    i=0;
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N1000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
    %%i=i+1;H5{i}=sprintf('Prior%s_M5_N5000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N10000_meanstd_bs128_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
    %%i=i+1;H5{i}=sprintf('Prior%s_M5_N50000_meanstd_bs256_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N100000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
    %%i=i+1;H5{i}=sprintf('Prior%s_M5_N500000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N1000000_meanstd_bs2048_ep2000_nh%d_nu40_do00_ls0_%s%s',prior_type,nh,pdf_type);
           
            
    i=i+1;H5{i}=sprintf('Prior%s_M5_N1000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N5000_meanstd_bs64_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M5_N10000_meanstd_bs128_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N50000_meanstd_bs256_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M5_N100000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N500000_meanstd_bs512_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M5_N1000000_meanstd_bs2048_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M5_N2000000_meanstd_bs2048_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M5_N5000000_meanstd_bs2048_ep2000_nh%d_nu40_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    
    for i=1:length(H5);
        %h=h5info(h5_ML);h.Datasets.Name;
        h5_file = sprintf('%s.h5',H5{i});
        h5_est_file = sprintf('%s%s_est.h5',H5{i},reftxt);
        h5_est_file = sprintf('%s%s_D%d_est.h5',H5{i},reftxt,useData);
        
        ML_mean{i} = h5read(h5_est_file,'/M_mean')';
        ML_std{i} = h5read(h5_est_file,'/M_std')';
        ML_pdf_est{i} = squeeze(h5read(h5_est_file,'/pdf_est'));
        ML_m_test{i} = h5read(h5_est_file,'/m_test');
        t_train(i) =  h5read(h5_file,'/t_train');
        loss{i} =  h5read(h5_file,'/loss');
        val_loss{i} =  h5read(h5_file,'/val_loss');
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        N_arr(i)=str2num(Nstr);
        
    end
    
    
    %% MEAN PDF
    figure(20+nh);clf;set_paper('portrait');clf
    %subplot(6,1,1);
    %imagesc(x,ML_m_test{i},ML_pdf_est{i});shading flat;colormap(1-gray)
    for i=1:length(H5)
        subplot(6,1,i);
        imagesc(x,ML_m_test{i},ML_pdf_est{i});shading flat;colormap(1-gray)
        caxis([0 0.03])
        hold on
        plot(x,MCMC_mean,'r-','LineWidth',2)
        hold off
        ylabel('\sigma(n_3) - thickness (m)')
        text(0.05, 0.9, sprintf('%s) N=%d',96+i,N_arr(i)),'Units','Normalized')
        set(gca,'ydir','normal')
        ylim([0,80])
    end
    xlabel('Distance along line (km)')
    
    print_mul(sprintf('Regression_M5_%s_pdf_mean_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))
    
    %% MEAN
    clear L
    figure(30+nh);clf;set_paper('portrait');clf
    subplot(3,1,1);
    plot(x,MCMC_mean,'color',colors(1,:),'LineWidth',3);
    L{1}='Sampling';
    hold on
    for i=1:length(H5)
        plot(x,ML_mean{i}+i*20,'-','color',colors(i+1,:),'LineWidth',1+length(H5)-i)
        L{i+1}=sprintf('%d',N_arr(i));
    end
    hold off
    xlabel('Distance along line (km)')
    ylabel('Thickness (m)')
    grid on
    xlim([x(1) x(end)])
    legend(L,'Location','NorthEastOutside')
    print_mul(sprintf('Regression_M5_%s_mean_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))
    if ~isempty(REF)
        i=i+1;
        REF_thick=sum(REF.M>2.35);
        hold on
        plot(x,REF_thick,'r-','LineWidth',2)
        hold off
        L{i+2}=sprintf('Reference');   
        print_mul(sprintf('Regression_M5_%s_mean_%s_nh%d_%s_ref_%s',prior_type,reftxt,nh,pdf_type,plt_txt))
    end
     
    %% MEAN SINGLE
    clear L
    figure(40+nh);clf;set_paper('portrait');clf
    subplot(3,1,1);
    plot(x,MCMC_mean,'color',colors(1,:),'LineWidth',3);
    L{1}='Sampling';
    hold on
    for i=length(H5)
        plot(x,ML_mean{i},'-','color',colors(end-1,:),'LineWidth',1)
        L{2}=sprintf('ML');
        %L{2}=sprintf('%d',N_arr(i));
    end
    hold off
    xlabel('Distance along line (km)')
    ylabel('Thickness (m)')
    grid on
    xlim([x(1) x(end)])
    legend(L,'Location','NorthEastOutside')
    print_mul(sprintf('Regression_M5_%s_mean_%s_nh%d_%s_N%d_%s',prior_type,reftxt,nh,pdf_type,N_arr(end),plt_txt))

    if ~isempty(REF)
        REF_thick=sum(REF.M>2.35);
        hold on
        %plot(x,REF_thick,'-','color',colors(end,:),'LineWidth',2)
        plot(x,REF_thick,'r-','LineWidth',2)
        hold off
        L{3}=sprintf('Reference');
        legend(L,'Location','NorthEastOutside')
        print_mul(sprintf('Regression_M5_%s_mean_%s_nh%d_%s_N%d_ref_%s',prior_type,reftxt,nh,pdf_type,N_arr(end),plt_txt))        
    end
    
    
    
    
end
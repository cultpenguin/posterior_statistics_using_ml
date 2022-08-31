close all
%i_prior=3;act='selu';ls=1;
%i_prior=3;act='relu';ls=1;

%useHTX=1;
%useHTX_data=1;
%useData=2;
%useBatchNormalization=1;

% if ~exist('useHTX');useHTX=1;end
% if ~exist('useHTX_data');useHTX_data=1;end
% if ~exist('useData');
%     useData=1; % 'Original' forward ref
%     useData=2; % Morrill data
% end
% 
% if ~exist('i_prior');i_prior=3;end
% 
% if ~exist('useRef');useRef=0;end
% if ~exist('useBatchNormalization');
%     useBatchNormalization=0; % ignore BatchNormaliation
%     useBatchNormalization=1;
% end 
% if ~exist('ls');ls=1;end % learning schedule
% if ~exist('act');% activation function
%     act='relu';
%     act='selu';
% end 
% if ~exist('nu');nu=40;end % number of units in layers
% 
plt_defs;
cax_std=[0 0.8];


use_prior = use_prior_types(i_prior);
prior_type = prior_types{i_prior};
disp([plt_txt,'-',prior_type])

%% GET REFERENCE SAMPLING RESULTS
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
MCMC_mean = h5read(h5_MCMC,'/M_est');
MCMC_std = h5read(h5_MCMC,'/M_std');


%% M1: Regression estimation
clear H5 ML_*
pdf_type=pdf_types{1};

%for nh=[2,4,8]
for nh=[8]
%     i=0;
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N1000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N5000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N10000_meanstd_bs128_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N50000_meanstd_bs256_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N100000_meanstd_bs512_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N500000_meanstd_bs512_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N1000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N2000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s',prior_type,nh,nu,ls,pdf_type,act);
  

%     i=0;
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N1000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N5000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N10000_meanstd_bs128_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N50000_meanstd_bs256_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N100000_meanstd_bs512_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N500000_meanstd_bs512_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     i=i+1;H5{i}=sprintf('Prior%s_M1_N1000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);
%     %i=i+1;H5{i}=sprintf('Prior%s_M1_N2000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useHTX,useHTX_data);

    i=0;
    i=i+1;H5{i}=sprintf('Prior%s_M1_N1000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M1_N5000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N10000_meanstd_bs128_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M1_N50000_meanstd_bs256_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N100000_meanstd_bs512_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M1_N500000_meanstd_bs512_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N1000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    i=i+1;H5{i}=sprintf('Prior%s_M1_N5000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);
    %i=i+1;H5{i}=sprintf('Prior%s_M1_N2000000_meanstd_bs2048_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);






    for i=1:length(H5);
        %h=h5info(h5_ML);h.Datasets.Name;
        h5_file = sprintf('%s.h5',H5{i});
        h5_est_file = sprintf('%s%s_D%d_est.h5',H5{i},reftxt,useData);
        ML_mean{i} = h5read(h5_est_file,'/M_mean')';
        ML_std{i} = h5read(h5_est_file,'/M_std')';
        try;
            ML_pdf_est{i} = h5read(h5_est_file,'/pdf_est');
        end
        t_train(i) =  h5read(h5_file,'/t_train');
        loss{i} =  h5read(h5_file,'/loss');
        val_loss{i} =  h5read(h5_file,'/val_loss');
    end
    
    %% MEAN
    figure(30+nh);clf;set_paper('portrait');clf
    for i=1:length(H5)
        subplot(6,1,i);
        morill_2d_plot(ML_mean{i},cax,cax_ytick,1);
        colormap(gca,cmap_rho)
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        text(0.05, 0.9, sprintf('%s) N=%s',96+i,Nstr),'Units','Normalized')
    end
    i=i+1;
    subplot(6,1,i);
    morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
    colormap(gca,cmap_rho)
    text(0.05, 0.9, sprintf('%s) Sampling',96+i),'Units','Normalized')
    
    if ~isempty(REF)
        i=i+1;
        subplot(6,1,i);
        morill_2d_plot(REF.M,cax,cax_ytick,1);
        colormap(gca,cmap_rho)
        text(0.05, 0.9, sprintf('%s) Reference model',96+i+1),'Units','Normalized')
    end
    print_mul(sprintf('Regression_M1_%s_compare_mean_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))
        
    % STD
    figure(40+nh);clf;set_paper('portrait')
    for i=1:length(H5)
        subplot(6,1,i);
        morill_2d_plot(ML_std{i},cax_std,cax_std_ytick,0,1:size(ML_std{i},2),'Std');
        colormap(gca,cmap_std)
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        text(0.05, 0.9, sprintf('%s) N=%s',96+i+length(H5)+1,Nstr),'Units','Normalized')
    end
    i=i+1;
    subplot(6,1,i);
    morill_2d_plot(MCMC_std,cax_std,cax_std_ytick,0,1:size(MCMC_std,2),'Std');
    colormap(gca,cmap_std)
    text(0.05, 0.9, sprintf('%s) Sampling',96+i+length(H5)+1),'Units','Normalized')
    print_mul(sprintf('Regression_M1_%s_compare_std_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))

    %% MEAN + TRANSP/STD
    figure(50+nh);clf;set_paper('portrait');clf
    for i=1:length(H5)
        subplot(6,1,i);
    
        A = (ML_std{i}- min_std)./(max_std-min_std);A(A>1)=1;A(A<0)=0;A=1-A;
        morill_2d_plot(ML_mean{i},cax,cax_ytick,1);
        alpha(A)            
        colormap(gca,cmap_rho)
        s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
        N_arr(i)=str2num(Nstr);
        text(0.05, 0.9, sprintf('%s) N=%s',96+i,Nstr),'Units','Normalized')
        drawnow;
    end
    i=i+1;
    subplot(6,1,i);
    A = (MCMC_std- min_std)./(max_std-min_std);A(A>1)=1;A(A<0)=0;A=1-A;
    morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
    alpha(A);
    colormap(gca,cmap_rho)
    text(0.05, 0.9, sprintf('%s) Sampling',96+i),'Units','Normalized')
    
    if ~isempty(REF)
        i=i+1;
        subplot(6,1,i+1);
        morill_2d_plot(REF.M,cax,cax_ytick,1);
        colormap(gca,cmap_rho)
        text(0.05, 0.9, sprintf('%s) Reference model',96+i),'Units','Normalized')
    end
    print_mul(sprintf('Regression_M1_%s_compare_meanalpha_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))    
    
    %% Mean,Std,Mean+Std sampling individual
    %cax_std=[0 0.8];
    %cmap_std=hsv;
    for p_type=1:3;
        
        if p_type==1;
            lab='Mean';labt=lower(lab);
        elseif p_type==2,
            lab='Std';labt=lower(lab);
        else
            lab='Mean+std'
            labt='meanalpha'
        end
        %for i=1:length(H5)
        for i=length(H5)
            figure(100+i+10*p_type);clf;set_paper('landscape');clf
            
            ia=97;
            if i_prior==2;
                ia=ia+(i_prior-1)*4;
            end
            ib=ia+1;
            
            subplot(3,1,1);
            morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
            if p_type==1;
                morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
                colormap(gca,cmap_rho)
            elseif p_type==2;
                morill_2d_plot(MCMC_std,cax_std,cax_std_ytick,0,1:size(ML_std{i},2),'Std');
                colormap(gca,cmap_std)
            else
                morill_2d_plot(MCMC_mean,cax,cax_ytick,1);
                A = (MCMC_std- min_std)./(max_std-min_std);A(A>1)=1;A(A<0)=0;A=1-A;
                alpha(A);
                colormap(gca,cmap_rho)
            end
            
            text(0.05, 0.9, sprintf('%s) %s Sampling',ia+(p_type-1)*2,lab),'Units','Normalized')
            
            subplot(3,1,2);
            if p_type==1
                morill_2d_plot(ML_mean{i},cax,cax_ytick,1);
                colormap(gca,cmap_rho)
            elseif p_type==2
                morill_2d_plot(ML_std{i},cax_std,cax_std_ytick,0,1:size(ML_std{i},2),'Std')                
                colormap(gca,cmap_std)
            else
                morill_2d_plot(ML_mean{i},cax,cax_ytick,1);
                A = (ML_std{i}- min_std)./(max_std-min_std);A(A>1)=1;A(A<0)=0;A=1-A;
                alpha(A)
                colormap(gca,cmap_rho)
            end
            s=strsplit(H5{i},'_');Nstr=s{3}(2:end);
            N_arr(i)=str2num(Nstr);
            text(0.05, 0.9, sprintf('%s) %s ML',ib+(p_type-1)*2,lab),'Units','Normalized')
            drawnow;
            
            print_mul(sprintf('Regression_M1_%s_compare_%s_%s_nh%d_%s_N%d_%s',prior_type,labt,reftxt,nh,pdf_type,N_arr(i),plt_txt))
        end
    end
    

    %%
    clear ha;
    figure(90+nh);clf;;set_paper;
    for i=1:length(H5)
        %ha(i)=semilogy(loss{i},'-','color',colors(i,:),'LineWidth',3)
        ha(i)=loglog(loss{i},'-','color',colors(i,:),'LineWidth',3)
        hold on
        plot(val_loss{i},'-','color',colors(i,:),'LineWidth',.2)        
    end
    hold off
    ylabel('Loss')
    xlabel('Iteration number')
    nend=length(loss{i});
    n=length(loss{1});
    dum1=val_loss{1}(ceil(n/2):end);
    n2=length(loss{end});
    dum2=loss{i}(ceil(n2/2):end);
    yl=[min(dum2) max(dum1)]; 
    if yl(2)<=yl(1),
        yl(2)=yl(1)*1.1;
    end
    dy=0.1*[-1,1].*diff(yl)  ;
    ylim(yl+dy)
    xlim([1 nend])
    grid on
    legend(ha,num2str(N_arr(:)))
    ppp(12,5,10,2,2)
    print_mul(sprintf('Regression_M1_%s_loss_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))    
   
    %%
    clear ha;
    figure(90+nh+1);clf;set_paper;
    for i=1:length(H5)
        ha(i)=semilogy(linspace(0,1,length(loss{i}))*t_train(i)/60,loss{i},'-','color',colors(i,:),'LineWidth',2)
        %ha(i)=loglog(linspace(0,1,length(loss{i}))*t_train(i)/60,loss{i},'-','color',colors(i,:),'LineWidth',2)
        hold on
        plot(linspace(0,1,length(loss{i}))*t_train(i)/60,val_loss{i},'-','color',colors(i,:),'LineWidth',.2)        
    end
    for i=1:length(H5)
        plot(t_train(i)/60,loss{i}(end),'k.','color',colors(i,:),'MarkerSize',62)
    end
    ylabel('Loss')
    xlabel('Training time (minutes)')
    legend(ha,num2str(N_arr(:)))
    ylim(yl+dy)
    xlim([0.1 max(t_train)/60])
    grid on
    ppp(12,7,10,2,2)
    print_mul(sprintf('Regression_M1_%s_losstime_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))    
    set(gca,'Xscale','log')
    print_mul(sprintf('Regression_M1_%s_losstime_log_%s_nh%d_%s_%s',prior_type,reftxt,nh,pdf_type,plt_txt))
    
end

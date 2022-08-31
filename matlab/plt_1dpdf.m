clear all;
close all
%plt_defs

load morrill_model
x=model.X(1,:)/1000;useABC=1;

useABC=1; % ABC
useABC=0; % METROPOLIS
i_data = 300;
act='elu' % Best in original implementation
act='relu';
useBatchNormalization=1;
useHTX=1;
useHTX_data=1;
%%
%txt='PriorC_N10000_meanstd_bs128_ep2000_nh8_nu80_do0';
%txt='PriorC_N5000000_meanstd_bs2048_ep2000_nh8_nu80_do0';

%txt='PriorA_N1000000_meanstd_bs2048_ep2000_nh8_nu80_do0';
%txt='PriorA_N5000000_meanstd_bs512_ep2000_nh8_nu80_do0';
%f_abc = '1D_P23_NO500_451_ABC500000_0000';

%txt='PriorB_N1000000_meanstd_bs2048_ep2000_nh8_nu80_do0';
%txt='PriorB_N500000_meanstd_bs512_ep2000_nh8_nu80_do0';
%f_abc = '1D_P51_NO500_451_ABC500000_0000';

txt='PriorC_M1_N5000000_meanstd_bs2048_ep2000_nh8_nu40_do00_ls1';
f_abc = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1';
%f_abc = '1D_P22_NO500_451_ABC1000000_0000_D2_HTX1_1';

%txt='PriorB_M1_N1000000_meanstd_bs2048_ep2000_nh8_nu40_do00_ls1';
%f_abc = '1D_P51_NO500_451_ABC5000000_0000';
%f_abc = '1D_P51_NO500_451_ABC1000000_0000';

%txt='PriorA_M1_N1000000_meanstd_bs2048_ep2000_nh8_nu40_do00_ls1';
%f_abc = '1D_P23_NO500_451_ABC5000000_0000';
%f_abc = '1D_P23_NO500_451_ABC1000000_0000';


f_mat = [f_abc,'_ME0_aT1_CN1_multiatt.mat'];

%f_mat='1D_P22_NO500_451_ABC2000000_0000_D2_HTX1_1_ME0_aT1_CN1_multiatt.mat';
f_mat = [f_abc,'_ME0_aT1_CN1_multiatt.mat'];


i=0;
i=i+1;f5{i}=sprintf('%s_Normal_%s_BN%d_HTX%d_%d_D2_est.h5',txt,act,useBatchNormalization,useHTX,useHTX_data);L{i}='Normal';
% %i=i+1;f5{i}=sprintf('%s_LogNormal_%s_BN%d_HTX%d_%d_D2_est.h5',txt,act,useBatchNormalization,useHTX,useHTX_data);L{i}='logNormal';
i=i+1;f5{i}=sprintf('%s_GeneralizedNormal_%s_BN%d_HTX%d_%d_D2_est.h5',txt,act,useBatchNormalization,useHTX,useHTX_data);L{i}='Generalized Normal';
%i=i+1;f5{i}=sprintf('%s_MixtureNormal1_%s_BN%d_HTX%d_%d_D2_est.h5',txt,act,useBatchNormalization,useHTX,useHTX_data);L{i}='Gaussian_1 Mixture';
i=i+1;f5{i}=sprintf('%s_MixtureNormal2_%s_BN%d_HTX%d_%d_D2_est.h5',txt,act,useBatchNormalization,useHTX,useHTX_data);L{i}='Gaussian_2 Mixture';
i=i+1;f5{i}=sprintf('%s_MixtureNormal3_%s_BN%d_HTX%d_%d_D2_est.h5',txt,act,useBatchNormalization,useHTX,useHTX_data);L{i}='Gaussian_3 Mixture';

%i=i+1;H5{i}=sprintf('Prior%s_M1_N1000_meanstd_bs64_ep2000_nh%d_nu%d_do00_ls%d_%s_%s_BN%d_HTX%d_%d',prior_type,nh,nu,ls,pdf_type,act,useBatchNormalization,useHTX,useHTX_data);

%% LOAD ML RESULTS
for i=1:length(f5)
    M_mean{i} = h5read(f5{i},'/M_mean')';
    M_std{i} = h5read(f5{i},'/M_std')';
    pdf_est{i} = h5read(f5{i},'/pdf_est');
    m_test{i} = h5read(f5{i},'/m_test');
end
   
%% GET SAMPLING RESULTS
show_reals = 1;
if show_reals==1;
    
    if ~exist('ABC')
        fprintf('Loading %s ... ',f_mat)
        M=load(f_mat);
        fprintf('DONE\n',f_mat)
        
        fprintf('Loading %s ... ',f_abc)
        load(f_abc,'ABC');
        fprintf('DONE\n',f_mat)
        
    end
    % Sample from posterior
    data = M.DALL{i_data}.data;
    %%
    if useHTX==1;
        for i=1:length(ABC.m)
            ABC.d{i}{1}(13)=ABC.m{i}{3};
        end
    end
    %%    
    
    n_reals=5000;        
    if useABC==1
        %use_adaptive_T=1;
        [logL,ev,T_est,ABC]=sippi_abc_logl(ABC,data);
        [post_reals, P_acc, i_use_all] = sippi_abc_post_sample(ABC, n_reals, T_est, logL);
        %[post_reals,logL]=abc_dummy(M.ABC,data,use_adaptive_T);
    else
        %% mcmc
        txt=[txt,'_metropolis'];
        Nite=120000;            
        Nite=100000;            
        Nchains=4;
        %Nite=10000000;            
        f_mcmc = sprintf('%s_d%d_N%d_NC%d_%s',f_abc(1:6),i_data,Nite,Nchains,txt);
        f_mcmc = sprintf('%s_d%d_N%d_NC%d',f_abc(1:6),i_data,Nite,Nchains);
        if exist([f_mcmc,'.mat'],'file');
            load(f_mcmc)
        else            
            
            options.mcmc.nite=Nite;
            options.mcmc.n_sample=n_reals;
            options.mcmc.i_plot=1000;
            options.mcmc.anneal.i_begin=1; % default, iteration number when annealing begins
            options.mcmc.anneal.i_end=ceil(Nite/20); %  iteration number when annealing stops
            options.mcmc.anneal.T_begin=15; % Start temperature for annealing
            options.mcmc.anneal.T_end=1; % End temperature for annealing
            

            options.mcmc.n_chains=10; % set number of chains (def=1)
            options.mcmc.T=ones(1,options.mcmc.n_chains);      % set temperature of chains [1:n_chains]
            options.mcmc.T=linspace(1,2,options.mcmc.n_chains);
            options.mcmc.chain_frequency_jump=0.1; % probability allowing a jump
                                            %  between two chains
   
            prior=ABC.prior;
            for ip=1:length(prior);
                prior{ip}.seq_gibbs.step_min=0.005;
            end
            for ip=1:length(prior);
                prior{ip}.seq_gibbs.i_update_step_max=ceil(Nite/10);
                prior{ip}.seq_gibbs.step=1;
            end
            % USE FLIGHT HEIGHT
            ABC.forward.htx_as_data=1;
            % FIX FLIGHT HEIGHT
            prior{3}.m0=data{1}.d_obs(13);prior{3}.std=0.01;
            prior{3}.seq_gibbs.step=0;
            prior{3}.seq_gibbs.i_update_step_max=0;
            
            [options,data,prior,forward,m_current]=sippi_metropolis(data,prior,ABC.forward,options)
            [reals,etype_mean,etype_var,post_reals{1},reals_ite]=sippi_get_sample(options.txt,1,15,1);
            post_reals{1} = post_reals{1}(ceil(options.mcmc.n_sample/10):end,:)';
            
            try
                [reals,etype_mean,etype_var,post_reals{2},reals_ite]=sippi_get_sample(options.txt,2,15,1);
                post_reals{2} = post_reals{2}(ceil(options.mcmc.n_sample/10):end,:)';
            end
            save(f_mcmc,'post_reals','options')
        end
        
        
    end
    dm_est=diff(m_test{1});dm_est=dm_est(1)

    %%
    nm=size(post_reals{1},1);
    mm=linspace(m_test{1}(1),m_test{1}(end),31);  
    
    %mm=linspace(m_test{1}(1),m_test{1}(end),151);        
    %mm=linspace(m_test{1}(1),m_test{1}(end),101);        
    dm_real=diff(mm);dm_real=dm_real(1);
    clear pdf_real;
    for im=1:nm
        pdf_real(im,:)=hist(post_reals{1}(im,:),mm);
        dm=mm(2)-mm(1);
        pdf_real(im,:) = pdf_real(im,:)./( sum(pdf_real(im,:).*dm));        
    end
    imagesc(mm,1:nm,pdf_real)
      
    LL{1}='Sampling';
    for il=1:length(L);
        LL{il+1}=L{il};
    end
end

%% PLOT
l_type{1}='-';l_type{2}='-';l_type{3}=':';l_type{4}=':';l_type{5}='-.';l_type{6}='-.';
col=[0 0 0; .5 .5 .5; 1 0 0; 0.7 0 0; 0 0 1; 1 0 0];

figure(1);set_paper('portrait');clf
is=0;
for i_plot = [10,50,125];
    is=is+1;
    subplot(3,1,is)
    
    if show_reals==1;
        bar(mm,pdf_real(i_plot,:))
        hold on
    end
    for i=1:length(f5)
        plot(m_test{i},pdf_est{i}(:,i_plot,i_data),l_type{i},'LineWidth',8-i,'color',col(i,:))
        hold on
        grid on
        title(sprintf('z=%d',i_plot))
        xlabel('Resistivity (ohm-m)')
        ylabel('pdf')
    end
    set(gca,'Xtick',-3:1); %// adjust manually; values in log scale
    set(gca,'Xticklabel',10.^get(gca,'Xtick')); %// use labels with linear values
    hold off
    if show_reals==1;
        legend(LL)
    else
        legend(L)
    end
        
end
print_mul(sprintf('%s_post_1d_pdf_id%d',txt,i_data))

%%
figure(2);set_paper('landscape');clf
Nf=length(f5);
if show_reals==1;
    subplot(2,Nf+1,1);
    imagesc(mm,1:nm,pdf_real)
    title(sprintf('%s) %s',96+1,LL{1}))
    caxis([0 3])
end
for i=1:Nf
    subplot(2,Nf+1,i+1);
    pdf_est_plot=pdf_est{i}(:,:,i_data)';
    
    iprob=find(abs(diff(diff(pdf_est_plot(:,100))))>0.09);
    iprob=iprob(find(iprob>10))+1;
    for j=1:length(iprob);
        pdf_est_plot(iprob(j),:)=(pdf_est_plot(iprob(j)+1,:)+pdf_est_plot(iprob(j)-1,:))/2;
    end
    
    imagesc(mm,1:nm,pdf_est_plot)
    title(sprintf('%s) %s',96+i+1,L{i}))    
end
    
for i=1:(Nf+1);
    subplot(2,Nf+1,i);
    if i==1
        ylabel('Depth (m)')
    end
    caxis([0 3])
    colormap([flipud(gray);hot])
    colormap(jet)   
    set(gca,'Xtick',-3:1:3); %// adjust manually; values in log scale
    set(gca,'Xticklabel',10.^get(gca,'Xtick')); %// use labels with linear values
    xlabel('Resistivity (Ohmm)')
end
colormap([flipud(gray);hot])
cb=colorbar_shift;
%colormap(hot)
print_mul(sprintf('%s_pdf_%d_REJ%d',txt,i_data,useABC))


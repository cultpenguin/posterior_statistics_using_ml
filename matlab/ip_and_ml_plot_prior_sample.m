clear all;close all
col=[1 0 0; 0 1 0; 0 0 1];

usePrior_arr = [23,51,22];
%usePrior_arr = [51];

%% load data
S = readSystem('hemSystem.txt');
D = readData('L10090.csv',S.nf);
if ~exist('noise_level'); noise_level=5; end
if ~exist('noise_base'); noise_base=5; end

load model
x=model.X(1,:)/1000;useABC=1;
i_data = 300;

data{1}.d_obs=D(i_data).obs;
data{1}.d_std = sqrt(((noise_level/100)*D(i_data).obs).^2 + noise_base^2);


%%
N=1000000;
N=5000000;
figure(1);set_paper('landscape');clf;    
for ip = 1:length(usePrior_arr);
    % load data
    usePrior = usePrior_arr(ip);
    f=sprintf('1D_P%d_NO500_451_ABC%5d_0000',usePrior,N);
    f_mat = [f,'.mat'];
    f_h5 = [f,'.h5'];
    D_est=load(sprintf('%s_ME0_aT1_CN1_multiatt.mat',f));
    disp(sprintf('Loading %s',f_mat))
    D_ABC=load(f_mat);
    
    %% prior reals
    m{1}=h5read(f_h5,'/M1');
    try; m{2}=h5read(f_h5,'/M2');end
    m{3}=h5read(f_h5,'/M3');
    
    %% post reals
    use_adaptive_T=1;
    %[m_post,logL]=abc_dummy(D_ABC.ABC,data,use_adaptive_T);
    [logL,ev,T_est,ABC]=sippi_abc_logl(D_ABC.ABC,data);
    [m_post, P_acc, i_use_all] = sippi_abc_post_sample(D_ABC.ABC, 1000, T_est, logL);
    try;m_post{2}=m{2}(:,i_use_all);end;
    m_post{3}=m{3}(:,i_use_all);
    thres=log10(1.6);
    for i=1:100;
        m_diff=abs(diff(m_post{1}(:,i)));
        m_new=zeros(size(m_diff));
        m_new(m_diff>thres)=1;
        m_post{3}(1:124,1)=m_new;
    end

    
    %% plot PRIOR
    %figure(usePrior);set_paper('landscape');clf;
    y=D_ABC.ABC.prior{1}.y;
    if length(y)==1;y=D_ABC.ABC.prior{1}.x;end
    nshow=11;
    
    
    subplot(3,2,(ip-1)*2+1)
    for i=1:nshow        
        rho_org = 0.1+m{1}(:,i)/8;        
        plot(i+rho_org,y,'k-','LineWidth',2)
        hold on        
        if usePrior==22
            for icat = [0,1,2]
                i0=find(m{2}(:,i)==icat);
                rho = rho_org.*NaN;
                rho(i0)=rho_org(i0);
                plot(i+rho,y,'.','color',col(icat+1,:))
            end
        end
        plot([i i],[y(1) y(end)],'k-')
        Pb = m{3}(:,i);
        for iy=1:length(y);
            if Pb(iy)==1;
                plot([i i+rho_org(iy)],[1 1].*y(iy),'k-','color',[1 1 1]*.3,'LineWidth',.5)
            end
        end
    end
    set(gca,'xtick',[1:1:nshow])
    xlim([0.5, nshow+1])
    ylim([0,max(y)])
    hold off
    set(gca,'ydir','revers')
    xlabel(sprintf('# in training data set, \\rho_%s(m)',64+ip))
    ylabel('Depth (m)')
    %ppp(14,5,12,1,1)
    text(-1,-10,sprintf('%s)',96+(ip-1)*2+1),'FontSize',18)
    
    subplot(3,2,(ip-1)*2+2)
    for i=1:nshow        
        rho_org = 0.1+m_post{1}(:,i)/8;        
        plot(i+rho_org,y,'k-','LineWidth',2)
        hold on        
        if usePrior==22
            for icat = [0,1,2]
                i0=find(m_post{2}(:,i)==icat);
                rho = rho_org.*NaN;
                rho(i0)=rho_org(i0);
            plot(i+rho,y,'.','color',col(icat+1,:))
            end
        end
        plot([i i],[y(1) y(end)],'k-')
        
        try
        Pb = m_post{3}(:,i);
        for iy=1:length(y);
            if Pb(iy)==1;
                plot([i i+rho_org(iy)],[1 1].*y(iy),'k-','color',[1 1 1]*.3,'LineWidth',.5)
            end
        end
        end
        
    end
    set(gca,'xtick',[1:1:nshow])
    xlim([0.5, nshow+1])
    ylim([0,max(y)])
    hold off
    set(gca,'ydir','revers')
    xlabel('# posterior realization')
    xlabel(sprintf('# posterior realization, \\sigma_%s(m)',64+ip))
    ylabel('Depth (m)')
    %ppp(14,5,12,1,1)
    text(-1,-10,sprintf('%s)',96+(ip-1)*2+2),'FontSize',18)
    
    print_mul(sprintf('P%d_training_prior',usePrior),1,1,0,600)
    
        
end

Ptxt=sprintf('_%d',usePrior_arr);
print_mul(sprintf('P%s_training_prior',Ptxt),1,1,0,600)


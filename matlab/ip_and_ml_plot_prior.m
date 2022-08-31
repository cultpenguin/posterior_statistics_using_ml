

col=[1 0 0; 0 1 0; 0 0 1];

usePrior_arr = [13,51,22];
usePrior_arr = [22];

for ip = 1:length(usePrior_arr);
    % load data
    usePrior = usePrior_arr(ip);
    %D=load(sprintf('1D_P%d_NO500_451_ABC200000_1000_ME0_aT1_CN1_multiatt.mat',usePrior));
    f=sprintf('1D_P%d_NO500_451_ABC1000000_0000',usePrior);
    f_mat = [f,'.mat']
    f_h5 = [f,'.h5']
    D_est=load(sprintf('%s_ME0_aT1_CN1_multiatt.mat',f));    
    D=load(f_mat);
    
    m{1}=h5read(f_h5,'/M1');
    m{2}=h5read(f_h5,'/M2');
    m{3}=h5read(f_h5,'/M3');
    
    %% plot PRIOR
    figure(usePrior);set_paper('landscape');clf;
    y=D.ABC.prior{1}.y;
    if length(y)==1;y=D.ABC.prior{1}.x;end
    nshow=11;
    for i=1:nshow
        
        rho_org = 0.1+D.ABC.m{i}{1}/8;
        rho_org = 0.1+m{1}(:,i)/8;
        
        plot(i+rho_org,y,'k-','LineWidth',2)
        hold on
        
        for icat = [0,1,2]
            i0=find(m{2}(:,i)==icat);
            rho = rho_org.*NaN;
            rho(i0)=rho_org(i0);
            plot(i+rho,y,'.','color',col(icat+1,:))
        end
        %plot(i+D.ABC.m{i}{2}/5,y,'g-')
        plot([i i],[y(1) y(end)],'k-')
        %plot(i+D.ABC.m{i}{3}/8,y,'k-')
        Pb = m{3}(:,i);
        %Pb = D.ABC.m{i}{3};
        for iy=1:length(y);
            if Pb(iy)==1;
                plot([i i+rho_org(iy)],[1 1].*y(iy),'k-')
            end
        end
        
        
    end
    set(gca,'xtick',[1:1:nshow])
    xlim([0.5, nshow+1])
    ylim([0,max(y)])
    hold off
    set(gca,'ydir','revers')
    xlabel('# in training data set')
    ylabel('Depth (m)')
    ppp(14,5,12,1,1)
    text(-.1,-4,sprintf('%s)',96+ip),'FontSize',18)
    print_mul(sprintf('P%d_training_prior',usePrior),1,1,0,600)
    
    
    %% plot DATA
    nshow=4;
    figure(100+usePrior);set_paper('landscape');clf;
    for i=1:nshow;
        subplot(3,nshow,i);
        d=D.ABC.d{i};
        d_noise = D.ABC.d_obs{i};
        p1=sippi_plot_data_fdem1d(d);
        hold on
        p2=sippi_plot_data_fdem1d(d_noise);
        set(p2(1),'linestyle','none')
        set(p2(2),'linestyle','none')
        
        set(p2(3),'Marker','.','MarkerSize',12)
        set(p2(4),'Marker','.','MarkerSize',12)
        
        set(p1(3),'Marker','o','MarkerSize',7)
        set(p1(4),'Marker','o','MarkerSize',7)
        %ylim([-200 3600])
        hold off
        
            
        
    end
    print_mul(sprintf('P%d_training_data',usePrior),1,1,0,600)
    subplot(3,nshow,1);
    text(10,4700,sprintf('%s)',96+ip),'FontSize',18)
    print_mul(sprintf('P%d_training_data_sub',usePrior),1,1,0,600)
    
    
    
    
end
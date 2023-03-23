% Estimation of posterior marginal mean, in Matlab
%% Load data
clear all 
if ~exist('M','var');

    file_training = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1.h5';
    file_sampling = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5';

    N=250000;
    N=min([N 5000000]);

    h=h5info(file_training);
    im=3; % M1- RHO
    %im=7; % M5 Thickness
    %im=8; % M6 Elevation
    
    id=2;
    N_total=h.Datasets(id).Dataspace.Size(2);
    Nd=h.Datasets(id).Dataspace.Size(1);
    Nm=h.Datasets(im).Dataspace.Size(1);
    M_txt = h.Datasets(im).Name;
    D_txt = h.Datasets(id).Name;
    disp(sprintf('Reading %s from %s',M_txt,file_training))
    M_org =  h5read(file_training,sprintf('/%s',M_txt),[1,1],[Nm,N])';
    disp(sprintf('Reading %s from %s',D_txt,file_training))
    D_org =  h5read(file_training,sprintf('/%s',D_txt),[1,1],[Nd,N])';

    h_obs=h5info(file_sampling);
    D_obs_org =  h5read(file_sampling,'/D_obs')';
    if im==3
        M_est_org =  h5read(file_sampling,'/M_est')';
    elseif im==7
        M_est_org =  h5read(file_sampling,'/T_est')';
        M_std_org =  h5read(file_sampling,'/T_std')';
    elseif im==8
        M_est_org =  h5read(file_sampling,'/EL_est')';
        M_est_org =  h5read(file_sampling,'/EL_obs')';
        M_std_org =  M_est_org.*0;
    end    
end

% normalize
% Calculate the mean and standard deviation of the data
M_mean = mean(M_org, 1); % Calculate the mean along the first dimension (rows)
M_std = std(M_org, 0, 1); % Calculate the standard deviation along the first dimension (rows)
D_mean = mean(D_org, 1); % Calculate the mean along the first dimension (rows)
D_std = std(D_org, 0, 1); % Calculate the standard deviation along the first dimension (rows)

M= (M_org - M_mean) ./ M_std;
D= (D_org - D_mean) ./ D_std;
D_obs = (D_obs_org - D_mean) ./ D_std;

perc_val=0.1;
N_val = ceil(N*perc_val);

M_val = M(1:N_val,:);
D_val = D(1:N_val,:);
M_train = M( (N_val+1):end, : );
D_train = D((N_val+1):end, : );


%%
if Nm==1
    Nhidden=4; %Number of hidden layers
    hiddenLayerSize = 20;% Define the number of neurons in each hidden layer
else
    Nhidden=6; %Number of hidden layers
    hiddenLayerSize = 80;% Define the number of neurons in each hidden layer
end


inputSize = Nd;% Define the input size
outputSize = Nm;% Define the output size

% Create the network layers
% Input layer
layers = [
    featureInputLayer(inputSize, 'Name', 'input')
]

% Hidden layer(s)
for il=1:Nhidden
layers = [layers 
    % hidden layer
    fullyConnectedLayer(hiddenLayerSize, 'Name', sprintf('fc%d',il))
    reluLayer('Name', sprintf('relu%d',il))
    ];
end

% Output layer
layers = [layers 
    fullyConnectedLayer(outputSize, 'Name','output')
    regressionLayer('Name','reg_output');  
    %fullyConnectedLayer(2*outputSize, 'Name','output')
    %GaussianNegativeLogLikelihoodLayer('gaussianNLL') % Add the custom loss layer
];

% Display the network layers
disp(layers);
lgraph = layerGraph(layers);


%% Create training options
MaxIteNotImproving=10; % max number of iteration where valiadtion loss is not decreasing..
MiniBatchSize = 128;
if N>=100000, MiniBatchSize = 512;end
if N>=1000000, MiniBatchSize = 1024;end

    
options = trainingOptions('adam', ... % or another optimizer
    'MaxEpochs', 120, ...
    'MiniBatchSize',     MiniBatchSize, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'ExecutionEnvironment','auto', ...                                                                                                                                                                                                 ', ...
    'ValidationData',{D_val,M_val}, ...
    'Plots', 'none', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,MaxIteNotImproving));
%    'Plots', 'training-progress');


%

%arrayDatastore = arrayDatastore(XTrain, YTrain, 'ReadSize', 'default');
%arrayDatastore = arrayDatastore({D_train, M_train}, 'ReadSize', 512);
% Create a custom regression response datastore
%customDatastore = CustomRegressionResponseDatastore(arrayDatastore);
%
% Train the network
%trainedNet = trainNetwork(customDatastore, layers, options);


% Train the network
[net, info] = trainNetwork(D_train, M_train, layers, options);

%%
txt = sprintf('%s_N%06d_Nhid%d_%03d',M_txt,N,Nhidden,hiddenLayerSize);

t1=now;
M_pred  = predict(net, D_obs);
t_est = (now-t1)*3600*24
M_pred = (M_pred.*M_std)+M_mean;


figure(1),clf;
%M= (M_org - M_mean) ./ M_std;
if Nm==1;
    subplot(3,1,[1:2]);
    plot(M_pred,'r-','LineWidth',2)
    hold on
    plot(M_est_org,'k-','LineWidth',1)
    plot(M_est_org+2*M_std_org,'k--','LineWidth',.5)
    plot(M_est_org-2*M_std_org,'k--','LineWidth',.5)
    hold off
    legend({'Prediction','Sampling'})
    ylabel(M_txt)
    grid on
else
    subplot(3,1,1);

    imagesc(10.^M_pred');caxis([1 350])
    imagesc(M_pred');caxis([1 3])
    colormap(cmap_geosoft)
    axis image
    colorbar
end
if Nm==1;
    %subplot(3,1,2);
    %plot(M_pred)
    %ylabel(M_txt)
else
    subplot(3,1,2);
    imagesc(10.^M_est_org');caxis([1 350])
    imagesc(M_est_org');caxis([1 3])
    colormap(cmap_geosoft)
    axis image
    colorbar
    %sgtitle(sprintf('Iteration #%d',i))
end
subplot(3,1,3);
semilogy(info.TrainingLoss,'k-');
hold on
plot(info.ValidationLoss,'r*');
hold off
legend({'Training','Validation'})
ylabel('Loss');xlabel('iteration #')
grid on
drawnow
sgtitle(['Estimation - ',txt],'Interpreter','None')
try;print_mul([txt,'_est']);end


%%
figure(2);clf;try;set_paper('Landscape');end
M_pred_val  = predict(net, D_val);
if Nm==1;
    subplot(1,2,1)
    plot((M_val.*M_std)+M_mean,(M_pred_val.*M_std)+M_mean,'.');
    xlabel('Real')
    ylabel('Predicted')
    axis image;
    grid on
    subplot(1,2,2)
    dd=(M_val.*M_std)+M_mean-((M_pred_val.*M_std)+M_mean);
    histogram(dd);
    text(0.1, 0.9, sprintf('Mean = %3.2f', mean(dd)),'Units','normalized')
    text(0.1, 0.8, sprintf('Std = %3.2f', std(dd)),'Units','normalized')
    xlabel('M_{val}-M_{pred-val}')    
    grid on
    
else
    plot(M_val(1:5,:)','k-','LineWidth',2);
    hold on
    plot(M_pred_val(1:5,:)','-');
    hold off
end
sgtitle(sprintf('Validation - %s',txt),'Interpreter','None')
try;print_mul([txt,'_val']);end

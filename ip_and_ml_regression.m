% Estimation of posterior marginal mean, in Matlab
clear all 
%% Load data
if ~exist('M','var');

    file_training = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1.h5';
    file_sampling = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5';

    N=10000;
    N=min([N 5000000]);
    Nm=125;
    Nd=13;

    h=h5info(file_training);
    M_org =  h5read(file_training,'/M1',[1,1],[Nm,N])';
    D1_org =  h5read(file_training,'/D1',[1,1],[Nd,N])';
    D2_org =  h5read(file_training,'/D2',[1,1],[Nd,N])';

    D_obs_org =  h5read(file_sampling,'/D_obs')';
    M_est_org =  h5read(file_sampling,'/M_est')';
end

% normalize
% Calculate the mean and standard deviation of the data
M_mean = mean(M_org, 1); % Calculate the mean along the first dimension (rows)
M_std = std(M_org, 0, 1); % Calculate the standard deviation along the first dimension (rows)
D_mean = mean(D2_org, 1); % Calculate the mean along the first dimension (rows)
D_std = std(D2_org, 0, 1); % Calculate the standard deviation along the first dimension (rows)

M= (M_org - M_mean) ./ M_std;
D= (D2_org - D_mean) ./ D_std;
D_obs = (D_obs_org - D_mean) ./ D_std;

perc_val=0.1;
N_val = ceil(N*perc_val);

M_val = M(1:N_val,:);
D_val = D(1:N_val,:);
M_train = M( (N_val+1):end, : );
D_train = D((N_val+1):end, : );


%%
Nhidden=6; %Number of hidden layers
hiddenLayerSize = 80;% Define the number of neurons in each hidden layer
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
t1=now;
M_pred  = predict(net, D_obs);
t_est = (now-t1)*3600*24
M_pred = (M_pred.*M_std)+M_mean;


figure(1),clf;
%M= (M_org - M_mean) ./ M_std;
subplot(3,1,1);
imagesc(10.^M_pred');caxis([1 350])
imagesc(M_pred');caxis([1 3])
colormap(cmap_geosoft)
axis image
colorbar
subplot(3,1,2);
imagesc(10.^M_est_org');caxis([1 350])
imagesc(M_est_org');caxis([1 3])
colormap(cmap_geosoft)
axis image
colorbar
%sgtitle(sprintf('Iteration #%d',i))
subplot(3,1,3);
plot(info.TrainingLoss,'k-');
hold on
plot(info.ValidationLoss,'r*');
hold off
legend({'Training','Validation'})
ylabel('Loss');xlabel('iteration #')
grid on
drawnow
sgtitle(sprintf('N=%d',N))


%%
figure(2);clf
M_pred_val  = predict(net, D_val);
plot(M_val(1:5,:)','k-','LineWidth',2);
hold on
plot(M_pred_val(1:5,:)','-');
hold off
%% REGRESSION

%% M1 : RESISTIVITY
clear all;close all
N_sampling=5000000;
ls=1;
useHTX=1;
useHTX_data=1;
useData=2;
useBatchNormalization=1;        
act_arr={'selu','relu'};
act_arr={'relu'};
for iact=1:length(act_arr);
    act=act_arr{iact}
    for i_prior=[1,2,3]
        %useBatchNormalization=1;
        close all
        plt_regression_M1
    end
end
return
%% M5 : 1D continuous (NEED WORK)
clear all;
nu=40;
act='selu';
plt_regression_M5
try
 act='relu';
 plt_regression_M5
end
%% CLASSIFICATION

%% M3 : LAYER INTERFACE
clear all;close all
N_sampling=5000000;
act_arr={'elu','selu','relu'};
for iact=1:length(act_arr);
    act=act_arr{iact}
    for i_prior=[2,3]
        close all;
        plt_classification_M3
    end
end
%% M4
clear all
N_sampling=5000000;
act_arr={'elu','selu','relu'};
%act_arr={'elu'};
for iact=1:length(act_arr);
    act=act_arr{iact}
    for i_prior=[1,2,3]
        close all
        plt_classification_M4
    end
end

%% 1D MARGINAL
clear all;close all
plt_1dpdf

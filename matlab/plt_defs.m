% pl_defs: Some defaults for plotting results

%% SOME DEFAULTS
pdf_types={'Normal','GeneralizedNormal','MixtureNormal2','MixtureNormal3'};


if ~exist('useHTX');useHTX=1;end
if ~exist('useHTX_data');useHTX_data=1;end
if ~exist('useData');
    useData=1; % 'Original' forward ref
    useData=2; % Morrill data
end


if ~exist('i_prior');i_prior=3;end

if ~exist('useRef');useRef=0;end
if ~exist('useBatchNormalization');
    useBatchNormalization=0; % ignore BatchNormaliation
    useBatchNormalization=1;
end 
if ~exist('ls','var');ls=1;end % learning schedule
if ~exist('act');% activation function
    act='relu';
    act='selu';
end 
if ~exist('nu');nu=40;end % number of units in layers

if ~exist('N_sampling')
    %N_sampling=1000000;
    %N_sampling=5000000;
    N_sampling=1000000;
end

plt_txt = sprintf('N%d_ls%d_%s',N_sampling,ls,act);
plt_txt = sprintf('N%d_ls%d_%s_BN%d_HTX%d_%d_D%d',N_sampling,ls,act,useBatchNormalization,useHTX,useHTX_data,useData);

use_prior_types=[23,51,22];
prior_types={'A','B','C'};



%% PLOT SETTINGS
load morrill_model
x=model.X(1,:)/1000;
plot_mov=0;
cmap_rho=jet;
cmap_kl=flipud(hot);
cmap_kl=cmap_geosoft;
cmap_kl=cmap_linear(flipud([0 0 0; 1 0 0;  0 1 0 ; 1 1 1]));
cmap_std=([flipud(gray);hot]);
cmap_prob = flipud(gray);
cmap_prob = cmap_linear([1 1 1; 0.5 .5 .5;1 0 0]);

cax=log10([10 320]);
cax_ytick=log10([10 20 40 80 160 320]);

cax_std=[0 1];
cax_std_ytick=[0,.5,1];



% transparency
%max_std = 1;min_std = 0.4;
max_std = 0.7;min_std = 0.2;
min_std=0.2 ; max_std=1;
    

colors=[
    0 0 0;
    1 0 0;
    0 1 0;
    0 0 1;
    1 0 1;
    0 1 1;
    1 0 1];

colors = copper(6);

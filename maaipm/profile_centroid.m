
clear;
%% load data

% mt is follows an uniform distribution on [base_part-rand_part/2,base_part+rand_part/2]
% m is options.support_size
% N is max_sample_size
global max_sample_size s_modalities d_modalities base_part rand_part;
max_sample_size=50; % the number of distributions
base_part = 100; % the number of the support points of each input distributions
rand_part = 0;
options.support_size=100; % the number of the support points of the barycenter

% if you want to use other data, comment the following line and replace test.d2 file
generate_data;

%% comment out this line to use fixed support points of barycenter
options.support_points= 'fixed';

%maxNumCompThreads(1);

s_modalities = 1;
d_modalities = [3]; % dimension of the support

filename='test.d2';

db_tmp = loaddata(max_sample_size, s_modalities, d_modalities, filename);
db=cell(1,1); db{1}=db_tmp{1}; % only use the first group

%% Additional Configuration
if isfield(options, 'support_points')
    options.init_method='kmeans'; % {'kmeans', 'mvnrnd'}
else
    options.init_method='mvnrnd'; % {'kmeans', 'mvnrnd'}
end
options.max_support_size=options.support_size;

%% Set Initialization
n=length(db);
c0=cell(n,1);
for s=1:n
    c0{s}=centroid_init(db{s}.stride, db{s}.supp, db{s}.w, options);% generate the support points of the barycenter
end

if(isfield(options, 'support_points'))
%% Compute Wasserstein Barycenter (default Gurobi)
options.method='default_gurobi'; % {'ibp','gurobi','badmm', 'admm', 'ibp'}
options.guouttolog = 1;
[c, OT]=Wasserstein_Barycenter(db, c0, options);

%% Compute Wasserstein Barycenter (Pre-specified support MAAIPM)
options.method='fixed_maaipm'; % {'ibp','gurobi','badmm', 'admm', 'ibp'}
options.ipmouttolog = 1;
options.ipmtol_primal_dual_gap = 5e-5;
if options.support_size > 2*(base_part +rand_part) %SLRM/DLRM
    options.largem = 1;
else
    options.largem = 0;
end
[c, OT]=Wasserstein_Barycenter(db, c0, options);
end

%% Compute Wasserstein Barycenter (IBP)
options.method='ibp'; % {'ibp','gurobi','badmm', 'admm', 'ibp'}
for vareps = [0.1,0.01,0.001]
    options.ibp_vareps=vareps;
    options.ibp_max_iters=1e5;
    % set ibp_change_tol higher to have faster performance
    options.ibp_tol=1e-5;
    if(~isfield(options, 'support_points'))
        options.ibp_max_iters=1e6;
        options.ibp_change_tol=1e-4;
        options.ibp_tol=1e-5;
    end

    [c, OT]=Wasserstein_Barycenter(db, c0, options);
end

if(~isfield(options, 'support_points'))
    %% Compute Wasserstein Barycenter (Free support MAAIPM)
        options.method='free_maaipm';
        options.free_MAAIPM_tol = 1e-4;
        options.ipmouttolog = 1;
        [c, OT]=Wasserstein_Barycenter(db, c0, options);
end

%% Compute Wasserstein Barycenter (B-ADMM)
options.method='badmm'; % {'ibp','gurobi','badmm', 'admm', 'ibp'}
options.badmm_tol=1e-5;
options.badmm_max_iters = 3000;
if(~isfield(options, 'support_points'))
    options.badmm_max_iters=2e4;
end
[c, OT]=Wasserstein_Barycenter(db, c0, options);

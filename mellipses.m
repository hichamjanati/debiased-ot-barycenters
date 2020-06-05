% This is a sample of the input format for the Wasserstein_barycenter.m
% It aims to manipulate the data of images (30 x 50 x 50) of nested ellipses
load('data/ellipses.mat', 'ellipses');
addpath('maaipm')
dimensions = size(ellipses);
N = dimensions(1);
width = dimensions(2);
n_features = width ^ 2;
d = 2;
db=cell(1,1);
w = cell(N,1);
db{1}.stride = zeros(1,N);

for i=1:N
    v  = find(ellipses(i,:,:));
    db{1}.stride(i) = length(v);
end

stride_cumsum = [0,cumsum(db{1}.stride)] ;
db{1}.w = zeros(1,stride_cumsum(end));
db{1}.supp = zeros(d,stride_cumsum(end));

for i=1:N
    A = reshape(ellipses(i,:,:),[width,width]);
    [row,col,v]  = find(A);
    db{1}.supp(:,stride_cumsum(i)+1:stride_cumsum(i+1)) = [row';col'];
    db{1}.w(stride_cumsum(i)+1:stride_cumsum(i+1)) = ones(1,db{1}.stride(i))./db{1}.stride(i);
end

c0 = cell(1,1);
c0{1}.w = ones(1,n_features)./n_features;

row = reshape(mod(0:n_features-1,width)+1,1,[]);
col = kron(1:width,ones(1,width));
c0{1}.supp = [row;col];

options.method='fixed_maaipm'; % {'ibp','gurobi','badmm', 'admm', 'ibp'}
options.ipmouttolog = 1;
options.ipmtol_primal_dual_gap = 1e-3;
options.largem = 1;
tic;
[c, OT]=Wasserstein_Barycenter(db, c0, options);
t = toc;

barycenter = reshape(c{1}.w,width,width);
save('data/barycenter.mat','barycenter', 't')


% options.method='ibp'; % {'ibp','gurobi','badmm', 'admm', 'ibp'}
% options.ipmouttolog = 1;
% options.ipmtol_primal_dual_gap = 1e-3;
% options.largem = 1;
% [c, OT]=Wasserstein_Barycenter(db, c0, options);
% barycenter = reshape(c{1}.w,width,width);
% save('data/barycenter-ibp.mat','barycenter');

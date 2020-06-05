function [c, OT] = Wasserstein_Barycenter(db, c0, options)
% Compute Wasserstein barycenter with different methods
% Input:
%  db -- a struct that stores n-phase discrete distribution data.
%           db{1...n} are data associated to different phases;
%           db{i}.stride is an array of the support sizes of individual
%           instance at phase i;
%           db{i}.supp is a matrix storing all support points across
%           instances by colums at phase i;
%           db{i}.w is an array storing all weights across instances at
%           phase i;
%           It is required that
%           length(db{i}.w)==size(db{i}.supp,2) = sum(db{i}.stride).
%  c0 -- a multi-phase discrete distribution as the initial start
%           c0.supp is a matrix storing support points by columns
%           c0.w is an array of weights. It can be automatically generated
%           if it is set to [].
%
%  options --
%          shared:
%           options.support_size: the desired support size of barycenter
%
%           options.support_points: if specified, the support of barycenter
%           is fixed to options.support_points. Then it is required that
%           options.support_size == size(options.support_points, 2)
%
%           options.init_method: {'kmeans', 'mvnrnd'} There are two options
%           for initialization of barycenter, if c0 is empty. One is
%           Kmeans, and the other is multivariate normal.
%
%           options.max_support_size: the maximum desired support size of
%           barycenter, typically set to 3*support_size(c).
%
%           options.method: {'lp', 'gd', 'badmm' (default), 'admm', 'ibp'}
%
%          method specific options:
%           'badmm':
%             options.badmm_max_iters (default, 3000)
%             options.badmm_rho (default, 2.0)
%             options.badmm_tau (default, 10)
%             options.baddm_tol (default, 1E-5)
%           'ibp':
%             options.ibp_max_iters (default, 10000)
%             options.ibp_vareps (default, 0.01)
%             options.ibp_tol (default, 1E-5)
%           'ipm'
%             options.tol_primal_dual_gap (default, 5E-5)
%             options.largem (default, 0)
%             options.itmax (default, 200)
%
% Output:
%  c  -- Wasserstein barycenter
%  OT -- matching matrix between c and each instance

fprintf('\n**********************************************\n')
resultfile = fopen('resultfile.txt','a+');

%max_stride = max(cellfun(@(x) max(x.stride), db));
%kantorovich_prepare(options.max_support_size,max_stride);

method='badmm';
if isfield(options, 'method')
    method = options.method;
end

n=length(db);
c=cell(n,1);
OT=cell(n,1);
for s=1:n
    tic;
    if strcmp(method, 'badmm')
        [c{s}, iter, optval,t,fea]=centroid_sphBregman(db{s}.stride, db{s}.supp, db{s}.w, c0{s}, options);
        fprintf('B_ADMM needs %5.2fs\n',t);
        fprintf(resultfile,'badmm, %d, %4.3f, %2.8f\n',iter, t, optval);
    elseif strcmp(method, 'free_maaipm')
        tic;
        T0=toc;
        [c{s}, iter, optval]=centroid_sphFreeMAAIPM(db{s}.stride, db{s}.supp, db{s}.w, c0{s}, options);
        t=toc;
        t = t - T0;
        fprintf('Free support MAAIPM needs %5.2fs\n',t);
        fprintf(resultfile,'primal IPM, %d, %4.3f, %2.8f\n',iter, t, optval);
    elseif strcmp(method, 'fixed_maaipm')
        [c{s}, iter, optval,t,fea]=centroid_sphFixedMAAIPM(db{s}.stride, db{s}.supp, db{s}.w, c0{s}, options);
        fprintf('Pre-specified support MAAIPM needs %5.2fs\n',t);
        fprintf(resultfile,'FIXIPM, %d, %4.3f, %2.8f\n',iter, t, optval);
    elseif strcmp(method, 'default_gurobi')
        [c{s}, iter, optval,t,fea]=centroid_sphdefaultGurobi(db{s}.stride, db{s}.supp, db{s}.w, c0{s}, options);
        fprintf('Default setting Gurobi needs %5.2fs\n',t);
        fprintf(resultfile,'default_gurobi, %d, %4.3f, %2.8f\n',iter, t, optval);
     elseif strcmp(method, 'ibp')
        [c{s}, iter, optval,t,fea]=centroid_sphIBP(db{s}.stride, db{s}.supp, db{s}.w, c0{s}, options);
        fprintf('IBP needs %5.2fs\n',t);
        fprintf(resultfile,'IBP, %d, %4.3f, %2.8f\n',iter, t, optval);
    end

    toc;
    fclose(resultfile);
end

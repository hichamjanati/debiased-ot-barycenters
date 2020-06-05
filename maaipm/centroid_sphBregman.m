function [c, iter, optval,timeall,fea] = centroid_sphBregman(stride, supp, w, c0, options)

d = size(supp,1);
n = length(stride);
m = length(w);
posvec=[1,cumsum(stride)+1];

if isempty(c0)
    c=centroid_init(stride, supp, w, options);
else
    c=c0;
end
support_size=length(c.w);

X = zeros(support_size, m);
Y = zeros(size(X)); Z = X;
spIDX_rows = zeros(support_size * m,1);
spIDX_cols = zeros(support_size * m,1);
for i=1:n
    [xx, yy] = meshgrid((i-1)*support_size + (1:support_size), posvec(i):posvec(i+1)-1);
    ii = support_size*(posvec(i)-1) + (1:(support_size*stride(i)));
    spIDX_rows(ii) = xx';
    spIDX_cols(ii) = yy';
end
spIDX = repmat(speye(support_size), [1, n]);


% initialization
for i=1:n
    Z(:,posvec(i):posvec(i+1)-1) = 1/(support_size*stride(i));
end
C = pdist2(c.supp', supp', 'sqeuclidean');

nIter = 3000;
if isfield(options, 'badmm_max_iters')
    nIter=options.badmm_max_iters;
end

if isfield(options, 'badmm_rho')
    rho = options.badmm_rho*median(median(pdist2(c.supp', supp', 'sqeuclidean')));
else
    rho = 2.*mean(mean(pdist2(c.supp', supp', 'sqeuclidean')));
end

if isfield(options, 'badmm_tau')
    tau=options.tau;
else
    tau=10;
end

if isfield(options, 'badmm_tol')
    badmm_tol=options.badmm_tol;
else
    badmm_tol=1E-5;
end


tic;
t0=0;

for iter = 1:nIter
    % update X
    X0 = X;
    X = Z .* exp((C+Y)/(-rho)) + eps;
    X = bsxfun(@times, X', w'./sum(X)')';

    % update Z
    Z0 = Z;
    Z = X .* exp(Y/rho) + eps;
    spZ = sparse(spIDX_rows, spIDX_cols, Z(:), support_size * n, m);
    tmp = full(sum(spZ, 2)); tmp = reshape(tmp, [support_size, n]);
    dg = bsxfun(@times, 1./tmp, c.w');
    dg = sparse(1:support_size*n, 1:support_size*n, dg(:));
    Z = full(spIDX * dg * spZ);

    % update Y
    Y0 = Y;
    Y = Y + rho * (X - Z);

    % update c.w
    w0 = c.w;
    tmp = bsxfun(@times, tmp, 1./sum(tmp));
    sumW = sum(sqrt(tmp),2)'.^2; % (R2)
    %sumW = sum(tmp,2)'; % (R1)
    c.w = sumW / sum(sumW);
    %c.w = Fisher_Rao_center(tmp');

    % update c.supp and compute C (lazy)
    if mod(iter, tau)==0 && ~isfield(options, 'support_points')
        c.supp = supp * X' ./ repmat(sum(X,2)', [d, 1]);
        C = pdist2(c.supp', supp', 'sqeuclidean');
    end


    % The constraint X=Z are not necessarily strongly enforced
    % during the update of w, which makes it suitable to reset
    % lagrangian multipler after a few iterations
    if (mod(iter, 10) == 0)

        %          Y(:,:) = 0;
        %           if primres > 10*dualres
        %             rho = 2 * rho;
        %             fprintf(' *2');
        %           elseif 10*primres < dualres
        %             rho = rho / 2;
        %             fprintf(' /2');
        %           end
    end


    % output
    if (mod(iter, 200) == 0)
        Xres = norm(X-X0,'fro')/(1+norm(X,'fro')+norm(X0,'fro'));
        primres = norm(X-Z,'fro')/(1+norm(X,'fro')+norm(Z,'fro'));
        Yres = norm(Y-Y0,'fro')/(1+norm(Y,'fro')+norm(Y0,'fro'));
        wres = norm(w0 - c.w)/(1+norm(w0)+norm(c.w));
        dualres = norm(Z-Z0,'fro')/(1+norm(Z0,'fro')+norm(Z,'fro'));
        runtime = toc;
        runtime = runtime - t0;
        fprintf('\t iter %d: optval = %f, primres = %f, dualres = %f, time = %fs', iter, sum(C(:).*X(:))/n, ...
            primres, dualres, runtime);
        fprintf('\n');
        if max([Xres, primres, Yres, wres, dualres])<badmm_tol
            break;
        end
    end

end

optval = sum(C(:).*X(:))/n;
timeall = toc;

%% fasibility
err1 = norm(sum(X)-w)/(1+norm(X,'fro')+norm(w));
upper = 0;
for i = 2:n+1
    upper = upper + norm(sum(X(:,posvec(i-1):posvec(i)-1)')-c.w)^2;
end
err2 = sqrt(upper)/(1+norm(X,'fro'));
err3 = abs(sum(c.w)-1)/(1+norm(c.w));
fea = max([err1,err2,err3]);
end

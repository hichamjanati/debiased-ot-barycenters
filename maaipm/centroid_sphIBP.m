function [c, iter, optval,timeall, fea] = centroid_sphIBP(stride, supp, w, c0, options)
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
C = pdist2(c.supp', supp', 'sqeuclidean');

nIter = 10000;
if isfield(options, 'ibp_max_iters')
    nIter = options.ibp_max_iters;
end

if isfield(options, 'ibp_vareps')
    rho = options.ibp_vareps * median(median(pdist2(c.supp', supp', 'sqeuclidean')));
else
    rho = 0.01 * median(median(pdist2(c.supp', supp', 'sqeuclidean')));
end

if isfield(options, 'ibp_tol')
    ibp_tol = options.ibp_tol;
else
    ibp_tol = 1E-5; % no updates of support
end

xi=exp(-C / rho);
xi(xi<1e-200)=1e-200; % add trick to avoid program breaking down
xi=sparse(spIDX_rows, spIDX_cols, xi(:), support_size * n, m);
v = ones(m, 1);
u =  0;
w1=w';
fprintf('\n');
obj=Inf;
tol=Inf;

tic;
for iter = 1:nIter

    t0 = toc;
    u_old = u;
    w_old = c.w;
    v_old = v;

    w0=repmat(c.w', n, 1);
    u=w0 ./ (xi*v);
    v=w1 ./ (xi'*u);
    c.w = geo_mean(reshape(u .* (xi * v), support_size, n), 2)';

    if (mod(iter, 10) == 0)
        tol = max([norm(u_old-u)/(1+norm(u_old)+norm(u));...
            norm(w_old-c.w)/(1+norm(c.w)+norm(w_old));norm(v-v_old)/(1+norm(v)+norm(v+v_old))]);
    end

    if tol < ibp_tol || iter>=nIter
        fprintf('iter = %d\n', iter);
        break;
    end

    if  ~isfield(options, 'support_points') && tol < options.ibp_change_tol
        c_back = c;
        X=full(spIDX * spdiags(u, 0, support_size*n, support_size*n) * xi * spdiags(v, 0, m, m));
        c.supp = supp * X' ./ repmat(sum(X,2)', [d, 1]);
        C = pdist2(c.supp', supp', 'sqeuclidean');
        xi=exp(-C / rho);
        xi(xi<1e-200)=1e-200; % add trick to avoid program breaking down
        xi=sparse(spIDX_rows, spIDX_cols, xi(:), support_size * n, m);
        v = ones(m, 1);
        last_obj=obj;
        obj=sum(C(:).*X(:))/n;
        fprintf('\t ibp careps = %f, iter %d: optobj = %f\n', options.ibp_vareps, iter, obj);
        if (obj>last_obj)
            c = c_back;
            fprintf('terminate!\n');
            break;
        end
        tol=Inf;
    end

    t2 = toc;
    if mod(iter,100) ==0
        t2 = t2- t0;
    end
end

if(isfield(options, 'support_points'))
    X=full(spIDX * spdiags(u, 0, support_size*n, support_size*n) * xi * spdiags(v, 0, m, m));
end
optval =sum(C(:).*X(:))/n;
fprintf('\t ibp careps = %f, iter %d: optobj = %f\n' , options.ibp_vareps ,iter,optval );
timeall = toc;

%% feasibility
err1 = norm(sum(X)-w)/(1+norm(X,'fro')+norm(w));
upper = 0;
for i = 2:n+1
    upper = upper + norm(sum(X(:,posvec(i-1):posvec(i)-1)')-c.w)^2;
end
err2 = sqrt(upper)/(1+norm(X,'fro'));
err3 = abs(sum(c.w)-1)/(1+norm(c.w));
fea = max([err1,err2,err3]);
end

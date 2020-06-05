function [IPMx, iter, optval,alltime,fea] = centroid_sphFixedMAAIPM(stride, supp, w, c0, options)
% Calculating Wasserstein Barycenter using Interior point method

N=length(stride);
M=length(w);
m_vec=stride;
m_vec = int64(m_vec) ;
m_vec_cumsum = [0,cumsum(m_vec)] ;


IPMx=c0;

m=length(IPMx.w);
n_row=M+N*(m-1)+1; % The total number of rows of A
n_col=(M+1)*m;     % The total number of columns of A
Mm=M*m;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Constructing b,c
b=zeros( M+N*m+1,1 );
b(1:M) = w';
b(M+N*m+1) = 1;
C = pdist2(IPMx.supp', supp', 'sqeuclidean');
c = [reshape(C,[],1); zeros(m,1)];
b(M+1:m:end-1) =[];% preconditioning;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Formulate the Constraints Coefficient Matrix
row1 = kron( (1:M)' , ones(m,1));
row2 = zeros( M*(m-1) , 1 );
for i=1:N
    index= (m_vec_cumsum(i)*(m-1)+1)  :  ( (m_vec_cumsum(i+1))*(m-1) );
    row2(index ) = kron( ones(m_vec(i) ,1) , ( M+(i-1)*(m-1)+1 : M+i*(m-1) )' );
end
row3 = n_row*ones(m,1);
row = [row1 ; row2 ; row3];

col1 = (1:Mm)' ;
col2 = (1:Mm)' - kron( (m*(1:M)+1-m)' , [1;zeros(m-1,1)] );
col2 = col2(find(col2));
col3 = (n_col -m+1 : n_col)' ;
col = [col1 ; col2 ; col3];

val = ones( Mm + M*(m-1)+m,1 );

A_true = sparse ( row , col , val  );
A_true( M+1: M+N*(m-1), n_col -m+2 : n_col) = -kron( ones(N,1), speye(m-1) );

clear col1 col2 col3 col row1 row2 row3 row val

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters for pathfollowing algorithm
if ~isfield(options,'ipmtol_primal_dual_gap')
    tol = 1e-4;
else
    tol = options.ipmtol_primal_dual_gap;
end

if ~isfield(options,'largem')
    option.largem = 0;
end

itmax = 200;
if  isfield(options,'itmax')
    itmax = option.itmax;
end

maxDiag = 5.e+14;
etaMin = .95;
bc = 1+max([norm(c), norm(b)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization

x = zeros( (M+1)*m, 1 );
for i=1:N
    pi = (1/m)*ones(m,1)*b( m_vec_cumsum(i)+1 : m_vec_cumsum(i+1) )';
    x( m*m_vec_cumsum(i)+1 : m*m_vec_cumsum(i+1) ) = pi(:);
end
x(Mm+1 : m*(M+1) ) = ones(m,1)/m ;

p = [ -ones(M,1) ; zeros(N*(m-1) , 1) ; -1];
s = (c' - p'*A_true)';

time_this=0;
alpha=100;
rel_gap = 999;


%%%%%%%%%%%%%%%%%%%%%%
%IPM begins
tic;
t0 = toc;
for iter=1:itmax
    if iter==1
        Rd = sparse([],[],[],(M+1)*m,1);
        Rp = sparse([],[],[],length(b),1);
        Rc = x.*s;  %X*S*e
        mu = mean(Rc); % complementary measure
        relResidual = sqrt(norm(Rd)^2 + norm(Rp)^2+ norm(Rc)^2) /bc;
        relResidual_old = relResidual ;
        relResidual_old_old = relResidual_old;
        rel_gap = 1e15;
        rel_gap_old = rel_gap;
        rel_gap_old_old = rel_gap_old;
    else
        relResidual_old_old = relResidual_old;
        relResidual_old = relResidual ;
        rel_gap_old_old = rel_gap_old;
        rel_gap_old = rel_gap;
    end


    rel_gap = (c'*x - b'*p)/(abs(b'*p)+abs(c'*x)+1);
    time_old = time_this;
    time_this = toc;
    time_iter = time_this - time_old;
    optval = c(1:end-m)'*x(1:end-m)/N;
    if options.ipmouttolog
        fprintf(1,'iter %2i: optval = %4.2f, mu = %9.2e, resid = %9.2e, rel_gap = %9.2e, time = %9.2e\n',...
            iter,optval, full(mu), full(relResidual) , rel_gap, time_iter );
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    if(relResidual <= tol && mu <= tol && rel_gap<tol)
        break;
    end

    d = min(maxDiag, x./s);
    t1 = x.*Rd-Rc;  % temporary variable 1
    t2 = -(Rp+A_true*(t1./s)); % temporary variable 2: right side of the equation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Solve linear system AD^2A dp == t2;
    if ~options.largem || rel_gap < 5e-4 % DLRM is less stable than SLRM
        first_solve; % get dp
    else
        doublelow_first_solve; % get dp
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dx = ( (A_true'*dp).*x+t1)./s;
    ds = -(s.*dx+Rc)./x;
    eta = max(etaMin,1-mu);

    %(14.21ab)
    alphax = -1/min( (min(dx./x)), -1  );
    alphas = -1/min( (min(ds./s)), -1  );
    %(14.22)
    mu_aff = (x+alphax.*dx)'*(s+alphas.*ds)/n_col;
    %(14.23)
    sigma = (mu_aff/mu)^3;

    Rc = Rc+dx.*ds -sigma*mu;
    t1 = x.*Rd -Rc;
    t2 = -( Rp + A_true*(t1./s) );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% solve linear system AD^2A dp == t2;
    if ~options.largem || rel_gap < 5e-4
        second_solve;% get dp
    else
        doublelow_second_solve;% get dp
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dx = ( (A_true'*dp).*x+t1)./s;
    ds = -(s.*dx+Rc)./x;
    [alpha, alphax, alphas] = steplength(x, s, dx, ds, eta); % steplength
    x = x + alphax * dx;
    s = s + alphas * ds;
    p = p + alphas * dp;


    Rc = x.*s;  %X*S*e
    mu = mean(Rc); % complementary measure
    relResidual  = sqrt(norm(Rd)^2 + norm(Rp)^2+ norm(Rc)^2) /bc;

    if  relResidual_old_old/relResidual<=1.01 && rel_gap_old_old/rel_gap <=1.01 && iter > 50
        fprintf('Cannot be better!\n')
        break;
    end
end


% Rd = A_true'*p+s-c;
% Rp = A_true*x-b;
alltime = toc;
optval = c(1:end-m)'*x(1:end-m)/N;
IPMx.w = x(end-m+1:end);
fprintf('final optimal value of IPM = %f\n', optval);

%% Calculating feasibility error
err1 = norm(A_true(1:M,:)*x-b(1:M))/(1+norm(x(1:end-m))+norm(b(1:M)));
err2 = norm(A_true(M+1:end-1,:)*x-b(M+1:end-1))/(1+norm(x(1:end-m))+norm(b(M+1:end-1)));
err3 = abs(sum(x(end-m+1:end))-1)/norm(x(end-m+1:end));
fea = max([err1,err2,err3]);
end

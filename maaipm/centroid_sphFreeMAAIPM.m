function [IPMx, iter, optval] = centroid_sphFreeMAAIPM(stride, supp, w, c0, options)
%Free support case MAAIPM

tic;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find the base matrix and prepare for eliminating the error
b_test  = rand( M+N*m+1,1 );
b_test(M+1:m:end-1) =[];
x_test = A_true\b_test;
ind  = find(x_test);
A_cor = A_true(:,ind);
clear b_test x_test


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters for pathfollowing algorithm
itmax = 300;
maxDiag = 5.e+14;
etaMin = .9995;

bc = 1+max([norm(c), norm(b)]);
begin_primal = 2;
time_primal_max = 100;
if ~isfield(options,'free_MAAIPM_tol')
    tol = 5e-4;
else
    tol = options.free_MAAIPM_tol;
end

if ~isfield(options,'free_MAAIPM_PDtol')
    PDtol = 1e-3;
else
    PDtol = options.free_MAAIPM_PDtol;
end


%%%%%%%%%%%%%%%%%%%%%%
%% iterations
PDoptval_old = inf;
bigloop = 0;
realloop = 0;
tic;
t0=0;


while (bigloop<=begin_primal + time_primal_max)
    realloop = realloop+1;
    bigloop = bigloop +1;
    if bigloop >= begin_primal + time_primal_max
        % It can be more accurate in the last iteration
        tol = 1e-5;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialization

    x = zeros( (M+1)*m, 1 );
    for i=1:N
        pi = (1/m)*ones(m,1)*b( m_vec_cumsum(i)+1 : m_vec_cumsum(i+1) )';
        x( m*m_vec_cumsum(i)+1 : m*m_vec_cumsum(i+1) ) = pi(:);
    end
    x(Mm+1 : m*(M+1) ) = ones(m,1)/m ;
    p = [ -ones(M,1) ; zeros(N*(m-1) , 1) ; -1];
    s = (c' - p'*A_true)';
    change = 0;


    %% Primal-dual IPM phase
    for iter=1:itmax
        if iter==1
            Rd = sparse([],[],[],(M+1)*m,1);
            Rp = sparse([],[],[],length(b),1);
            Rc = x.*s;  %X*S*e
            mu = mean(Rc);
            relResidual = sqrt(norm(Rd)^2 + norm(Rp)^2+ norm(Rc)^2) /bc;
        else
        end

        rel_gap = (c'*x - b'*p)/(abs(b'*p)+abs(c'*x)+1);
        if(relResidual <= tol/4 && mu <= tol/4 && rel_gap<tol/4)
            break;
        end

        d = min(maxDiag, x./s);
        t1 = x.*Rd-Rc;
        t2 = -(Rp+A_true*(t1./s));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        first_solve; % get dp
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        dx = ( (A_true'*dp).*x+t1)./s;
        ds = -(s.*dx+Rc)./x;
        eta = max(etaMin,1-mu); %

        alphax = -1/min( (min(dx./x)), -1  );
        alphas = -1/min( (min(ds./s)), -1  );
        mu_aff = (x+alphax.*dx)'*(s+alphas.*ds)/n_col;
        sigma = (mu_aff/mu)^3;

        Rc = Rc+dx.*ds -sigma*mu;
        t1 = x.*Rd -Rc;
        t2 = -( Rp + A_true*(t1./s) );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        second_solve;% get dp
        %solve equation AD^2A dp == t2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        dx = ( (A_true'*dp).*x+t1)./s;
        ds = -(s.*dx+Rc)./x;
        [alpha, alphax, alphas] = steplength(x, s, dx, ds, eta);
        x = x + alphax * dx;
        s = s + alphas * ds;
        p = p + alphas * dp;

        %         Rd = A_true'*p+s-c;
        %         Rp = A_true*x-b;
        Rc = x.*s;  %X*S*e
        mu = mean(Rc);
        relResidual  = sqrt(norm(Rd)^2 + norm(Rp)^2+ norm(Rc)^2) /bc;

        % Get a good starting point for thr primal IPM phase
        if bigloop >= begin_primal && mu < 1e-2/(bigloop-begin_primal+1)^(1/2) && change == 0
            changex = x;
            changemu = mu;
            change = 1;
        end
    end


    %% Intermidiate process
    PDoptval = c(1:end-m)'*x(1:end-m)/N;

    time_iter = toc;
    time_iter = time_iter - t0;
    t0 = toc;
    if options.ipmouttolog
        fprintf(1,'loop %2i: optval = %4.2f, time = %9.2e\n',...
            realloop,PDoptval, time_iter );
    end


    if(PDoptval > PDoptval_old)
        % If the decreasing property doesn't hold, terminate
        % the IPM sraightway!
        x = PDx;
        break;
    end


    if bigloop > begin_primal + 1 && PDoptval > PDoptval_old * (1-tol)
        % If the decrease part between primal IPM is too small, terminate
        % the IPM!
        bigloop=max([begin_primal + time_primal_max bigloop]);
    end

    PDx = x;
    PDoptval_old = PDoptval;


    if bigloop < begin_primal
        % Normally, update c and implement primal-dual IPM
        X = reshape(x(1:end-m),m,[]);
        dim = size(supp,1);
        IPMx.supp = supp * X' ./ repmat(sum(X,2)', [dim, 1]);
        C = pdist2(IPMx.supp', supp', 'sqeuclidean');
        c= [reshape(C,[],1); zeros(m,1)];
        PDCoptval = c(1:end-m)'*x(1:end-m)/N;
        if (PDoptval - PDCoptval)/PDCoptval > PDtol && bigloop == begin_primal - 1
            % if the optval is decreasing at a high rate, continue
            % primal-dual IPM phase
            if bigloop < begin_primal
                begin_primal = begin_primal +1;
            end
        end
        continue;
    end

    if (bigloop > begin_primal + time_primal_max)
        break;
    end

    if (bigloop == begin_primal)
        fprintf("Enter the primal IPM phase!\n")
    end

    %% Primal IPM phase

    x = changex;
    mu = changemu;
    rate = 0.65;
    for iter=1:100
        d = min(maxDiag, x.*x);
        t2 = A_true*(x.*(x.*c-mu)); %大户
        first_solve; % get y
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dp = x + (x.*x.*(A_true'*dp - c))/mu;
        dp(ind) = dp(ind) - A_cor\(A_true*dp);
        alpha = min([-0.99/(min(dp./x)),1]);
        x = x +  alpha * dp;
        mu = mu * rate;
        %% change support points
        Poptval_old=c(1:end-m)'*x(1:end-m)/N;
        c_old = c;
        X = reshape(x(1:end-m),m,[]);
        dim = size(supp,1);
        IPMx.supp = supp * X' ./ repmat(sum(X,2)', [dim, 1]);
        C = pdist2(IPMx.supp', supp', 'sqeuclidean');
        c = [reshape(C,[],1); zeros(m,1)];
        Poptval=c(1:end-m)'*x(1:end-m)/N;
        change_c = (norm(c_old-c))/(norm(c_old)+norm(c)+1);
        change_val = (Poptval-Poptval_old)/(1+Poptval+Poptval_old);

        % Terminate the primal IPM phase
        if  (change_c < 1e-2 && change_val <1e-2 &&mu < 1e-6/(bigloop-begin_primal+1))%1e-6
            break;
        end
    end
end

%% After the IPM
X = reshape(x(1:end-m),m,[]);
dim = size(supp,1);
IPMx.supp = supp * X' ./ repmat(sum(X,2)', [dim, 1]);
C = pdist2(IPMx.supp', supp', 'sqeuclidean');
c= [reshape(C,[],1); zeros(m,1)];
optval = c(1:end-m)'*x(1:end-m)/N;
end

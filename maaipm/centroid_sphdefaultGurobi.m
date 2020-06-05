function [Gu_x, iter, optval,alltime,fea] = centroid_sphdefaultGurobi(stride, supp, w, c0, options)
% The algorithmic prototype of Wasserstein Barycenter using gurobi
%
tic;
global temp
N=length(stride);
M=length(w);
m_vec=stride;
m_vec = int64(m_vec) ;
m_vec_cumsum = [0,cumsum(m_vec)] ;

if isempty(c0)
    Gu_x=centroid_init(stride, supp, w, options);
else
    Gu_x=c0;
end
m=length(Gu_x.w);

n_row=M+N*(m-1)+1; % The total number of rows of A
n_col=(M+1)*m;     % The total number of columns of A
Mm=M*m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Constructing b,c
b=zeros( M+N*m+1,1 );
b(1:M) = w';
b(M+N*m+1) = 1;
C = pdist2(Gu_x.supp', supp', 'sqeuclidean');
c = [reshape(C,[],1); zeros(m,1)]; %这里的C有可能是C'
b(M+1:m:end-1) =[];% preconditioning;

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
A_true( M+1: M+N*(m-1), n_col -m+2 : n_col) = -kron( ones(N,1), speye(m-1) );%此处的A_ture做过了行变化，去掉了无关行。

clear col1 col2 col3 col row1 row2 row3 row val
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization

x = zeros( (M+1)*m, 1 );
for i=1:N
    pi = (1/m)*ones(m,1)*b( m_vec_cumsum(i)+1 : m_vec_cumsum(i+1) )';
    x( m*m_vec_cumsum(i)+1 : m*m_vec_cumsum(i+1) ) = pi(:);
end
x(Mm+1 : m*(M+1) ) = ones(m,1)/m ;

p = [ -ones(M,1) ; zeros(N*(m-1) , 1) ; -1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve with Gurobi
model.A = A_true;
model .rhs = b;
model .obj = c;
model.sense = '=';
params.OutputFlag = options.guouttolog;
params.method=2;
% Usually, Gurobi exploits 16 threads
% params.Threads =16; 

result = gurobi(model,params);
gux= result.x;
guval=result.objval;
iter = 0;

Gu_x.w = gux(end-m+1:end);
Guopt = c(1:end-m)'*gux(1:end-m)/N;
fprintf('final optimal value of default gurobi = %f\n', Guopt);
optval = Guopt;
iter = result.baritercount;

alltime = toc;


%% feasibility
x=gux;
err1 = norm(A_true(1:M,:)*x-b(1:M))/(1+norm(x(1:end-m))+norm(b(1:M)));
err2 = norm(A_true(M+1:end-1,:)*x-b(M+1:end-1))/(1+norm(x(1:end-m))+norm(b(M+1:end-1)));
err3 = abs(sum(x(end-m+1:end))-1)/norm(x(end-m+1:end));
fea = max([err1,err2,err3]);
end




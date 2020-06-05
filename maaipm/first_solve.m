
%%%%%%%%%%%%%%%%%%%%%%%% SLRM Solve the first equation %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The first SLRM in MAAIPM

% The following 1 row: reshaping the vector diag(D) into a m x M matrix
d_pile = reshape( d(1:Mm), m, M );

% The following 2 rows: formulating the diagonal matrix B_1, the vector y, and the scalar c; (line 1 of Algorithm 1)
B_1_diag = sum(d_pile )';
B_1_diag_cell = mat2cell(B_1_diag, m_vec, 1)';
cc= sum(d(n_col-m+1:n_col));
y = d(n_col-m+2: n_col)/cc;

% The following 3 rows: computing T = B_2^T B_1^(-1) in the cell formulation; (line 2 of Algorithm 1)
d_pile(1,:)=[];
d_pile_cell = mat2cell( d_pile, m-1, m_vec );
T = cellfun( @(x,y) x./y', d_pile_cell, B_1_diag_cell,  'UniformOutput',0 );

% The following 2 rows: computing A_1^(-1) = diag{ A_11^(-1), ...,A_NN^(-1)}; (line 2 of Algorithm 2)
A_1_inv_cell = cellfun( @(x,y) diag(sum(x,2)) - y*x' , d_pile_cell,T, 'UniformOutput',0  );
A_1_inv_cell = cellfun(  @inv, A_1_inv_cell,  'UniformOutput',false );

%The following 1 row: computing BB = diag(y) - (1/c)yy^T; (line 3 of Algorithm 1)
BB = diag( d(n_col-m+2: n_col) ) - d(n_col-m+2: n_col)*d(n_col-m+2: n_col)'/cc;

% The following 2 rows: computing trN = Y^(-1) + A_11^(-1) + ... + A_NN^(-1) ; ( to be used in line 5 of Algorithm 2)
inv_sum = sum( cat(3,A_1_inv_cell{:} ),3 );
trN = (inv_sum + inv(BB));

ys = kron( ones(N,1), y ); % to be used

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Start solving
xx=t2;

% The following 4 rows: computing xx = V_1 * xx ; (line 4 of Algorithm 1)
tmp = xx(1:M);
tmp_cell = mat2cell(tmp, m_vec, 1)';
tmp_cell = cellfun(@mtimes,T,tmp_cell,'UniformOutput',false  );
xx(M+1 : n_row-1) =xx(M+1 : n_row-1)  - cat(1,tmp_cell{:} );

%The following 1 row: computing  xx = V_2 * xx;   (line 4 of Algorithm 1)
xx(M+1: n_row -1) = xx(M+1: n_row -1) + kron( ones(N,1),y )*xx(n_row);

% The following 1 row: computing  xx(1:M) = B_1^(-1) xx(1:M);  (line 5 of Algorithm 1)
xx(1:M) =  xx(1:M)./B_1_diag;

% The following 1 row: (line 6 of Algorithm 1)
xx(n_row) = xx(n_row)/cc;

% The following 4 rows: computing xx(M+1: n_row-1) = A_1^(-1) * xx(M+1: n_row-1); (line 3 of Algorithm 2)
tilde_x = xx(M+1: n_row-1);
tilde_x_cell = mat2cell(tilde_x, (m-1)*ones(N,1),1 )';
tilde_x_cell = cellfun(@mtimes, A_1_inv_cell, tilde_x_cell, 'UniformOutput',false  );
tilde_x = cat(1,tilde_x_cell{:});

tilde_x_1 = tilde_x;
% The following 3 rows: computing  tilde_x_2 = U^T * tilde_x; (line 4 of Algorithm 2 )
tilde_x_2 = zeros( N*(m-1),1 );
tilde_x_2( 1:(N-1)*(m-1) )=  tilde_x( 1:(N-1)*(m-1) );
tilde_x_2( (N-1)*(m-1)+1: N*(m-1) ) =  sum( reshape( tilde_x, m-1, N), 2 );

%The following 2 rows: (line 5 and 6 of Algorithm 2)
tilde_x_2 = [ zeros((N-1)*(m-1) ,1 )  ; trN\tilde_x_2( (N-1)*(m-1)+1: N*(m-1) )];

%The following 1 row: computing  tilde_x_2 = U*tilde_x_2; (line 7 of Algorithm 2)
tilde_x_2( 1: (N-1)*(m-1) ) = kron( ones(N-1,1), tilde_x_2( (N-1)*(m-1)+1: N*(m-1) ) );

%The following 3 rows: computing  tilde_x_2 = A_1^(-1) * tilde_x_2; (line 7 of Algorithm 2)
tilde_x_2_cell = mat2cell(tilde_x_2, (m-1)*ones(N,1),1 )';
tilde_x_2_cell = cellfun(@mtimes, A_1_inv_cell, tilde_x_2_cell, 'UniformOutput',false );
tilde_x_2 = cat(1,tilde_x_2_cell{:});

%The following 1 row: (line 8 of Algorithm 2)
tilde_x = tilde_x_1 - tilde_x_2;

xx(M+1: n_row-1) = tilde_x;

%The following 1 row: computing xx = V_2^T * xx ; (line 8 of Algorithm 1)
xx(n_row) = xx(n_row)+dot( ys, xx(M+1 : n_row-1) );

%The following 4 rows: computing xx = V_1^T * xx ; (line 8 of Algorithm 1)
tmp_cell = mat2cell(xx(M+1 : n_row-1), (m-1)*ones(N,1),1 )';
tmp_cell = cellfun(@(x,y) x'*y, T,tmp_cell,'UniformOutput',false );
tmp = cat(1,tmp_cell{:});
xx(1:M) = xx(1:M) - tmp;

dp = xx;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End solving %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

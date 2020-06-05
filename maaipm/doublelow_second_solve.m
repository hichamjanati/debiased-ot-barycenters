%%%%%%%%%%%%%%%%%%%%%%%% DLRM Solve the second equation %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The second DLRM in MAAIPM

xx = t2;

% The following 4 rows: computing xx = V_1 * xx ; (line 4 of Algorithm 1)
tmp = xx(1:M);
tmp_cell = mat2cell(tmp, m_vec, 1)';
tmp_cell = cellfun(@mtimes,T,tmp_cell,'UniformOutput',false  );
xx(M+1 : n_row-1) = xx(M+1 : n_row-1)  - cat(1,tmp_cell{:} );

%The following 1 row: computing  xx = V_2 * xx;   (line 4 of Algorithm 1)
xx(M+1: n_row -1) = xx(M+1: n_row -1) + kron( ones(N,1),y )*xx(n_row);

% The following 1 row: computing  xx(1:M) = B_1^(-1) xx(1:M);  (line 5 of Algorithm 1)
xx(1:M) = xx(1:M)./B_1_diag;

% The following 1 row: (line 6 of Algorithm 1)
xx(n_row) = xx(n_row)/cc;

% The following 7 rows: computing xx(M+1: n_row-1) = A_1^(-1) * xx(M+1: n_row-1); (line 3 of Algorithm 2 and Equation (11) )
tilde_x = xx(M+1: n_row-1);
tilde_x_cell = mat2cell(tilde_x, (m-1)*ones(N,1),1 )';
tilde_x_cell_1 = cellfun(@mtimes, B2_B3inv_cell,tilde_x_cell,'UniformOutput',false );
tilde_x_cell_1 = cellfun(@mtimes, center_inv_cell,tilde_x_cell_1,'UniformOutput',false );
tilde_x_cell_1 = cellfun(@(x,y) x'*y, B2_B3inv_cell,tilde_x_cell_1,'UniformOutput',false );
tilde_x_cell = cellfun(@(x,y,z) x./y +z ,  tilde_x_cell, B3_diag_cell, tilde_x_cell_1 ,'UniformOutput',false );
tilde_x = cat(1,tilde_x_cell{:});

tilde_x_1 = tilde_x;
% The following 3 rows: computing  tilde_x_2 = U^T * tilde_x; (line 4 of Algorithm 2 )
tilde_x_2 = zeros( N*(m-1),1 );
tilde_x_2( 1:(N-1)*(m-1) ) = tilde_x( 1:(N-1)*(m-1) );
tilde_x_2( (N-1)*(m-1)+1: N*(m-1) ) = sum( reshape( tilde_x, m-1, N), 2 );

%The following 2 rows: (line 5 and 6 of Algorithm 2)
tilde_x_2 = [ zeros((N-1)*(m-1) ,1 )  ; trN\tilde_x_2( (N-1)*(m-1)+1: N*(m-1) )];

%The following 1 row: computing  tilde_x_2 = U*tilde_x_2; (line 7 of Algorithm 2)
tilde_x_2( 1: (N-1)*(m-1) ) = kron( ones(N-1,1), tilde_x_2( (N-1)*(m-1)+1: N*(m-1) ) );

%The following 6 rows: computing  tilde_x_2 = A_1^(-1) * tilde_x_2; (line 7 of Algorithm 2 and Equation (11) )
tilde_x_2_cell = mat2cell(tilde_x_2, (m-1)*ones(N,1),1 )';
tilde_x_2_cell_1 = cellfun(@mtimes, B2_B3inv_cell,tilde_x_2_cell,'UniformOutput',false );
tilde_x_2_cell_1 = cellfun(@mtimes, center_inv_cell,tilde_x_2_cell_1,'UniformOutput',false );
tilde_x_2_cell_1 = cellfun(@(x,y) x'*y, B2_B3inv_cell,tilde_x_2_cell_1,'UniformOutput',false );
tilde_x_2_cell = cellfun(@(x,y,z) x./y +z ,  tilde_x_2_cell, B3_diag_cell, tilde_x_2_cell_1 ,'UniformOutput',false );
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

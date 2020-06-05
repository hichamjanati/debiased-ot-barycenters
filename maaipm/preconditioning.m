
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preconditioning
row = zeros(M,1);
for k=1:N
    row( m_vec_cumsum(k)+1 : m_vec_cumsum(k+1) ) = (k-1)*m*ones( m_vec(k),1 ) +M+1;
end
col = (1:M)';
val = -ones( M,1 );
L_1 = sparse(row, col, val , M+N*m+1 ,M+N*m+1) + speye( M+N*m+1  );

tilde_L_2 = speye(m); tilde_L_2(1 , 2:m) = ones(1,m-1);
L_2 = blkdiag( speye(M), kron( speye(N) , tilde_L_2 ) , 1 );

L_3 = speye( M+N*m+1);
L_3( M+1:m:M+(N-1)*m+1 , M+N*m+1 ) = ones(N,1);

row = ( 1: M+(N)*(m-1)+1 )';
col2 =kron( ones(N,1) , (2:m)' )+kron( (0:m:(N-1)*m)', ones(m-1,1) ) +M;
col = [ (1:M)' ;col2  ; M+N*m+1 ];
val = ones( M+N*(m-1)+1 , 1 );
P = sparse( row, col , val);

b = L_1*b; b=L_2*b ; b=L_3*b; b = P*b;
clear L_1 L_2 L_3 P
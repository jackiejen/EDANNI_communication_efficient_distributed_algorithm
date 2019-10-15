function [s,w,history] = admm_solver(B, w0, theta, rho, rho1, alpha, gamma, wt, w_old, gt, flag)
%% lasso Solve lasso problem via ADMM
%Solve the subproblem via ADMM:
%
%   minimize -1/n*sum w'*B(1,i,:,:)*B(1,i,:,:)'w + <gt,w-wt> + theta||w||_1
%   + I_{w<=1}(w) + rho/2||w-wt||_2^2
% The solution is returned in the variable s. 
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
% [z, history] = lasso(A, b, lambda, rho, alpha);
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% The code is based on the algorithm in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

% t_start = tic;
% tic
QUIET    = 1;
MAX_ITER = 400; 
ABSTOL   = 1e-11;
RELTOL   = 1e-9;
[m, n, n1, n2] = size(B);
mn = m*n;
B1_avg2 = zeros(n1,n1);
for i = 1:n
    B1_avg2 = B1_avg2 + (-2)/n*squeeze(B(1,i,:,:))*squeeze(B(1,i,:,:))';
end
% save a matrix-vector multiply

s = randn(n1,1); 
s = proj(s, 1);
u= zeros(n1,1); 

% cache the factorization
% [L, U] = factor(A, rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end
for k = 1:MAX_ITER
%     k
    % x-update
    [~,w] = solver2(w0, s, u, B, theta, gamma, 0, rho, rho1, wt, gt, flag);
    obj_try(k) = ISTAlikelihood_ADMMsubprob(w,B, 0, wt) + theta*norm(w,1);
    % z-update with relaxation
    zold = s;
    w_hat = alpha*w + (1 - alpha)*zold;
    s = proj(w_hat + u, 1); 
    
    % u-update
    % u + alpha*x - alpha*z; u~ = u~ + rho*(x - z)
    u = u + (w_hat - s); 
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = obj_try(k);%
%     obj_xx(k)  = objective(A, b, theta, x, x);
    %duality gap
    history.r_norm(k)  = norm(w - s);
    %adjacent decay of z
    history.s_norm(k)  = norm(-rho1*(s - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(w), norm(-s));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho1*u);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    differential_part = B1_avg2'*(s-wt) + rho*(s-wt) + gt;
    if (norm(B1_avg2'*(s-wt) + theta*subgrad(s,differential_part) + rho*(s-wt) + gt) < norm(w_old-wt))||(k>1 && history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
    
end
k
% figure; plot(obj_xx)
% toc
% if ~QUIET
%     toc(t_start);
% end
% end

end

% function p = objective(A, b, lambda, x, s)
%     p = ( 1/2*sum((A*x - b).^2) + lambda*norm(s,1) );
% end

% function z = shrinkage(x, kappa)
%     z = max( 0, x - kappa ) - max( 0, -x - kappa );
% end
function u = subgrad(s,differential_part)
    u = sign(s);
    for i = 1 : length(s)
        if u(i) == 0 && differential_part(i) < 0
            u(i) = 1;
        elseif u(i) == 0 && differential_part(i) > 0
            u(i) = -1;
        end
    end
end
function z = proj(x, kappa)
    if norm(x) <= kappa
        z = x;
    else
        z = x/norm(x);
    end
end

function [L, U] = factor(A, rho)
    [m, n] = size(A);
    % if skinny
    if ( m >= n )    
       % return a lower triangular matrix s.t. L*L' = A
       L = chol( A'*A + rho*speye(n), 'lower' ); 
     % if fat   
    else           
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end
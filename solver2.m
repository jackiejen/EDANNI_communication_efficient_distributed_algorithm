function [ fn, wn, ferror, iter, xdiff, funcdiff ] = solver2( w0, s, u, B, theta, gamma, accel, rho, rho1, wt, gt, flag)
% The function is used to solve the subproblems in the admm_solver 
%%
% tic
[m, n, n1, n2] = size(B);
if size(wt,1) ~= n1
    error('B, w are not of appropriate dimensions')
end

if gamma <= 0 | gamma >= 1,
    error('gamma not in (0,1)')
end

if theta < 0,
    error('lambda must be nonnegative')
end

%
%INITIALIZE
%
iter        = 0;
xdiff       = 1;
funcdiff    = 1;
wcurrent    = wt;
stepsize    = 1/(rho+rho1);
if flag ~= 0 
    max_iter = 6;
else
    max_iter = 1;
end 
mn = m*n;
%
% RUN UPDATE LOOP
%%
while iter < max_iter && xdiff > 1e-10 && funcdiff > 1e-8
    %---------------------------update the gradient------------------------
    grad_sum2 = zeros(n1,1);
    for i = 1:n
        grad_sum2 = grad_sum2 + (-2)/n*squeeze(B(1,i,:,:))*squeeze(B(1,i,:,:))'*wcurrent;
    end
    grad_sum1 = gt; 
    % compute the overall gradient
    gradvalue = grad_sum1 + grad_sum2 + rho*(wcurrent - wt) + rho1*u + rho1*(wcurrent - s);
    %-----------------------------update w---------------------------------
%     wprevious = wcurrent;
%     wcurrent  = softThreshold(wprevious - stepsize*gradvalue, theta*stepsize);
    wprevious               = wcurrent;  
    [~, wcurrent]    = backtrack_prox2(B,wcurrent, s, u, theta, gamma, stepsize,gradvalue,gt, rho, rho1, wt);
    
    %update terminating values;
    funcdiff    = ISTAlikelihood_ADMMsubprob(wprevious,B(1,:,:,:),rho,wt)  + gt'*(wprevious - wt) + rho/2*norm(wprevious - wt)^2 + rho1*u'*(wprevious - s) + rho1/2*norm(wprevious - s)^2 ...
        + theta*norm(wprevious,1) - ISTAlikelihood_ADMMsubprob(wcurrent,B(1,:,:,:),rho,wt) - gt'*(wcurrent - wt) - rho/2*norm(wcurrent - wt)^2 - rho1*u'*(wcurrent - s) ...
        - rho1/2*norm(wcurrent - s)^2 - theta*norm(wcurrent,1);
    if ~accel 
        searchPoint = wcurrent;
    else
        searchPoint = wcurrent + iter / (iter + 2) * (wcurrent - wprevious);
    end
%     
    iter        = iter + 1;
    xdiff       = norm(wcurrent - wprevious);
end
% iter

wn = wcurrent;
ferror = ISTAlikelihood_ADMMsubprob(wcurrent,B,rho,wt);
fn = ferror + theta*norm(wcurrent,1);
% toc

end


% function gt = ISTA_grad_sum(grad_sum1, B1 ,wt, wcurrent, s, u, rho, rho1, n1, n)
% % B1i =  squeeze(B(1,i,:,:))
%     grad_sum2 = zeros(n1,1);
%     for i = 1:n
%         grad2(:,i) = squeeze(B1(i,:,:))*squeeze(B1(i,:,:))'*wt + squeeze(B1(i,:,:))*squeeze(B1(i,:,:))'*wcurrent;
%         grad_sum2 = grad_sum2 + (-2)/(n)*grad2(:,i);
%     end % e
%     gt = grad_sum1 + grad_sum2 + rho*(wcurrent - wt) + rho1*u + rho1*(wcurrent - s); % gt is very small
% end



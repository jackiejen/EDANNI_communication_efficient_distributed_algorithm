function EDANNI_example
% This code is based on EDANNI in the paper "A Provably Communication-Efficient Asynchronous
% Distributed Inference Method for Convex and Nonconvex Problems"
% It solves an example of the sparse PCA problem. The framework of the algorithm for 
% the LASSO problem is similar with a different subproblem, which can be solved by the same solver.
%%
clear all
m = 6; 
n = 15; 
n1 = 100; 
n2 = 40;
% sparsity density
p1 = 200/(n1*n2);      
tau = 0;
A = floor(0.1*m);
% probability to update
probu = 0.2;
probu1 = 0.3;
gamma = 0.5;
rho1 = 1;
% Choose EDANNI
Danni_flag = 1;
max_iter = 500;
% starting point of FISTA
rng(1)
w0 = randn(n1,1);
w0 = w0/norm(w0);
% regularization parameter 
theta = 0.1;
tic
%===================generate B and define rho============================
mat_sum = zeros(n1,n1);
mn = m*n;
k = 30;
rng(k)
dirname=['Syn_compare_conv',num2str(m),'_',num2str(n),'_',num2str(k),'_n1_',num2str(n1),'_n2_',num2str(n2)];
cmd_mkdir=['mkdir ' dirname];
if ~exist(dirname, 'dir')
    system(cmd_mkdir); 
end
for j = 1:m
    for i = 1:n
        B(j,i,:,:) = full(sprandn(n1, n2, p1));
        mat_sum = mat_sum + (-1)/n/2* squeeze(B(j,i,:,:))*squeeze(B(j,i,:,:))';
    end
end
rho = 2*abs(eigs(mat_sum,1)); 
w(:,1) = w0;
obj(1) = ISTAlikelihood_ADMMsubprob(w(:,1),B, 0, w(:,1)) + theta*norm(w0,1);
for j = 1:m
    wj(:,j,1) = w(:,1); 
end
w_old = w; 
%%
%--------------------------- update beta -------------------------------
d = zeros(m,1);
all_idx = (1:m)';
t = 1;
% grad_sum = norm(squeeze(reshape(X,[m*n*p,1])));
while (t <= max_iter && (t<=1 || norm(obj(t) - obj(t-1))>=1e-13)) 
    fprintf('objective value at iteration %d: %.10f \n', t, obj(t))
    At =[];
    At = find(d >= tau);
    Atc = setdiff(all_idx, At);
    while(length(At) <= A)
        l_half = floor(length(Atc)/2);
        rnd = rand(l_half,1); 
        rnd1 = rand(length(Atc) - l_half,1); 
        % randomly select idx to be updated; 
        idx_new = Atc(find(rnd>probu)); 
        idx_new1 = Atc(find(rnd1>probu1)) + l_half; 
        At = [At;idx_new;idx_new1];
        Atc = setdiff(all_idx, At);
    end
    %==========================update gt==================================
    grad_sum1 = zeros(n1,1);
    for i = 1:n
        for j = 1:m
            grad_sum1 = grad_sum1 + (-2)/(mn)*squeeze(B(j,i,:,:))*squeeze(B(j,i,:,:))'*wj(:,j,t);
        end
        grad_sum1 = grad_sum1 + 2/n*squeeze(B(1,i,:,:))*squeeze(B(1,i,:,:))'*wj(:,j,t);
    end % e
    gt = grad_sum1;
    %beta(:,t+1) = SpaRSA(b,A,lambda); used to be ISTA(x0...
    w_try = admm_solver(B, w0, theta, rho, rho1, 1, gamma, w(:,t), w_old, gt, Danni_flag);
    obj_try = ISTAlikelihood_ADMMsubprob(w_try,B, 0, w(:,t)) + theta*norm(w_try,1);
    %obj_try itself
    [obj(t+1),idx_try] = min(obj_try);
    w_old = w(:,t);
    w(:,t+1) = w_try(:,idx_try);
    
   %-----------------------------------------------------------------------
   %========================do the update of wj===========================
    for i=1:length(At)
        d(At(i)) = 0;
        % update to the newest w(:,t+1)
        wj(:,At(i),t+1) = w(:,t+1); 
    end
    for i=1:length(Atc)
        d(Atc(i)) = d(Atc(i)) + 1;
        %===================keep the last value======================
        wj(:,Atc(i),t+1) = wj(:,Atc(i),t);
    end
    %----------------------------------------------------------------------
    t = t+1;
    if obj(t-1) - obj(t)<8e-12 && obj(t-1) - obj(t) > 0
        break;
    end
end
w1 = w(:,t);
toc
save([dirname,'\variable_EDANNI_',int2str(max_iter)], 'w', 'obj', 'B', 'w0', 'theta', 'rho', 'rho1', 'gamma', 'gt', 'Danni_flag')
end
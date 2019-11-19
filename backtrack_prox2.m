function [ newStep, trialPoint ] = backtrack_prox2(B,xprevious, s, u, theta, gamma, stepsize, gradvalue, gt, rho, rho1, wt)
% Simple backtracking method used in solver2.m algorithm 
% Do backtracking to choose a good step and compute the corresponding P_L(x_(k-1))
n1 =  size(B,3);
B1 = B(1,:,:,:);

numTrials   = 0;
newStep     = stepsize;
% newStep
trialPoint  = softThreshold(xprevious - newStep*gradvalue, theta*newStep);
% Taylor expansion
difference  = ISTAlikelihood_ADMMsubprob(trialPoint, B1, rho, wt) + gt'*(trialPoint - wt) + rho/2*norm(trialPoint - wt)^2+ rho1*u'*(trialPoint - s)+rho1/2*norm(trialPoint - s)^2 ...
    - ISTAlikelihood_ADMMsubprob(xprevious,B1, rho, wt) - gt'*(xprevious - wt)- rho/2*norm(xprevious - wt)^2 - rho1*u'*(xprevious - s) - rho1/2*norm(xprevious - s)^2 ...
    - (trialPoint-xprevious)'*gradvalue-1/(2*newStep)*norm(trialPoint-xprevious)^2;
while numTrials < 0 && difference > 5e-4,
    newStep = newStep*gamma;
    numTrials = numTrials + 1;
    % compute P_L(x_(k-1)) L is the new step
    trialPoint  = softThreshold(xprevious - newStep*gradvalue, theta*newStep);
    difference  = ISTAlikelihood_ADMMsubprob(trialPoint, B1, rho, wt) + gt'*(trialPoint - wt) + rho/2*norm(trialPoint - wt)^2 + rho1*u'*(trialPoint - s) + rho1/2*norm(trialPoint - s)^2 ...
    - ISTAlikelihood_ADMMsubprob(xprevious, B1, rho, wt) - gt'*(xprevious - wt) - rho/2*norm(xprevious - wt)^2 - rho1*u'*(xprevious - s) - rho1/2*norm(xprevious - s)^2 - (trialPoint-xprevious)'*gradvalue-1/(2*newStep)*norm(trialPoint-xprevious)^2;
%     difference
end
% difference
% numTrials
% newStep
% trialPoint

if numTrials == 100,
    error('backtracking failed')
end

end

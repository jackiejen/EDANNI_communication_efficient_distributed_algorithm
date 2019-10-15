function [ value ] = ISTAlikelihood_ADMMsubprob( w, B, rho, wt)
%This function implements the different likelihood function that we see in
%the ISTA method.
m = size(B,1);
n = size(B,2);
qua_sum = 0;
mn = m*n;
% size(squeeze(B(1,1,:,:)))
% size(w)
for ii = 1:n
    for jj = 1:m
        qua_sum = qua_sum + (-1)/mn*w'*squeeze(B(jj,ii,:,:))*squeeze(B(jj,ii,:,:))'*w;
    end
end

value = qua_sum;

end


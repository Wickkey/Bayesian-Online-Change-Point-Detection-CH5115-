load('NMRlogWell.mat');

N = length(y);

%Given data.
lambda = 250;
hazard = 1/lambda;
mu0 = 1.15;
k = 0.01;
alpha = 20;
beta = 2;

%Initializing few useful values.
scale = beta*(k+1)/(alpha*k);
params = zeros(N+1,5);
params(1,:) = [alpha, beta, k, mu0, scale];
message = ones(N+1,N+1);
run_length = zeros(N,1);
% nth row of params stores the following, 1->alpha_n, 2->beta_n, 3->k_n,
% 4->mu_n, 5->(scale parameter)^2 for t-distribution.

for i=1:N
    x = y(i);
    
    %UPM predictive is a Non-central t-distribution.
    UPM = pdf('tLocationScale', x, params(1:i,4), params(1:i,5).^0.5, 2*params(1:i,1));
    
    growth_prob = message(i,1:i)'.*UPM*(1 - hazard);
    change_prob = sum(message(i,1:i)'.*UPM*hazard);
    normalizer = sum(growth_prob) + change_prob;
    prob_vec = [change_prob/normalizer; growth_prob/normalizer];
    
    [~,run_length(i)] = max(prob_vec);
    if i==N
        break;
    end
    params(1:i+1,:) = update_params(x,params,i);
    message(i+1,1:i+1) = prob_vec;
end

plot(1:N, run_length,'b-','LineWidth',1.5);
set(gca,'fontsize',12,'fontweight','bold');
xlabel('Data index','fontsize',13,'fontweight','bold');
ylabel('Run Length','fontsize',13,'fontweight','bold');
title('Run Length Prediction','fontsize',14,'fontweight','bold');


% function post_val = UPM_Pred(x,params,n)
%     post_val = zeros(n,1);
%     
%     for j=1:n
%         helparam = params(j,1) * params(j,3)/(params(j,2) * (params(j,3)+1));
%         power_term = -(2*params(j,1) + 1)/2;
%         post_val(j) = log(pi^-0.5) + (gammaln((2*params(j,1)+1)/2)- gammaln(params(j,1)))...
%             + 0.5*log(helparam/(2*params(j,1)))...
%             * power_term*log(1 + helparam*((x - params(j,4))^2)/(2*params(j,1)));
%     end
%     % params has 4 values. 1->alpha_n, 2->beta_n, 3->k_n, 4->mu_n
%     % Returns log(UPM) values -> each value in the vector is log of the
%     % actual value.
% end

function new_params = update_params(x,params,n)
    new_params = zeros(n+1,5);
    new_params(1,:) = params(1,:);
    for i=2:n+1
        new_params(i,1) = params(i-1,1) + 0.5;
        new_params(i,3) = params(i-1,3) + 1;
        new_params(i,4) = (params(i-1,4)*params(i-1,3) + x)/new_params(i,3);
        new_params(i,2) = params(i-1,2) + params(i-1,3)*((x-params(i-1,4))^2)/(2*(params(i-1,3) + 1));
        new_params(i,5) = new_params(i,2)*(new_params(i,3)+1)/(new_params(i,1)*new_params(i,3));
    end
end
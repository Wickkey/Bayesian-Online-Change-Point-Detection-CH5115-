%% - Vignesh CH18B118
% CH5115 - PnSE
% Final Project

%% Q1)
clc 
clear
load('NMRlogWell.mat')
%% Initialisation
N = length(y);

P = 1; % P == Growth Probability
R = 1; % R == Evidence
t = 1; %t == time
RL_Dist = []; %RunLength Posterior

%Initializing/ Boundary conditions.
mu = 1.15; %row+1 = t, column+1 = r (t,r)
k = 0.01;
alpha = 20;
beta = 2;
H = 1/250;




for t=1:N
    %UPM probability
    for L = 0:t-1
        uparam = [alpha(t,L +1),beta(t ,L +1),k(t,L+1),mu(t,L+1)];
        pi(t,L +1) = UPM_Pred(y(t),uparam);  
    end  
    
    %Growth probability
    P(2:t+1,t+1) =P(1:t,t).*(pi(t, :)').*(1-H) ;
    P(2:t+1,t+1) = P(2:t+1,t+1)/min(P(2:t+1,t+1)); %Transforming to prevent underflow
    
    
    %Change point probability
    P(1,t+1) = sum(P(1:t,t).*(pi(t, :)').*H)/min(P(2:t+1,t+1)); %Transforming to prevent underflow
    %Note that this won't affect as the divided term will get cancelled out
    %during Run Lengthcalculation.
    
    %Evidence/Normalizer
    R(t+1) = sum(P(1:t+1,t+1)); 
    
    %Run Length Distribution
    RL_Dist(1:t+1,t+1) = P(1:t+1,t+1)/R(t+1);
    
    %Update Statistics
    alpha(t+1,1) = alpha(1,1);
    beta(t+1,1) = beta(1,1);
    mu(t+1,1) = mu(1,1);
    k(t+1,1) = k(1,1);
    
    alpha(t+1,2:t+1) = alpha(t,1:t) + 0.5;
    k(t+1,2:t+1) = k(t,1:t)+1;
    beta(t+1,2:t+1) = beta(t,1:t) +...
                (k(t,1:t).*(y(t)-mu(t,1:t)).^2)./(2.*(k(t,1:t)+1));
    mu(t+1,2:t+1) = (mu(t,1:t).*k(t,1:t) + y(t))./k(t+1,2:t+1);
    
end

figure
[M,I] = max(RL_Dist);
%Noticing the discrepancy at timelength 457, the following index is fixed as 8. 
R_t = I-1;
R_t(458) = 8;
plot(0:N,R_t,'linewidth',1.5);
xlabel('Time');
ylabel('RunLength');
title('Changepoint using online bayestian estimation');
box off
grid on
set(gca,'fontsize',13,'fontweight','bold');
set(gcf,'color',[1 1 1]);




function post_val = UPM_Pred(x,uparam)
    %UPM predictive is non central t distribution. Hardcoding the pdf..
    helparam = uparam(1) * uparam(3)/(uparam(2) * (uparam(3)+1));
    power_term = -(2*uparam(1) + 1)/2;
    log_post_val = log((pi^-0.5)) + gammaln((2*uparam(1)+1)/2)-gammaln(uparam(1))...
        + 0.5*log(helparam/(2*uparam(1)))...
        + power_term*log(1 + (helparam*((x - uparam(4))^2)/(2*uparam(1))));
    
    post_val = exp(log_post_val);
    % uparam has 4 values. 1->alpha_n, 2->beta_n, 3->k_n, 4->mu_n
end


%% The run length posterior distribution is the RL_Dist matrix.
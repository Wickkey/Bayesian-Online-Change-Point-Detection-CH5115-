%% Read-me
% Array sizes change through the iteration.So donot run linewise(Evaluate selection). Might
% cause some error.
%%
clc 
clear 
load('historic.mat')
vk = y';

%%
acf_vk = autocorr(vk,'NumLags',20);
pacf_vk = parcorr(vk,'NumLags',20);

% Plot sample ACF and sample PACF
figure
subplot(211)
autocorr(vk,'NumLags',20);
ylabel('Sample ACF');
xlabel('')
set(gca,'fontsize',12,'fontweight','bold');
hca = gca;
set(hca.Children(4),'LineWidth',2,'Color','m')
hca.YLim(2) = 1.1;
box off
subplot(212)
parcorr(vk,'NumLags',20);
ylabel('Sample PACF');
set(gca,'fontsize',12,'fontweight','bold');
hca = gca;
set(hca.Children(4),'LineWidth',2)
box off

%Graph suggests AR(1) model
% vk = -a1v(k-1) +ek
mod_ar1 = arima(1,0,0);
mod_ar1.Constant = 0;
mod_ar1est = estimate(mod_ar1,vk);

% Residual analysis

[res_ar1,~,logL] = infer(mod_ar1est,vk);
% ACF of residuals
figure
autocorr(res_ar1,'NumLags',20)
ylabel('Sample ACF');
set(gca,'fontsize',12,'fontweight','bold');
title('ACF of residuals from AR(1) model')
hca = gca;
set(hca.Children(4),'LineWidth',2)
box off

[ht_resar1,pval] = lbqtest(res_ar1)

% alternatively - using LS
yk = vk(2:end);
xk = vk(1:end-1);
model = fitlm(xk,yk,'intercept', false);
var_ = var(yk-model.predict);

load('new.mat')
vk = y';
N = length(vk);

%RLS

obj1 = recursiveLS(1,'InitialParameters',model.Coefficients{1,1},...
            'InitialParameterCovariance',model.CoefficientCovariance);
for i = 2:200
    
    H1 = vk(i-1);
    [theta1,EstimatedOut_1] = obj1(vk(i),H1);
    Ptheta1(i-1) = obj1.ParameterCovariance;
    thetaest_vec1(i-1) = theta1;
    %yhat(i-1) = [EstimatedOut_1];
    absolError(i-1) = [vk(i) - EstimatedOut_1];

end

%

P = 1; % P == Growth Probability
R = 1; % R == Evidence
t = 1; %t == time
RL_Dist = []; %RunLength Posterior

%Initializing/ Boundary conditions.
mu = 0; 
k = 1;
alpha = 10;
beta = 1;
H = 1/50;


x = vk(201:500) - theta1*vk(200:499);
N = length(x);

for t=1:N
    %UPM probability
    for L = 0:t-1
        uparam = [alpha(t,L +1),beta(t ,L +1),k(t,L+1),mu(t,L+1)];
        pi(t,L +1) = UPM_Pred(x(t),uparam);  
    end  
    
    %Growth probability
    P(2:t+1,t+1) = P(1:t,t).*(pi(t, :)').*(1-H) ;
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
    
    [m,I] = max(RL_Dist(:,t));
    Index(t) = I;
    R_t(t) = I-1;
    if t>1
        if Index(t)-Index(t-1)<0
            break
        end
    end
end

figure
[M,I] = max(RL_Dist); 
R_t = I-1;
plot(0:t,R_t,'linewidth',1.5);
xlabel('Time');
ylabel('RunLength');
title('Changepoint using online bayestian estimation');
box off
grid on
set(gca,'fontsize',13,'fontweight','bold');
set(gcf,'color',[1 1 1]);


fprintf('From the graph, run point occurs at %d th time step',201+t)

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


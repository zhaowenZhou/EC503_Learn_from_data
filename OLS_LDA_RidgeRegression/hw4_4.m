% ENG EC 503 (Ishwar) Fall 2021
% HW 4
% <Zhaowen Zhou zzwzhou@bu.edu>
% HW 4_4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc
load('prostateStnd.mat');

%% Normalize train data
Xtrain_mean = mean(Xtrain); % mean of each feature
Xtrain_std = std(Xtrain); % std of each feature
Ytrain_mean = mean(ytrain);
Ytrain_std = std(ytrain);
Zxtrain = zeros(67, 8);
Zytrain = (ytrain(:,1) - repmat(Ytrain_mean,67,1))/Ytrain_std;
for i = 1:1:8
    Zxtrain(:,i) = (Xtrain(:,i)-repmat(Xtrain_mean(:,i), 67, 1))/Xtrain_std(:,i);
end


%% Train ridge regressiong model
% lambda = e^-5 ...... e^10
lambda_array = zeros(16, 1);
Zx_mean = zeros(1, 8);
C = eye(67) - (ones(67, 1)*ones(67, 1).')/67;
Szx = Zxtrain.'*C*Zxtrain;
Szxy = Zxtrain.'*C*ytrain;
Wridge_array = zeros(8,16);
for k = 1:1:16
    lambda_array(k,1) = exp(k-6);
    Wridge_array(:,k) = (((lambda_array(k,1)/67)*eye(8) + Szx)^-1)*Szxy;
end
bridge_array = zeros(1, 16);
for k = 1:1:16
    bridge_array(1, k) = mean(Zytrain) - (Wridge_array(:,k).')*mean(Zxtrain).';
end


%% plot the coefficients versus lnlambda
figure(1);
hold on;
for k=1:8
    plot(-5:10,Wridge_array(k,:), 'LineWidth',2);
end
title('Ridge coefficients as sa function of ln(lambda)', ...
    'FontSize', 16);
xlabel('ln(lambda)');
ylabel('Coefficients');
legend(names(1:end-1));


%% Plot the MSE versus lnlambda
Zxtest = zeros(30, 8);
Zytest = (ytest(:,1) - repmat(Ytrain_mean,30,1))/Ytrain_std;
for i = 1:1:8
    Zxtest(:,i) = (Xtest(:,i)-repmat(Xtrain_mean(:,i), 30, 1))/Xtrain_std(:,i);
end
MSEtest_array = zeros(1, 16);
for k = 1:16
    mse = 0;
    for t = 1:30
        y_hat = Wridge_array(:,k).'*Zxtest(t,:).' + bridge_array(1,k);
        mse_yhat = (y_hat - Zytest(t,1))^2;
        mse = mse + mse_yhat;
    end
    MSEtest_array(1, k) = mse;
end
MSEtest_array = MSEtest_array/30;
MSEtrain_array = zeros(1, 16);
for k = 1:16
    mse_train = 0;
    for t = 1:67
        y_hat_train = Wridge_array(:,k).'*Zxtrain(t,:).' + bridge_array(1,k);
        mse_yhat_train = (y_hat_train - Zytrain(t,1))^2;
        mse_train = mse_train + mse_yhat_train;
    end
    MSEtrain_array(1, k) = mse_train;
end
MSEtrain_array = MSEtrain_array/67;
figure(4);
hold on;
plot(-5:10,MSEtrain_array, 'LineWidth',2);
plot(-5:10,MSEtest_array, 'LineWidth',2);
title('Ridge MSE versus ln(lambda)', ...
    'FontSize', 18);
xlabel('ln(lambda)');
ylabel('MSE');
legend('training MSE', 'test MSE', 'Location','southeast');
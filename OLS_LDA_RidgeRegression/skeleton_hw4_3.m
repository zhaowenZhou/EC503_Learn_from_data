%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2021
% HW 4
% <Zhaowen Zhou zzwzhou@bu.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50; % numbers of examples
n2 = 100;
n = n1+n2;
mu1 = [1; 2]; % mean vector for class
mu2 = [3; 2];

% Generate dataset (i) 

% lambda1 = 1;
% lambda2 = 0.25;
% theta = 0*pi/6;
% [X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
% X1 = X(:, Y==1);
% X2 = X(:, Y==2);
% 
% figure(1);subplot(2,2,1);
% scatter(X1(1,:),X1(2,:),'o','fill','b');
% grid;axis equal;hold on;
% xlabel('x_1');ylabel('x_2');
% title(['\theta = ',num2str(0),'\times \pi/6']);
% scatter(X2(1,:),X2(2,:),'^','fill','r');
% axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii)
lambda1 = 1;
lambda2 = 0.25;
theta = pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,2);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%datasets(iii)
% theta = pi/3;
% [X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
% X1 = X(:, Y==1);
% X2 = X(:, Y==2);
% 
% figure(1);subplot(2,2,3);
% scatter(X1(1,:),X1(2,:),'o','fill','b');
% grid;axis equal;hold on;
% xlabel('x_1');ylabel('x_2');
% title(['\theta = ',num2str(2),'\times \pi/6']);
% scatter(X2(1,:),X2(2,:),'^','fill','r');
% axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%datasets(iv)
% lambda1 = 0.25;
% lambda = 1;
% theta = pi/6;
% [X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
% X1 = X(:, Y==1);
% X2 = X(:, Y==2);
% 
% figure(1);subplot(2,2,4);
% scatter(X1(1,:),X1(2,:),'o','fill','b');
% grid;axis equal;hold on;
% xlabel('x_1');ylabel('x_2');
% title(['\theta = ',num2str(1),'\times \pi/6']);
% scatter(X2(1,:),X2(2,:),'^','fill','r');
% axis equal;


%% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal_power;
    noise_power_array(i) = noise_power;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
figure(2); subplot(2, 2, 1);
plot(phi_array, signal_power_array)
[max_power, signal_index] = max(signal_power_array);
argmax_signal = phi_array(signal_index);
xlabel('phi');
ylabel('signal power');
title('\signal power versus phi, the argmax signal power is', num2str(argmax_signal));

figure(2); subplot(2, 2, 2);
plot(phi_array, noise_power_array)
[min_noise, noise_index] = min(noise_power_array);
argmin_noise = phi_array(noise_index);
xlabel('phi');
ylabel('noise power');
title('\noise power versus phi, the argmin noise power is', num2str(argmin_noise));

figure(2); subplot(2, 2, 3);
plot(phi_array, snr_array)
[max_snr, snr_index] = max(snr_array);
argmax_snr = phi_array(snr_index);
xlabel('phi');
ylabel('snr');
title('\snr versus phi, the argmax snr is', num2str(argmax_snr));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 

signal_noise_snr(X, Y, 0, true)
signal_noise_snr(X, Y, pi/6, true)
signal_noise_snr(X, Y, pi/3, true)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);

% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
meanx2 = [sum(X(1, (n1+1):(n1+n2)))/n2; sum(X(2, (n1+1):(n1+n2)))/n2];
meanx1 = [sum(X(1, (1):(n1)))/n1; sum(X(2, 1:(n1)))/n1];
arrow_x = meanx1(1);
arrow_y = meanx1(2);
arrow_u1 = meanx2(1) - meanx1(1);
arrow_v1 = meanx2(2) - meanx1(2);
arrow_u2 = w_LDA(1) + meanx1(1);
arrow_v2 = w_LDA(2) + meanx1(2);

figure(9)
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

quiver(arrow_x, arrow_y, arrow_u1, arrow_v1, 'k', 'LineWidth', 2)
quiver(arrow_x, arrow_y, arrow_u2, arrow_v2, 'g', 'LineWidth', 2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot

X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
figure(19)
plot(b_array, ccr_array);
xlabel('the value of b');
ylabel('the value of CCR');
title('b versus CCR');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2000)
quiver(arrow_x, arrow_y, arrow_u2, arrow_v2, 'g', 'LineWidth', 2)
hold on
quiver(arrow_x, arrow_y, cos((38/48)*pi)+arrow_x, sin((38/48)*pi)+arrow_y, 'k', 'LineWidth', 2)




%%
function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2


%%%%%%%%%%%%%%%%%%%%%%
X = zeros(2, n1+n2);
Y = zeros(1, n1+n2);
S = [cos(theta) sin(theta); sin(theta) -cos(theta)]*[lambda1 0;0 lambda2]*[cos(theta) sin(theta); sin(theta) -cos(theta)];
X(:, 1:n1) = mvnrnd(mu1, S, n1).';
X(:, (n1+1):(n1+n2)) = mvnrnd(mu2, S, n2).';
Y(:, 1:n1) = 1;
Y(:, (n1+1):(n1+n2)) = 2;
%%%%%%%%%%%%%%%%%%%%%%

end
%%
function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)
n1 = 50;
n2 = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step1: calculate w
n = n1 + n2; 
w = [cos(phi);sin(phi)];

%Step2: Calculate the signal power [wT(meanx2 - meanx1)]^2
meanx2 = [sum(X(1, (n1+1):(n1+n2)))/n2; sum(X(2, (n1+1):(n1+n2)))/n2];
meanx1 = [sum(X(1, (1):(n1)))/n1; sum(X(2, 1:(n1)))/n1];
signal = (w.'*(meanx2 - meanx1))^2;

%Step3: Calculate the noise power
sum_x2_hat = zeros(2, 2);
for k = n1+1 : n1+n2
    sum_x2_hat = sum_x2_hat + (X(:, k)-meanx2)*(X(:, k)-meanx2).';
end
s_x2 = sum_x2_hat/n2;

sum_x1_hat = zeros(2, 2);
for k = 1 : n1
    sum_x1_hat = sum_x1_hat + (X(:, k)-meanx1)*(X(:, k)-meanx1).';
end
s_x1 = sum_x1_hat/n1;

noise = w.'*((n1/n)*s_x1 + (n2/n)*s_x2)*w;

%Step4: Calculate SNR
snr = signal/noise;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    X_projected_phi_class1 = w.'*X(:, 1:n1);
    X_projected_phi_class2 = w.'*X(:, (n1+1):(n1+n2));
    [pdf1,z1] = ksdensity(X_projected_phi_class1);
    figure
    plot(pdf1,z1)
    hold on;
    [pdf2,z2] = ksdensity(X_projected_phi_class2);
    plot(pdf2,z2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title('Estimated class density estimates of data projected along \phi = num2str(phi) \times \pi/6. Ground-truth \phi = \pi/6')
end

end
%%
function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = 50;
n2 = 100;
n = n1+n2;
meanx2 = [sum(X(1, (n1+1):(n1+n2)))/n2; sum(X(2, (n1+1):(n1+n2)))/n2];
meanx1 = [sum(X(1, (1):(n1)))/n1; sum(X(2, 1:(n1)))/n1];
sum_x2_hat = zeros(2, 2);
for k = n1+1 : n1+n2
    sum_x2_hat = sum_x2_hat + (X(:, k)-meanx2)*(X(:, k)-meanx2).';
end
s_x2 = sum_x2_hat/n2;

sum_x1_hat = zeros(2, 2);
for k = 1 : n1
    sum_x1_hat = sum_x1_hat + (X(:, k)-meanx1)*(X(:, k)-meanx1).';
end
s_x1 = sum_x1_hat/n1;

s_x_avg = (n1/n)*s_x1 + (n2/n)*s_x2;

w_LDA = (s_x_avg)^(-1)*(meanx2 - meanx1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
correct_num = 0;
n1 = 50;
n2 = 100;
for i = 1:1:n1+n2
    calc_res = w_LDA.'*X(:,i)+b;
    if calc_res <= 0
        pred_res = 1;
    else
        pred_res = 2;
    end
    if pred_res == Y(i)
        correct_num = correct_num + 1;
    else
        correct_num = correct_num + 0;
    end
end
ccr = correct_num/(n1+n2);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


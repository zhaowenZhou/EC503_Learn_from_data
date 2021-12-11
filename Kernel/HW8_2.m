clear all;
clc;
%% Load and visualize the data
load('kernel-svm-2rings.mat')
figure(1)
gscatter(x(1, :), x(2, :), y, 'cr');
title('data visualization')
hold on



%% Compute K using RBF in SSGD
[~, n] = size(x);
K = zeros(n, n);
bandwidth = 0.5;
for i = 1:n
    for j = 1:n
        K(i, j) = exp((-1/(2*bandwidth^2))*norm(x(:, i) - x(:, j))^2);
    end
end

%% Plot the cost plot
phi = zeros(n + 1, 1);
phi_array = ssgd(phi, x, y, 256, K);
cost_array = cost(phi_array, K, y, 256);
figure(2)
plot(1:101, cost_array)
xlabel('the number of iterations')
ylabel('The total cost')
title('The cost against iteration number t')


%% Plot the CCR plot
[CCR_array, y_hat] = CCR(phi_array, x, y);
figure(3)
plot(1:101, CCR_array)
xlabel('the number of iterations')
ylabel('The value of CCR')
title('CCR against iteration number')

%% Confusion matrix
conf_mat = confusionmat(y, y_hat);


%% Visualize the decision boundary
[~, I] = min(cost_array);
phi_final = phi_array(:, I);
[Xgrid, Ygrid] = meshgrid([-2:0.06:2], [-2:0.06:2]);
X_boundary = [Xgrid(:), Ygrid(:)];
X_boundary = X_boundary.';
[~, N] = size(X_boundary);
Y_boundary = zeros(N, 1);
for point = 1:N
    Ktest = zeros(n, 1);
    for row = 1:n
        Ktest(row, 1) = exp((-1/(2*bandwidth^2))*norm(x(:, row) - X_boundary(:, point))^2);
    end
    Ktest_ext = [Ktest; 1];
    if phi_final.' * Ktest_ext >= 0
        Y_boundary(point, 1) = 1;
    else
        Y_boundary(point, 1) = -1;
    end
end
% X_boundary = X_boundary(:, Y_boundary == 1);
X_boundary_contour1 = reshape(X_boundary(1, :), [67, 67]);
X_boundary_contour2 = reshape(X_boundary(2, :), [67, 67]);
Y_boundary_contour = reshape(Y_boundary, [67, 67]);
figure(1)
gscatter(X_boundary(1, :), X_boundary(2, :), Y_boundary, 'cr')
figure(5)
gscatter(x(1, :), x(2, :), y, 'cr');
hold on
contour(X_boundary_contour1, X_boundary_contour2, Y_boundary_contour)
xlabel('x1')
ylabel('x2')
title('the scatter plot of points and decision boundary')
legend('label -1', 'label 1', 'the decision boundary')


%% SSGD of kernal SVM
function phi_array = ssgd(phi, X, label, nC, K)
[~, n] = size(X);
K_ext = [K; ones(1, n)];
phi_array = zeros(n + 1, 100);
for t = 1:1000
    st = 0.256/t;
    % choose sample index
    j = randi([1 n], 1, 1);
    % compute subgradient
    v = [K zeros(n, 1); zeros(1, n + 1)] * phi;
    if label(j, 1)*phi.'*K_ext(:, j) < 1
        v = v - nC*label(j, 1)*K_ext(:, j);
    end
    phi = phi - st*v;
    if mod(t, 10) == 0
        phi_array(:, t/10) = phi;
    end
end
phi_array = [zeros(n + 1, 1) phi_array];
end
    

%% Cost function
function cost_array = cost(phi_array, K, label, nC)
n = 200;
C = nC/n;
cost_array = zeros(101, 1);
K_ext = [K; ones(1, n)];
for t = 1:101
    f0 = (1/2) * phi_array(:, t).' * [K zeros(n, 1); zeros(1, n + 1)] * phi_array(:, t);
    fj = zeros(n, 1);
    for j = 1:n
        if (1 - label(j)*phi_array(:, t).'*K_ext(:, j)) > 0
            fj(j, 1) = C * (1 - label(j)*phi_array(:, t).'*K_ext(:, j));
        else
            fj(j, 1) = 0;
        end
    end
    cost = f0 + sum(fj);
    cost = cost/n;
    cost_array(t, 1) = cost;
end
end
    

%% CCR function
function [CCR_array, y_hat] = CCR(phi_array, X, label)
[~, n] = size(X);
CCR_array = zeros(101, 1);
bandwidth = 0.5;
for t = 1:101
    y_hat = zeros(n, 1);
    for j = 1:n
        K = zeros(n, 1);
        for p = 1:n
            K(p, 1) = exp((-1/(2*bandwidth^2))*norm(X(:, p) - X(:, j))^2);
        end
        K_ext = [K; 1];
        sign = phi_array(:, t).' * K_ext;
        if sign >= 0
            y_hat(j, 1) = 1;
        else
            y_hat(j, 1) = -1;
        end
    end
    CCR_array(t, 1) = (sum(label - y_hat == 0)/n);
end
end        
        
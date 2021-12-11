clear all; clc
load('kernel-kmeans-2rings.mat')
%% Visualize the data
figure(1)
scatter(data(:, 1), data(:, 2))
xlabel('x1')
ylabel('x2')
title('Dataset')

%% Data Preprocessing
X = data.';


%% Compute K
[~, n] = size(X);
K = zeros(n, n);
bandwidth_square = 0.16;
for i = 1:n
    for j = 1:n
        K(i, j) = exp((-1/(2*bandwidth_square))*norm(X(:, i) - X(:, j))^2);
    end
end


%% initialize data
threshold = 1 * 10^-5;
ite_num = 100;
%% Algorithm
iteration = 1;
u1 = rand(n, 1);
u2 = rand(n, 1);
u1 = u1/norm(u1);
u2 = u2/norm(u2);
y_hat = zeros(n, 1);
e = eye(n, n);
WCSS_prev = 0;
while iteration < ite_num
    % update labels
    for j = 1:n
        ej = e(:, j);
        [~, I] = min([(ej - u1).' * K * (ej - u1), (ej - u2).' * K * (ej - u2)]);
        y_hat(j, 1) = I;
    end
    % update coefficient vectors
    n1 = sum(y_hat == 1);
    n2 = sum(y_hat == 2);
    if n1 == 0 && n2 ~= 0
        u1 = zeros(n, 1);
        u2 = [y_hat == 2]/n2;
    elseif n1 ~= 0 && n2 == 0
        u1 = [y_hat == 1]/n1;
        u2 = zeros(n, 1);
    else
        u1 = [y_hat == 1]/n1;
        u2 = [y_hat == 2]/n2;
    end
%     Calculate the WCSS
    WCSS = 0;
    for d = 1:n
        ed = e(:, d);
        if y_hat(d, 1) == 1
            WCSS = WCSS + (ed - u1).' * K * (ed - u1);
        else
            WCSS = WCSS + (ed - u2).' * K * (ed - u2);
        end
    end
    if WCSS - WCSS_prev < 0
        if abs(WCSS - WCSS_prev) <= threshold
            break
        elseif abs(WCSS - WCSS_prev) > threshold
            WCSS_prev = WCSS;
        end
    else
        WCSS_prev = WCSS;
    end
    iteration = iteration + 1;
end

%% Compute the y_hat and visualize it
figure(2)
gscatter(data(:, 1), data(:, 2), y_hat)
title('The clustering result')
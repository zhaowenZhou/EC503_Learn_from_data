load('iris.mat')
%% plot the histogram
Y_label = [Y_label_train; Y_label_test];
edges = [0.9 1.1 1.9 2.1 2.9 3.1];
figure(1)
histogram(Y_label, edges)
xlabel('class m')
ylabel('the number of points belong to class m')

%% empirical correlation coefficients for all pairs of features
X_data_train = X_data_train.';
X_data_test = X_data_test.';
X_data = [X_data_train X_data_test]; % d*n matrix
X_data_R = X_data.';
X_mean = mean(X_data_R, 2);
X_tilde = X_data_R - X_mean;
var = zeros(150, 1);
for i = 1:150
    var(i, 1) = sqrt(sum((X_data_R(i,:) - X_mean(i)).^2));
end
var = diag(var);
X_s = X_tilde.'/var;
R = X_s.'*X_s;


%% draw the scatter plot of distinct features
figure(2)
sgtitle('scatter plot for all distinct pairs of features')
iteration = 1;
for i = 1:3
    for j = i+1 : 4
        subplot(2, 3, iteration)
        scatter(X_data(i, :), X_data(j, :))
        title({['feature1 = ', num2str(i)], ['feature2 = ', num2str(j)]})
        iteration = iteration + 1;
    end
end

%% draw the cost function
theta = zeros(5, 3);
X_ext = [X_data_train; ones(1, 105)];
cost = zeros(6000, 1);
cost_array = zeros(300, 1);
t_array = (1:300);
for t = 1:6000
    theta = gradient_descent(theta, X_ext, t, Y_label_train);
    cost = total_cost(theta, X_ext, Y_label_train);
    if mod(t, 20) == 0
        cost_array(t/20, 1) = cost;
    end
end
cost_array = cost_array/105;
figure(3)
plot(t_array, cost_array)
xlabel('iteration')
ylabel('the total cost')
title('the cost of train data against iteration')

%% Plot the CCR of train dataset
theta = zeros(5, 3);
theta_array = zeros(5, 900);
for t = 1:6000
    theta = gradient_descent(theta, X_ext, t, Y_label_train);
    if mod(t, 20) == 0
        theta_array(:, ((t/20)*3-2):((t/20)*3)) = theta;
    end
end
theta_X_ext = theta_array.' * X_ext;
CCR = zeros(300, 1);
for iteration = 1:3:898
    theta_X_for_CCR = theta_X_ext(iteration:iteration+2, :);
    [~, I_train] = max(theta_X_for_CCR);
    I_train = I_train.';
    CCR_count = I_train - Y_label_train;
    CCR(((iteration+2)/3), 1) = sum(CCR_count(:) == 0);
end
CCR = CCR/105;
figure(4)
plot(t_array, CCR)
xlabel('the iteration')
ylabel('CCR of train data')
title('CCR of train data against iteration')

%% plot the CCR of test dataset
theta_test = zeros(5, 3);
theta_test_array = zeros(5, 900);
for t = 1:6000
    theta_test = gradient_descent(theta_test, X_ext, t, Y_label_train);
    if mod(t, 20) == 0
        theta_test_array(:, ((t/20)*3-2):((t/20)*3)) = theta_test;
    end
end
X_test_ext = [X_data_test; ones(1, 45)];
theta_X_ext_test = theta_test_array.' * X_test_ext;
CCR_test = zeros(300, 1);
for iteration = 1:3:898
    theta_X_for_CCR_test = theta_X_ext_test(iteration:iteration+2, :);
    [~, I_test] = max(theta_X_for_CCR_test);
    I_test = I_test.';
    CCR_count_test = I_test - Y_label_test;
    CCR_test(((iteration+2)/3), 1) = sum(CCR_count_test(:) == 0);
end
CCR_test = CCR_test/45;
figure(5)
plot(t_array, CCR_test)
xlabel('iteration')
ylabel('the CCR of test data')
title('the CCR of test data against iteraion')
    

%% Plot the log loss
logloss_test = zeros(300, 1);
for iteration = 1:300
    fj_theta_test = zeros(45, 1);
    onethetax_test = zeros(45, 1);
    for j = 1:45
        for l = 1:3
            if l == Y_label_test(j,1)
                onethetax_test(j,1) = onethetax_test(j,1) + theta_test_array(:,3*iteration-3+l).'*X_test_ext(:,j);
            end
        end
    end
    for j = 1:45
        fj_theta_test(j, 1) = log(exp(theta_test_array(:,3*iteration).'*X_test_ext(:,j))+exp(theta_test_array(:,3*iteration-1).'*X_test_ext(:,j))+exp(theta_test_array(:,3*iteration-2).'*X_test_ext(:,j))) - onethetax_test(j, 1);
    end
    logloss_test(iteration, 1) = sum(fj_theta_test);
end
logloss_test = logloss_test/45;
figure(6)
plot((1:300), logloss_test)
xlabel('iteraion')
ylabel('the log loss of test data')
title('the log loss of test data against iteraion')


%% The training confusion mat and test confusion mat
conf_mat_train = confusionmat(Y_label_train, I_train);
conf_mat_test = confusionmat(Y_label_test, I_test);



%% decision region
theta_use = theta_array(:, 898:900); %5*3
figure(7)
sgtitle('the decision of different dimension')
iteration = 1;
for i = 1:3
    for j = i+1 : 4
        X_data_region = [X_data; ones(1, 150)]; %5*150
        X_data_region([i, j], :) = 0; %5*150
        region_pred = theta_use.'*X_data_region;
        [~, I] = max(region_pred); %I为150个点的label
        subplot(2, 3, iteration)
        X_data_region([i, j], :) = [];
        gscatter(X_data_region(1, :), X_data_region(2, :), I)
        title({['x',num2str(i), '=0'], ['x',num2str(j),'=0']})
        iteration = iteration + 1;
    end
end
%% gradient descent function
function THETA = gradient_descent(theta, X, t, Y_label_train) 
[~, n] = size(X);
j = randi([1 n], 1, 1);
for k = 1:3
    p = exp(theta(:, k).'*X(:, j))/(exp(theta(:,1).'*X(:, j))+exp(theta(:,2).'*X(:, j))+exp(theta(:,3).'*X(:, j)));
    if p <= 10^(-10)
        p = 10^(-10);
    end
    if k == Y_label_train(j,1)
        v = 2*0.1*theta(:, k)+105*(p-1)*X(:, j);
    else
        v = 2*0.1*theta(:, k)+105*(p)*X(:, j);
    end
    theta(:, k) = theta(:, k) - (0.01/t)*v;
end
THETA = theta;
end

%% cost function
function cost = total_cost(theta,X, Y_label_train)
theta_square = sum(theta.*theta);
f0 = 0.1*sum(theta_square);
%theta_X = theta.' * X;                  %the column is xj*all theta
fj_theta = zeros(105, 1);
onethetax = zeros(105, 1);
for j = 1:105
    for l = 1:3
        if l == Y_label_train(j,1)
            onethetax(j,1) = onethetax(j,1) + theta(:,l).'*X(:,j);
        end
    end
end
for j = 1:105
    fj_theta(j, 1) = log(exp(theta(:,1).'*X(:,j))+exp(theta(:,2).'*X(:,j))+exp(theta(:,3).'*X(:,j))) - onethetax(j, 1);
end
cost = f0 + sum(fj_theta);
end



        
    


    
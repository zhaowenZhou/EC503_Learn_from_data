%% Process the data
load('iris.mat')
X_data_train = X_data_train(:, [2, 4]);
X_data_test = X_data_test(:, [2, 4]);
train_ext_origin = [X_data_train ones(105, 1)].';
test_ext_origin = [X_data_test ones(45, 1)].';
Y_label_test_origin = Y_label_test;
Y_label_train_origin = Y_label_train;


%% Visualize the data
figure(9)
subplot(1, 2, 1)
gscatter(X_data_train(:, 1), X_data_train(:, 2), Y_label_train)
title('the points distribution and the hyperplane of train data')
hold on
subplot(1, 2, 2)
gscatter(X_data_test(:, 1), X_data_test(:, 2), Y_label_test)
hold on
title('the points distribution and the hyperplane of test data')
%% Plot the cost 1/n * g(theta)
figure(1)
sgtitle('the cost of different binary classifier')

Y_label_train = zeros(70, 1);
Y_label_train(1:35, 1) = -1;
Y_label_train(36:70, 1) = 1;

theta = zeros(3, 1);

t_array = 1:201;
% % class 1 and 2
% train_ext1 = [train_ext_origin(:, (Y_label_train_origin == 1)) train_ext_origin(:, (Y_label_train_origin == 2))];
% 
% %class 1 and 3
% train_ext2 = [train_ext_origin(:, (Y_label_train_origin == 1)) train_ext_origin(:, (Y_label_train_origin == 3))];
% 
% %class 2 and 3
% train_ext3 = [train_ext_origin(:, (Y_label_train_origin == 2)) train_ext_origin(:, (Y_label_train_origin == 3))];

train_ext1 = train_ext_origin(:, 1:70);
train_ext2 = train_ext_origin(:, [1:35 71:105]);
train_ext3 = train_ext_origin(:, 36:105);

%draw the subplot of class 1 and 2
theta_array_1 = ssgd(theta, train_ext1, Y_label_train, 1.2);
cost_array_1 = cost(theta_array_1, train_ext1, Y_label_train, 1.2);
subplot(2, 2, 1)
plot(t_array, cost_array_1)
xlabel('the number of iteration')
ylabel('the total cost')
title('the cost against iteration of class 1 and 2')

%draw the subplot of class 1 and 3
theta_array_2 = ssgd(theta, train_ext2, Y_label_train, 1.2);
cost_array_2 = cost(theta_array_2, train_ext2, Y_label_train, 1.2);
subplot(2, 2, 2)
plot(t_array, cost_array_2)
xlabel('the number of iteration')
ylabel('the total cost')
title('the cost against iteration of class 1 and 3')

%draw the subplot of class2 and 3
theta_array_3 = ssgd(theta, train_ext3, Y_label_train, 1.2);
cost_array_3 = cost(theta_array_3, train_ext3, Y_label_train, 1.2);
subplot(2, 2, 3)
plot(t_array, cost_array_3)
xlabel('the number of iteration')
ylabel('the total cost')
title('the cost against iteration of class 2 and 3')


%% Plot the train CCR of binary classifier
figure(2)
sgtitle('the CCR of different class')

%class 1 and 2
[CCR_array_1, yj_hat1] = CCR(theta_array_1, train_ext1, Y_label_train);
subplot(2, 2, 1)
plot((1:200), CCR_array_1)
xlabel('the number of iteration')
ylabel('CCR')
title('the CCR against iteration of class 1 and 2')

%class 1 and 3
[CCR_array_2, yj_hat2] = CCR(theta_array_2, train_ext2, Y_label_train);
subplot(2, 2, 2)
plot((1:200), CCR_array_2)
xlabel('the number of iteration')
ylabel('CCR')
title('the CCR against iteration of class 1 and 3')

%class 2 and 3
[CCR_array_3, yj_hat3] = CCR(theta_array_3, train_ext3, Y_label_train);
subplot(2, 2, 3)
plot((1:200), CCR_array_3)
xlabel('the number of iteration')
ylabel('CCR')
title('the CCR against iteration of class 2 and 3')

%% Plot the test CCR
Y_label_test = zeros(30, 1);
Y_label_test(1:15, 1) = -1;
Y_label_test(16:30, 1) = 1;

figure(3)
sgtitle('the test data CCR of different class')
test_ext1 = test_ext_origin(:, 1:30);
test_ext2 = test_ext_origin(:, [1:15 31:45]);
test_ext3 = test_ext_origin(:, 16:45);

[testCCR_array_1, testyj_hat1] = CCR(theta_array_1, test_ext1, Y_label_test);
[testCCR_array_2, testyj_hat2] = CCR(theta_array_2, test_ext2, Y_label_test);
[testCCR_array_3, testyj_hat3] = CCR(theta_array_3, test_ext3, Y_label_test);

subplot(2, 2, 1)
plot((1:200), testCCR_array_1)
xlabel('the iteration')
ylabel('CCR')
title('the test data CCR of class 1 and 2')

subplot(2, 2, 2)
plot((1:200), testCCR_array_2)
xlabel('the iteration')
ylabel('CCR')
title('the test data CCR of class 1 and 3')

subplot(2, 2, 3)
plot((1:200), testCCR_array_3)
xlabel('the iteration')
ylabel('CCR')
title('the test data CCR of class 2 and 3')


%% Confusion matrix
conf_mat_train1 = confusionmat(Y_label_train, yj_hat1);
conf_mat_train2 = confusionmat(Y_label_train, yj_hat2);
conf_mat_train3 = confusionmat(Y_label_train, yj_hat3);
conf_mat_test1 = confusionmat(Y_label_test, testyj_hat1);
conf_mat_test2 = confusionmat(Y_label_test, testyj_hat2);
conf_mat_test3 = confusionmat(Y_label_test, testyj_hat3);


%% Preposses before all pair
theta12 = theta_array_1(:, 201);
theta13 = theta_array_2(:, 201);
theta23 = theta_array_3(:, 201);
theta_allpair = [theta12 theta13 theta23];
theta_xtrain = theta_allpair.' * train_ext_origin;
theta_xtest = theta_allpair.' * test_ext_origin;
figure(4)
gscatter(X_data_train(:, 1), X_data_train(:, 2), Y_label_train_origin)
title('class distribution using all pairs')
x = linspace(0, 5);
y12 = -(theta12(1)/theta12(2))*x - (theta12(3)/theta12(2));
line(x, y12, 'Color','g')
hold on
y13 = -(theta13(1)/theta13(2))*x - (theta13(3)/theta13(2));
line(x, y13, 'Color', 'y')
hold on
y23 = -(theta23(1)/theta23(2))*x - (theta23(3)/theta23(2));
line(x, y23, 'Color', 'm')
legend('clas1', 'class2', 'class3', 'line of class 12', 'line of class 13', 'line of class 23')

%% All pair CCR
[allpair_train, CCR_valuetrain] = AllpairCCR(theta_xtrain, Y_label_train_origin);
[allpair_test, CCR_valuetest] = AllpairCCR(theta_xtest, Y_label_test_origin);


%% All pair confusion mat
Allpairtrainmat = confusionmat(Y_label_train_origin, allpair_train);
Allpairtestmat = confusionmat(Y_label_test_origin, allpair_test);
%% Plot the hyperplane
figure(9)
subplot(1, 2, 1)
x = linspace(0, 5);
y12 = -(theta12(1)/theta12(2))*x - (theta12(3)/theta12(2));
line(x, y12, 'Color','g')
hold on
y13 = -(theta13(1)/theta13(2))*x - (theta13(3)/theta13(2));
line(x, y13, 'Color', 'y')
hold on
y23 = -(theta23(1)/theta23(2))*x - (theta23(3)/theta23(2));
line(x, y23, 'Color', 'm')
legend('clas1', 'class2', 'class3', 'class12', 'class13', 'class23')
subplot(1, 2, 2)
x = linspace(0, 5);
y12 = -(theta12(1)/theta12(2))*x - (theta12(3)/theta12(2));
line(x, y12, 'Color','g')
hold on
y13 = -(theta13(1)/theta13(2))*x - (theta13(3)/theta13(2));
line(x, y13, 'Color', 'y')
hold on
y23 = -(theta23(1)/theta23(2))*x - (theta23(3)/theta23(2));
line(x, y23, 'Color', 'm')
legend('clas1', 'class2', 'class3', 'class12', 'class13', 'class23')
%% SSGD function
% compute the theta array for t = 1:2*10^5
function theta_array = ssgd(theta, X_ext, label, C)
[d, n] = size(X_ext);
theta_array = zeros(3, 200);
for t = 1:2*10^5
    j = randi([1 n], 1, 1);
    st = 0.5/t;
    v = [eye(d - 1) zeros(d - 1, 1); zeros(1, d)] * theta;
    if label(j, 1)*theta.'*X_ext(:, j) < 1
        v = v - n*C*label(j, 1)*X_ext(:, j);
    end
    theta = theta - st*v;
    if mod(t, 1000) == 0
        theta_array(:, t/1000) = theta;
    end
end
theta_array = [zeros(3, 1) theta_array];
end
%% Cost function
% Compute the cost array for different theta against iteraion
function cost_array = cost(theta_array, X_ext, label, C)
[~,n] = size(X_ext);
cost_array = zeros(201, 1);
for t = 1:201
    w = theta_array(1:2, t);
    f0_theta = (1/2)*norm(w)*norm(w);
    fj_theta = zeros(n, 1);
    for i = 1:n
        if (1 - label(i)*theta_array(:, t).'*X_ext(:, i)) > 0
            fj_theta(i, 1) = C * (1 - label(i)*theta_array(:, t).'*X_ext(:, i));
        else
            fj_theta(i, 1) = 0;
        end
    end
    cost = f0_theta + sum(fj_theta);
    cost = cost/n;
    cost_array(t, 1) = cost;
end
end

%% CCR function
% Input theta array compute the CCR array
function [CCR_array, yj_hat] = CCR(theta_array, X_ext, label)
[~, T] = size(theta_array);
CCR_array = zeros(T-1, 1);
theta_array = theta_array(:, 2:T);
[~, n] = size(X_ext); 
for t = 1:(T-1)
    theta = theta_array(:, t);
    yj_hat = zeros(n, 1);
    for j = 1:n
        if (theta.' * X_ext(:, j)) >= 0
            yj_hat(j, 1) = 1;
        else
            yj_hat(j, 1) = -1;
        end
    end
    CCR_array(t, 1) = (sum(label - yj_hat == 0)/n);
end
end


%% All pair CCR function
function [allpair_label, CCRvalue] = AllpairCCR(theta_x, Y_label)
[rol, col] = size(theta_x);
for i = 1:rol
    for j = 1:col
        if i == 1
            if theta_x(i, j) <= 0
                theta_x(i, j) = 1;
            else
                theta_x(i, j) = 2;
            end
        end
        if i == 2
            if theta_x(i, j) <= 0
                theta_x(i, j) = 1;
            else
                theta_x(i, j) = 3;
            end
        end
        if i == 3
            if theta_x(i, j) <= 0
                theta_x(i, j) = 2;
            else
                theta_x(i, j) = 3;
            end
        end
    end
end
allpair_label = mode(theta_x);
diff = allpair_label.' - Y_label;
n = size(diff);
sum_zero = 0;
for k = 1:n
    if diff(k) == 0
        sum_zero = sum_zero + 1;
    end
end
CCRvalue = sum_zero/col;
end


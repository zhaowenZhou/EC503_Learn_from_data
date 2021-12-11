% EC 503 Learning from Data
% Fall semester, 2021
% Homework 2
% by (Zhaowen Zhou)
%
% Nearest Neighbor Classifier
%
% Problem 2.5e
clc, clear
tic
fprintf("==== Loading data_mnist_train.mat\n");
load("data_mnist_train.mat");
fprintf("==== Loading data_mnist_test.mat\n");
load("data_mnist_test.mat");

% show test image
imshow(reshape(X_train(200,:), 28,28)')

% determine size of dataset
[Ntrain, dims] = size(X_train);
[Ntest, ~] = size(X_test); %test point sample数

% precompute components

% Note: To improve performance, we split our calculations into
% batches. A batch is defined as a set of operations to be computed
% at once. We split our data into batches to compute so that the 
% computer is not overloaded with a large matrix.
batch_size = 500;  % fit 4 GB of memory
num_batches = Ntest / batch_size; % 分成了num_batches个batch
X_train_sqr=sum(X_train.^2,2); % 60000*1
X_train_dist=repmat(X_train_sqr.',500,1); %500*60000

% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
Y_pred=zeros(Ntest,1); % 每一个点的Y预测值,10000*1
for bn = 1:num_batches % 循环batch,test point
  batch_start = 1 + (bn - 1) * batch_size; % 定义这个batch下的计算从哪一行开始
  batch_stop = batch_start + batch_size - 1; % 定义这个batch下的计算在哪一行结束
  X_test_sqr=sum(X_test(batch_start:batch_stop,:).^2,2); %500*1
  X_test_dist=repmat(X_test_sqr,1,60000);
  CV=-2*X_test(batch_start:batch_stop,:)*X_train.';
  dist=CV+X_train_dist+X_test_dist; % 500*60000
  [dist_value,index]=min(dist,[],2); %index是一个500*1的vector
  Y_pred(batch_start:batch_stop,1)=Y_train(index);
end
conf_mat=confusionmat(Y_test(:),Y_pred)
CCR=sum(diag(conf_mat))/Ntest
toc
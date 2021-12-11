% EC 503 Learning from Data
% Fall semester, 2021
% Homework 2
% by (Zhaowen Zhou)
%
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d


clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain);
xlabel('feature1');
ylabel('feature2');
title('X distribution chart');


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);
% compute probabilities of being in class 2 for each point on grid
[N,M]=size(Xtrain);
probability=zeros(Ntest,1);
for i=1:Ntest
    dist=zeros(N,1);
    for j=1:N
        dist(j,:)=norm(Xtrain(j,:)-Xtest(i,:));
    end
    [B,index]=sort(dist);
    temp=ytrain(index);
    class=temp(1:K,:);
    class_2=sum(class==2);
    probability(i,:)=class_2/length(class);
end

% Figure for class 2
figure
class2ProbonGrid = reshape(probability,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('x1')
ylabel('x2')
title('P10NN(y=2|x)')


% repeat steps above for class 3 below
probability_class_3=zeros(Ntest,1);
for i=1:Ntest
    dist=zeros(N,1);
    for j=1:N
        dist(j,:)=norm(Xtrain(j,:)-Xtest(i,:));
    end
    [B,index]=sort(dist);
    temp=ytrain(index);
    class=temp(1:K,:);
    class_3=sum(class==3);
    probability_class_3(i,:)=class_3/length(class);
end
figure
class2ProbonGrid = reshape(probability_class_3,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('x1')
ylabel('x2')
title('P10NN(y=3|x)')

%% c) Class label predictions
K = 1 ; % K = 1 case
ypred=zeros(Ntest,1);
% compute predictions 
for i=1:Ntest
    dist=zeros(N,1);
    for j=1:N
        dist(j,:)=norm(Xtrain(j,:)-Xtest(i,:));
    end
    [B,index]=sort(dist);
    temp=ytrain(index);
    ypred(i,1)=temp(K);
end
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('x1')
ylabel('x2')
title('Prediction using 1NN')

% repeat steps above for the K=5 case. Include code for this below.
K=5;
ypred_5NN=zeros(Ntest,1);
for i=1:Ntest
    dist=zeros(N,1);
    for j=1:N
        dist(j,:)=norm(Xtrain(j,:)-Xtest(i,:));
    end
    [B,index]=sort(dist);
    temp=ytrain(index);
    class_5NN=temp(1:K,:);
    ypred_5NN(i,1)=mode(class_5NN);
end
figure
gscatter(Xgrid(:),Ygrid(:),ypred_5NN,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
xlabel('x1')
ylabel('x2')
title('Prediction using 5NN')
%% d) LOOCV CCR computations


for k=1:2:11
    ypred_LOOCV=zeros(N,1);
    for p=1:1:200 %validation point
        dist_LOOCV=zeros(N,1);
        for q=1:1:200 % compute distance
            dist_LOOCV(q,:)=norm(Xtrain(p,:)-Xtrain(q,:));
        end
        [B_LOOCV,index]=sort(dist_LOOCV);
        temp_LOOCV=ytrain(index); %sort ylabel
        ypred_LOOCV(p,:)=mode(temp_LOOCV(2:(k+1),:));
    end
    conf_mat = confusionmat(ytrain, ypred_LOOCV);
    CCR=sum(diag(conf_mat))/N;
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end
figure
plot([1:2:11],CCR_values)
xlabel('K')
ylabel('CCR of K')
title('The CCR of different K')


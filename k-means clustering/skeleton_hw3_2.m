% EC 503 - HW 3 - Fall 2021
% K-Means starter code
% This code is a combination of all problems in 3.2, it might seems
% unreasonable but a good use of this code with choice can solve all
% problems in 3.2
clear, clc, close all;

%% Generate Gaussian data:
% Add code below:
u1 = [3 3];
u2 = [-4 -1];
u3 = [2 -4];
Sigma1 = 0.02*eye(2);
Sigma2 = 0.05*eye(2);
Sigma3 = 0.07*eye(2);
R1 = mvnrnd(u1, Sigma1, 50);
R2 = mvnrnd(u2, Sigma2, 50);
R3 = mvnrnd(u3, Sigma3, 50);
figure
scatter(R1(:,1), R1(:,2), 'r')
hold on
scatter(R2(:,1), R2(:,2), 'g')
hold on
scatter(R3(:,1), R3(:,2), 'b')
legend('C1','C2','C3')
xlabel('x1')
ylabel('x2')
title('3 clusters with Gaussian distribution')
hold off
DATA = cat(1,R1,R2,R3);

%% Generate NBA data:
% Add code below:
all_stats = readmatrix('NBA_stats_2018_2019.xlsx');
NBA_DATA = all_stats(2:589,[5 7]);

% HINT: readmatrix might be useful here

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.

%% K-Means implementation
% Add code below

K = 3;
MU_init = [-0.14 3.15 -3.28; 2.61 -0.84 -1.48];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
ones(length(DATA),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    %% code for calculate the distance between R points and current cluster mean
   DATA_square = repmat(sum(DATA.*DATA,2),1,K);
   MU_previous_square = repmat(sum(MU_previous.*MU_previous,1),length(DATA),1);
   Dist = -2*DATA*MU_previous + DATA_square + MU_previous_square;
   [Dist_min,Data_label] = min(Dist,[],2); % 得到了每一个数据的label
    

 
   
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    % cluster the points with same label
    for l = 1:K
        MU_current(:,l) = mean(DATA(Data_label == l,:)).'; %
    end  
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    MU_diff = MU_current - MU_previous;
    MU_diffsquare = sum((MU_diff.^2),2);
    if (max(MU_diffsquare) < convergence_threshold)
        converged=1;
    else
        MU_previous = MU_current;
    end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        gscatter(DATA(:,1),DATA(:,2),Data_Label(:,1))
        
        %% If converged, get WCSS metric
        % Add code below
        wcss_cluster = sum(Dist_min);
        if i == 1
            WCSS_cluster = wcss_cluster;
        else
            WCSS_cluster = [WCSS_cluster wcss_cluster]; %这个是一个K下全部的wcss
        end
                % 找到一个K下不同的initial point中最小的wcss
    end            
    wcss_min = min(WCSS_cluster); %在initial point循环完成后找到最小的
    if K == 2
        WCSS_min = wcss_min;
    else
        WCSS_min = [WCSS_min wcss_min];
    end        
end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




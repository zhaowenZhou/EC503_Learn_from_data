clear, clc, close all;
% Problem 3.2c
%% Generate Gaussian data:
% Add code below:
u1 = [2 2];
u2 = [-2 2];
u3 = [0 -3.25];
Sigma1 = 0.02*eye(2);
Sigma2 = 0.05*eye(2);
Sigma3 = 0.07*eye(2);
R1 = mvnrnd(u1, Sigma1, 50);
R2 = mvnrnd(u2, Sigma2, 50);
R3 = mvnrnd(u3, Sigma3, 50);
DATA = cat(1,R1,R2,R3);


%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.

%% K-Means implementation
K = 3;
NeedData=(DATA(randi(length(DATA),1,K),:)).';
MU_init = NeedData;
WCSS = zeros(10,1);
t = 1; %第几个initial point
for i = 1:28:3 %循环不同的分类初始值
    MU_previous = MU_init(:,i:i+2);
    MU_current = MU_init(:,i:i+2);

    % initializations
    labels = [1 2 3];
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        %% code for calculate the distance between R points and current cluster mean
        R1_square = repmat(sum(R1.*R1, 2), 1, 3); %50*3的矩阵
        R2_square = repmat(sum(R2.*R2, 2), 1, 3); %50*3的矩阵
        R3_square = repmat(sum(R3.*R3, 2), 1, 3); %50*3的矩阵
        MU_previous_square = repmat(sum(MU_previous.*MU_previous, 1), 50, 1); %50*3的矩阵
        Dist_R1 = -2*R1*MU_previous + R1_square + MU_previous_square;
        Dist_R2 = -2*R2*MU_previous + R2_square + MU_previous_square;
        Dist_R3 = -2*R3*MU_previous + R3_square + MU_previous_square;

        %% assign labels for each points
        % for R1
        [R1minD,R1Label] = min(Dist_R1,[],2); %R1Label is the all label for R1 points
        %for R2
        [R2minD,R2Label] = min(Dist_R2,[],2); %R1Label is the all label for R1 points
        %for R3
        [R3minD,R3Label] = min(Dist_R3,[],2); %R1Label is the all label for R1 points

        %% CODE - Mean Updating - Update the cluster means
        % Write code below here:
        % cluster the points with same label
        label1 = cat(1, R1(R1Label == 1,:), R2(R2Label == 1,:), R3(R3Label == 1,:));
        label2 = cat(1, R1(R1Label == 2,:), R2(R2Label == 2,:), R3(R3Label == 2,:));
        label3 = cat(1, R1(R1Label == 3,:), R2(R2Label == 3,:), R3(R3Label == 3,:));
        % calculate the mean
        MU_current(:,1) = mean(label1,1);
        MU_current(:,2) = mean(label2,1);
        MU_current(:,3) = mean(label3,1);

        %% CODE 4 - Check for convergence 
        % Write code below here:
        MU_diff = MU_current - MU_previous;
        if (MU_diff < convergence_threshold)
            converged=1;
        else
            MU_previous = MU_current;
        end

        %% CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            all_label = cat(1, label1, label2, label3);
            label_ = cat(1,ones(size(label1,1),1),2*ones(size(label2,1),1),3*ones(size(label3,1),1));
            all_label_ = cat(2, all_label, label_);
            if i == 1
                ALL_LABEL_ = all_label_;
            else
                ALL_LABEL_ = [ALL_LABEL all_label_];
            end
            %% If converged, get WCSS metric
            % Add code below
            size_label = cat(1, size(label1),size(label2),size(label3));
            label1_square = repmat(sum(label1.*label1, 2), 1, 3);
            label2_square = repmat(sum(label2.*label2, 2), 1, 3); 
            label3_square = repmat(sum(label3.*label3, 2), 1, 3); %*3的矩阵
            
            MU_current_square1 = repmat(sum(MU_current.*MU_current, 1),size(label1,1), 1);
            MU_current_square2 = repmat(sum(MU_current.*MU_current, 1),size(label2,1), 1);
            MU_current_square3 = repmat(sum(MU_current.*MU_current, 1),size(label3,1), 1);
            
            Dist_label1 = -2*label1*MU_current + label1_square + MU_current_square1;
            Dist_label2 = -2*label2*MU_current + label2_square + MU_current_square2;
            Dist_label3 = -2*label3*MU_current + label3_square + MU_current_square3;
            wcss = sum(Dist_label1) + sum(Dist_label2) + sum(Dist_label3);
            if i == 1
                WCSS = wcss;
            else
                WCSS = [WCSS wcss];
            end
        end
    end
    t = t+1;
end
[minwcss, No.t] = min(WCSS);
fprintf('\nConverged.\n')
figure
gscatter(ALL_LABEL_(:,1),ALL_LABEL_(:,2),ALL_LABEL_(:,3))
xlabel('x1')
ylabel('x2')
title('The result of kmeans clustering')
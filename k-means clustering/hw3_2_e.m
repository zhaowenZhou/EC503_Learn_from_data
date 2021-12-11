% EC 503 - HW 3 - Fall 2021
% K-Means starter code

clear, clc, close all;

%% Generate NBA data:
% Add code below:
all_stats = readmatrix('NBA_stats_2018_2019.xlsx');
DATA = all_stats(2:589,[5 7]);
%for K = 2:1:10 %循环几个cluster
K = 10;
NeedData=(DATA(randi(length(DATA),1,K),:)).';% data是原始数据，randi(a,1,b)是从data数据库的前a行抽取b行的随机样本，并保存在Need Data数据库中。
MU_init = NeedData; % all initializations points 2*30 matrix
labels = ones(length(DATA),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;
for i = 1:9*K:K %循环一个cluster中不同的initial point
    MU_previous = MU_init(:,i:i+K-1);
    MU_current = MU_init(:,i:i+K-1); %分配initial point

    % 分配完initial point后开始实施算法步骤
    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)
        %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % 计算距离
        % 运算结果为150*K的矩阵，数据到这K个点的距离
        DATA_square = repmat(sum(DATA.*DATA,2),1,K);
        MU_previous_square = repmat(sum(MU_previous.*MU_previous,1),length(DATA),1);
        Dist = -2*DATA*MU_previous + DATA_square + MU_previous_square;
        [Dist_min,Data_label] = min(Dist,[],2); % 得到了每一个数据的label
        %% CODE - Mean Updating - Update the cluster means
        % 更新current
        for l = 1:K
            MU_current(:,K) = mean(DATA(Data_label == K));
        end           
        %% CODE 4 - Check for convergence 
        MU_diff = MU_current - MU_previous;
        MU_diff(isnan(MU_diff) == 1) = 0;
        if (MU_diff < convergence_threshold)
            converged=1;
        else
            MU_previous = MU_current;
        end

        %% CODE 5 - Plot clustering results if converged:
        if (converged == 1)
%               fprintf('\nConverged.\n')
%               figure
%               gscatter(DATA(:,2),DATA(:,1),Data_label)
            %% If converged, get WCSS metric
            %这个算的是不同的initial point的wcss
            wcss_cluster = sum(Dist_min);
            if i == 1
                WCSS_cluster = wcss_cluster;
                datalabel = Data_label;
            else
                WCSS_cluster = [WCSS_cluster wcss_cluster]; %这个是一个K下全部的wcss
                datalabel = [datalabel Data_label];
            end
            % 找到一个K下不同的initial point中最小的wcss
        end            
    end 
end
[wcss_min, wcss_index] = min(WCSS_cluster); %在initial point循环完成后找到最小的
gscatter(DATA(:,1),DATA(:,2),datalabel(:,wcss_index))
xlabel('MPG')
ylabel('PPG')
title('Cluster of NBA player stats')





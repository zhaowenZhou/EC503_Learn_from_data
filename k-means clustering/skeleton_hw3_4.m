% EC 503 - HW 3 - Fall 2021
% DP-Means starter code
%Some obstacles occured when running the code, and the solution is inspired
%by the resource from
%"https://github.com/zxy9815/MatlabMachineLearning/blob/master/Clustering/DP_centers.m"
clear, clc, close all,

%% Generate Gaussian data:
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
all_stats = readmatrix('NBA_stats_2018_2019.xlsx');
NBA_DATA = all_stats(2:589,[5 7]);

%% DP Means method:

% Parameter Initializations
l = [44 100 450];
convergence_threshold = 1;
num_points = length(NBA_DATA);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
for count = 1:3
    LAMBDA = l(count);
    K = 1;
    K_prev = K;


    % sets of points that make up clusters
    L = {};
    L = [L [1:num_points]];

    % Class indicators/labels
    Z = ones(1,num_points); 

    % means
    MU = [];
    MU = [MU; mean(NBA_DATA,1)]; % 新元素会形成新的一行
    MU_prev = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initializations for algorithm:
    converged = 0;
    t = 0;
    while (converged == 0) 
        t = t + 1;
        fprintf('Current iteration: %d...\n',t)
        K_prev = K;
        MU_prev = MU;

        for i = 1:length(NBA_DATA) %对数据集中按点遍历
            dist = [];
        %% CODE 1 - Calculate distance from current point to all currently existing clusters
            for j = 1:K
                disti = (NBA_DATA(i,1) - MU(j,1)).^2 + (NBA_DATA(i,2) - MU(j,2)).^2;
                dist = [dist disti];
            end
            [disti_min, ilabel] = min(dist);
            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            if disti_min > LAMBDA
                K = K+1;
                Z(i) = K;
                MU = [MU; NBA_DATA(i, :)];
            else
                Z(i) = ilabel;
            end
        end
        % 完成上述步骤就是完成了划分 MU和每个点的label都有了

            %% CODE 4 - Recompute means per cluster
            % Write code below here:
        clusters = zeros(K, 2);
        for c = 1:K
            clusters(c, 1) = mean(NBA_DATA(Z == c, 1));
            clusters(c, 2) = mean(NBA_DATA(Z == c, 2));
        end
        MU = clusters;

            %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
            % Write code below here:
        if K_prev == K && (max(max(MU - MU_prev))) < convergence_threshold
            converged = 1;
        end
            %% CODE 6 - Plot final clusters after convergence 
            % Write code below here:

            if (converged)
                figure
                gscatter(NBA_DATA(:,1),NBA_DATA(:,2),Z(:).')
                xlabel('x1')
                ylabel('x2')
                title(['the result of DP means of lambda = ', num2str(LAMBDA),' '])
                %%%%
            end    
    end
end




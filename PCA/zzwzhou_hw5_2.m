function [lambda_top5, k_] = zzwzhou_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    faces = load_faces();
    faces = faces.';
    %%%%% TODO

    %% Compute mean face and the covariance matrix of faces
    % compute X_tilde
    %%%%% TODO
    mean_faces = mean(faces, 2);
    % Compute covariance matrix using X_tilde
    %%%%% TODOx
    X_tilde = faces - mean_faces;
    S_faces = (1/400)*(X_tilde * X_tilde.');
%     faces_hat = faces * (eye(400) - (1/400)*ones(400, 400)*(ones(400, 400).'));
%     S_faces = (faces_hat * faces_hat.')/400;                             %d*d matrix
    
    %% Compute the eigenvalue decomposition of the covariance matrix
    %%%%% TODO
    [vector, value] = eig(S_faces);

    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    %%%%% TODO
    [evalue, index] = sort(diag(value), 'descend');
    lambda_top5 = evalue(1:5, :);
    e_value = value(index, index);
    e_vector = vector(:, index);  
    %% Compute the principal components: Y
    %%%%% TODO

%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(faces(:, 120), img_size)))
    title('the #120 face')
    % Visualize the mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(uint8(reshape(mean_faces, img_size)))
    title('the mean face')
%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    
    % Report the top 5 eigenvalues
    % lambda_top5 = ?; %%%%% TODO
    
    % Plot the eigenvalues in from largest to smallest
    d = 450;
    k = 1:d;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')

    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    %%%%% TODO
    plot(k, evalue(k));
    xlabel('the value of k')
    ylabel('the value of lambda k')
    title('the eigenvalue against k')
    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    
    subplot(1,2,2)
    %%%%% TODO
    sum_evalue_k = zeros(d, 1);
    for i = 1:d
        sum_evalue_k(i, 1) = sum(evalue(1:i, 1));
    end
    fraction_k = round((sum_evalue_k)/sum(evalue), 2); % round to 2 decimals
    plot(k, fraction_k)
    xlabel('the value of k')
    ylabel('the values of fraction of variance explained by k principal components')
    title('the fraction against k')
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    %%%%% TODO (Hint: ismember())
    [~, res_index] = ismember(ef, fraction_k);
    
    k_ = k(1, res_index); %%%%% TODO
    
%% Q5.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = faces(:,test_img_idx);    
    % Compute eigenface coefficients
    %%%% TODO
    y_hat = zeros(500, 1);
    for i = 1:1:500
        y_hat(i, 1) = e_vector(:, i).'*(test_img - mean_faces);
    end
    x_hat = zeros(10304, 7);
    iteration = 1;
    for K = [1 2 6 29 105 179 300]
        p = 1;
        y_sum_hat = zeros(10304, 1);
        while p <= K
            y_sum_hat = y_sum_hat + y_hat(p, :)*e_vector(:, p);
            p = p + 1;
        end
        x_hat(:, iteration) = mean_faces + y_sum_hat;
        iteration = iteration + 1;
    end
        
            
        
        
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to the mean face

    % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.

    %%%% TODO 
    
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    subplot(3, 3, 1)
    imshow(uint8(reshape(mean_faces, img_size)))
    title('mean face')
    K = [ 1 2 6 29 105 179 300];
    for f = 1:1:7
        subplot(3, 3, 1+f)
        imshow(uint8(reshape(x_hat(:, f), img_size)))
        title({'the image when k = ', num2str(K(1, f))})
    end
    
    subplot(3, 3, 9)
    imshow(uint8(reshape(test_img, img_size)))
    title('the test image');

%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of quantile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,~] = size(X);  % X是n*d的矩阵
    X = X.'; % X变成了d*n的矩阵
    
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    mean_digit = mean(X, 2);
    X_tilde = X - mean_digit;
    % Compute covariance using X_tilde
    %%%%% TODO
    S_X = (1/400)*(X_tilde * X_tilde.');
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    [vector, value] = eig(S_X);
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    [evalue, index] = sort(diag(value), 'descend');
    e_value = value(index, index);
    e_vector = vector(:, index);  
    %% Compute principal components
    %%%%% TODO
    y_pca = e_vector.' * X_tilde;
    %% Computing the first 2 pricipal components
    %%%%% TODO
    y_pca_1 = y_pca(1, :).';
    y_pca_2 = y_pca(2, :).';
    % finding percentiles points
    percentiles_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - quantile_points())
    P_th_percentile_1 = percentile_values(y_pca_1, percentiles_vals);
    P_th_percentile_2 = percentile_values(y_pca_2, percentiles_vals);
    % Finding the cartesian product of quantile points to find grid corners
    %%%%% TODO
    [temp1,temp2] = meshgrid(P_th_percentile_1,P_th_percentile_2);
    Cartesian = [temp1(:) temp2(:)];
    Cartesian = flip(Cartesian, 2);
    
    %% Find images whose PCA coordinates are closest to the grid coordinates 
    y_pca_dist = y_pca((1:2), :);
    y_pca_dist_sq = sum((y_pca_dist.*y_pca_dist));
    cartesian_sq = sum((Cartesian.*Cartesian), 2);
    y_pca_Carte_dist = repmat(y_pca_dist_sq, 25, 1) + repmat(cartesian_sq, 1, 658) - 2*Cartesian*y_pca_dist;
    [min_dist, I] = min(y_pca_Carte_dist, [], 2);
    I = unique(I);
    
    %%%%% TODO

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 120th image
    subplot(1,2,1)
    %%%%% TODO
    imshow(reshape(X(:, 120), img_size))
    title('the 120th image')
    % Mean digital image
    subplot(1,2,2)
    %%%%% TODO
    imshow(reshape(mean_digit, img_size))
    title('the mean image')
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    subplot(1, 2, 1)
    hold on
    grid on
    scatter(y_pca(1,:), y_pca(2,:))
    xlabel('the second principle component')
    ylabel('the first principle component')
    yticks(P_th_percentile_1)
    xticks(P_th_percentile_2)
    subplot(1, 2, 2)
    hold on
    grid on
    scatter(y_pca(1,:), y_pca(2,:)), scatter(y_pca(1, I), y_pca(2, I), 'r', 'filled')
    xlabel('the second principle component')
    ylabel('the first principle component')
    title('red points are points nearest to the intersection')
    yticks(P_th_percentile_1)
    xticks(P_th_percentile_2)    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the quantile points and highlight the image points that are closest to the 
    %% quantile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)

    hold off
    title('Image points closest to percentile grid corners')
    
    
    figure(6)
    sgtitle('Images closest to percentile grid corners')
    hold on
    for k = 1:25
        subplot(5, 5, k)
        imshow(reshape(X(:, I(k, :)), img_size))
        title({'the number of the image is = ', num2str(I(k, 1))})
    end

    
    %%%%% TODO
    
    hold off    
end
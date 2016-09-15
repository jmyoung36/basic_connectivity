% set data directory
% for work desktop machine
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/';
% for macbook air
%data_dir = '/Users/jonyoung/Data/Connectivity data/';

% read in some data
data = csvread([data_dir, 'M_timecourse_connectivity_data.csv']);
labels = data(:, 1);
connectivity_data = data(:, 2:end);

% optional - only include positive connections
connectivity_data(connectivity_data < 0) = 0;

% optional - set diagonal to 0
% diag_inds = find(eye(90) == 1);
% connectivity_data(:, diag_inds) = 0;

% correct labels for GPML
labels(labels == 0) = -1;

% to hold predicted probabilities
%ps = zeros(length(labels), 1);

% get a set of k-fold cross-validations
% k = 10;
% kf = kFold(length(labels), k);

% to hold predicted results and labels
ps = [];
all_test_labels = [];

% get a set of k mccv splits
k = 200;
kf = kMCCV(length(labels), k, 0.1);

% results
predicted_p = zeros(length(labels));

tril_inds = find(~tril(ones(90, 90)));

for i = 1:k
    
    training_indices = kf(i).training_indices;
    testing_indices = kf(i).testing_indices;
    training_data = connectivity_data(training_indices, :);
    testing_data = connectivity_data(testing_indices, :);
    training_labels = labels(training_indices);
    testing_labels = labels(testing_indices);
    
    training_connectivity_matrices = squeeze(reshape(training_data, [size(training_indices) 90 90]));
    testing_connectivity_matrices = squeeze(reshape(testing_data, [size(testing_indices) 90 90]));
    base_connectivity_matrix = squeeze(mean(training_connectivity_matrices, 1));
    
    disp('Creating groupwise base connectivity matrix...')
    
    % create a group base connectivity matrix
    for j = 1:20
    
        % project all matrices into the tangent space
        tangent_matrices = zeros(size(training_connectivity_matrices)); 
        for k = 1:length(training_indices)
            
            tangent_matrices(k, : , :) = (base_connectivity_matrix ^ 0.5) * logm((base_connectivity_matrix ^ -0.5) * squeeze(training_connectivity_matrices(k, :, :)) * (base_connectivity_matrix ^ -0.5)) * (base_connectivity_matrix ^ 0.5);        
            
        end
        
        % calculate the tangent space mean
        tangent_space_base_connectivity_matrix = squeeze(mean(tangent_matrices, 1));
        
        % project the new tangent space mean back to the manifold
        base_connectivity_matrix = (base_connectivity_matrix ^ 0.5) * expm((base_connectivity_matrix ^ -0.5) * tangent_space_base_connectivity_matrix * (base_connectivity_matrix ^ -0.5)) * (base_connectivity_matrix ^ 0.5);       
        
    end
    
    disp('Applying matrix whitening transport...')
    
    % apply whitening transport and projection for training and testing data
    training_connectivity_matrices_transformed = zeros(size(training_connectivity_matrices));
    testing_connectivity_matrices_transformed = zeros(size(testing_connectivity_matrices));
    
    for j = 1:length(training_indices)
        
        training_connectivity_matrices_transformed(j, :, :) = logm((base_connectivity_matrix ^ -0.5) * squeeze(training_connectivity_matrices(j, : ,:)) * (base_connectivity_matrix ^ -0.5));   
        
    end
    for j = 1:length(testing_indices)
        
        testing_connectivity_matrices_transformed(j, :, :) = logm((base_connectivity_matrix ^ -0.5) * squeeze(testing_connectivity_matrices(j, : ,:)) * (base_connectivity_matrix ^ -0.5));   
        
    end
    
    disp('Reshaping transported matrices to vectors...')
    
    % reshape to vectors and pull out lower triangle to form transformed
    % connectivity data
    training_data_transformed = reshape(training_connectivity_matrices_transformed, [length(training_indices) 8100]);
    training_data_transformed = training_data_transformed(:, tril_inds);
    testing_data_transformed = reshape(testing_connectivity_matrices_transformed, [length(testing_indices) 8100]); 
    testing_data_transformed = testing_data_transformed(:, tril_inds);
    
    % create the actual kernels
    K_train = {training_data_transformed * training_data_transformed'};
    K_test = {testing_data_transformed * testing_data_transformed'};
    K_cross = {training_data_transformed * testing_data_transformed'};
    
    disp('Running GP classification...')
    
    % set up GP
    meanfunc = @meanConst_K; hyp.mean = 0;
    covfunc = @covLINMKL; hyp.cov = [0 0];
    likfunc = @likErf;
    hyp
    hyp_opt = minimize(hyp, @gp_K, -200, @infEP_K, covfunc, likfunc, meanfunc, K_train, training_labels);
    hyp_opt
    [a b c d lp post] = gp_K(hyp_opt, @infEP_K, covfunc, likfunc, meanfunc, K_train, training_labels, K_test, K_cross, ones(length(testing_labels), 1));
    %ps(testing_indices) = exp(lp);
    ps = [ps; exp(lp)];
    all_test_labels = [all_test_labels; testing_labels];
    preds = exp(lp);
    preds(preds > 0.5) = 1;
    preds(preds < 0.5) = -1;
    [acc, sens, spec] = accStats(testing_labels, preds);
    acc
    sens
    spec
end

ps(ps > 0.5) = 1;
ps(ps < 0.5) = -1;
%[acc, sens, spec] = accStats(labels, ps);
[acc, sens, spec] = accStats(all_test_labels, ps);
acc
sens
spec




    

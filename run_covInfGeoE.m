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

% correct labels for GPML
labels(labels == 0) = -1;

% create kernel
K = zeros(size(connectivity_data, 1));

for i = 1:size(connectivity_data,1);
    for j = 1:i
        
        % reshape connectivity vectors in to matrices
        G1 = connectivity_data(i, :);
        G2 = connectivity_data(j, :);
        M1 = reshape(G1, [90, 90]);
        M2 = reshape(G2, [90, 90]);
        
        % calculate and store geodesic distances
        element = (norm(logm((M1 ^ (-0.5)) * M2 * (M1 ^ (-0.5))), 'fro')) ^ 2;
        K(i, j) = element;
        K(j, i) = element;     
    end    
end

% get a set of k-fold cross-validations
k = 10;
kf = kFold(length(labels), k);

% results
predicted_p = zeros(length(labels));

for i = 1:k
    
    training_indices = kf(i).training_indices;
    testing_indices = kf(i).testing_indices;
    %K_train = {K(training_indices, training_indices)};
    %K_test = {K(testing_indices, training_indices)};
    K_train = {connectivity_data(training_indices, :) * connectivity_data(training_indices, :)'};
    K_test = {connectivity_data(testing_indices, :) * connectivity_data(training_indices, :)'};
    training_labels = labels(training_indices);
    testing_labels = labels(testing_indices);
    
    % get mean value of K_train to initialise
    mean_dist = sqrt(mean2(K_train{1}));
    
    % set up GP
    meanfunc = @meanConst_K; hyp.mean = 0;
    covfunc = @covLINMKL; hyp.cov = [0 0];
    likfunc = @likErf;
    %hyp = minimize(hyp, @gp_K, -40, @infEP_K, meanfunc, covfunc, likfunc, K_train, training_labels);
    hyp_opt = minimize(hyp, @gp_K, -200, @infEP_K, covfunc, likfunc, meanfunc, K_train, training_labels);

    
end




    

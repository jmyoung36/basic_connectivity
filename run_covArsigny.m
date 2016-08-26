% set data directory
% for work desktop machine
%data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/';
% for macbook air
data_dir = '/Users/jonyoung/Data/Connectivity data/';

% read in some data
data = csvread([data_dir, 'M_timecourse_connectivity_data.csv']);
labels = data(:, 1);
connectivity_data = data(:, 2:end);

% optional - only include positive connections
%connectivity_data(connectivity_data < 0) = 0;

% correct labels for GPML
labels(labels == 0) = -1;

% create transformed connectivity data
transformed_connectivity_data = zeros(size(connectivity_data));

for i = 1:size(connectivity_data,1);
        
    % reshape connectivity vectors in to matrices
    M = reshape(connectivity_data(i, :), [90, 90]);
        
    % calcuate the log
    logConnectivity = logm(M);
    
    % store back into the new data array, keeping only the lower triangle
    transformed_connectivity_data(i, :) = reshape(logConnectivity, [1 8100]);

end    

% pull out lower triangle elements
tril_inds = find(~tril(ones(90, 90)));
transformed_connectivity_data = transformed_connectivity_data(:, tril_inds);
size(transformed_connectivity_data)

% to hold predicted probabilities
ps = zeros(length(labels), 1);

% get a set of k-fold cross-validations
k = 10;
kf = kFold(length(labels), k);

% results
predicted_p = zeros(length(labels));

for i = 1:k
    
    training_indices = kf(i).training_indices;
    testing_indices = kf(i).testing_indices;
    K_train = {connectivity_data(training_indices, :) * connectivity_data(training_indices, :)'};
    K_test = {connectivity_data(testing_indices, :) * connectivity_data(testing_indices, :)'};
    K_cross = {connectivity_data(training_indices, :) * connectivity_data(testing_indices, :)'};
%     K_train = {transformed_connectivity_data(training_indices, :) * transformed_connectivity_data(training_indices, :)'};
%     K_test = {transformed_connectivity_data(testing_indices, :) * transformed_connectivity_data(testing_indices, :)'};
%     K_cross = {transformed_connectivity_data(training_indices, :) * transformed_connectivity_data(testing_indices, :)'};
    training_labels = labels(training_indices);
    testing_labels = labels(testing_indices);
    
    % get mean value of K_train to initialise
    mean_dist = sqrt(mean2(K_train{1}));
    
    % set up GP
    meanfunc = @meanConst_K; hyp.mean = 0;
    covfunc = @covLINMKL; hyp.cov = [0 -100];
    likfunc = @likErf;
    hyp_opt = minimize(hyp, @gp_K, -200, @infEP_K, covfunc, likfunc, meanfunc, K_train, training_labels);
    [a b c d lp post] = gp_K(hyp_opt, @infEP_K, covfunc, likfunc, meanfunc, K_train, training_labels, K_test, K_cross, ones(length(testing_labels), 1));
    ps(testing_indices) = exp(lp);
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
[acc, sens, spec] = accStats(labels, ps);
acc
sens
spec




    

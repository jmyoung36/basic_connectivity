% set data directory
% for work desktop machine
data_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/QUIC_connectivity/';
% for macbook air
%data_dir = '/Users/jonyoung/Data/Connectivity data/';

% read in some data
data = csvread([data_dir, 'labeled_log_connectivity_data_25.csv']);
labels = data(:, 1);
connectivity_data = data(:, 2:end);

% optional - only include positive connections
%connectivity_data(connectivity_data < 0) = 0;

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
k = 10;
kf = kMCCV(length(labels), k, 0.1);

% results
predicted_p = zeros(length(labels));

size(labels)
size(connectivity_data)

ps = [];
all_test_labels = [];

for i = 1:k
    
    training_indices = kf(i).training_indices;
    testing_indices = kf(i).testing_indices;
    training_labels = labels(training_indices);
    testing_labels = labels(testing_indices);
    training_data = connectivity_data(training_indices, :);
    testing_data = connectivity_data(testing_indices, :);
    
    % set up GP
    meanfunc = @meanConst; hyp.mean = 0;
    covfunc = @covSEard; hyp.cov = ones(301, 1);
    likfunc = @likErf;
    hyp_opt = minimize(hyp, @gp, -200, @infEP, meanfunc, covfunc, likfunc, training_data, training_labels);
    [a b c d lp post] = gp(hyp_opt, @infEP, meanfunc, covfunc, likfunc, training_data, training_labels, testing_data, ones(length(testing_labels), 1));
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





    

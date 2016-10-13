% set data directory
% for work desktop machine
data_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/QUIC_connectivity/';
% for macbook air
%data_dir = '/Users/jonyoung/Data/Connectivity data/';

n_regions = 15;

% read in some data
data = csvread([data_dir, 'sq_IG_dists_', int2str(n_regions), '_regions.csv']);
labels = data(:, 1);
K = data(:, 2:end);

% correct labels for GPML
labels(labels == 0) = -1;

% to hold predicted probabilities
%ps = zeros(length(labels), 1);


% to hold predicted results and labels
ps = [];
all_test_labels = [];

% get a set of k mccv splits
k = 200;
kf = kMCCV(length(labels), k, 0.1);

% results
predicted_p = zeros(length(labels));

for i = 1:k
    
    training_indices = kf(i).training_indices;
    testing_indices = kf(i).testing_indices;
    K_train = {K(training_indices, training_indices)};
    K_cross = {K(training_indices, testing_indices)};
    K_test = {K(testing_indices, testing_indices)};
    training_labels = labels(training_indices);
    testing_labels = labels(testing_indices);
    
    % get mean value of K_train to initialise
    mean_dist = sqrt(mean2(K_train{1}));
    
    % set up GP
    meanfunc = @meanConst_K; hyp.mean = 0;
    covfunc = @covSEMKL; hyp.cov = [log(mean_dist) 0 0];
    likfunc = @likErf;
    hyp_opt = minimize(hyp, @gp_K, -200, @infEP_K, covfunc, likfunc, meanfunc, K_train, training_labels);
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





    

% set data directory
% for work desktop machine
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/';
% for macbook air
%data_dir = '/Users/jonyoung/Data/Connectivity data/';

% read in some data
data = csvread([data_dir, 'M_connectivity_data.csv']);
%data = data(1:20, :);
labels = data(:, 1);
connectivity_data = data(:, 2:end);

% optional - remove negative connections
connectivity_data(connectivity_data < 0) = 0;

% correct labels for GPML
labels(labels == 0) = -1;

% to hold predicted probabilities
ps = zeros(length(labels), 1);

% get a set of k-fold cross-validations
k = 10;
kf = kFold(length(labels), k);

for i = 1:k
    
    training_indices = kf(i).training_indices;
    testing_indices = kf(i).testing_indices;
    training_data = connectivity_data(training_indices, :);
    testing_data = connectivity_data(testing_indices, :);
    training_labels = labels(training_indices);
    testing_labels = labels(testing_indices);
    
    % find correct initial value for sigma
    mean_sq_dist = 0;
    c = 0;

    for j = 1:size(training_data,1);
        for k = 1:j;
        
            G1 = training_data(j, :);
            G2 = training_data(k, :);
            M1 = reshape(G1, [90, 90]);
            M2 = reshape(G2, [90, 90]);
            D1 = diag(sum(M1, 2));
            D2 = diag(sum(M2, 2));
            L1 = D1 - M1;
            L2 = D2 - M2;
            min_diag = min([min(diag(L1)), min(diag(L2))]);
            gamma = 10;
            S1 = L1 + (gamma * eye(90));
            S2 = L2 + (gamma * eye(90));
            dist = norm((logm(S1) - logm(S2)), 'fro');
            mean_sq_dist = mean_sq_dist + dist ^ 2;
            c = c+1;        
            
        end    
    end
    
    % set up GP
    mean_sq_dist = mean_sq_dist/c;

    % set up GP classification
    % initialise g to 0 (i.e. log(1) ) and s to log(mean_sq_dist)
    meanfunc = @meanConst; hyp.mean = 0;
    covfunc = @covLogE; gamma = 10; sigma = mean_sq_dist; hyp.cov = log([gamma sigma]);
    likfunc = @likErf;
    hyp.cov
    hyp_opt = minimize(hyp, @gp, -100, @infEP, meanfunc, covfunc, likfunc, training_data, training_labels);
    hyp_opt.cov
    [a b c d lp] = gp(hyp_opt, @infEP, meanfunc, covfunc, likfunc, training_data, training_labels, testing_data, ones(length(testing_labels), 1));
    p = exp(lp);
    ps(testing_indices) = p;
    p(p > 0.5) = 1;
    p(p < 0.5) = -1; 
    [acc, sens, spec] = accStats(testing_labels, p);
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







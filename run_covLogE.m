% set data directory
% for work desktop machine
%data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/';
% for macbook air
data_dir = '/Users/jonyoung/Data/Connectivity data/';

% read in some data
data = csvread([data_dir, 'M_connectivity_data.csv']);
labels = data(:, 1);
connectivity_data = data(:, 2:end);

connectivity_data(connectivity_data < 0) = 0;

% correct labels for GPML
labels(labels == 0) = -1;

% find correct initial value for sigma
mean_sq_dist = 0;
c = 0;

for i = 1:size(connectivity_data,1);
    for j = 1:i
        
        G1 = connectivity_data(i, :);
        G2 = connectivity_data(j, :);
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

mean_sq_dist = mean_sq_dist/c;

% set up GP classification
% initialise g to 0 (i.e. log(1) ) and s to log(mean_sq_dist)
meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covLogE; gamma = 1.0; sigma = mean_sq_dist; hyp.cov = log([gamma sigma]);
likfunc = @likErf;

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, connectivity_data, labels);
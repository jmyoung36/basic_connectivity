% set data directory
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/';

% read in some data
data = csvread([data_dir, 'M_connectivity_data.csv']);
labels = data(:, 1);
connectivity_data = data(:, 2:end);

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
        M1(M1 < 0) = 0;
        M2(M2 < 0) = 0;
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
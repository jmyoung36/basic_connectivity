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

mean_sq_dist = 4;

element = calcLogE_test(connectivity_data(1, :), connectivity_data(2, :), exp(0), log(mean_sq_dist));

connectivity_data_mini = connectivity_data(1:4, :);

foo = pdist(connectivity_data_mini, @(G1, G2)calcLogE_test(G1, G2, exp(0), log(mean_sq_dist)));
% foo = zeros(4);
% for i = (1:4);
%     for j = (1:4);
%         G1 = connectivity_data_mini(i, :);
%         G2 = connectivity_data_mini(j, :);
%         foo1 = calcLogE_test(G1, G2, exp(0), log(mean_sq_dist));
%         foo(i, j) = foo1;
%         foo(j, i) = foo1;
%     end
% end
%foo1 = pdist(connectivity_data_mini, @(G1, G2)simpleDist_test(G1, G2, exp(0)));

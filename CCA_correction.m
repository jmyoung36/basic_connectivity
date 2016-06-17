% read in connectivity data
connectivity_data = csvread('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/connectivity_data.csv');

% read in labels
label_data = csvread('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/K_edge.csv');
labels = label_data(:,1);
labels(labels == 0) = -1;

% trick to get lower triangle only
% build matrix of same size as connectivity matrix with ones in lower
% triangle and zeros in diagonal and upper triangle
M = zeros([90, 90]) + tril(ones([90, 90])) - eye(90);
ind = find(M);
connectivity_data = connectivity_data(:, ind);

% split into blocks for CCA
X = connectivity_data(1:140, :);
Y = connectivity_data(141:280,:);
Y_extra = connectivity_data(281:end,:);

% do CCA
[A,B,r] = canoncorr(X,Y);
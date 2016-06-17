% read in connectivity data
connectivity_data = csvread('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/connectivity_data.csv');

% set up centre indicator variable
centres = [zeros(140,1); ones(193,1)];

% read in labels
label_data = csvread('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/K_edge.csv');
labels = label_data(:,1);

% trick to get lower triangle only
% build matrix of same size as connectivity matrix with ones in lower
% triangle and zeros in diagonal and upper triangle
M = zeros([90, 90]) + tril(ones([90, 90])) - eye(90);
ind = find(M);
connectivity_data = connectivity_data(:, ind);

% shuffle everything
r = randperm(length(centres));
labels = labels(r);
centres = centres(r);
connectivity_data = connectivity_data(r, :);

% confound matrix L is linear kernel on centres
L = centres * centres';

% transpose connectivity_data for ccSVM
X = connectivity_data';

%the setting up to select parameter lambda and C
LambdaRange = [1e-8,1e-4,1e-2,1,1e+2,1e+4,1e+8];
CRange = 2.^[-8,-4,-2,0,2,4,8];
kfold = 2;

%%TO choose training and test dataset
[n,m] = size(X);
% CVO = cvpartition(m,'k',10);
% test = find(CVO.test(1));
% train = find(CVO.training(1)); 

% n-fold CV loop
n_folds = 10;
step_size = ceil(m/n_folds);
accs_corrected = zeros(n_folds, 1);
preds_corrected = zeros(m, 1);
accs = zeros(n_folds, 1);
preds = zeros(m, 1);
for i = 1:n_folds
    
    start_ind = (i-1) * step_size + 1;
    stop_ind = min(start_ind + step_size - 1, m);
    train = [1:m];
    test = [start_ind:stop_ind];
    train(start_ind:stop_ind) = [];
    
    %parameter selection by kfold cross validation based only on training data
    [lambda,C] = ParameterSetting(X(:,train),labels(train),L(train,train),LambdaRange,CRange,kfold);

    %to do prediction on test data using ccSVM
    [Predict_label,dec,accuracy,ccauc,w] = ccSVM(X,train,test,labels,L,lambda,C);


    %to do prediction on test data using standard SVM, setting lambda as 0
    [Predict_label_2,dec_2,accuracy_2,svmauc_2,w_2] = ccSVM(X,train,test,labels,L,0,C);
    
    % store results
    accs_corrected(i) = accuracy(1);
    preds_corrected(test) = Predict_label;   
    accs(i) = accuracy_2(1);
    preds(test) = Predict_label_2;   
end





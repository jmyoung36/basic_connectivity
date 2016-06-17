% read in connectivity data
connectivity_data = csvread('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/connectivity_data.csv');

% set up centre indicator variable
centres = [zeros(140,1); ones(193,1)];

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

% shuffle everything
r = randperm(length(centres));
labels = labels(r);
centres = centres(r);
connectivity_data = connectivity_data(r, :);

% concatenate centres and into a single matrix
data = [connectivity_data, centres];

% masks
connectivity_mask = [ones(4005, 1); zeros(1, 1)];
centre_mask = [zeros(4005, 1); ones(1, 1)];

% n-fold CV loop
n = size(data, 1);
n_folds = 10;
step_size = ceil(n/n_folds);
p = zeros(n, 1);

for i = 1:n_folds
    
    start_ind = (i-1) * step_size + 1;
    stop_ind = min(start_ind + step_size - 1, n);
    train = [1:n];
    test = [start_ind:stop_ind];
    train(start_ind:stop_ind) = [];
    
    train_data = data(train, :);
    test_data = data(test, :);
    train_labels = labels(train);
    test_labels = labels(test);
    
    % set up GP covariance function
    % base functions and hyps for centre
    cov_lin_cent={@covLIN};
    cov_SE_cent={@covSEiso}; hyp_cov_SE_cent = [0; 0];
    cov_const_cent={@covConst}; hyp_cov_const_cent = 0;
    cov_delta_cent={@covDelta}; hyp_cov_delta_cent = 0;

    % mask the functions for the centre
    cov_lin_cent_masked={@covMask,{centre_mask,cov_lin_cent{:}}};
    cov_SE_cent_masked={@covMask,{centre_mask,cov_SE_cent{:}}};
    cov_const_cent_masked={@covMask,{centre_mask,cov_const_cent{:}}};
    cov_delta_cent_masked={@covMask,{centre_mask,cov_delta_cent{:}}};

    % weight the cov_lin_cent_masked function
    cov_lin_cent_masked_scaled={@covScale,{cov_lin_cent_masked{:}}}; hyp_cov_lin_cent_masked_scaled = 0;

    % base function for connectivity
    cov_lin_conn={@covLIN};
    %cov_SE_conn={@covSEiso}; hyp_cov_SE_conn = [0; 0];

    % mask it
    cov_lin_conn_masked={@covMask,{connectivity_mask,cov_lin_conn{:}}};
    %cov_SE_conn_masked={@covMask,{connectivity_mask,cov_SE_conn{:}}};

    % weight the cov_lin_cent_masked function
    cov_lin_conn_masked_scaled={@covScale,{cov_lin_conn_masked{:}}}; hyp_cov_lin_conn_masked_scaled = 0;

    % finally add all the covariance functions
    cov_sum_scaled={@covSum, {cov_lin_conn_masked_scaled, cov_lin_cent_masked_scaled, cov_SE_cent_masked, cov_const_cent_masked, cov_delta_cent_masked}};
    %cov_sum_scaled={@covSum, {cov_SE_conn_masked, cov_lin_cent_masked_scaled, cov_SE_cent_masked, cov_const_cent_masked, cov_delta_cent_masked}};

    % and collect the hyperparameters
    hyp_cov_sum_scaled = [hyp_cov_lin_conn_masked_scaled; hyp_cov_lin_cent_masked_scaled; hyp_cov_SE_cent; hyp_cov_const_cent; hyp_cov_delta_cent];
    %hyp_cov_sum_scaled = [hyp_cov_SE_conn; hyp_cov_lin_cent_masked_scaled; hyp_cov_SE_cent; hyp_cov_const_cent; hyp_cov_delta_cent];

    % zero-mean function
    meanfunc = @meanConst; hyp.mean = 0;

    cov=cov_sum_scaled;
    likfunc = @likErf;
    hyp.cov = hyp_cov_sum_scaled;
        
    % set hyperparameters
    hyp = minimize(hyp, @gp, -200, @infEP, meanfunc, cov, likfunc, train_data, train_labels);
    
    % make predictions
    [a b c d lp] = gp(hyp, @infEP, meanfunc, cov, likfunc, train_data, train_labels, test_data, ones(length(test), 1));
    
    % store results
    p(test) = exp(lp);

end

foo = p > 0.5;
foo = double(foo);
foo(foo == 0) = -1;
foo1 = foo == labels;
sum(foo1)
        
% read in connectivity data
connectivity_data = csvread('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/connectivity_data.csv');

% set up centre indicator variable
centres = [zeros(140,1); ones(193,1)];

% trick to get lower triangle only
% build matrix of same size as connectivity matrix with ones in lower
% triangle and zeros in diagonal and upper triangle
M = zeros([90, 90]) + tril(ones([90, 90])) - eye(90);
ind = find(M);
connectivity_data = connectivity_data(:, ind);

% set up 10-fold CV loop
n = size(connectivity_data, 1);
step_size = ceil(n/10);

% set up matrix to store results
estimated_results = zeros(size(connectivity_data));

% loop through all columns (connections)
for i=1:size(connectivity_data, 2)
    
    i
    targets = connectivity_data(:,i);
    
    % set up GP covariance function
    % base functions and hyps
    cov_lin={@covLIN};
    cov_SE={@covSEiso}; hyp_cov_SE = [0; 0];
    cov_const={@covConst}; hyp_cov_const = 0;
    cov_delta={@covDelta}; hyp_cov_delta = 0;
    
    % weight the cov_lin function
    cov_lin_scaled={@covScale,{cov_lin{:}}}; hyp_cov_lin_scaled = 0;        

    % finally add all the covariance functions
    cov_sum_scaled={@covSum,{cov_lin_scaled ,cov_SE, cov_const, cov_delta}};

    % and collect the hyperparameters
    hyp_cov_sum_scaled = [hyp_cov_lin_scaled; hyp_cov_SE; hyp_cov_const; hyp_cov_delta];

    % zero-mean function
    meanfunc = @meanZero;

    cov=cov_sum_scaled;
    likfunc = @likGauss;
    hyp.cov = hyp_cov_sum_scaled;
    hyp.lik = 0;
        
    % set hyperparameters
    hyp = minimize(hyp, @gp, -200, @infExact, meanfunc, cov, likfunc, centres, targets);
        
    % make predictions
    [mF s2F] = gp(hyp, @infExact, meanfunc, cov, likfunc, centres, targets, centres);
     
    % store predictions
    estimated_results(:, i) = mF;
        
end

% subtract the estimated results from the original data to find the
% resiudals
residuals = connectivity_data - estimated_results;

% save the residuals
csvwrite('/home/jonyoung/IoP_data/Data/connectivity_data/kernels/adjusted_connectivity_data.csv', residuals);




function K = covLogE(hyp, x, z, i)

% Log-Euclidean kernel for graphs represented as equivalent adjacency
% matrices with correspondence between nodes, based on L. Dodero, Hà Quang 
% Minh, M. San Biagio, V. Murino and D. Sona, Kernel-based Classification 
% For Brain Connectivity Graphs On The Riemannian Manifold Of Positive 
% Definite Matrices. International Symposium on Biomedical Imaging 
% (ISBI 2015)
%
% covariance function is parameterized as:
%
% k(x1, x2) = exp(-dLogE(L1, L2)^2 / sigma^2)
%
% where the L is the graph Laplacian of adjacency matrix x plus gamma * I, 
% and dLogE of two matrices M1, M2 is the Frobenius norm of 
%
% matrix_log(M1) - matrix_log(M2)
% 
% The hyperparameters are:
%
% hyp = [ log(sigma)
%         log(gamma)  ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

% function calculating a log-Euclidean kernel element for a single pair of
% graphs G1 and G2
function element = calcLogE(G1, G2, sigma, gamma)
    
% reshape G1 and G2 from 1 by n^2 vectors n by n square matrices M
n = sqrt(size(G1, 2));
M1 = reshape(G1, [n, n]);
M2 = reshape(G2, [n, n]);

% remove graph loops - set diagonal to 0
M1 = M1 - eye(size(M1));
M2 = M2 - eye(size(M2));

% calculate degree matrices D
D1 = diag(sum(M1, 2));
D2 = diag(sum(M2, 2));

% form graph Laplacians L as the degree matrix minus the original matrix
L1 = D1 - M1;
L2 = D2 - M2;

% calculate regularised Laplacians S as L + gamma I
S1 = L1 + (gamma * eye(n));
S2 = L2 + (gamma * eye(n));

% calculate diff-log ( logm(S1) - logm(S2) ) and its norm 
diff_log = logm(S1) - logm(S2);
norm_diff_log = norm(diff_log, 'fro');

% return the kernel function result
element = exp((-1 * (norm_diff_log ^ 2)) / sigma);

end

% function calculating the gradient of a log-Euclidean kernel element for 
% with respect to ln(gamma) for a single pair of graphs G1 and G2
% return a coefficient of the corresponding kernel elements as kernel
% element itself will already have been calculated
function c = gammaGrad(G1, G2, sigma, gamma, j, k)
    
if j == k
    
    c = 0;
    
else
   
    % reshape G1 and G2 from 1 by n^2 vectors n by n square matrices M
    n = sqrt(size(G1, 2));
    M1 = reshape(G1, [n, n]);
    M2 = reshape(G2, [n, n]);

    % calculate degree matrices D
    D1 = diag(sum(M1, 2));
    D2 = diag(sum(M2, 2));

    % form graph Laplacians L as the degree matrix minus the original matrix
    L1 = D1 - M1;
    L2 = D2 - M2;

    % calculate regularised Laplacians S as L + gamma I
    S1 = L1 + (gamma * eye(n));
    S2 = L2 + (gamma * eye(n));

    % calculate diff-log ( logm(S1) - logm(S2) ) and its norm as we will them
    % it more than once
    diff_log = logm(S1) - logm(S2);
    norm_diff_log = norm(diff_log, 'fro');

    % calculate intermediate matrix A
    A = gamma * (inv(S1) - inv(S2));

    % calculate intermediate value b
    b = sum(sum(diff_log .*A)) / norm_diff_log;

    % calculate and return intermediate value c
    c = (-2 * b * norm_diff_log) / sigma;

end

end

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

g = hyp(1);
s = hyp(2);
gamma = exp(g);                                
sigma = exp(s);

% precompute K as we will always use it
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    %K = sq_dist(x'/ell);
    %K = pdist(x, @(G1, G2)calcLogE(G1, G2, sigma, gamma));
    % can't use pdist as it requires the distance function to take an 
    % arbitrary number of rows. So use a loop instead.
    
    % allocate memory
    K = zeros(size(x,1));
    
    % calculate element by element
    for j = 1:size(x,1)
        for k=1:j            
            element = calcLogE(x(j, :), x(k,:), sigma, gamma);
            K(j, k) = element;
            K(k, j) = element;
        end
    end
  else                                                   % cross covariances Kxz
    %K = sq_dist(x'/ell,z'/ell);
    %K = pdist2(x, z, @(G1, G2)calcLogE(G1, G2, sigma, gamma));
    % can't use pdist2 for the same reason we can't use pdist so again use
    % a loop instead
    
    % preallocate memory
    K = zeros(size(x, 1), size(z, 1));
    for j = 1:size(x, 1)
        for k = 1:size(z, 1)
            K(j, k) = calcLogE(x(j, :), z(k,:), sigma, gamma);
        end
    end
    
  end
end

% return modes
if nargin<4                                                        % covariances
  K = K;
else                                                               % derivatives
  if i==1                                                          % wrt gamma
    %K = sf2*exp(-K/2).*K;
    % loop again
    % calculate a matrix of coefficients and multiply it by K
    K_coeffs = zeros(size(x, 1));
    for j = 1:size(x,1)
        for k=1:j            
            element = gammaGrad(x(j, :), x(k,:), sigma, gamma, j, k);
            K_coeffs(j, k) = element;
            K_coeffs(k, j) = element;
        end
    end
    %K = K.* pdist(x, @(G1, G2)gammaGrad(G1, G2, sigma, gamma));
    K = K.*K_coeffs;
    %K(1:5, 1:5)
    %disp('gamma grads')
    %K(1:5, 1:5)
  elseif i==2                                                       % wrt sigma
    %K = 2*sf2*exp(-K/2);end
    K = - K.* log(K);
    %disp('sigma grads')
    %K(1:5, 1:5)
    
  else
    error('Unknown hyperparameter')
  end
end

end












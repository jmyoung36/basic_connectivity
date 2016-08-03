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

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(x'/ell);
  else                                                   % cross covariances Kxz
    K = sq_dist(x'/ell,z'/ell);
  end
end

if nargin<4                                                        % covariances
  K = sf2*exp(-K/2);
else                                                               % derivatives
  if i==1
    K = sf2*exp(-K/2).*K;
  elseif i==2
    K = 2*sf2*exp(-K/2);
  else
    error('Unknown hyperparameter')
  end
end

% function calculating a log-Euclidean kernel element for a single pair of
% graphs G1 and G2
function element = calc_LogE(G1, G2, sigma, gamma)

% reshape G1 and G2 from 1 by N^2 vectors N by N square matrices


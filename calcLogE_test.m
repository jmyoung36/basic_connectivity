% function calculating a log-Euclidean kernel element for a single pair of
% graphs G1 and G2
function element = calcLogE_test(G1, G2, sigma, gamma)

size(G1), size(G2)
G1(:, 1:2)
G2(:, 1:2)

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


% 
% foo = norm((logm(S1) - logm(S2)), 'fro')
% foo1 = (foo ^ 2) /sigma
% foo2 = exp(foo1)

% return the kernel function result
%element = exp(-(norm((logm(S1) - logm(S2)), 'fro') ^ 2) / (sigma);
element = exp(-1 * (norm((logm(S1) - logm(S2)), 'fro') ^ 2) / sigma);
element


end
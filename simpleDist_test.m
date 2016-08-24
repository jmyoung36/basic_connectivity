% function calculating a log-Euclidean kernel element for a single pair of
% graphs G1 and G2
function element = simpleDist_test(G1, G2, sigma)

size(G1)
size(G2)

% reshape G1 and G2 from 1 by n^2 vectors n by n square matrices M
n = sqrt(size(G1, 2));
M1 = reshape(G1, [n, n]);
M2 = reshape(G2, [n, n]);

element = norm(M1 - M2, 'fro') + sigma;

end
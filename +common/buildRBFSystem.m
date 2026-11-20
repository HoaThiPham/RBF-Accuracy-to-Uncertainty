function S = buildRBFSystem( ...
    X, y, kernel, epsilon, reg_param, geometry)
%BUILDRBFSYSTEM Assemble the regularized augmented RBF system.
%
%   S = common.buildRBFSystem( ...
%       X, y, kernel, epsilon, reg_param)
%
%   S = common.buildRBFSystem( ...
%       X, y, kernel, epsilon, reg_param, geometry)
%
% INPUTS
% ------
% X
%       n-by-p coordinate matrix.
%
% y
%       n-by-1 observation vector.
%
% kernel
%       RBF kernel name.
%
% epsilon
%       Shape/scale parameter used by shape-parameter kernels.
%
% reg_param
%       Regularization parameter applied directly to the
%       RBF block.
%
% geometry
%       Optional precomputed structure containing:
%
%           geometry.R
%           geometry.P
%
% The augmented regularized RBF system is
%
%       [Phi + reg_param*I    P] [lambda] = [y]
%       [P'                   0] [ beta ]   [0]
%
% Regularization is applied only to the RBF block Phi.


% =============================================================
% 1. Dimensions and observations
% =============================================================

[n,p] = ...
    size(X);

q = ...
    p + 1;


y = ...
    y(:);


if numel(y) ~= n

    error( ...
        'X and y must contain the same number of observations.');

end


% =============================================================
% 2. Fixed geometry
% =============================================================

if nargin < 6 || ...
   isempty(geometry)

    R = ...
        pdist2(X,X);


    P = ...
        [ones(n,1), X];

else

    R = ...
        geometry.R;


    P = ...
        geometry.P;

end


% =============================================================
% 3. RBF kernel matrix
% =============================================================

Phi = ...
    common.applyRBFKernel( ...
        R, ...
        epsilon, ...
        kernel);


if ~isequal(size(Phi),[n,n]) || ...
   ~isreal(Phi) || ...
   any(~isfinite(Phi(:)))

    error( ...
        'RBF kernel evaluation returned an invalid Phi matrix.');

end


% =============================================================
% 4. Regularized augmented system
% =============================================================

A_reg = ...
    [Phi + reg_param*eye(n), P;
     P',                     zeros(q)];


% =============================================================
% 5. Right-hand side
% =============================================================

rhs = ...
    [y; zeros(q,1)];


% =============================================================
% 6. Output
% =============================================================

S = ...
    struct();


S.A_reg = ...
    A_reg;


S.rhs = ...
    rhs;


S.Phi = ...
    Phi;


S.P = ...
    P;


S.kernel = ...
    kernel;


S.epsilon = ...
    epsilon;


S.reg_param = ...
    reg_param;

end
function [yhat, B] = predictRBF( ...
    Xeval, Xfit, coeffs, kernel, epsilon, geometry)
%PREDICTRBF Evaluate a solved RBF model.
%
%   yhat = common.predictRBF(...)
%   [yhat,B] = common.predictRBF(...)
%
%   Xeval    : m-by-p query coordinates
%   Xfit     : n-by-p RBF centre coordinates
%   coeffs   : [lambda; beta]
%   kernel   : RBF kernel
%   epsilon  : shape parameter for shaped kernels
%   geometry : optional precomputed evaluation geometry with
%              fields R_eval and P_eval
%
%   B        : optional evaluation matrix for final diagnostics


% =============================================================
% 1. Dimensions and coefficients
% =============================================================

n = ...
    size(Xfit,1);

p = ...
    size(Xfit,2);

q = ...
    p + 1;

m = ...
    size(Xeval,1);


coeffs = ...
    coeffs(:);


if numel(coeffs) ~= (n + q)

    error( ...
        ['Coefficient vector must contain exactly ' ...
         'n + p + 1 elements.']);

end


% =============================================================
% 2. Evaluation geometry
% =============================================================

if nargin < 6 || ...
   isempty(geometry)

    R = ...
        pdist2(Xeval,Xfit);


    P_eval = ...
        [ones(m,1), Xeval];

else

    R = ...
        geometry.R_eval;


    P_eval = ...
        geometry.P_eval;

end


if ~isequal(size(R),[m,n]) || ...
   ~isequal(size(P_eval),[m,q])

    error( ...
        'Evaluation geometry has inconsistent dimensions.');

end

if ~isreal(P_eval) || ...
   any(~isfinite(P_eval(:)))

    error( ...
        'Evaluation polynomial matrix P_eval must be real and finite.');

end

% =============================================================
% 3. RBF evaluation block
% =============================================================

Phi_eval = ...
    common.applyRBFKernel( ...
        R, ...
        epsilon, ...
        kernel);


if ~isequal(size(Phi_eval),[m,n]) || ...
   ~isreal(Phi_eval) || ...
   any(~isfinite(Phi_eval(:)))

    error( ...
        'RBF evaluation returned an invalid Phi_eval matrix.');

end


% =============================================================
% 4. Coefficients
% =============================================================

lambda = ...
    coeffs(1:n);


beta = ...
    coeffs(n+1:end);


% =============================================================
% 5. Prediction
% =============================================================

yhat = ...
    Phi_eval * lambda + ...
    P_eval * beta;


% =============================================================
% 6. Optional evaluation matrix
% =============================================================

if nargout > 1

    B = ...
        [Phi_eval, P_eval];

end

end
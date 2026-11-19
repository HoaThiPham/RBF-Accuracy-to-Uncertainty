function Evaluation = evaluateBootstrapVarianceCalibration( ...
    residual, ...
    bootstrapVariance, ...
    alpha, ...
    beta)

%EVALUATEBOOTSTRAPVARIANCECALIBRATION
%
% Evaluate bootstrap-based predictive variance before and after
% calibration.
%
% Calibration:
%
%       sigma_i^2 = beta * (v_i + alpha)
%
% INPUTS
% ------
% residual
%       Prediction residuals.
%
%       For training:
%           target-excluded prediction residuals.
%
%       For testing:
%           unseen-test prediction residuals.
%
% bootstrapVariance
%       Finite-sample-adjusted bootstrap variance.
%
% alpha
%       Additive calibration parameter, alpha >= 0.
%
% beta
%       Multiplicative calibration parameter, beta > 0.
%
% No optimization is performed in this function.
% The same function is used for global and local RBF models.


% =============================================================
% 1. Inputs
% =============================================================

e = ...
    residual(:);


v = ...
    bootstrapVariance(:);


if numel(e) ~= numel(v)

    error( ...
        'residual and bootstrapVariance must have equal length.');

end


if isempty(e)

    error( ...
        'Evaluation input cannot be empty.');

end


if any(~isfinite(e))

    error( ...
        'residual must contain only finite values.');

end


if any(~isfinite(v)) || ...
   any(v < 0)

    error( ...
        'bootstrapVariance must be finite and nonnegative.');

end


if ~isscalar(alpha) || ...
   ~isfinite(alpha) || ...
   alpha < 0

    error( ...
        'alpha must be a finite nonnegative scalar.');

end


if ~isscalar(beta) || ...
   ~isfinite(beta) || ...
   beta <= 0

    error( ...
        'beta must be a finite positive scalar.');

end


% =============================================================
% 2. Residual statistics
% =============================================================

e2 = ...
    e.^2;


RMSE = ...
    sqrt(mean(e2));


if ~isfinite(RMSE) || ...
   RMSE <= 0

    error( ...
        'RMSE must be finite and positive.');

end


% =============================================================
% 3. Before calibration
% =============================================================

ASE_Pre = ...
    sqrt(mean(v));


ASE_RMSE_Pre = ...
    ASE_Pre / RMSE;


% RMSSE is formally defined only for positive predictive
% variances. Do not modify zero bootstrap variances artificially.

if all(v > 0)

    RMSSE_Pre = ...
        sqrt(mean(e2 ./ v));

else

    RMSSE_Pre = ...
        NaN;

end


% =============================================================
% 4. Calibrated predictive variance
%
% Eq. (15):
%
%       sigma_i^2 = beta * (v_i + alpha)
% =============================================================

VarianceCalibrated = ...
    beta .* ...
    (v + alpha);


if any(~isfinite(VarianceCalibrated)) || ...
   any(VarianceCalibrated <= 0)

    error( ...
        'Calibration produced invalid predictive variance.');

end


StdCalibrated = ...
    sqrt(VarianceCalibrated);


% =============================================================
% 5. After calibration
% =============================================================

ASE = ...
    sqrt(mean(VarianceCalibrated));


ASE_RMSE = ...
    ASE / RMSE;


RMSSE = ...
    sqrt(mean(e2 ./ VarianceCalibrated));


% =============================================================
% 6. Output
% =============================================================

Evaluation = ...
    struct();


Evaluation.n = ...
    numel(e);


Evaluation.RMSE = ...
    RMSE;


% Before calibration

Evaluation.ASE_Pre = ...
    ASE_Pre;


Evaluation.ASE_RMSE_Pre = ...
    ASE_RMSE_Pre;


Evaluation.RMSSE_Pre = ...
    RMSSE_Pre;


% After calibration

Evaluation.ASE = ...
    ASE;


Evaluation.ASE_RMSE = ...
    ASE_RMSE;


Evaluation.RMSSE = ...
    RMSSE;


% Pointwise uncertainty

Evaluation.VarianceBootstrap = ...
    v;


Evaluation.VarianceCalibrated = ...
    VarianceCalibrated;


Evaluation.StdCalibrated = ...
    StdCalibrated;


end
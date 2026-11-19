function Calibration = calibrateBootstrapVariance( ...
    residual, bootstrapVariance, opts)

%CALIBRATEBOOTSTRAPVARIANCE
%
% Estimate alpha and beta from TRAINING data only.
%
% Calibration model:
%
%       sigma_i^2 = beta * (v_i + alpha)
%
% Objective:
%
%       J(alpha,beta)
%       =
%       gamma1 * abs(ASE/RMSE - 1)
%       +
%       gamma2 * abs(RMSSE - 1)
%
% Constraints:
%
%       alpha >= 0
%       beta  > 0
%
% The function is independent of global/local RBF formulation.


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
        'Calibration input cannot be empty.');

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


% =============================================================
% 2. Options
% =============================================================

if nargin < 3 || ...
        isempty(opts)

    opts = ...
        struct();

end


if ~isfield(opts,'gamma1') || ...
        isempty(opts.gamma1)

    opts.gamma1 = ...
        0.5;

end


if ~isfield(opts,'gamma2') || ...
        isempty(opts.gamma2)

    opts.gamma2 = ...
        0.5;

end


if abs(opts.gamma1 + opts.gamma2 - 1) > 1e-12 || ...
        opts.gamma1 < 0 || ...
        opts.gamma2 < 0

    error( ...
        'gamma1 and gamma2 must be nonnegative and sum to 1.');

end


% -------------------------------------------------------------
% Bounds
%
% Preferred:
%
%   opts.alpha_bounds = [alphaMin alphaMax]
%   opts.beta_bounds  = [betaMin  betaMax]
%
% Backward compatibility:
%
%   opts.bounds(1,:) -> alpha
%   opts.bounds(2,:) -> beta
% -------------------------------------------------------------

if ~isfield(opts,'alpha_bounds') || ...
        isempty(opts.alpha_bounds)

    if isfield(opts,'bounds') && ...
            size(opts.bounds,1) >= 2

        opts.alpha_bounds = ...
            opts.bounds(1,:);

    else

        error( ...
            'opts.alpha_bounds must be provided.');

    end

end


if ~isfield(opts,'beta_bounds') || ...
        isempty(opts.beta_bounds)

    if isfield(opts,'bounds') && ...
            size(opts.bounds,1) >= 2

        opts.beta_bounds = ...
            opts.bounds(2,:);

    else

        error( ...
            'opts.beta_bounds must be provided.');

    end

end


alphaBounds = ...
    opts.alpha_bounds(:).';


betaBounds = ...
    opts.beta_bounds(:).';


if numel(alphaBounds) ~= 2 || ...
        alphaBounds(1) < 0 || ...
        ~isfinite(alphaBounds(1)) || ...
        ~isfinite(alphaBounds(2)) || ...
        alphaBounds(2) <= alphaBounds(1)

    error( ...
        ['opts.alpha_bounds must be [alphaMin alphaMax], ' ...
         'with alphaMin >= 0.']);

end


if numel(betaBounds) ~= 2 || ...
        betaBounds(1) <= 0 || ...
        ~isfinite(betaBounds(1)) || ...
        ~isfinite(betaBounds(2)) || ...
        betaBounds(2) <= betaBounds(1)

    error( ...
        ['opts.beta_bounds must contain finite positive ' ...
         'increasing bounds.']);

end


if ~isfield(opts,'max_gen') || ...
        isempty(opts.max_gen)

    opts.max_gen = ...
        100;

end


if ~isfield(opts,'pop_size') || ...
        isempty(opts.pop_size)

    opts.pop_size = ...
        100;

end


if ~isfield(opts,'RNG_SEED') || ...
        isempty(opts.RNG_SEED)

    opts.RNG_SEED = ...
        1;

end


if ~isfield(opts,'verbose') || ...
        isempty(opts.verbose)

    opts.verbose = ...
        false;

end


if ~isfield(opts,'boundary_tol') || ...
        isempty(opts.boundary_tol)

    opts.boundary_tol = ...
        1e-3;

end


% =============================================================
% 3. Quantities fixed during GA
% =============================================================

e2 = ...
    e.^2;


RMSE2 = ...
    mean(e2);


RMSE = ...
    sqrt(RMSE2);


if ~isfinite(RMSE) || ...
        RMSE <= 0

    error( ...
        'Training RMSE must be finite and positive.');

end


meanV = ...
    mean(v);


% =============================================================
% 4. Numerically scaled optimization variables
%
% x(1) = alpha / alphaScale
% x(2) = log10(beta)
%
% This changes only the numerical parameterization, not the
% mathematical calibration model or objective.
% =============================================================

alphaScale = ...
    max([meanV,RMSE2,eps]);


lb = ...
    [ ...
    alphaBounds(1)/alphaScale, ...
    log10(betaBounds(1)) ...
    ];


ub = ...
    [ ...
    alphaBounds(2)/alphaScale, ...
    log10(betaBounds(2)) ...
    ];


objective = ...
    @(x) localCalibrationObjective( ...
        x, ...
        alphaScale, ...
        v, ...
        e2, ...
        meanV, ...
        RMSE, ...
        opts.gamma1, ...
        opts.gamma2);




% =============================================================
% 5. GA settings
% =============================================================

popSize = ...
    opts.pop_size;


elite = ...
    max(2,ceil(0.05*popSize));


gaOptions = ...
    optimoptions( ...
        'ga', ...
        'Display','off', ...
        'UseParallel',false, ...
        'UseVectorized',false, ...
        'PopulationSize',popSize, ...
        'EliteCount',elite, ...
        'CrossoverFraction',0.8, ...
        'SelectionFcn',@selectionstochunif, ...
        'MutationFcn',@mutationadaptfeasible, ...
        'FitnessScalingFcn',@fitscalingrank, ...
        'FunctionTolerance',1e-8, ...
        'MaxGenerations',opts.max_gen, ...
        'MaxStallGenerations',30);


% =============================================================
% 6. Reproducible GA without changing caller RNG state
% =============================================================

rngState = ...
    rng;


cleanupRNG = ...
    onCleanup(@() rng(rngState)); %#ok<NASGU>


rng( ...
    opts.RNG_SEED, ...
    'twister');


% =============================================================
% 7. Optimize alpha and beta
% =============================================================

tGA = ...
    tic;


[xStar,fval,exitflag,gaOutput] = ...
    ga( ...
        objective, ...
        2, ...
        [], ...
        [], ...
        [], ...
        [], ...
        lb, ...
        ub, ...
        [], ...
        gaOptions);


wallTime = ...
    toc(tGA);


alpha = ...
    xStar(1) * alphaScale;


beta = ...
    10.^xStar(2);


% =============================================================
% 8. Final training evaluation
%
% evaluateBootstrapVarianceCalibration must use:
%
%       VarianceCalibrated = beta .* (v + alpha)
%
% =============================================================

Train = ...
    common.evaluateBootstrapVarianceCalibration( ...
        e, ...
        v, ...
        alpha, ...
        beta);


% -------------------------------------------------------------
% Recalculate objective independently for verification
% -------------------------------------------------------------

Jcheck = ...
    opts.gamma1 * ...
    abs(Train.ASE_RMSE - 1) + ...
    opts.gamma2 * ...
    abs(Train.RMSSE - 1);


% =============================================================
% 9. Boundary diagnostics
% =============================================================

alphaSpan = ...
    alphaBounds(2) - alphaBounds(1);


zBetaBounds = ...
    log10(betaBounds);


zBeta = ...
    log10(beta);


zBetaSpan = ...
    zBetaBounds(2) - zBetaBounds(1);


nearAlphaLower = ...
    abs(alpha-alphaBounds(1)) <= ...
    opts.boundary_tol * alphaSpan;


nearAlphaUpper = ...
    abs(alpha-alphaBounds(2)) <= ...
    opts.boundary_tol * alphaSpan;


nearBetaLower = ...
    abs(zBeta-zBetaBounds(1)) <= ...
    opts.boundary_tol * zBetaSpan;


nearBetaUpper = ...
    abs(zBeta-zBetaBounds(2)) <= ...
    opts.boundary_tol * zBetaSpan;


% =============================================================
% 10. Output
% =============================================================

Calibration = ...
    struct();


Calibration.alpha = ...
    alpha;


Calibration.beta = ...
    beta;


Calibration.Loss = ...
    fval;


Calibration.LossCheck = ...
    Jcheck;


Calibration.gamma1 = ...
    opts.gamma1;


Calibration.gamma2 = ...
    opts.gamma2;


Calibration.Train = ...
    Train;


Calibration.alpha_bounds = ...
    alphaBounds;


Calibration.beta_bounds = ...
    betaBounds;


Calibration.alphaScale = ...
    alphaScale;


Calibration.nearAlphaLower = ...
    nearAlphaLower;


Calibration.nearAlphaUpper = ...
    nearAlphaUpper;


Calibration.nearBetaLower = ...
    nearBetaLower;


Calibration.nearBetaUpper = ...
    nearBetaUpper;


Calibration.exitflag = ...
    exitflag;


Calibration.generations = ...
    gaOutput.generations;


Calibration.funccount = ...
    gaOutput.funccount;


Calibration.wallTime = ...
    wallTime;


Calibration.success = ...
    isfinite(alpha) && ...
    alpha >= 0 && ...
    isfinite(beta) && ...
    beta > 0 && ...
    isfinite(fval) && ...
    isfinite(Jcheck) && ...
    abs(fval-Jcheck) <= ...
    1e-8 * max(1,abs(fval));


% =============================================================
% 11. Optional display
% =============================================================

if opts.verbose

    fprintf('\nVariance calibration\n');

    fprintf( ...
        'alpha          = %.12g\n', ...
        alpha);

    fprintf( ...
        'beta           = %.12g\n', ...
        beta);

    fprintf( ...
        'Loss           = %.12g\n', ...
        fval);

    fprintf( ...
        'ASE/RMSE       = %.8f\n', ...
        Train.ASE_RMSE);

    fprintf( ...
        'RMSSE          = %.8f\n', ...
        Train.RMSSE);

    fprintf( ...
        'GA evaluations = %d\n', ...
        gaOutput.funccount);

    fprintf( ...
        'Wall time      = %.3f s\n', ...
        wallTime);

end


% =============================================================
% LOCAL OBJECTIVE
% =============================================================

function J = localCalibrationObjective( ...
    x,alphaScale,v,e2,meanV,RMSE,gamma1,gamma2)

% -------------------------------------------------------------
% Recover physical calibration parameters
% -------------------------------------------------------------

alpha = ...
    x(1) * alphaScale;


beta = ...
    10.^x(2);


% -------------------------------------------------------------
% Eq. (15)
%
% sigma_i^2 = beta * (v_i + alpha)
% -------------------------------------------------------------

vShift = ...
    v + alpha;


if any(~isfinite(vShift)) || ...
        any(vShift <= 0) || ...
        ~isfinite(beta) || ...
        beta <= 0

    J = ...
        Inf;

    return

end


% -------------------------------------------------------------
% Calibration diagnostics
%
% The calibrated pointwise variance is
%
%       varianceCalibrated = beta .* (v + alpha)
%
% Therefore:
%
%       ASE
%       =
%       sqrt(mean(varianceCalibrated))
%       =
%       sqrt(beta * (meanV + alpha))
%
% and
%
%       RMSSE
%       =
%       sqrt(mean(e2 ./ varianceCalibrated))
%       =
%       sqrt(mean(e2 ./ (v + alpha)) / beta)
%
% These algebraically equivalent expressions avoid constructing
% varianceCalibrated during every GA objective evaluation.
% -------------------------------------------------------------

ASE = ...
    sqrt( ...
        beta * ...
        (meanV + alpha));


RMSSE = ...
    sqrt( ...
        mean(e2 ./ vShift) / beta);


if ~isfinite(ASE) || ...
        ~isfinite(RMSSE)

    J = ...
        Inf;

    return

end


J = ...
    gamma1 * ...
    abs(ASE/RMSE - 1) + ...
    gamma2 * ...
    abs(RMSSE - 1);

end


end
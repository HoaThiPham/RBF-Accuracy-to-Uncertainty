function [C_global, y_cv_pred, y_diff_cv, cv_rmse, fold_rmse, ...
          C_ga, cv_rmse_ga, trainingInfo] = ...
    globalRBFTrainingCV_TuneReg( ...
        X, y, kernel, method, C_min, C_max, Reg_min, Reg_max, ...
    kfolds, verbose, RNG_SEED, options)

%GLOBALRBFTRAININGCV
%
% Global RBF training and cross-validation with optimization
% of the shape coefficient C and regularization parameter RegParam.
% Either parameter may be fixed by setting its lower and upper
% bounds equal.
%
% For shape-parameter kernels, C and RegParam may be optimized
% or fixed. For shape-free kernels, only RegParam is relevant;
% C is internally fixed to a dummy value.
%
% For shape-parameter kernels:
%
%       epsilon = C * d
%
% For shape-free kernels:
%
%       epsilon = 1
%
% For shape-parameter kernels, d is calculated from the training
% geometry of each fixed CV fold and reused for every candidate.
%

%
%
% INPUTS
% ------
% X
%       [n x d] raw training coordinates.
%
%       A global coordinate scale is calculated once:
%
%           S_global = MeanDistant(X)
%
%       and the coordinates used throughout CV are:
%
%           X_scaled = X / S_global
%
% y
%       [n x 1] training observations.
%
% kernel
%       RBF kernel name.
%
%       Shape-parameter kernels currently include:
%           'gaussian'
%           'multiquadric'
%

%
% method
%       Optimization method retained for API compatibility.
%       GA is used for parameter optimization.
%
% C_min, C_max
%       Lower and upper bounds for C.
%       If C_min == C_max, C is fixed at that value.
%
% Reg_min, Reg_max
%       Lower and upper bounds for RegParam.
%       If Reg_min == Reg_max, RegParam is fixed at that value.
%
% kfolds
%       Requested number of CV folds.
% adjust_kfolds(...) caps this number at the sample size and
% verifies that each training fold satisfies the polynomial-tail
% requirement.
%       Default:
%
%           kfolds = 10
%
% verbose
%       Logical flag controlling command-window messages.
%       Default:
%
%           verbose = true
%
% RNG_SEED
%       Random seed controlling reproducibility of the fixed CV
%       partition and GA.
%
%       Default:
%
%           RNG_SEED = 1
%
%       If RNG_SEED = [], the current MATLAB random stream is retained
%       and rng(...) is not reset.
%
% options
%       Structure containing numerical solver/stabilization options.
%
%
% IMPORTANT DESIGN PRINCIPLES
% ---------------------------
% 1. CV prediction error alone selects C and RegParam
%
% 2. Numerical conditioning is used only for numerical stabilization
%    and diagnostics; CN/ECN/EVN are NOT optimization criteria.
%
% 3. The same fixed CV partition is reused for every candidate
%    (C, RegParam) pair.
%
% 4. Every (C, RegParam) objective evaluation is independently
%    evaluated and retained; only fixed fold geometry is reused.
%
% 5. Genuine large RMSE values are retained exactly.
%
% 6. Failed evaluations use Inf only for the GA objective,
%    while the scientific RMSE trace stores NaN.
%
% 7. Boundary checking is diagnostic only and does not alter
%    C or RegParam.
%
%
% OUTPUTS
% -------
% C_global
%       Selected global coefficient C.

%
% y_cv_pred
%       Out-of-fold CV predictions.
%
% y_diff_cv
%       Signed CV residual:
%
%           y_diff_cv = y - y_cv_pred
%
% cv_rmse
%       Overall pooled CV RMSE.
%
% fold_rmse
%       RMSE for each CV fold.
%
% C_ga
%       Complete C evaluation history for shape-parameter kernels.

%
% cv_rmse_ga
%       Genuine CV RMSE corresponding to C_ga.
%       Failed evaluations are stored as NaN.

% trainingInfo
%       Training, preprocessing, CV, optimization, fallback,
%       boundary, and selected-run information.
%
%       The structure has the same fields for all kernels.

% =============================================================


% =============================================================
% 1. Defaults
% =============================================================


if nargin < 9 || isempty(kfolds)
    kfolds = 10;
end

if nargin < 10 || isempty(verbose)
    verbose = true;
end

if nargin < 11
    RNG_SEED = 1;
end

if nargin < 12 || isempty(options)
    options = struct();
end

if ~isfield(options,'solver') || ~isstruct(options.solver)
    options.solver = struct();
end

if ~isfield(options.solver,'cond_lsq') || ...
        isempty(options.solver.cond_lsq)

    options.solver.cond_lsq = 1e12;

end

% Relative residual is not calculated during parameter search.
if ~isfield(options.solver,'compute_rel_residual') || ...
   isempty(options.solver.compute_rel_residual)

    options.solver.compute_rel_residual = false;

end

if isempty(method)
    method = "ga";
end

% =============================================================
% Input validation
% =============================================================

if ~strcmpi(char(method),'ga')

    error('Only method="ga" is currently supported.');

end


if ~isnumeric(X) || ...
   isempty(X) || ...
   ~ismatrix(X) || ...
   ~isreal(X) || ...
   any(~isfinite(X(:)))
error('X must be a nonempty finite real numeric matrix.');

end


if ~isnumeric(y) || ...
   isempty(y) || ...
   ~isvector(y) || ...
   ~isreal(y)

    error('y must be a nonempty real numeric vector.');

end


y = ...
    y(:);

if any(~isfinite(y))

    error('y must contain only finite values.');

end


if numel(y) ~= size(X,1)

    error( ...
        'X and y must contain the same number of observations.');

end

if ~isnumeric(kfolds) || ...
   ~isscalar(kfolds) || ...
   ~isreal(kfolds) || ...
   ~isfinite(kfolds) || ...
   kfolds < 2 || ...
   kfolds ~= round(kfolds)
    error('kfolds must be a finite integer >= 2.');

end

if ~isnumeric(Reg_min) || ...
   ~isscalar(Reg_min) || ...
   ~isreal(Reg_min) || ...
   ~isfinite(Reg_min) || ...
   Reg_min <= 0

    error('Reg_min must be a finite positive real numeric scalar.');

end


if ~isnumeric(Reg_max) || ...
   ~isscalar(Reg_max) || ...
   ~isreal(Reg_max) || ...
   ~isfinite(Reg_max) || ...
   Reg_max < Reg_min

    error('Reg_max must be a finite real numeric scalar >= Reg_min.');

end
% =============================================================
% 2. Reproducibility
%
% Called ONCE before creation of the fixed CV partition and GA.
%
% No rng(...) call should occur inside evaluateGlobalCV.
% =============================================================

if ~isempty(RNG_SEED)
    rng(RNG_SEED,'twister');
end


% =============================================================
% 3. Data size and valid K-fold number
%
% Ordering:
%
% validation
%   -> RNG
%   -> size
%   -> adjust_kfolds
%   -> global coordinate scaling
%   -> fixed CV
% =============================================================


[n, d] = size(X);

k_used = adjust_kfolds(n, d, kfolds);


if verbose && k_used ~= kfolds

 fprintf(['[info] kfolds reduced from %d to %d ' ...
         'because kfolds cannot exceed the sample size.\n'], ...
         kfolds, k_used);

end

% =============================================================
% 4. Global coordinate scaling
%
% S_global is calculated from the ORIGINAL full training geometry.
% =============================================================


S_global = ...
    common.MeanDistant(X);


if ~isscalar(S_global) || ...
   ~isfinite(S_global) || ...
   S_global <= 0

    error(['MeanDistant(X) must return a finite positive ' ...
           'coordinate scale.']);

end


X_scaled = X ./ S_global;

% =============================================================
% 5. Kernel type
% =============================================================

useShape = ...
    common.kernelHasShapeParameter(kernel);

if ~useShape

    % Shape-free kernels do not use C.
    % Keep the GA interface uniform by fixing the dummy C variable.
    C_min = 1;
    C_max = 1;

end


% =============================================================
% 6. ONE fixed CV partition
%
% Created once and reused throughout the complete training/CV run.
% =============================================================

cvp_fixed = cvpartition( ...
    size(X_scaled,1), ...
    'KFold', ...
    k_used);


% =============================================================
% Fixed CV-fold cache
%
% Fold indices, data subsets, and geometry-dependent quantities
% are calculated once and reused for every (C, RegParam) candidate.
% =============================================================

cvCache = repmat(struct( ...
    'idxVal',   [], ...
    'Xfit',     [], ...
    'Xval',     [], ...
    'yfit',     [], ...
    'yval',     [], ...
    'R',        [], ...
    'P',        [], ...
    'R_eval',   [], ...
    'P_eval',   [], ...
    'd',        NaN, ...
    'isOK',     false), ...
    k_used,1);


for f = 1:k_used

 idxTrain = training(cvp_fixed,f);
    idxVal   = test(cvp_fixed,f);

    Xfit = X_scaled(idxTrain,:);
    Xval = X_scaled(idxVal,:);

    R = pdist2(Xfit,Xfit);
    P = [ones(size(Xfit,1),1),Xfit];

    R_eval = pdist2(Xval,Xfit);
    P_eval = [ones(size(Xval,1),1),Xval];


    yfit = y(idxTrain);
    yval = y(idxVal);


  

    % ---------------------------------------------------------
    % Fixed fold-specific geometry scale
    % ---------------------------------------------------------

  if useShape

    [~,d_fold] = ...
        common.buildEpsilon(Xfit,1);

else

    d_fold = NaN;

end


    % ---------------------------------------------------------
    % Fixed polynomial admissibility
    % ---------------------------------------------------------

    [isOK, ~] = ...
        common.checkPolynomialAdmissibility(Xfit);


    % ---------------------------------------------------------
    % Store fixed fold information
    % ---------------------------------------------------------

    cvCache(f).idxVal   = idxVal;

    cvCache(f).Xfit = Xfit;
    cvCache(f).Xval = Xval;

    cvCache(f).yfit = yfit;
    cvCache(f).yval = yval;

    cvCache(f).R = R;
    cvCache(f).P = P;

    cvCache(f).d = d_fold;
    cvCache(f).isOK = isOK;

    cvCache(f).R_eval = R_eval;
    cvCache(f).P_eval = P_eval;

end

% =============================================================
% Training information
% Same fields are returned for ALL kernels.

% =============================================================

trainingInfo = struct();

% Common information
trainingInfo.kernel       = kernel;
trainingInfo.useShape     = useShape;
trainingInfo.reg_param = [];

trainingInfo.Reg_min = Reg_min;
trainingInfo.Reg_max = Reg_max;
trainingInfo.Reg_ga = [];
trainingInfo.options      = options;

trainingInfo.RNG_SEED     = RNG_SEED;
trainingInfo.k_requested  = kfolds;
trainingInfo.k_used       = k_used;
trainingInfo.S_global     = S_global;
trainingInfo.cvp_fixed    = cvp_fixed;

% CV information
trainingInfo.finalCVRMSE       = [];
trainingInfo.bestRMSE          = [];
trainingInfo.selectedFoldInfo  = [];

% Shape-related information
trainingInfo.C_global     = [];
trainingInfo.C_best       = [];
trainingInfo.C_min        = [];
trainingInfo.C_max        = [];
trainingInfo.C_ga         = [];
trainingInfo.cv_rmse_ga   = [];

% GA/search information
trainingInfo.gaBestRMSE              = [];
trainingInfo.searchBestRMSE          = [];
trainingInfo.usedFallback            = [];
trainingInfo.exitflag                = [];
trainingInfo.method                  = [];
trainingInfo.converged               = [];
trainingInfo.acceptedLimitStop       = [];

trainingInfo.nEvaluations            = [];
trainingInfo.nGAEvaluations          = [];
trainingInfo.nFallbackEvaluations    = [];
trainingInfo.nValidEvaluations       = [];
trainingInfo.nFailedEvaluations      = [];

trainingInfo.generations             = [];
trainingInfo.funccount               = [];
trainingInfo.message                 = [];

trainingInfo.finalSearchRMSEDifference = [];

% Boundary diagnostics
trainingInfo.boundaryTolerance = [];
trainingInfo.relativePosition  = [];
trainingInfo.nearLowerBound    = [];
trainingInfo.nearUpperBound    = [];

trainingInfo.RegRelativePosition = [];
trainingInfo.RegNearLowerBound = [];
trainingInfo.RegNearUpperBound = [];

trainingInfo.C_fixed = [];
trainingInfo.Reg_fixed = [];


trainingInfo.success = false;

% Computational information
trainingInfo.nRBFSolves = 0;

% GA settings
trainingInfo.GA = [];


trainingInfo.gaConverged = [];
trainingInfo.fallbackSucceeded = [];

% =============================================================
% 8. Parameter bounds
% =============================================================
if ~isnumeric(C_min) || ...
   ~isscalar(C_min) || ...
   ~isreal(C_min) || ...
   ~isfinite(C_min) || ...
   C_min <= 0

    error('C_min must be a finite positive real numeric scalar.');

end


if ~isnumeric(C_max) || ...
   ~isscalar(C_max) || ...
   ~isreal(C_max) || ...
   ~isfinite(C_max) || ...
   C_max < C_min

    error('C_max must be a finite real numeric scalar >= C_min.');

end


C_min_eff = ...
    C_min;

C_max_eff = ...
    C_max;


% =============================================================
% 9. Log-space search interval
%
%       z = log10(C)
%
% therefore
%
%       C = 10^z
% =============================================================


zCmin = log10(C_min_eff);
zCmax = log10(C_max_eff);

zRmin = log10(Reg_min);
zRmax = log10(Reg_max);

% Equal lower and upper bounds fix that GA variable.
% =============================================================

C_fixed = ...
    (C_min_eff == C_max_eff);

Reg_fixed = ...
    (Reg_min == Reg_max);





if useShape

    if ~C_fixed && ~Reg_fixed

        nvars = 2;
        lb = [zCmin, zRmin];
        ub = [zCmax, zRmax];

    elseif ~C_fixed

        nvars = 1;
        lb = zCmin;
        ub = zCmax;

    elseif ~Reg_fixed

        nvars = 1;
        lb = zRmin;
        ub = zRmax;

    else

        nvars = 0;
        lb = [];
        ub = [];

    end

else

    if Reg_fixed

        nvars = 0;
        lb = [];
        ub = [];

    else

        nvars = 1;
        lb = zRmin;
        ub = zRmax;

    end

end

allFixed = ...
    (nvars == 0);
% =============================================================
% Fixed-variable flags
%


% =============================================================
% 10. Complete evaluation history
%
% No cache, merging, averaging, or removal of repeated values.
% Every objective call is retained in actual evaluation order.
% =============================================================

C_eval_log = [];
Reg_eval_log = [];
E_eval_log = [];


% =============================================================
% 11. Failed-objective penalty
%
% Failed evaluations use Inf only for the GA objective.
% Scientific RMSE history stores NaN.
% =============================================================

FAIL_PENALTY = ...
    Inf;
% =============================================================
% Fallback-grid settings
% =============================================================

nC = 10;
nR = 10;

if C_fixed
    nCgrid = 1;
else
    nCgrid = nC;
end

if Reg_fixed
    nRgrid = 1;
else
    nRgrid = nR;
end

% =============================================================
% 12. GA settings
%
% Preserve baseline settings exactly.
% =============================================================

popSize = ...
    40;

elite = ...
    max(2, ceil(0.05 * popSize));


opts = optimoptions( ...
    'ga', ...
    'Display',              'off', ...
    'UseParallel',          false, ...
    'UseVectorized',        false, ...
    'PopulationSize',       popSize, ...
    'EliteCount',           elite, ...
    'CrossoverFraction',    0.8, ...
    'SelectionFcn',         @selectionstochunif, ...
    'MutationFcn',          @mutationadaptfeasible, ...
    'FitnessScalingFcn',    @fitscalingrank, ...
    'FunctionTolerance',    1e-8, ...
    'MaxGenerations',       60, ...
    'MaxStallGenerations',  20);


% =============================================================
% GA settings for reproducibility / computational reporting
% =============================================================

trainingInfo.GA = struct();

trainingInfo.GA.Display = 'off';

trainingInfo.GA.PopulationSize = popSize;
trainingInfo.GA.EliteCount = elite;
trainingInfo.GA.CrossoverFraction = 0.8;

trainingInfo.GA.SelectionFcn = 'selectionstochunif';
trainingInfo.GA.MutationFcn = 'mutationadaptfeasible';
trainingInfo.GA.FitnessScalingFcn = 'fitscalingrank';

trainingInfo.GA.FunctionTolerance = 1e-8;
trainingInfo.GA.MaxGenerations = 60;
trainingInfo.GA.MaxStallGenerations = 20;

trainingInfo.GA.UseParallel = false;
trainingInfo.GA.UseVectorized = false;

if useShape

    if ~C_fixed && ~Reg_fixed

        trainingInfo.GA.SearchVariable = ...
            {'log10(C)','log10(RegParam)'};

    elseif ~C_fixed

        trainingInfo.GA.SearchVariable = ...
            {'log10(C)'};

    elseif ~Reg_fixed

        trainingInfo.GA.SearchVariable = ...
            {'log10(RegParam)'};

    else

        trainingInfo.GA.SearchVariable = ...
            {};

    end

else

    if Reg_fixed

        trainingInfo.GA.SearchVariable = ...
            {};

    else

        trainingInfo.GA.SearchVariable = ...
            {'log10(RegParam)'};

    end

end

trainingInfo.GA.nvars = nvars;

if useShape
    trainingInfo.GA.C_fixed = C_fixed;
else
    trainingInfo.GA.C_fixed = NaN;
end

trainingInfo.GA.Reg_fixed = Reg_fixed;
trainingInfo.GA.Skipped = allFixed;

trainingInfo.GA.FallbackGridSize = ...
    [nCgrid nRgrid];


trainingInfo.GA.FailurePenalty = FAIL_PENALTY;
% =============================================================
% 13. Initialize optimization information
% =============================================================

usedFallback = ...
    false;

gaMessage = ...
    '';

ga_best_rmse = ...
    NaN;

search_best_rmse = ...
    NaN;

exitflag = ...
    NaN;

output = ...
    struct();

nGAEvaluations = ...
    0;

nFallbackEvaluations = ...
    0;

nRBFSolves = 0; 

% number of RBF systems actually sent to solveLinearSystem during CV search + 
% fallback + final selected-parameter CV
% =============================================================
% 14. Parameter optimization
% =============================================================

if allFixed

    % ---------------------------------------------------------
    % All relevant parameters are fixed.
    % Do not start GA.
    % Final CV is evaluated once in Section 17.
    % ---------------------------------------------------------

    if useShape

        C_global = ...
            C_min_eff;

    else

        C_global = 1;

    end

    reg_param = ...
        Reg_min;

    ga_best_rmse = ...
        NaN;

    nGAEvaluations = ...
        0;

    nFallbackEvaluations = ...
        0;

    exitflag = ...
        NaN;


else

    % ---------------------------------------------------------
    % Run GA
    % ---------------------------------------------------------

    [~, best_rmse, exitflag, output] = ga( ...
        @ga_fitness_log_z, ...
        nvars, ...
        [], [], [], [], ...
        lb, ub, ...
        [], ...
        opts);


    ga_best_rmse = ...
        best_rmse;


    % ---------------------------------------------------------
    % Number of genuine GA objective calls
    % ---------------------------------------------------------

    nGAEvaluations = ...
        numel(E_eval_log);


    idxGA = ...
        (1:nGAEvaluations).';


    validGA = ...
        isfinite(E_eval_log(idxGA));


    % ---------------------------------------------------------
    % Decide whether fallback is required
    % ---------------------------------------------------------

    if ~isfinite(exitflag) || ...
       exitflag < 0 || ...
       ~any(validGA)

        usedFallback = ...
            true;

        gaMessage = ...
            'GA did not return a valid scientific CV solution.';


        if verbose

            fprintf( ...
                '[warn] GA fallback: %s\n', ...
                gaMessage);

        end


    else

        % -----------------------------------------------------
        % Select best genuine GA evaluation
        % -----------------------------------------------------

        validIdx = ...
            idxGA(validGA);


        [search_best_rmse, jbest] = ...
            min(E_eval_log(validIdx));


        ibest = ...
            validIdx(jbest);


        reg_param = ...
            Reg_eval_log(ibest);


        if useShape

            C_global = ...
                C_eval_log(ibest);

        else

            C_global = 1;

        end


        % -----------------------------------------------------
        % GA termination information
        % -----------------------------------------------------

        if verbose

            if exitflag == 0

                fprintf(['[note] GA stopped by limit ' ...
                    '(exitflag=0); using best found parameter pair.\n']);

            else

                fprintf( ...
                    '[note] GA exitflag=%d.\n', ...
                    exitflag);

            end

        end

    end


    % =========================================================
    % 15. Invalid GA result -> logarithmic-grid fallback
    % =========================================================

    if usedFallback

        % -----------------------------------------------------
        % Construct fallback grids
        % -----------------------------------------------------

        if useShape

            if C_fixed

                C_coarse = ...
                    C_min_eff;

            else

                C_coarse = ...
                    logspace( ...
                        zCmin, ...
                        zCmax, ...
                        nCgrid);

            end

        end


        if Reg_fixed

            Reg_coarse = ...
                Reg_min;

        else

            Reg_coarse = ...
                logspace( ...
                    zRmin, ...
                    zRmax, ...
                    nRgrid);

        end


        % -----------------------------------------------------
        % Evaluate fallback grid
        % -----------------------------------------------------

        if useShape

            if ~C_fixed && ~Reg_fixed

                for iC = 1:numel(C_coarse)

                    for iR = 1:numel(Reg_coarse)

                        ga_fitness_log_z( ...
                            [log10(C_coarse(iC)), ...
                             log10(Reg_coarse(iR))]);

                    end

                end


            elseif ~C_fixed

                for iC = 1:numel(C_coarse)

                    ga_fitness_log_z( ...
                        log10(C_coarse(iC)));

                end


            elseif ~Reg_fixed

                for iR = 1:numel(Reg_coarse)

                    ga_fitness_log_z( ...
                        log10(Reg_coarse(iR)));

                end

            end


        else

            for iR = 1:numel(Reg_coarse)

                ga_fitness_log_z( ...
                    log10(Reg_coarse(iR)));

            end

        end


        nFallbackEvaluations = ...
            numel(E_eval_log) - ...
            nGAEvaluations;



% -----------------------------------------------------
% Select best genuine evaluation from complete search
% -----------------------------------------------------

validAll = ...
    isfinite(E_eval_log);


if ~any(validAll)

    error(['Neither GA nor the logarithmic-grid ' ...
        'fallback found a valid parameter pair.']);

end


validIdx = ...
    find(validAll);


[search_best_rmse, jbest] = ...
    min(E_eval_log(validAll));


ibest = ...
    validIdx(jbest);


reg_param = ...
    Reg_eval_log(ibest);


if useShape

    C_global = ...
        C_eval_log(ibest);

else

    C_global = 1;

end
    end

end

if ~usedFallback

    nFallbackEvaluations = ...
        0;

end

% Complete parameter-RMSE evaluation trace
% =============================================================
% 16. Selected parameter for final CV
% =============================================================



% Shape-free kernels do not have a meaningful C.
if ~useShape
    C_global_eval = 1;
else
    C_global_eval = C_global;
end
% =============================================================
% 17. Final CV ONCE at selected (C, RegParam)
%
% IMPORTANT:
%
%   - X_scaled is used
%   - same cvp_fixed
%   - no RNG reset
%   - for shape-parameter kernels, fold-specific d is reused
%     from cvCache and epsilon_f = C_global*d_f
%   - for shape-free kernels, epsilon = 1
%

% =============================================================
% =============================================================
% Final CV evaluation at the selected parameters
%
% Relative residual is calculated only for the selected
% configuration, not during GA/fallback optimization.
% =============================================================

finalOptions = ...
    options;

finalOptions.solver.compute_rel_residual = ...
    true;

cvResult = globalrbf.evaluateGlobalCV( ...
    X_scaled, ...
    y, ...
    kernel, ...
    C_global_eval, ...
    cvp_fixed, ...
    reg_param, ...
    finalOptions, ...
    cvCache);

nRBFSolves = ...
    nRBFSolves + cvResult.nRBFSolves;

if ~cvResult.success || ...
   ~isscalar(cvResult.RMSE) || ...
   ~isfinite(cvResult.RMSE) || ...
   numel(cvResult.yhat) ~= numel(y) || ...
   any(~isfinite(cvResult.yhat)) || ...
   numel(cvResult.fold_rmse) ~= k_used || ...
   any(~isfinite(cvResult.fold_rmse))

    error( ...
        'Final CV evaluation failed or returned invalid outputs.');

end




% =============================================================
% 18. Selected-parameter outputs
% =============================================================

y_cv_pred = ...
    cvResult.yhat;


% Signed residual convention:
%
%       observed - predicted
%
y_diff_cv = ...
    y - y_cv_pred;


cv_rmse = ...
    cvResult.RMSE;


fold_rmse = ...
    cvResult.fold_rmse;


% -------------------------------------------------------------
% Fixed-parameter run: Section 17 is the only CV evaluation.
% Store it as the complete parameter/RMSE trace.
% -------------------------------------------------------------

if allFixed

    search_best_rmse = cv_rmse;

    if useShape
        C_eval_log = C_global;
    else
        C_eval_log = NaN;
    end

    Reg_eval_log = reg_param;
    E_eval_log   = cv_rmse;

end

C_ga = ...
    C_eval_log;

Reg_ga = ...
    Reg_eval_log;

cv_rmse_ga = ...
    E_eval_log;


% -------------------------------------------------------------
% Search-trace consistency checks
% -------------------------------------------------------------

nTrace = ...
    numel(cv_rmse_ga);


if numel(C_ga) ~= nTrace || ...
   numel(Reg_ga) ~= nTrace

    error( ...
        'C, RegParam, and RMSE search traces have inconsistent lengths.');

end


if allFixed

    if nTrace ~= 1

        error( ...
            'Fixed-parameter run must contain exactly one CV evaluation.');

    end

else

    if nTrace ~= ...
            (nGAEvaluations + nFallbackEvaluations)

        error( ...
            ['Search-trace length does not match GA + ' ...
             'fallback evaluation counts.']);

    end

end


% =============================================================
% 19. Selected shape parameter and search trace
% =============================================================

if ~useShape
    C_global = NaN;
end

trainingInfo.C_global = ...
    C_global;

trainingInfo.reg_param = reg_param;

trainingInfo.Reg_min = Reg_min;
trainingInfo.Reg_max = Reg_max;

trainingInfo.Reg_ga = Reg_ga;


trainingInfo.C_best = ...
    C_global;

if useShape

    trainingInfo.C_min = C_min_eff;
    trainingInfo.C_max = C_max_eff;

else

    trainingInfo.C_min = NaN;
    trainingInfo.C_max = NaN;

end


% =============================================================
% Complete (C, RegParam)-RMSE search trace
% =============================================================

trainingInfo.C_ga = ...
    C_ga;

trainingInfo.cv_rmse_ga = ...
    cv_rmse_ga;

% =============================================================
% 20. Selected-parameter CV and optimization summary
% =============================================================

trainingInfo.finalCVRMSE = ...
    cv_rmse;

trainingInfo.bestRMSE = ...
    cv_rmse;

trainingInfo.selectedFoldInfo = ...
    cvResult.foldInfo;


% =============================================================
% Optimization information
% =============================================================

trainingInfo.gaBestRMSE = ...
    ga_best_rmse;

trainingInfo.searchBestRMSE = ...
    search_best_rmse;



% =============================================================
% 21. Search method / convergence
% =============================================================

trainingInfo.usedFallback = ...
    usedFallback;


trainingInfo.exitflag = ...
    exitflag;


if allFixed
    trainingInfo.method = 'fixed parameters';
elseif usedFallback
    trainingInfo.method = 'GA + coarse-grid fallback';
else
    trainingInfo.method = 'GA';
end


% GA convergence status.
trainingInfo.gaConverged = ...
    ~allFixed && ...
    ~usedFallback && ...
    isfinite(exitflag) && ...
    exitflag > 0;

trainingInfo.acceptedLimitStop = ...
    ~allFixed && ...
    ~usedFallback && ...
    isfinite(exitflag) && ...
    exitflag == 0;

if usedFallback

fallbackIdx = ...
    nGAEvaluations + (1:nFallbackEvaluations);

    trainingInfo.fallbackSucceeded = ...
        any(isfinite(cv_rmse_ga(fallbackIdx)));

else

    trainingInfo.fallbackSucceeded = ...
        false;

end

trainingInfo.converged = ...
    allFixed || ...
    trainingInfo.gaConverged;



% =============================================================
% 22. Evaluation counts
% =============================================================

trainingInfo.nEvaluations = ...
    numel(cv_rmse_ga);


trainingInfo.nGAEvaluations = ...
    nGAEvaluations;


trainingInfo.nFallbackEvaluations = ...
    nFallbackEvaluations;


trainingInfo.nValidEvaluations = ...
    sum(isfinite(cv_rmse_ga));


trainingInfo.nFailedEvaluations = ...
    sum(~isfinite(cv_rmse_ga));


trainingInfo.nRBFSolves = ...
    nRBFSolves;
% =============================================================
% 23. MATLAB GA information
% =============================================================

if isfield(output, 'generations')

    trainingInfo.generations = ...
        output.generations;

else

    trainingInfo.generations = ...
        NaN;

end


if isfield(output, 'funccount')

    trainingInfo.funccount = ...
        output.funccount;

else

    trainingInfo.funccount = ...
        NaN;

end


if usedFallback

    trainingInfo.message = ...
        gaMessage;

elseif isfield(output, 'message')

    trainingInfo.message = ...
        output.message;

else

    trainingInfo.message = ...
        '';

end


% =============================================================
% 24. Search-versus-final-CV consistency diagnostic
%
% With:
%
%   - same cvp_fixed
%   - no RNG reset
%   - deterministic CV evaluation
%
% this difference should normally be very small.
% =============================================================

trainingInfo.finalSearchRMSEDifference = ...
    abs(cv_rmse - search_best_rmse);

rmseTol = ...
    1e-10 * max(1,abs(cv_rmse));

if isfinite(search_best_rmse) && ...
   trainingInfo.finalSearchRMSEDifference > rmseTol

    warning( ...
        'globalRBF:RMSEConsistency', ...
        ['Final CV RMSE differs from selected search RMSE: ' ...
         'search=%.17g, final=%.17g, diff=%.3e'], ...
        search_best_rmse, ...
        cv_rmse, ...
        trainingInfo.finalSearchRMSEDifference);

end

% =============================================================
% 25. Boundary diagnostics
%
% DIAGNOSTIC ONLY.
%
% These diagnostics must NEVER:
%
%   - alter selected parameters
%   - expand bounds automatically
%   - reject parameters
%   - affect the optimization objective
% =============================================================

BOUND_TOL = 0.02;

trainingInfo.boundaryTolerance = BOUND_TOL;


% -------------------------------------------------------------
% C boundary diagnostic in log10 space
% -------------------------------------------------------------

if useShape

    trainingInfo.C_fixed = ...
        C_fixed;

else

    trainingInfo.C_fixed = ...
        NaN;

end

if C_fixed

    trainingInfo.relativePosition = NaN;
    trainingInfo.nearLowerBound = false;
    trainingInfo.nearUpperBound = false;

else

    zbestC = ...
        log10(C_global);

    relativePosition = ...
        (zbestC - zCmin) / ...
        (zCmax - zCmin);

    trainingInfo.relativePosition = ...
        relativePosition;

    trainingInfo.nearLowerBound = ...
        relativePosition <= BOUND_TOL;

    trainingInfo.nearUpperBound = ...
        relativePosition >= (1 - BOUND_TOL);

end


% -------------------------------------------------------------
% RegParam boundary diagnostic in log10 space
% -------------------------------------------------------------

trainingInfo.Reg_fixed = ...
    Reg_fixed;

if Reg_fixed

    trainingInfo.RegRelativePosition = NaN;
    trainingInfo.RegNearLowerBound = false;
    trainingInfo.RegNearUpperBound = false;

else

    zbestReg = ...
        log10(reg_param);

    regRelativePosition = ...
        (zbestReg - zRmin) / ...
        (zRmax - zRmin);

    trainingInfo.RegRelativePosition = ...
        regRelativePosition;

    trainingInfo.RegNearLowerBound = ...
        regRelativePosition <= BOUND_TOL;

    trainingInfo.RegNearUpperBound = ...
        regRelativePosition >= (1 - BOUND_TOL);

end


trainingInfo.success = true;

% =============================================================
% 26. Nested GA objective
% =============================================================

   function f = ga_fitness_log_z(zvec)

    % -----------------------------------------------------
    % GA variables -> physical parameters
    % -----------------------------------------------------

if useShape

    if ~C_fixed && ~Reg_fixed

        C = ...
            10.^zvec(1);

        reg_param_candidate = ...
            10.^zvec(2);

    elseif ~C_fixed

        C = ...
            10.^zvec(1);

        reg_param_candidate = ...
            Reg_min;

    elseif ~Reg_fixed

        C = ...
            C_min_eff;

        reg_param_candidate = ...
            10.^zvec(1);

    else

        C = ...
            C_min_eff;

        reg_param_candidate = ...
            Reg_min;

    end

else

    C = 1;

    if Reg_fixed

        reg_param_candidate = ...
            Reg_min;

    else

        reg_param_candidate = ...
            10.^zvec(1);

    end

end


    % -----------------------------------------------------
    % Recalculate this candidate independently.
    %
    % IMPORTANT:
    %
    %   - X_scaled is used
    %   - same cvp_fixed is used
    %   - no RNG reset
    % -----------------------------------------------------

result = ...
    globalrbf.evaluateGlobalCV( ...
        X_scaled, ...
        y, ...
        kernel, ...
        C, ...
        cvp_fixed, ...
        reg_param_candidate, ...
        options, ...
        cvCache);

nRBFSolves = ...
    nRBFSolves + result.nRBFSolves;

rmse_actual = ...
    result.RMSE;

if result.success && ...
   isfinite(rmse_actual)

    f = ...
        rmse_actual;

else

    f = ...
        FAIL_PENALTY;

    rmse_actual = ...
        NaN;

end

    % -----------------------------------------------------
    % Preserve EVERY evaluation at full precision.
    % -----------------------------------------------------

if useShape

    C_eval_log(end+1,1) = C;

else

    C_eval_log(end+1,1) = NaN;

end
    Reg_eval_log(end+1,1) = ...
        reg_param_candidate;

    E_eval_log(end+1,1) = ...
        rmse_actual;

   end

end
% =============================================================
% Local helper: adjust CV folds
%


function k_used = adjust_kfolds(n, d, kreq)

    if n < 2

        error( ...
            'At least two observations are required for CV.');

    end


    % K cannot exceed the number of observations.
    k_used = ...
        min(n, kreq);


    % Minimum training-set size over all K folds.
    min_train = ...
        n - ceil(n / k_used);


    if min_train < (d + 1)

        error(['Requested %d-fold CV gives a minimum training size ' ...
               'of %d, but at least %d observations are required ' ...
               'for the polynomial tail. Increase kfolds or the ' ...
               'number of observations.'], ...
               k_used, min_train, d + 1);

    end

end


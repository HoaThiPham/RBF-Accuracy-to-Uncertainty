function Bootstrap = globalRBFBootstrapForTesting( ...
    X,y,X_test,trainingInfo, ...
    nBootstrap,RNG_SEED,bootstrapOptions)

%GLOBALRBFBOOTSTRAPFORTESTING
%
% Bootstrap prediction uncertainty for the FINAL global RBF model.
%
% The final prediction model is based on all n training observations.
%
% For every bootstrap realization:
%
%   1. if NumSpatialSegments<=1, resample all n training observations
%      WITH replacement exactly as in the original implementation;
%      otherwise partition the full fitting set once into adaptive radial
%      annuli about its spatial centroid and apply the SAME resampling
%      operation independently within each annulus;
%   2. preserve bootstrap sample size n;
%   3. retain selected C_global and RegParam;
%   4. calculate epsilon_boot=C_global*d_boot for shaped kernels;
%   5. refit RBF coefficients;
%   6. predict ALL fixed test locations.
%
% Before the final pointwise mean/variance is calculated, apply the same
% sequential dominance safeguard used by Local RBF: repeatedly remove the
% current largest squared-deviation contributor only while it contributes
% more than DominanceLimit (default 0.80) of the current total squared
% deviation, with at most floor(MaxDominantFraction*B) removals
% (default MaxDominantFraction=0.03).
%
% Therefore each test location receives B bootstrap predictions.
%
% Step (3):
%
%   var(z_est^B) =
%       sum_b (z_est^(b)-mean(z_est^B))^2 / (B-1)
%
% Step (4):
%
%   var(z_est^o) =
%       var(z_est^B) * n/(n-1)
%
% where n is the number of observations used by the original
% full-data RBF prediction and resampled in every bootstrap.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 5 || isempty(nBootstrap)
    nBootstrap = 200;
end


if nargin < 6 || isempty(RNG_SEED)
    RNG_SEED = 1;
end


if nargin < 7 || isempty(bootstrapOptions)
    bootstrapOptions = struct();
end


if ~isfield(bootstrapOptions,'Verbose') || ...
        isempty(bootstrapOptions.Verbose)

    bootstrapOptions.Verbose = true;
end


if ~isfield(bootstrapOptions,'MaxAttemptsFactor') || ...
        isempty(bootstrapOptions.MaxAttemptsFactor)

    bootstrapOptions.MaxAttemptsFactor = 10;
end


if ~isfield(bootstrapOptions,'StoreIndices') || ...
        isempty(bootstrapOptions.StoreIndices)

    bootstrapOptions.StoreIndices = false;
end


if ~isfield(bootstrapOptions,'NumSpatialSegments') || ...
        isempty(bootstrapOptions.NumSpatialSegments)

    bootstrapOptions.NumSpatialSegments = 1;
end


if ~isfield(bootstrapOptions,'MinPointsPerAnnulus') || ...
        isempty(bootstrapOptions.MinPointsPerAnnulus)

    bootstrapOptions.MinPointsPerAnnulus = 5;
end


if ~isfield(bootstrapOptions,'DominanceLimit') || ...
        isempty(bootstrapOptions.DominanceLimit)

    bootstrapOptions.DominanceLimit = 0.80;
end


if ~isfield(bootstrapOptions,'MaxDominantFraction') || ...
        isempty(bootstrapOptions.MaxDominantFraction)

    bootstrapOptions.MaxDominantFraction = 0.03;
end


verbose = ...
    logical(bootstrapOptions.Verbose);

storeIndices = ...
    logical(bootstrapOptions.StoreIndices);

maxAttemptsFactor = ...
    bootstrapOptions.MaxAttemptsFactor;


numSpatialSegments = ...
    bootstrapOptions.NumSpatialSegments;


minPointsPerAnnulus = ...
    bootstrapOptions.MinPointsPerAnnulus;


dominanceLimit = ...
    bootstrapOptions.DominanceLimit;


maxDominantFraction = ...
    bootstrapOptions.MaxDominantFraction;


useSpatialBootstrap = ...
    numSpatialSegments > 1;


% =============================================================
% 2. Validate data
% =============================================================

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


if size(X,1) ~= numel(y)
    error('X and y must contain the same number of observations.');
end


if any(~isfinite(y))
    error('y contains NaN or Inf.');
end


if ~isnumeric(X_test) || ...
   isempty(X_test) || ...
   ~ismatrix(X_test) || ...
   ~isreal(X_test) || ...
   any(~isfinite(X_test(:)))

    error('X_test must be a nonempty finite real numeric matrix.');
end


if size(X_test,2) ~= size(X,2)

    error('X and X_test must have the same coordinate dimension.');
end


if ~isscalar(nBootstrap) || ...
   ~isfinite(nBootstrap) || ...
   nBootstrap < 2 || ...
   nBootstrap ~= round(nBootstrap)

    error('nBootstrap must be an integer >= 2.');
end


if ~isscalar(RNG_SEED) || ...
   ~isfinite(RNG_SEED) || ...
   RNG_SEED < 0 || ...
   RNG_SEED > (2^32-1) || ...
   RNG_SEED ~= round(RNG_SEED)

    error(['RNG_SEED must be a finite nonnegative integer ' ...
           'not exceeding 2^32-1.']);
end


if ~isscalar(maxAttemptsFactor) || ...
   ~isfinite(maxAttemptsFactor) || ...
   maxAttemptsFactor < 1

    error('MaxAttemptsFactor must be a finite scalar >= 1.');
end


if ~isscalar(verbose)
    error('bootstrapOptions.Verbose must be scalar.');
end


if ~isscalar(storeIndices)
    error('bootstrapOptions.StoreIndices must be scalar.');
end


if ~isnumeric(numSpatialSegments) || ...
   ~isscalar(numSpatialSegments) || ...
   ~isfinite(numSpatialSegments) || ...
   numSpatialSegments < 0 || ...
   numSpatialSegments ~= round(numSpatialSegments)

    error(['bootstrapOptions.NumSpatialSegments must be a ' ...
           'nonnegative integer. Values 0 and 1 use the exact ' ...
           'current ordinary bootstrap; values >1 enable annular resampling.']);
end


if ~isnumeric(minPointsPerAnnulus) || ...
   ~isscalar(minPointsPerAnnulus) || ...
   ~isfinite(minPointsPerAnnulus) || ...
   minPointsPerAnnulus < 1 || ...
   minPointsPerAnnulus ~= round(minPointsPerAnnulus)

    error(['bootstrapOptions.MinPointsPerAnnulus must be a ' ...
           'positive integer.']);
end


if ~isnumeric(dominanceLimit) || ...
   ~isscalar(dominanceLimit) || ...
   ~isfinite(dominanceLimit) || ...
   dominanceLimit <= 0 || ...
   dominanceLimit >= 1

    error(['bootstrapOptions.DominanceLimit must be a finite ' ...
           'scalar strictly between 0 and 1.']);
end


if ~isnumeric(maxDominantFraction) || ...
   ~isscalar(maxDominantFraction) || ...
   ~isfinite(maxDominantFraction) || ...
   maxDominantFraction < 0 || ...
   maxDominantFraction >= 1

    error(['bootstrapOptions.MaxDominantFraction must be a finite ' ...
           'scalar in [0,1).']);
end


% =============================================================
% 3. Extract selected FINAL model configuration
% =============================================================

requiredFields = { ...
    'kernel', ...
    'reg_param', ...
    'S_global', ...
    'options'};


for k = 1:numel(requiredFields)

    if ~isfield(trainingInfo,requiredFields{k})

        error( ...
            'trainingInfo.%s is required.', ...
            requiredFields{k});
    end

end


if isfield(trainingInfo,'success') && ...
        ~trainingInfo.success

    error('trainingInfo does not represent a successful training run.');
end


kernel = ...
    char(trainingInfo.kernel);


useShape = ...
    common.kernelHasShapeParameter(kernel);


if isfield(trainingInfo,'useShape')

    if ~isscalar(trainingInfo.useShape) || ...
       logical(trainingInfo.useShape) ~= useShape

        error('trainingInfo.useShape is inconsistent with kernel.');
    end

end


reg_param = ...
    trainingInfo.reg_param;


if ~isscalar(reg_param) || ...
   ~isfinite(reg_param) || ...
   reg_param <= 0

    error('trainingInfo.reg_param must be finite and positive.');
end


S_global = ...
    trainingInfo.S_global;


if ~isscalar(S_global) || ...
   ~isfinite(S_global) || ...
   S_global <= 0

    error('trainingInfo.S_global must be finite and positive.');
end


options = ...
    trainingInfo.options;


if ~isfield(options,'solver') || ...
        ~isstruct(options.solver)

    options.solver = struct();
end


if ~isfield(options.solver,'cond_lsq') || ...
        isempty(options.solver.cond_lsq)

    options.solver.cond_lsq = 1e12;
end


if ~isscalar(options.solver.cond_lsq) || ...
   ~isfinite(options.solver.cond_lsq) || ...
   options.solver.cond_lsq <= 0

    error('options.solver.cond_lsq must be finite and positive.');
end


if useShape

    if ~isfield(trainingInfo,'C_global') || ...
       ~isscalar(trainingInfo.C_global) || ...
       ~isfinite(trainingInfo.C_global) || ...
       trainingInfo.C_global <= 0

        error('A finite positive trainingInfo.C_global is required.');
    end

    C_global = ...
        trainingInfo.C_global;

else

    C_global = 1;

end


% =============================================================
% 4. Reuse ORIGINAL global coordinate scaling
% =============================================================

X_scaled = ...
    X ./ S_global;


X_test_scaled = ...
    X_test ./ S_global;


n = ...
    size(X_scaled,1);


nTest = ...
    size(X_test_scaled,1);


if n <= 1
    error('At least two observations are required.');
end


% =============================================================
% 5. Start timing
% =============================================================

tTotal = ...
    tic;


% =============================================================
% 6. Precompute geometry ONCE
%
% No pdist2 is called inside bootstrap realizations.
% =============================================================

tGeometry = ...
    tic;


R_all = ...
    pdist2(X_scaled,X_scaled);


P_all = ...
    [ones(n,1),X_scaled];


R_test_all = ...
    pdist2(X_test_scaled,X_scaled);


P_test = ...
    [ones(nTest,1),X_test_scaled];


geometryWallTime = ...
    toc(tGeometry);


% =============================================================
% 7. Original full-data model geometry
% =============================================================

dOriginal = NaN;
epsilonOriginal = NaN;


if useShape

    [epsilonOriginal,dOriginal] = ...
        common.buildEpsilon( ...
            X_scaled, ...
            C_global);

end


% =============================================================
% 8. Fixed original geometry cache
% =============================================================

geometry = struct();


geometry.R = ...
    R_all;


geometry.P = ...
    P_all;


geometry.R_eval = ...
    R_test_all;


geometry.P_eval = ...
    P_test;


% =============================================================
% 9. Allocate outputs
% =============================================================

Predictions = ...
    nan(nTest,nBootstrap);


% Diagnostics retain the original 1-by-bootstrap layout in both modes,
% because one bootstrap RBF fit predicts all test targets together.
dBootstrap = ...
    nan(1,nBootstrap);

epsilonBootstrap = ...
    nan(1,nBootstrap);

condReg = ...
    nan(1,nBootstrap);

regPhi = ...
    nan(1,nBootstrap);

solverUsed = ...
    cell(1,nBootstrap);

nUnique = ...
    nan(1,nBootstrap);


if storeIndices

    BootstrapIndices = ...
        cell(1,nBootstrap);

else

    BootstrapIndices = ...
        [];

end


% Spatial partition is built ONCE for the complete fitting set.
if useSpatialBootstrap

    SpatialCenterScaled = ...
        mean(X_scaled,1);


    SpatialCenter = ...
        SpatialCenterScaled .* S_global;


    distanceToCenter = ...
        sqrt(sum((X_scaled - SpatialCenterScaled).^2,2));


    [segments,segmentInfo] = ...
        globalrbf.buildAdaptiveRadialSegments( ...
            distanceToCenter, ...
            numSpatialSegments, ...
            minPointsPerAnnulus);


    NumSegmentsEffective = ...
        segmentInfo.NumSegmentsEffective;


    MinPointsFeasible = ...
        segmentInfo.MinPointsFeasible;


    MinPointsSatisfied = ...
        segmentInfo.MinPointsSatisfied;


    SegmentCountsRequested = ...
        segmentInfo.RequestedCounts;


    SegmentCountsEffective = ...
        segmentInfo.EffectiveCounts;


    RadiusEdgesRequestedScaled = ...
        segmentInfo.RequestedRadiusEdges;


    RadiusEdgesEffectiveScaled = ...
        segmentInfo.EffectiveRadiusEdges;


    RadiusEdgesRequested = ...
        segmentInfo.RequestedRadiusEdges .* S_global;


    RadiusEdgesEffective = ...
        segmentInfo.EffectiveRadiusEdges .* S_global;

else

    segments = [];
    SpatialCenterScaled = [];
    SpatialCenter = [];
    NumSegmentsEffective = [];
    MinPointsFeasible = [];
    MinPointsSatisfied = [];
    SegmentCountsRequested = [];
    SegmentCountsEffective = [];
    RadiusEdgesRequestedScaled = [];
    RadiusEdgesEffectiveScaled = [];
    RadiusEdgesRequested = [];
    RadiusEdgesEffective = [];

end


maxAttempts = ...
    ceil(maxAttemptsFactor*nBootstrap);


% =============================================================
% 10. Bootstrap final prediction model
% =============================================================

tFits = ...
    tic;


stream = ...
    RandStream( ...
        'mt19937ar', ...
        'Seed', ...
        RNG_SEED);


b = 0;
attempts = 0;

rejectAdmissibility = 0;
rejectSolve = 0;
rejectPrediction = 0;

nRBFSolves = 0;


while b < nBootstrap && ...
        attempts < maxAttempts

    attempts = ...
        attempts + 1;


    if useSpatialBootstrap

        % Same current bootstrap-with-replacement operation, applied
        % independently within the common fitting-set radial annuli.
        idxBoot = ...
            globalrbf.resampleRadialSegments( ...
                segments, ...
                stream, ...
                n);

    else

        % EXACT current/original bootstrap path. NumSpatialSegments equal
        % to 0 or 1 executes this literal randi operation.
        idxBoot = ...
            randi( ...
                stream, ...
                n, ...
                [n,1]);

    end


    [yhat,bootInfo] = ...
        common.bootstrapRBFReplicate( ...
            X_scaled, ...
            y, ...
            X_test_scaled, ...
            idxBoot, ...
            kernel, ...
            C_global, ...
            reg_param, ...
            options, ...
            geometry);


    nRBFSolves = ...
        nRBFSolves + ...
        bootInfo.nRBFSolves;


    if ~bootInfo.success

        switch bootInfo.failureStage

            case 'admissibility'
                rejectAdmissibility = rejectAdmissibility + 1;

            case 'solve'
                rejectSolve = rejectSolve + 1;

            case 'prediction'
                rejectPrediction = rejectPrediction + 1;

            otherwise
                error('Unknown bootstrap failure stage.');
        end

        continue
    end


    b = ...
        b + 1;


    Predictions(:,b) = ...
        yhat;


    dBootstrap(b) = ...
        bootInfo.d_boot;


    epsilonBootstrap(b) = ...
        bootInfo.epsilon;


    condReg(b) = ...
        bootInfo.cond_reg;


    regPhi(b) = ...
        bootInfo.reg_phi;


    solverUsed{b} = ...
        bootInfo.solver_used;


    nUnique(b) = ...
        bootInfo.nUnique;


    if storeIndices

        BootstrapIndices{b} = ...
            idxBoot;

    end

end


if b < nBootstrap

    error([ ...
        'Only %d/%d valid testing bootstrap realizations were ' ...
        'obtained after %d attempts.'], ...
        b,nBootstrap,attempts);
end


fitWallTime = ...
    toc(tFits);


totalWallTime = ...
    toc(tTotal);


% =============================================================
% 12. Completion check
% =============================================================

if any(~isfinite(Predictions(:)))

    error(['Testing bootstrap prediction matrix is incomplete ' ...
           'or contains NaN/Inf.']);
end


% =============================================================
% 13. Pointwise bootstrap statistics
% =============================================================

PredictionMean = ...
    nan(nTest,1);


BootstrapVariance = ...
    nan(nTest,1);


nDominant = ...
    zeros(nTest,1);


% -------------------------------------------------------------
% Sequential dominance control before Step (3), Eq. (25).
%
% The complete accepted B-realization matrix remains unchanged.
% Dominance control acts only on the pointwise statistics vector,
% exactly as in the Local-RBF bootstrap implementation.
% -------------------------------------------------------------

for iTarget = 1:nTest

    [predPointStats,nDominant(iTarget)] = ...
        applyDominanceControlGlobal( ...
            Predictions(iTarget,:), ...
            dominanceLimit, ...
            maxDominantFraction);


    PredictionMean(iTarget) = ...
        mean(predPointStats);


    % MATLAB normalization 0 uses denominator B_retained-1.
    BootstrapVariance(iTarget) = ...
        var(predPointStats,0);

end


BootstrapStd = ...
    sqrt(BootstrapVariance);


% -------------------------------------------------------------
% Step (4), Eq. (26)
%
% The original final RBF prediction uses n observations.
% Every bootstrap realization therefore resamples n observations.
% -------------------------------------------------------------

nUsed = ...
    n;


AdjustmentFactor = ...
    nUsed / (nUsed - 1);


AdjustedVariance = ...
    BootstrapVariance .* AdjustmentFactor;


AdjustedStd = ...
    sqrt(AdjustedVariance);


nValid = ...
    sum(isfinite(Predictions),2);


% =============================================================
% 14. Numerical summaries
% =============================================================

nBackslash = ...
    sum(strcmp(solverUsed(:),'backslash'));


nLsqminnorm = ...
    sum(strcmp(solverUsed(:),'lsqminnorm'));


condRegAll = ...
    condReg(isfinite(condReg));


if isempty(condRegAll)

    condRegMin = NaN;
    condRegMedian = NaN;
    condRegMax = NaN;

else

    condRegMin = ...
        min(condRegAll);

    condRegMedian = ...
        median(condRegAll);

    condRegMax = ...
        max(condRegAll);

end


nRejected = ...
    rejectAdmissibility + ...
    rejectSolve + ...
    rejectPrediction;


% =============================================================
% 15. Output
% =============================================================

Bootstrap = struct();


% -------------------------------------------------------------
% Selected model
% -------------------------------------------------------------

Bootstrap.kernel = ...
    kernel;


Bootstrap.useShape = ...
    useShape;


Bootstrap.reg_param = ...
    reg_param;


Bootstrap.S_global = ...
    S_global;


if useShape

    Bootstrap.C_global = ...
        C_global;

else

    Bootstrap.C_global = ...
        NaN;

end


% -------------------------------------------------------------
% Complete prediction/statistical information
% -------------------------------------------------------------

Bootstrap.Predictions = ...
    Predictions;


Bootstrap.Mean = ...
    PredictionMean;


% Step (3), Eq. (25)
Bootstrap.BootstrapVariance = ...
    BootstrapVariance;


Bootstrap.BootstrapStd = ...
    BootstrapStd;


% Step (4), Eq. (26)
Bootstrap.AdjustedVariance = ...
    AdjustedVariance;


Bootstrap.AdjustedStd = ...
    AdjustedStd;


% Final variance/std reported by the method
Bootstrap.Variance = ...
    AdjustedVariance;


Bootstrap.Std = ...
    AdjustedStd;


Bootstrap.nUsed = ...
    nUsed;


Bootstrap.AdjustmentFactor = ...
    AdjustmentFactor;


Bootstrap.nValid = ...
    nValid;


% -------------------------------------------------------------
% Sequential dominance-control information
% -------------------------------------------------------------

Bootstrap.DominanceControl = ...
    struct();


Bootstrap.DominanceControl.DominanceLimit = ...
    dominanceLimit;


Bootstrap.DominanceControl.MaxDominantFraction = ...
    maxDominantFraction;


Bootstrap.DominanceControl.MaxAllowedDominant = ...
    min( ...
        floor(maxDominantFraction*nBootstrap), ...
        max(0,nBootstrap-2));


Bootstrap.DominanceControl.PointMarked = ...
    (nDominant > 0);


Bootstrap.DominanceControl.PointIndex = ...
    find(nDominant > 0);


Bootstrap.DominanceControl.NDominantByPoint = ...
    nDominant;


Bootstrap.DominanceControl.nDominantPoints = ...
    nnz(nDominant > 0);


Bootstrap.DominanceControl.totalDominant = ...
    sum(nDominant);


if isempty(nDominant)

    Bootstrap.DominanceControl.maxDominantAtPoint = ...
        0;

else

    Bootstrap.DominanceControl.maxDominantAtPoint = ...
        max(nDominant);

end


% -------------------------------------------------------------
% Original full-data prediction geometry
% -------------------------------------------------------------

Bootstrap.Original.nFit = ...
    n;


Bootstrap.Original.nTest = ...
    nTest;


Bootstrap.Original.d = ...
    dOriginal;


Bootstrap.Original.epsilon = ...
    epsilonOriginal;


% -------------------------------------------------------------
% Bootstrap diagnostics
% -------------------------------------------------------------

Bootstrap.Diagnostics.d_bootstrap = ...
    dBootstrap;


Bootstrap.Diagnostics.epsilon_bootstrap = ...
    epsilonBootstrap;


Bootstrap.Diagnostics.cond_reg = ...
    condReg;


Bootstrap.Diagnostics.reg_phi = ...
    regPhi;


Bootstrap.Diagnostics.solver_used = ...
    solverUsed;


Bootstrap.Diagnostics.nUnique = ...
    nUnique;


Bootstrap.Diagnostics.Layout = ...
    'bootstrap';


Bootstrap.Diagnostics.nBackslash = ...
    nBackslash;


Bootstrap.Diagnostics.nLsqminnorm = ...
    nLsqminnorm;


Bootstrap.Diagnostics.cond_reg_min = ...
    condRegMin;


Bootstrap.Diagnostics.cond_reg_median = ...
    condRegMedian;


Bootstrap.Diagnostics.cond_reg_max = ...
    condRegMax;


% -------------------------------------------------------------
% Computation/rejection information
% -------------------------------------------------------------

Bootstrap.Computation.nAttempts = ...
    attempts;


Bootstrap.Computation.nRejected = ...
    nRejected;


Bootstrap.Computation.nRejectedAdmissibility = ...
    rejectAdmissibility;


Bootstrap.Computation.nRejectedSolve = ...
    rejectSolve;


Bootstrap.Computation.nRejectedPrediction = ...
    rejectPrediction;


Bootstrap.Computation.nRBFSolves = ...
    nRBFSolves;



Bootstrap.Computation.geometryWallTime = ...
    geometryWallTime;


Bootstrap.Computation.fitWallTime = ...
    fitWallTime;


Bootstrap.Computation.totalWallTime = ...
    totalWallTime;


% -------------------------------------------------------------
% Reproducibility/settings
% -------------------------------------------------------------

Bootstrap.Settings.nBootstrap = ...
    nBootstrap;


Bootstrap.Settings.RNG_SEED = ...
    RNG_SEED;


Bootstrap.Settings.MaxAttemptsFactor = ...
    maxAttemptsFactor;


Bootstrap.Settings.StoreIndices = ...
    storeIndices;


Bootstrap.Settings.NumSpatialSegments = ...
    numSpatialSegments;


Bootstrap.Settings.MinPointsPerAnnulus = ...
    minPointsPerAnnulus;


Bootstrap.Settings.DominanceLimit = ...
    dominanceLimit;


Bootstrap.Settings.MaxDominantFraction = ...
    maxDominantFraction;


Bootstrap.Settings.DominanceControlMethod = ...
    'sequential_largest_variance_contributor';


Bootstrap.Settings.SpatialBootstrapEnabled = ...
    useSpatialBootstrap;


if useSpatialBootstrap

    Bootstrap.Settings.ResamplingLayout = ...
        'fitting_set_centered_annular_shared';

else

    Bootstrap.Settings.ResamplingLayout = ...
        'original_all_test_shared';

end


if useSpatialBootstrap

    Bootstrap.Spatial.NumSegmentsRequested = ...
        numSpatialSegments;


    Bootstrap.Spatial.NumSegmentsEffective = ...
        NumSegmentsEffective;


    Bootstrap.Spatial.MinPointsPerAnnulus = ...
        minPointsPerAnnulus;


    Bootstrap.Spatial.MinPointsFeasible = ...
        MinPointsFeasible;


    Bootstrap.Spatial.MinPointsSatisfied = ...
        MinPointsSatisfied;


    Bootstrap.Spatial.CenterScaled = ...
        SpatialCenterScaled;


    Bootstrap.Spatial.Center = ...
        SpatialCenter;


    Bootstrap.Spatial.SegmentCountsRequested = ...
        SegmentCountsRequested;


    Bootstrap.Spatial.SegmentCountsEffective = ...
        SegmentCountsEffective;


    Bootstrap.Spatial.RadiusEdgesRequestedScaled = ...
        RadiusEdgesRequestedScaled;


    Bootstrap.Spatial.RadiusEdgesEffectiveScaled = ...
        RadiusEdgesEffectiveScaled;


    Bootstrap.Spatial.RadiusEdgesRequested = ...
        RadiusEdgesRequested;


    Bootstrap.Spatial.RadiusEdgesEffective = ...
        RadiusEdgesEffective;

end


Bootstrap.Settings.options = ...
    options;


if storeIndices

    Bootstrap.BootstrapIndices = ...
        BootstrapIndices;

end


Bootstrap.success = ...
    all(nValid == nBootstrap);

end

% =============================================================
% Local helper
% =============================================================

function [pStats,nDominant] = applyDominanceControlGlobal( ...
    p,dominanceLimit,maxDominantFraction)
%APPLYDOMINANCECONTROLGLOBAL
% Apply the same pointwise sequential dominance safeguard used by the
% Local-RBF bootstrap. At each iteration, the prediction with the largest
% current squared deviation from the current mean is removed only when
% its contribution to the current total squared deviation exceeds the
% prescribed dominance limit. At most floor(maxDominantFraction*B)
% predictions are removed, while retaining at least two predictions.

pStats = ...
    p(:).';


Boriginal = ...
    numel(pStats);


maxDominant = ...
    min( ...
        floor(maxDominantFraction*Boriginal), ...
        max(0,Boriginal-2));


nDominant = ...
    0;


while nDominant < maxDominant

    meanP = ...
        mean(pStats);


    sqDev = ...
        (pStats - meanP).^2;


    totalSS = ...
        sum(sqDev);


    if ~isfinite(totalSS) || ...
       totalSS <= 0

        break;

    end


    [maxSqDev,idxExtreme] = ...
        max(sqDev);


    dominance = ...
        maxSqDev / totalSS;


    if ~isfinite(dominance) || ...
       dominance <= dominanceLimit

        break;

    end


    pStats(idxExtreme) = ...
        [];


    nDominant = ...
        nDominant + 1;

end

end


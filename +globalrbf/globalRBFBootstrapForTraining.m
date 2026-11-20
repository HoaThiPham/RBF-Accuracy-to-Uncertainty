function Bootstrap = globalRBFBootstrapForTraining( ...
    X,y,trainingInfo,cvp_fixed, ...
    nBootstrap,RNG_SEED,bootstrapOptions)

%GLOBALRBFBOOTSTRAPFORTRAINING
%
% Bootstrap prediction uncertainty at GLOBAL RBF training points.
%
% CRITICAL:
%   The ORIGINAL fixed CV partition is reused.
%   NO new cvpartition is generated.
%
% For each original fixed CV fold:
%
%   1. retain the original fitting observations;
%   2. retain the original validation observations;
%   3. if NumSpatialSegments<=1, resample the complete fitting set
%      WITH replacement exactly as in the original implementation;
%      otherwise partition the fitting set once into adaptive radial annuli
%      about its spatial centroid and apply the SAME resampling operation
%      independently within each annulus;
%   4. preserve the fitting-set sample size;
%   5. retain fixed selected C_global and RegParam;
%   6. calculate epsilon_boot=C_global*d_boot for shaped kernels;
%   7. refit RBF coefficients;
%   8. predict the SAME original validation locations.
%
% Before the final pointwise mean/variance is calculated, apply the same
% sequential dominance safeguard used by Local RBF: repeatedly remove the
% current largest squared-deviation contributor only while it contributes
% more than DominanceLimit (default 0.80) of the current total squared
% deviation, with at most floor(MaxDominantFraction*B) removals
% (default MaxDominantFraction=0.03).
%
% Each original training point therefore receives B bootstrap
% predictions from the model corresponding to its original CV fold.
%
% Step (3):
%
%   var(z_est^B) =
%       sum_b (z_est^(b)-mean(z_est^B))^2 / (B-1)
%
% Step (4):
%
%   var(z_est^o) =
%       var(z_est^B) * n_used/(n_used-1)
%
% where n_used is the number of observations used to construct
% that RBF prediction and subsequently resampled in bootstrap.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 4 || isempty(cvp_fixed)

    if isfield(trainingInfo,'cvp_fixed') && ...
            ~isempty(trainingInfo.cvp_fixed)

        cvp_fixed = ...
            trainingInfo.cvp_fixed;

    else

        error(['cvp_fixed is required. The training bootstrap ' ...
               'must reuse the original fixed CV partition.']);
    end

end


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
% 3. Extract selected RBF model configuration
% =============================================================

requiredFields = { ...
    'kernel', ...
    'reg_param', ...
    'S_global', ...
    'k_used', ...
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


k_used = ...
    trainingInfo.k_used;


if ~isscalar(k_used) || ...
   ~isfinite(k_used) || ...
   k_used < 2 || ...
   k_used ~= round(k_used)

    error('trainingInfo.k_used is invalid.');
end


options = ...
    trainingInfo.options;


% -------------------------------------------------------------
% Same solver threshold as main workflow.
% -------------------------------------------------------------

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

    % Internal dummy only.
    C_global = 1;

end


% =============================================================
% 4. Reuse ORIGINAL global coordinate scaling
% =============================================================

X_scaled = ...
    X ./ S_global;


n = ...
    size(X_scaled,1);


% =============================================================
% 5. Validate ORIGINAL fixed CV partition
%
% Absolutely no cvpartition(...) call appears in this function.
% =============================================================

if cvp_fixed.NumTestSets ~= k_used

    error(['cvp_fixed.NumTestSets is inconsistent with ' ...
           'trainingInfo.k_used.']);
end


idxCheck = ...
    test(cvp_fixed,1);


if numel(idxCheck) ~= n

    error(['cvp_fixed does not correspond to the supplied ' ...
           'training dataset.']);
end


% -------------------------------------------------------------
% If available, explicitly confirm that the supplied partition
% equals the one actually used during model training.
% -------------------------------------------------------------

if isfield(trainingInfo,'cvp_fixed') && ...
        ~isempty(trainingInfo.cvp_fixed)

    cvp_training = ...
        trainingInfo.cvp_fixed;


    if cvp_training.NumTestSets ~= k_used

        error('trainingInfo.cvp_fixed is inconsistent with k_used.');
    end


    for f = 1:k_used

        if ~isequal( ...
                test(cvp_fixed,f), ...
                test(cvp_training,f))

            error(['Supplied cvp_fixed differs from the partition ' ...
                   'used during model training.']);
        end

    end

end


% =============================================================
% 6. Extract fixed CV structure
% =============================================================

foldID = ...
    zeros(n,1);

validationCount = ...
    zeros(n,1);

nFitFold = ...
    zeros(k_used,1);

nValFold = ...
    zeros(k_used,1);


for f = 1:k_used

    idxTrain = ...
        training(cvp_fixed,f);

    idxVal = ...
        test(cvp_fixed,f);


    if any(idxTrain & idxVal)

        error( ...
            'Training and validation sets overlap in fold %d.', ...
            f);
    end


    if any(~(idxTrain | idxVal))

        error( ...
            'Fold %d does not cover all observations.', ...
            f);
    end


    foldID(idxVal) = ...
        f;


    validationCount(idxVal) = ...
        validationCount(idxVal) + 1;


    nFitFold(f) = ...
        sum(idxTrain);


    nValFold(f) = ...
        sum(idxVal);

end


if any(validationCount ~= 1)

    error(['Each observation must occur exactly once in the ' ...
           'fixed CV validation sets.']);
end


if any(nFitFold <= 1)

    error('At least two fitting observations are required in every fold.');
end


% =============================================================
% 7. Start timing
% =============================================================

tTotal = ...
    tic;


% =============================================================
% 8. Precompute geometry ONCE
%
% This is a major time saving because bootstrap realizations
% only index the cached matrices instead of calling pdist2.
% =============================================================

tGeometry = ...
    tic;


R_all = ...
    pdist2(X_scaled,X_scaled);


P_all = ...
    [ones(n,1),X_scaled];


geometryWallTime = ...
    toc(tGeometry);


% =============================================================
% 9. Allocate output arrays
% =============================================================

Predictions = ...
    nan(n,nBootstrap);


% Original fixed-fold geometry
dOriginalFold = ...
    nan(k_used,1);

epsilonOriginalFold = ...
    nan(k_used,1);


% Accepted-bootstrap diagnostics retain the original fold-by-bootstrap
% layout for BOTH ordinary and annular resampling. One bootstrap RBF fit
% predicts all validation targets in the fold.
dBootstrap = ...
    nan(k_used,nBootstrap);

epsilonBootstrap = ...
    nan(k_used,nBootstrap);

condReg = ...
    nan(k_used,nBootstrap);

regPhi = ...
    nan(k_used,nBootstrap);

solverUsed = ...
    cell(k_used,nBootstrap);

nUnique = ...
    nan(k_used,nBootstrap);


% Rejection/computational information remains fold based.
nAttempts = ...
    zeros(k_used,1);

nRejected = ...
    zeros(k_used,1);

nRejectedAdmissibility = ...
    zeros(k_used,1);

nRejectedSolve = ...
    zeros(k_used,1);

nRejectedPrediction = ...
    zeros(k_used,1);

nRBFSolves = ...
    zeros(k_used,1);


% Spatial-partition metadata is computed ONCE per original CV fitting set.
if useSpatialBootstrap

    pCoord = ...
        size(X_scaled,2);

    SpatialCenterScaled = ...
        nan(k_used,pCoord);

    SpatialCenter = ...
        nan(k_used,pCoord);

    NumSegmentsEffective = ...
        nan(k_used,1);

    MinPointsFeasible = ...
        false(k_used,1);

    MinPointsSatisfied = ...
        false(k_used,1);

    SegmentCountsRequested = ...
        cell(k_used,1);

    SegmentCountsEffective = ...
        cell(k_used,1);

    RadiusEdgesRequestedScaled = ...
        cell(k_used,1);

    RadiusEdgesEffectiveScaled = ...
        cell(k_used,1);

    RadiusEdgesRequested = ...
        cell(k_used,1);

    RadiusEdgesEffective = ...
        cell(k_used,1);

else

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


% Optional exact bootstrap samples preserve the original fold-by-bootstrap
% storage layout in both modes.
if storeIndices

    BootstrapIndices = ...
        cell(k_used,nBootstrap);

else

    BootstrapIndices = ...
        [];

end


FoldRNG_SEED = ...
    zeros(k_used,1);


maxAttempts = ...
    ceil(maxAttemptsFactor*nBootstrap);


% =============================================================
% 10. Bootstrap original fixed CV folds
% =============================================================

tFits = ...
    tic;


for f = 1:k_used

    % ---------------------------------------------------------
    % EXACT original CV fitting/validation observations
    % ---------------------------------------------------------

    idxTrain = ...
        training(cvp_fixed,f);

    idxVal = ...
        test(cvp_fixed,f);


    trainID = ...
        find(idxTrain);

    valID = ...
        find(idxVal);


    Xfit = ...
        X_scaled(trainID,:);

    yfit = ...
        y(trainID);

    Xval = ...
        X_scaled(valID,:);


    m = ...
        size(Xfit,1);


    % ---------------------------------------------------------
    % Original fold geometry
    % ---------------------------------------------------------

    if useShape

        [epsilonOriginalFold(f),dOriginalFold(f)] = ...
            common.buildEpsilon( ...
                Xfit, ...
                C_global);

    end


    % ---------------------------------------------------------
    % Cache original fold geometry
    % ---------------------------------------------------------

    geometry = struct();


    geometry.R = ...
        R_all(trainID,trainID);


    geometry.P = ...
        P_all(trainID,:);


    geometry.R_eval = ...
        R_all(valID,trainID);


    geometry.P_eval = ...
        P_all(valID,:);


    % ---------------------------------------------------------
    % Build ONE common spatial partition for this fitting set.
    % This is outside the bootstrap loop and is shared by all validation
    % targets and all bootstrap replicates in this CV fold.
    % ---------------------------------------------------------

    if useSpatialBootstrap

        centerScaled = ...
            mean(Xfit,1);


        distanceToCenter = ...
            sqrt(sum((Xfit - centerScaled).^2,2));


        [segments,segmentInfo] = ...
            globalrbf.buildAdaptiveRadialSegments( ...
                distanceToCenter, ...
                numSpatialSegments, ...
                minPointsPerAnnulus);


        SpatialCenterScaled(f,:) = ...
            centerScaled;


        SpatialCenter(f,:) = ...
            centerScaled .* S_global;


        NumSegmentsEffective(f) = ...
            segmentInfo.NumSegmentsEffective;


        MinPointsFeasible(f) = ...
            segmentInfo.MinPointsFeasible;


        MinPointsSatisfied(f) = ...
            segmentInfo.MinPointsSatisfied;


        SegmentCountsRequested{f} = ...
            segmentInfo.RequestedCounts;


        SegmentCountsEffective{f} = ...
            segmentInfo.EffectiveCounts;


        RadiusEdgesRequestedScaled{f} = ...
            segmentInfo.RequestedRadiusEdges;


        RadiusEdgesEffectiveScaled{f} = ...
            segmentInfo.EffectiveRadiusEdges;


        RadiusEdgesRequested{f} = ...
            segmentInfo.RequestedRadiusEdges .* S_global;


        RadiusEdgesEffective{f} = ...
            segmentInfo.EffectiveRadiusEdges .* S_global;

    else

        segments = [];

    end


    % ---------------------------------------------------------
    % Independent reproducible random stream for this fold
    % ---------------------------------------------------------

    foldSeed = ...
        mod( ...
            double(RNG_SEED) + double(f-1), ...
            2^32);


    FoldRNG_SEED(f) = ...
        foldSeed;


    stream = ...
        RandStream( ...
            'mt19937ar', ...
            'Seed', ...
            foldSeed);


    % ---------------------------------------------------------
    % Generate exactly B ACCEPTED bootstrap predictions.
    % Ordinary and spatial modes differ ONLY in idxBoot generation.
    % ---------------------------------------------------------

    b = 0;
    attempts = 0;

    rejectAdmissibility = 0;
    rejectSolve = 0;
    rejectPrediction = 0;

    rbfSolvesFold = 0;


    while b < nBootstrap && ...
            attempts < maxAttempts

        attempts = ...
            attempts + 1;


        if useSpatialBootstrap

            % Same current bootstrap-with-replacement operation, applied
            % independently inside each common fitting-set annulus.
            idxBoot = ...
                globalrbf.resampleRadialSegments( ...
                    segments, ...
                    stream, ...
                    m);

        else

            % EXACT current/original bootstrap path. NumSpatialSegments
            % equal to 0 or 1 executes this literal randi operation.
            idxBoot = ...
                randi( ...
                    stream, ...
                    m, ...
                    [m,1]);

        end


        [yhat,bootInfo] = ...
            common.bootstrapRBFReplicate( ...
                Xfit, ...
                yfit, ...
                Xval, ...
                idxBoot, ...
                kernel, ...
                C_global, ...
                reg_param, ...
                options, ...
                geometry);


        rbfSolvesFold = ...
            rbfSolvesFold + ...
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


        Predictions(valID,b) = ...
            yhat;


        dBootstrap(f,b) = ...
            bootInfo.d_boot;


        epsilonBootstrap(f,b) = ...
            bootInfo.epsilon;


        condReg(f,b) = ...
            bootInfo.cond_reg;


        regPhi(f,b) = ...
            bootInfo.reg_phi;


        solverUsed{f,b} = ...
            bootInfo.solver_used;


        nUnique(f,b) = ...
            bootInfo.nUnique;


        if storeIndices

            BootstrapIndices{f,b} = ...
                idxBoot;

        end

    end


    if b < nBootstrap

        error([ ...
            'Only %d/%d valid bootstrap realizations were ' ...
            'obtained for fold %d after %d attempts.'], ...
            b,nBootstrap,f,attempts);
    end


    nAttempts(f) = ...
        attempts;


    nRejectedAdmissibility(f) = ...
        rejectAdmissibility;


    nRejectedSolve(f) = ...
        rejectSolve;


    nRejectedPrediction(f) = ...
        rejectPrediction;


    nRejected(f) = ...
        rejectAdmissibility + ...
        rejectSolve + ...
        rejectPrediction;


    nRBFSolves(f) = ...
        rbfSolvesFold;


    if verbose

        fprintf( ...
            ['[bootstrap training] fold %d/%d: ' ...
             '%d accepted, %d rejected, %d RBF solves.\n'], ...
            f,k_used, ...
            nBootstrap, ...
            nRejected(f), ...
            nRBFSolves(f));

    end

end


fitWallTime = ...
    toc(tFits);


totalWallTime = ...
    toc(tTotal);


% =============================================================
% 11. Integrity check
% =============================================================

if any(~isfinite(Predictions(:)))

    error(['Training bootstrap prediction matrix is incomplete ' ...
           'or contains NaN/Inf.']);
end


% =============================================================
% 12. Pointwise bootstrap statistics
% =============================================================

PredictionMean = ...
    nan(n,1);


BootstrapVariance = ...
    nan(n,1);


nDominant = ...
    zeros(n,1);


% -------------------------------------------------------------
% Sequential dominance control before Step (3), Eq. (25).
%
% The complete accepted B-realization matrix remains unchanged.
% Dominance control acts only on the pointwise statistics vector,
% exactly as in the Local-RBF bootstrap implementation.
% -------------------------------------------------------------

for iTarget = 1:n

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
% nUsed(i) is the number of observations used to construct
% prediction i and subsequently resampled for its bootstrap.
% -------------------------------------------------------------

nUsed = ...
    nFitFold(foldID);


AdjustmentFactor = ...
    nUsed ./ (nUsed - 1);


AdjustedVariance = ...
    BootstrapVariance .* AdjustmentFactor;


AdjustedStd = ...
    sqrt(AdjustedVariance);


nValid = ...
    sum(isfinite(Predictions),2);


% =============================================================
% 13. Numerical summaries
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


% =============================================================
% 14. Output
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
% Complete bootstrap prediction information
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


% Final variance/std used by the manuscript.
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
% Fixed CV information
% -------------------------------------------------------------

Bootstrap.CV.cvp_fixed = ...
    cvp_fixed;


Bootstrap.CV.foldID = ...
    foldID;


Bootstrap.CV.k_used = ...
    k_used;


Bootstrap.CV.nFitFold = ...
    nFitFold;


Bootstrap.CV.nValFold = ...
    nValFold;


Bootstrap.CV.d_original_fold = ...
    dOriginalFold;


Bootstrap.CV.epsilon_original_fold = ...
    epsilonOriginalFold;


% -------------------------------------------------------------
% Bootstrap numerical diagnostics
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
    'fold_by_bootstrap';


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
% Computation/rejections
% -------------------------------------------------------------

Bootstrap.Computation.nAttempts = ...
    nAttempts;


Bootstrap.Computation.nRejected = ...
    nRejected;


Bootstrap.Computation.nRejectedAdmissibility = ...
    nRejectedAdmissibility;


Bootstrap.Computation.nRejectedSolve = ...
    nRejectedSolve;


Bootstrap.Computation.nRejectedPrediction = ...
    nRejectedPrediction;


Bootstrap.Computation.nRBFSolves = ...
    nRBFSolves;



Bootstrap.Computation.totalAttempts = ...
    sum(nAttempts);


Bootstrap.Computation.totalRejected = ...
    sum(nRejected);


Bootstrap.Computation.totalRBFSolves = ...
    sum(nRBFSolves);


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


Bootstrap.Settings.FoldRNG_SEED = ...
    FoldRNG_SEED;


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
        'original_fold_shared';

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


function [y_pred_test, test_diff, test_rmse, TestingResult] = ...
    LocalRBFTesting( ...
        TrainingResult, ...
        Xtest, ytest, ...
        near_threshold, idw_radius, ...
        options, geometryCache)
%LOCALRBFTESTING
%
% Independent testing / prediction for a trained local-RBF model.
%
% The training-derived coordinate scale is reused:
%
%       Xtest_scaled = Xtest / TrainingResult.S_global
%


% Transfer strategy is controlled by:
%
%       options.transfer.method
%
%   'hybrid'  : nearest when nearest distance <= near_threshold;
%               otherwise N-nearest IDW
%   'idw'     : always N-nearest IDW
%   'nearest' : always nearest
%   'ensemble': nearest when nearest distance <= near_threshold;
%               otherwise N-nearest COMPLETE donor tuples are solved
%               independently and their predictions are combined by the
%               transfer weights
%
% where:
%
%       N = options.transfer.nIDWSources
%
% Backward-compatible defaults:
%
%       method                = 'hybrid'
%       nIDWSources           = 7
%       parameterTransferMode = 'log_log'
%
% parameterTransferMode is used by IDW/hybrid and is inactive for
% nearest/ensemble.
%
% idw_radius is retained in the public interface/metadata for
% backward compatibility and is not used for source selection.

% The following settings are transferred:
%
%       radius
%       RegParam
%
% and, for shaped kernels:
%
%       C
%
% For shaped kernels, epsilon is NOT transferred. It is
% recomputed from the actual query neighbourhood:
%
%       epsilon_q = C_q * d_q
%
% For shape-free kernels:
%
%       C_q       = NaN
%       d_q       = NaN
%       epsilon_q = 1
%
% Residual convention:
%
%       test_diff = observed - predicted
%
% Diagnostic-output responsibility:
%
%   - TRAINING diagnostics are numerical diagnostics calculated from the
%     final selected local-RBF system at each training point after the final
%     radius cap is chosen;
%   - this function exports TEST/query-system diagnostics only;
%   - common.calculateConditionDiagnostics may internally return
%     cond_evaluation_train, but LocalRBFTesting does not expose that field.
%
% IMPORTANT:
%   - no optimization is performed here;
%   - no RNG is used here;
%   - training-derived S_global is reused exactly;
%   - query-to-training distances are calculated ONCE;
%   - training-to-training geometry is reused when geometryCache
%     is supplied, otherwise reconstructed ONCE;
%   - nearest/IDW/hybrid retain the historical transferred-setting path;
%   - ensemble does NOT average radius, C, or RegParam; it solves each
%     unique complete donor tuple once and combines tuple predictions;
%   - neighbourhood construction uses the transferred radius for the
%     historical scalar path and each tuple radius for ensemble;
%   - minimum-neighbour fallback is handled by getNeighborhood;
%   - full diagnostics required by the selected test system are calculated
%     once per test point;
%   - no large distance/system matrices are stored;
%   - options.execution.saveStage2All=false retains/constructs only the
%     compulsory Stage-2 testing outputs required by CORE/NUMERICAL/RUNTIME;
%   - when saveStage2All=false and transfer mode is ensemble,
%     options.execution.saveEnsemblePointDiagnostics selects:
%         false : retain only the compulsory point-level conservative maxima;
%         true  : additionally retain lightweight per-unique-tuple numerical
%                 diagnostics for each far-ensemble test point in CORE;
%   - options.execution.saveStage2All=true additionally constructs the
%     complete retained transfer/neighbourhood/coefficient/diagnostic DETAIL;
%     the lightweight CORE-only copy is suppressed in this mode;
%   - optional containers are never constructed and discarded later.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 3

    ytest = ...
        [];

end


if nargin < 6 || ...
   isempty(options)

    options = ...
        struct();

end


if nargin < 7

    geometryCache = ...
        [];

end


if ~isstruct(options)

    error( ...
        'options must be a structure.');

end


if ~isfield(options,'solver') || ...
   ~isstruct(options.solver)

    options.solver = ...
        struct();

end


if ~isfield(options.solver,'cond_lsq') || ...
   isempty(options.solver.cond_lsq)

    options.solver.cond_lsq = ...
        1e12;

end


% =============================================================
% 1.1 Transfer options
% =============================================================

if isfield(options,'transfer') && ...
   ~isempty(options.transfer) && ...
   ~isstruct(options.transfer)

    error( ...
        'options.transfer must be a structure.');

end


% Backward compatibility only
transferMode = ...
    'hybrid';

nIDWSources = ...
    7;


parameterTransferMode = ...
    'log_log';


if isfield(options,'transfer') && ...
   isstruct(options.transfer)

    if isfield(options.transfer,'method') && ...
       ~isempty(options.transfer.method)

        transferMode = ...
            options.transfer.method;

    end


    if isfield(options.transfer,'nIDWSources') && ...
       ~isempty(options.transfer.nIDWSources)

        nIDWSources = ...
            options.transfer.nIDWSources;

    end


    if isfield(options.transfer,'parameterTransferMode') && ...
       ~isempty(options.transfer.parameterTransferMode)

        parameterTransferMode = ...
            options.transfer.parameterTransferMode;

    end

end


if ~(ischar(transferMode) || ...
     (isstring(transferMode) && ...
      isscalar(transferMode)))

    error( ...
        ['options.transfer.method must be ''hybrid'', ''idw'', ' ...
         '''nearest'', or ''ensemble''.']);

end


transferMode = ...
    lower( ...
        strtrim( ...
            char(transferMode)));


if ~any(strcmp( ...
        transferMode, ...
        {'hybrid','idw','nearest','ensemble'}))

    error( ...
        ['options.transfer.method must be ''hybrid'', ''idw'', ' ...
         '''nearest'', or ''ensemble''.']);

end


if ~isnumeric(nIDWSources) || ...
   ~isscalar(nIDWSources) || ...
   ~isreal(nIDWSources) || ...
   ~isfinite(nIDWSources) || ...
   nIDWSources < 1 || ...
   nIDWSources ~= round(nIDWSources)

    error( ...
        ['options.transfer.nIDWSources must be ' ...
         'a positive integer.']);

end


nIDWSources = ...
    double(nIDWSources);

% Selected test systems require relative residual diagnostics.
options.solver.compute_rel_residual = ...
    true;

% =============================================================
% 2. Training-result checks
% =============================================================

if ~isstruct(TrainingResult)

    error( ...
        'TrainingResult must be a structure.');

end


requiredTrainingFields = { ...
    'success', ...
    'kernel', ...
    'useShape', ...
    'X', ...
    'X_raw', ...
    'y', ...
    'S_global', ...
    'minNeighbors', ...
    'Final'};


for q = 1:numel(requiredTrainingFields)

    if ~isfield( ...
            TrainingResult, ...
            requiredTrainingFields{q})

        error( ...
            'TrainingResult is missing required field "%s".', ...
            requiredTrainingFields{q});

    end

end


if ~TrainingResult.success || ...
   ~isstruct(TrainingResult.Final) || ...
   ~isfield(TrainingResult.Final,'success') || ...
   ~TrainingResult.Final.success

    error( ...
        'TrainingResult does not contain a successful final model.');

end


% Stage-2 storage policy.  The explicit Testing options take precedence.
% When the batch caller passes numerical-only options, fall back to the
% training record so Stage-2 DETAIL follows the same run-level request.
saveStage2All = ...
    getSaveStage2All( ...
        options, ...
        TrainingResult);


saveEnsemblePointDiagnosticsRequested = ...
    getSaveEnsemblePointDiagnostics( ...
        options, ...
        TrainingResult);


saveEnsemblePointDiagnostics = ...
    ~saveStage2All && ...
    saveEnsemblePointDiagnosticsRequested;


% Execution/storage controls must never reach the numerical solver.
numericalOptions = ...
    canonicalNumericalOptions( ...
        options);


% =============================================================
% 3. Training information
% =============================================================

Xtrain = ...
    TrainingResult.X;


XtrainRaw = ...
    TrainingResult.X_raw;


ytrain = ...
    TrainingResult.y(:);


[nTrain,p] = ...
    size(Xtrain);


if size(XtrainRaw,1) ~= nTrain || ...
   size(XtrainRaw,2) ~= p

    error( ...
        'TrainingResult.X and X_raw are inconsistent.');

end


if numel(ytrain) ~= nTrain

    error( ...
        'TrainingResult.y is inconsistent with training coordinates.');

end


S_global = ...
    TrainingResult.S_global;

if ~isnumeric(S_global) || ...
   ~isscalar(S_global) || ...
   ~isreal(S_global) || ...
   ~isfinite(S_global) || ...
   S_global <= 0

    error( ...
        'TrainingResult.S_global must be finite and positive.');

end


kernel = ...
    TrainingResult.kernel;


useShape = ...
    TrainingResult.useShape;


min_neighbors = ...
    TrainingResult.minNeighbors;



% =============================================================
% 3.1 Verify training coordinate-scale consistency
% =============================================================

XtrainExpected = ...
    XtrainRaw ./ ...
    S_global;


scaleTol = ...
    1e-12 .* ...
    max( ...
        [1; abs(XtrainExpected(:))]);


if any( ...
        abs( ...
            Xtrain(:) - ...
            XtrainExpected(:)) > ...
        scaleTol)

    error( ...
        ['TrainingResult.X is inconsistent with ' ...
         'TrainingResult.X_raw / S_global.']);

end


% =============================================================
% 3.2 Verify final-radius unit consistency
% =============================================================

F = ...
    TrainingResult.Final;


if isfield(F,'radius') && ...
   isfield(F,'radius_scaled')

    if ~isequal( ...
            size(F.radius), ...
            size(F.radius_scaled))

        error( ...
            ['TrainingResult.Final.radius and ' ...
             'radius_scaled have inconsistent sizes.']);

    end


    radiusScale = ...
        max( ...
            [1; ...
             abs(F.radius(:)); ...
             abs(F.radius_scaled(:))]);


    tolRadius = ...
        1e-12 .* ...
        radiusScale;


    if any( ...
            abs( ...
                F.radius(:) - ...
                F.radius_scaled(:)) > ...
            tolRadius)

        error( ...
            ['TrainingResult.Final.radius is inconsistent ' ...
             'with radius_scaled.']);

    end

end


if isfield(F,'radius') && ...
   isfield(F,'radius_raw')

    expectedRadiusRaw = ...
        F.radius(:) .* ...
        S_global;


    if ~isequal( ...
            size(F.radius(:)), ...
            size(F.radius_raw(:)))

        error( ...
            ['TrainingResult.Final.radius and radius_raw ' ...
             'have inconsistent sizes.']);

    end


    radiusRawScale = ...
        max( ...
            [1; ...
             abs(expectedRadiusRaw(:)); ...
             abs(F.radius_raw(:))]);


    tolRadiusRaw = ...
        1e-12 .* ...
        radiusRawScale;


    if any( ...
            abs( ...
                F.radius_raw(:) - ...
                expectedRadiusRaw(:)) > ...
            tolRadiusRaw)

        error( ...
            ['TrainingResult.Final.radius_raw is inconsistent ' ...
             'with radius * S_global.']);

    end

end

% =============================================================
% 4. Test-coordinate checks
% =============================================================

if ~isnumeric(Xtest) || ...
   isempty(Xtest) || ...
   ~ismatrix(Xtest) || ...
   ~isreal(Xtest) || ...
   any(~isfinite(Xtest(:)))

    error( ...
        'Xtest must be a nonempty finite real numeric matrix.');

end


[nTest,pTest] = ...
    size(Xtest);


if pTest ~= p

    error( ...
        'Training and testing coordinate dimensions differ.');

end


% =============================================================
% 5. Optional test observations
% =============================================================

hasObservations = ...
    ~isempty(ytest);


if hasObservations

    if ~isnumeric(ytest) || ...
       ~isvector(ytest) || ...
       ~isreal(ytest) || ...
       numel(ytest) ~= nTest || ...
       any(~isfinite(ytest(:)))

        error( ...
            ['ytest must be empty or a finite real vector ' ...
             'with one value per test point.']);

    end


    ytest = ...
        ytest(:);


else

    ytest = ...
        nan(nTest,1);

end


% =============================================================
% 6. Transfer-distance checks
%
% Public inputs use ORIGINAL coordinate units.
% =============================================================

if ~isnumeric(near_threshold) || ...
   ~isscalar(near_threshold) || ...
   ~isreal(near_threshold) || ...
   ~isfinite(near_threshold) || ...
   near_threshold <= 0

    error( ...
        'near_threshold must be a finite positive real scalar.');

end


if ~isnumeric(idw_radius) || ...
   ~isscalar(idw_radius) || ...
   ~isreal(idw_radius) || ...
   ~isfinite(idw_radius) || ...
   idw_radius <= 0

    error( ...
        'idw_radius must be a finite positive real scalar.');

end


% =============================================================
% 7. Apply TRAINING-derived coordinate scale
% =============================================================

XtestScaled = ...
    Xtest ./ ...
    S_global;


nearThresholdScaled = ...
    near_threshold ./ ...
    S_global;


idwRadiusScaled = ...
    idw_radius ./ ...
    S_global;

if any(~isfinite(XtestScaled(:))) || ...
   ~isfinite(nearThresholdScaled) || ...
   nearThresholdScaled <= 0 || ...
   ~isfinite(idwRadiusScaled) || ...
   idwRadiusScaled <= 0

    error( ...
        'Test-coordinate scaling produced invalid values.');

end


% =============================================================
% 8. Geometry
%
% D_query_train is calculated ONCE.
%
% Training geometry is reused when available.
% =============================================================

tGeometry = ...
    tic;


D_query_train = ...
    pdist2( ...
        XtestScaled, ...
        Xtrain);


P_query = ...
    [ones(nTest,1), ...
     XtestScaled];


geometryReused = ...
    ~isempty(geometryCache);


if geometryReused

    if ~isstruct(geometryCache) || ...
       ~isfield(geometryCache,'D') || ...
       ~isfield(geometryCache,'P')

        error( ...
            'geometryCache must contain fields D and P.');

    end


    if ~isnumeric(geometryCache.D) || ...
       ~isequal( ...
           size(geometryCache.D), ...
           [nTrain nTrain])

        error( ...
            'geometryCache.D has an invalid size.');

    end


    if ~isnumeric(geometryCache.P) || ...
       ~isequal( ...
           size(geometryCache.P), ...
           [nTrain p+1])

        error( ...
            'geometryCache.P has an invalid size.');

    end


    D_train_train = ...
        geometryCache.D;


    P_train = ...
        geometryCache.P;


else

    D_train_train = ...
        pdist2( ...
            Xtrain, ...
            Xtrain);


    P_train = ...
        [ones(nTrain,1), ...
         Xtrain];

end


geometryWallTime = ...
    toc(tGeometry);


% =============================================================
% 9. Transfer final trained settings to test points
% =============================================================

tTransfer = ...
    tic;


Transfer = ...
    localrbf.transferSettingsFromTrainToTest( ...
        Xtrain, ...
        XtestScaled, ...
        TrainingResult.Final, ...
        kernel, ...
        nearThresholdScaled, ...
        idwRadiusScaled, ...
        D_query_train, ...
        saveStage2All, ...
        transferMode, ...
        nIDWSources, ...
        parameterTransferMode);


transferWallTime = ...
    toc(tTransfer);


% Extract only the transfer quantities required by the compulsory path.
% These are small canonical vectors/cells.  When DETAIL is disabled the
% complete Transfer structure is released before point evaluation.
transferQuerySuccess = Transfer.querySuccess(:);
transferMethod = Transfer.transferMethod;
transferMessageByQuery = Transfer.messageByQuery;

radius_scaled = Transfer.radius(:);
C_transferred = Transfer.C(:);
Reg_transferred = Transfer.reg_param(:);

nNearestTransfers = Transfer.nNearestTransfers;
nIDWTransfers = Transfer.nIDWTransfers;

if isfield(Transfer,'nEnsembleTransfers')

    nEnsembleTransfers = ...
        Transfer.nEnsembleTransfers;

else

    nEnsembleTransfers = ...
        0;

end

nTransferFailures = Transfer.nFailedQueries;


% Canonical transfer-mode metadata after transfer-function normalization.
parameterTransferModeUsed = ...
    Transfer.parameterTransferMode;


% Ensemble tuple state is compulsory only when the requested transfer mode
% is ensemble.  Near-threshold ensemble queries are labelled 'nearest' and
% continue through the historical scalar path.  Only far ensemble queries
% use these tuple cells.
isEnsembleMode = ...
    strcmp(transferMode,'ensemble');


if isEnsembleMode

    ensembleUniqueWeight = ...
        Transfer.ensembleUniqueWeight;

    ensembleUniqueRadius = ...
        Transfer.ensembleUniqueRadius;

    ensembleUniqueRegParam = ...
        Transfer.ensembleUniqueRegParam;

    ensembleUniqueC = ...
        Transfer.ensembleUniqueC;

    ensembleNUniqueTuples = ...
        Transfer.ensembleNUniqueTuples(:);

else

    ensembleUniqueWeight = ...
        cell(nTest,1);

    ensembleUniqueRadius = ...
        cell(nTest,1);

    ensembleUniqueRegParam = ...
        cell(nTest,1);

    ensembleUniqueC = ...
        cell(nTest,1);

    ensembleNUniqueTuples = ...
        zeros(nTest,1);

end


% =============================================================
% 10. Raw/scaled transfer aliases required by compulsory output
%
% Only radius_raw is compulsory.  The large source-distance/source-radius
% alias cells are DETAIL-only and therefore are constructed only when
% saveStage2All=true.
% =============================================================

radius_raw = radius_scaled .* S_global;

if saveStage2All

    Transfer.near_threshold_scaled = nearThresholdScaled;
    Transfer.near_threshold_raw = near_threshold;

    Transfer.idw_radius_scaled = idwRadiusScaled;
    Transfer.idw_radius_raw = idw_radius;

    Transfer.radius_scaled = Transfer.radius;
    Transfer.radius_raw = radius_raw;

    Transfer.nearestDistance_scaled = Transfer.nearestDistance;
    Transfer.nearestDistance_raw = Transfer.nearestDistance .* S_global;

    Transfer.sourceDistance_scaled = Transfer.sourceDistance;
    Transfer.sourceDistance_raw = ...
        scaleCellVectors(Transfer.sourceDistance,S_global);

    Transfer.sourceRadius_scaled = Transfer.sourceRadius;
    Transfer.sourceRadius_raw = ...
        scaleCellVectors(Transfer.sourceRadius,S_global);

else

    % The detailed source indices/distances/weights are not required by
    % Stage-2 compulsory output.  Release them before the test-point loop.
    clear Transfer

end


% =============================================================
% 11. Compulsory outputs
% =============================================================

y_pred_test = nan(nTest,1);
test_diff = nan(nTest,1);
test_rmse = NaN;

TestingResult = struct();

TestingResult.success = false;
TestingResult.predictionSuccess = false;
TestingResult.metricSuccess = false;
TestingResult.message = '';

TestingResult.kernel = kernel;
TestingResult.useShape = useShape;
TestingResult.nTrain = nTrain;
TestingResult.nTest = nTest;
TestingResult.nDimensions = p;
TestingResult.hasObservations = hasObservations;
TestingResult.S_global = S_global;
TestingResult.transferMode = ...
    transferMode;

TestingResult.nIDWSources = ...
    nIDWSources;


TestingResult.parameterTransferMode = ...
    parameterTransferModeUsed;


TestingResult.ensembleConditionAggregation = ...
    'independent-max-across-unique-tuples';


TestingResult.saveEnsemblePointDiagnosticsRequested = ...
    saveEnsemblePointDiagnosticsRequested;


TestingResult.saveEnsemblePointDiagnostics = ...
    saveEnsemblePointDiagnostics;


TestingResult.geometryWallTime = geometryWallTime;
TestingResult.transferWallTime = transferWallTime;

% Stage-2 CORE parameter fields.
TestingResult.transferMethod = transferMethod;
TestingResult.radius_raw = radius_raw;
TestingResult.C = C_transferred;
TestingResult.reg_param = Reg_transferred;
TestingResult.neighbor_count = nan(nTest,1);

% Stage-2 NUMERICAL fields.
TestingResult.cond_raw = nan(nTest,1);
TestingResult.cond_reg = nan(nTest,1);
TestingResult.cond_effective = nan(nTest,1);
TestingResult.cond_evaluation_test = nan(nTest,1);

% Numerical reliability is compulsory in BOTH compact and DETAIL modes.
% Scalar systems retain their own values. Far-ensemble systems retain the
% independent maximum across unique tuple solves.
TestingResult.rel_residual = nan(nTest,1);
TestingResult.backward_error = nan(nTest,1);

TestingResult.solver_used = cell(nTest,1);
TestingResult.point_success = false(nTest,1);

% Stage-2 batch counters.
TestingResult.nRBFSolves = 0;
TestingResult.nNearestTransfers = nNearestTransfers;
TestingResult.nIDWTransfers = nIDWTransfers;
TestingResult.nEnsembleTransfers = nEnsembleTransfers;
TestingResult.nTransferFailures = nTransferFailures;

% Number of unique complete tuples solved for each far-ensemble query.
% It is zero for the historical scalar path, including near-threshold
% queries when transferMode='ensemble'.
TestingResult.ensemble_n_unique_tuples = ...
    ensembleNUniqueTuples;


% Compact ensemble tuple settings are compulsory for reproducibility and
% fixed-setting reuse.  They are retained only for ensemble mode and only
% as the small unique-tuple vectors already required during prediction.
if isEnsembleMode

    TestingResult.ensembleUniqueWeight = ...
        ensembleUniqueWeight;

    TestingResult.ensembleUniqueRadiusRaw = ...
        scaleCellVectors( ...
            ensembleUniqueRadius, ...
            S_global);

    TestingResult.ensembleUniqueRegParam = ...
        ensembleUniqueRegParam;

    TestingResult.ensembleUniqueC = ...
        ensembleUniqueC;

end


TestingResult.ensemble_cond_raw_max_tuple_index = ...
    nan(nTest,1);

TestingResult.ensemble_cond_reg_max_tuple_index = ...
    nan(nTest,1);

TestingResult.ensemble_cond_effective_max_tuple_index = ...
    nan(nTest,1);

TestingResult.ensemble_rel_residual_max_tuple_index = ...
    nan(nTest,1);

TestingResult.ensemble_backward_error_max_tuple_index = ...
    nan(nTest,1);


if saveEnsemblePointDiagnostics && ...
   isEnsembleMode

    TestingResult.ensemblePointDiagnostics = ...
        cell(nTest,1);

end


% =============================================================
% 12. DETAIL-only output allocation
%
% HARD RULE:
%   Nothing in this section exists when saveStage2All=false.
% =============================================================

if saveStage2All

    TestingResult.Xtest_raw = Xtest;
    TestingResult.Xtest = XtestScaled;
    TestingResult.Xtest_scaled = XtestScaled;
    TestingResult.ytest = ytest;

    TestingResult.near_threshold_raw = near_threshold;
    TestingResult.near_threshold_scaled = nearThresholdScaled;
    TestingResult.idw_radius_raw = idw_radius;
    TestingResult.idw_radius_scaled = idwRadiusScaled;
    TestingResult.minNeighbors = min_neighbors;

    TestingResult.Transfer = Transfer;

    % Scaled-radius aliases are DETAIL-only duplicates of the compulsory
    % original-unit radius vector.
    TestingResult.radius = radius_scaled;
    TestingResult.radius_scaled = radius_scaled;

    TestingResult.geometryReused = geometryReused;

    TestingResult.distanceMatrixInfo = struct();
    TestingResult.distanceMatrixInfo.queryTrainSize = size(D_query_train);
    TestingResult.distanceMatrixInfo.trainTrainSize = size(D_train_train);
    TestingResult.distanceMatrixInfo.stored = false;

    TestingResult.Point = ...
        repmat(emptyTestPoint(),nTest,1);

    TestingResult.pointMessage = cell(nTest,1);
    TestingResult.pointWallTime = zeros(nTest,1);

    TestingResult.d_local = nan(nTest,1);
    TestingResult.epsilon = nan(nTest,1);

    TestingResult.neighbor_idx = cell(nTest,1);
    TestingResult.neighbor_distances = cell(nTest,1);
    TestingResult.requestedRadius = nan(nTest,1);
    TestingResult.effectiveRadius = nan(nTest,1);
    TestingResult.selectedMaxDistance = nan(nTest,1);
    TestingResult.nWithinRadius = nan(nTest,1);
    TestingResult.nAddedForMinimum = nan(nTest,1);
    TestingResult.usedMinNeighborFallback = false(nTest,1);
    TestingResult.effectiveRadius_raw = nan(nTest,1);
    TestingResult.selectedMaxDistance_raw = nan(nTest,1);

    TestingResult.cond_ratio = nan(nTest,1);
    TestingResult.delta_log10_cond = nan(nTest,1);
    TestingResult.rank_eff = nan(nTest,1);
    TestingResult.rank_tol = nan(nTest,1);
    TestingResult.sigma_min_retained = nan(nTest,1);
    TestingResult.coeffs = cell(nTest,1);
    TestingResult.lambda = cell(nTest,1);
    TestingResult.beta = cell(nTest,1);

    conditionDiag = ...
        repmat(emptyConditionDiagnostic(),nTest,1);

else

    % Explicitly avoid constructing all DETAIL containers.
    conditionDiag = [];

end


% =============================================================
% 13. Internal computational counters
%
% Scalars are working state.  DETAIL-only counters are published only
% when saveStage2All=true.
% =============================================================

nSuccessfulRBFSolves = 0;
nFailedRBFSolves = 0;
nPolynomialFailures = 0;
nNeighborhoodFailures = 0;
nPredictionFailures = 0;


% =============================================================
% 14. Evaluate every test point
% =============================================================

tTesting = tic;

for iTest = 1:nTest

    tPoint = tic;

    % ---------------------------------------------------------
    % DETAIL point record is not even initialized unless requested.
    % ---------------------------------------------------------

    if saveStage2All

        Pi = emptyTestPoint();

        Pi.iTest = iTest;
        Pi.xTest_raw = Xtest(iTest,:);
        Pi.xTest = XtestScaled(iTest,:);
        Pi.xTest_scaled = XtestScaled(iTest,:);
        Pi.hasObservation = hasObservations;
        Pi.observation = ytest(iTest);

        Pi.transferSuccess = transferQuerySuccess(iTest);
        Pi.transferMethod = transferMethod{iTest};

        Pi.nearestTrainingIndex = ...
            Transfer.nearestTrainingIndex(iTest);

        Pi.nearestDistance = ...
            Transfer.nearestDistance(iTest);

        Pi.nearestDistance_raw = ...
            Transfer.nearestDistance_raw(iTest);

        Pi.sourceTrainingIndex = ...
            Transfer.sourceTrainingIndex{iTest};

        Pi.sourceDistance = ...
            Transfer.sourceDistance{iTest};

        Pi.sourceDistance_raw = ...
            Transfer.sourceDistance_raw{iTest};

        Pi.sourceWeight = ...
            Transfer.sourceWeight{iTest};

        Pi.nTransferSources = ...
            Transfer.nTransferSources(iTest);

        Pi.radius = radius_scaled(iTest);
        Pi.radius_scaled = radius_scaled(iTest);
        Pi.radius_raw = radius_raw(iTest);
        Pi.C = C_transferred(iTest);
        Pi.reg_param = Reg_transferred(iTest);


        if strcmp(transferMethod{iTest},'ensemble')

            Pi.ensembleNUniqueTuples = ...
                ensembleNUniqueTuples(iTest);

        end

    end


    % =========================================================
    % 14.1 Transfer information
    % =========================================================

    transferSuccess = ...
        transferQuerySuccess(iTest);

    radius_i = ...
        radius_scaled(iTest);

    C_i = ...
        C_transferred(iTest);

    reg_i = ...
        Reg_transferred(iTest);


    if ~transferSuccess

        pointMessage = ...
            transferMessageByQuery{iTest};

        if saveStage2All

            Pi.failureStage = 'setting transfer';
            Pi.message = pointMessage;
            Pi.totalWallTime = toc(tPoint);

            TestingResult.Point(iTest) = Pi;
            TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
            TestingResult.pointMessage{iTest} = Pi.message;

        end

        continue

    end


    % =========================================================
    % 14.1A Ensemble prediction branch
    %
    % IMPORTANT:
    %   - only FAR ensemble queries enter here;
    %   - near-threshold ensemble queries have transferMethod='nearest'
    %     and continue through the historical scalar path unchanged;
    %   - complete tuples are solved independently;
    %   - exact duplicate tuples were already merged by transfer;
    %   - failed tuples are NOT dropped and weights are NOT renormalized;
    %   - repeated tuple radii reuse the same neighbourhood/geometry.
    % =========================================================

    if strcmp(transferMethod{iTest},'ensemble')

        ER = ...
            evaluateEnsembleTestPointLocal( ...
                iTest, ...
                Xtrain, ...
                ytrain, ...
                XtestScaled, ...
                kernel, ...
                useShape, ...
                min_neighbors, ...
                D_query_train, ...
                D_train_train, ...
                P_train, ...
                P_query, ...
                numericalOptions, ...
                ensembleUniqueWeight{iTest}, ...
                ensembleUniqueRadius{iTest}, ...
                ensembleUniqueRegParam{iTest}, ...
                ensembleUniqueC{iTest}, ...
                S_global, ...
                saveStage2All, ...
                saveEnsemblePointDiagnostics);


        TestingResult.nRBFSolves = ...
            TestingResult.nRBFSolves + ...
            ER.nRBFSolves;


        nSuccessfulRBFSolves = ...
            nSuccessfulRBFSolves + ...
            ER.nSuccessfulRBFSolves;


        nFailedRBFSolves = ...
            nFailedRBFSolves + ...
            ER.nFailedRBFSolves;


        nPolynomialFailures = ...
            nPolynomialFailures + ...
            ER.nPolynomialFailures;


        nNeighborhoodFailures = ...
            nNeighborhoodFailures + ...
            ER.nNeighborhoodFailures;


        nPredictionFailures = ...
            nPredictionFailures + ...
            ER.nPredictionFailures;


        if ~ER.success

            if saveEnsemblePointDiagnostics

                TestingResult.ensemblePointDiagnostics{iTest} = ...
                    ER.TupleDiagnostic;

            end


            if saveStage2All

                Pi.failureStage = ...
                    ER.failureStage;

                Pi.message = ...
                    ER.message;

                Pi.ensemble = ...
                    ER.TupleDetail;

                Pi.ensembleCondRawMaxTupleIndex = ...
                    ER.condRawMaxTupleIndex;

                Pi.ensembleCondRegMaxTupleIndex = ...
                    ER.condRegMaxTupleIndex;

                Pi.ensembleCondEffectiveMaxTupleIndex = ...
                    ER.condEffectiveMaxTupleIndex;

                Pi.ensembleRelResidualMaxTupleIndex = ...
                    ER.relResidualMaxTupleIndex;

                Pi.ensembleBackwardErrorMaxTupleIndex = ...
                    ER.backwardErrorMaxTupleIndex;

                Pi.totalWallTime = ...
                    toc(tPoint);


                TestingResult.Point(iTest) = ...
                    Pi;

                TestingResult.pointWallTime(iTest) = ...
                    Pi.totalWallTime;

                TestingResult.pointMessage{iTest} = ...
                    Pi.message;

            end

            continue

        end


        yhat = ...
            ER.yhat;


        y_diff_i = ...
            NaN;


        if hasObservations

            y_diff_i = ...
                ytest(iTest) - ...
                yhat;

        end


        pointElapsed = ...
            toc(tPoint);


        % -----------------------------------------------------
        % Compulsory ensemble output
        % -----------------------------------------------------

        TestingResult.point_success(iTest) = ...
            true;


        y_pred_test(iTest) = ...
            yhat;


        if hasObservations

            test_diff(iTest) = ...
                y_diff_i;

        end


        TestingResult.neighbor_count(iTest) = ...
            ER.maxNeighborCount;


        % For an ensemble there is no single RBF matrix.  Each canonical
        % condition field is therefore the WORST value across the unique
        % constituent tuple systems, calculated independently by metric.
        TestingResult.cond_raw(iTest) = ...
            ER.maxCondition.cond_raw;


        TestingResult.cond_reg(iTest) = ...
            ER.maxCondition.cond_reg;


        TestingResult.cond_effective(iTest) = ...
            ER.maxCondition.cond_effective;


        TestingResult.rel_residual(iTest) = ...
            ER.maxCondition.rel_residual;


        TestingResult.backward_error(iTest) = ...
            ER.maxCondition.backward_error;


        % There is no single solver for an ensemble prediction.
        TestingResult.solver_used{iTest} = ...
            'ensemble';


        TestingResult.ensemble_cond_raw_max_tuple_index(iTest) = ...
            ER.condRawMaxTupleIndex;


        TestingResult.ensemble_cond_reg_max_tuple_index(iTest) = ...
            ER.condRegMaxTupleIndex;


        TestingResult.ensemble_cond_effective_max_tuple_index(iTest) = ...
            ER.condEffectiveMaxTupleIndex;


        TestingResult.ensemble_rel_residual_max_tuple_index(iTest) = ...
            ER.relResidualMaxTupleIndex;


        TestingResult.ensemble_backward_error_max_tuple_index(iTest) = ...
            ER.backwardErrorMaxTupleIndex;


        if saveEnsemblePointDiagnostics

            TestingResult.ensemblePointDiagnostics{iTest} = ...
                ER.TupleDiagnostic;

        end


        % -----------------------------------------------------
        % DETAIL ensemble output
        % -----------------------------------------------------

        if saveStage2All

            Pi.ensemble = ...
                ER.TupleDetail;


            Pi.ensembleCondRawMaxTupleIndex = ...
                ER.condRawMaxTupleIndex;


            Pi.ensembleCondRegMaxTupleIndex = ...
                ER.condRegMaxTupleIndex;


            Pi.ensembleCondEffectiveMaxTupleIndex = ...
                ER.condEffectiveMaxTupleIndex;


            Pi.ensembleRelResidualMaxTupleIndex = ...
                ER.relResidualMaxTupleIndex;


            Pi.ensembleBackwardErrorMaxTupleIndex = ...
                ER.backwardErrorMaxTupleIndex;


            Pi.neighbor_count = ...
                ER.maxNeighborCount;


            Pi.cond_raw = ...
                ER.maxCondition.cond_raw;


            Pi.cond_reg = ...
                ER.maxCondition.cond_reg;


            Pi.cond_effective = ...
                ER.maxCondition.cond_effective;


            % No single ensemble value is defined for the remaining
            % tuple-specific diagnostics.  They remain fully available in
            % Pi.ensemble(j).condition when DETAIL is retained.
            Pi.cond_ratio = ...
                NaN;


            Pi.delta_log10_cond = ...
                NaN;


            Pi.rank_eff = ...
                NaN;


            Pi.rank_tol = ...
                NaN;


            Pi.sigma_min_retained = ...
                NaN;


            Pi.cond_evaluation_test = ...
                NaN;


            Pi.rel_residual = ...
                ER.maxCondition.rel_residual;


            Pi.backward_error = ...
                ER.maxCondition.backward_error;


            Pi.solver_used = ...
                'ensemble';


            Pi.yhat = ...
                yhat;


            if hasObservations

                Pi.y_diff = ...
                    y_diff_i;

                Pi.residual = ...
                    y_diff_i;

                Pi.AE = ...
                    abs(y_diff_i);

                Pi.squaredResidual = ...
                    y_diff_i.^2;

            end


            Pi.success = ...
                true;


            Pi.failureStage = ...
                '';


            Pi.message = ...
                '';


            Pi.totalWallTime = ...
                pointElapsed;


            TestingResult.Point(iTest) = ...
                Pi;


            TestingResult.pointWallTime(iTest) = ...
                pointElapsed;


            TestingResult.pointMessage{iTest} = ...
                '';


            % Ensemble condition numbers are conservative independent
            % maxima across the constituent tuple systems.  Other
            % diagnostics have no unique point-level ensemble definition
            % and therefore remain NaN in this summary record.
            conditionDiag(iTest).cond_raw = ...
                ER.maxCondition.cond_raw;

            conditionDiag(iTest).cond_reg = ...
                ER.maxCondition.cond_reg;

            conditionDiag(iTest).cond_effective = ...
                ER.maxCondition.cond_effective;

            conditionDiag(iTest).cond_ratio = ...
                NaN;

            conditionDiag(iTest).delta_log10_cond = ...
                NaN;

            conditionDiag(iTest).rank_eff = ...
                NaN;

            conditionDiag(iTest).rank_tol = ...
                NaN;

            conditionDiag(iTest).sigma_min_retained = ...
                NaN;

            % INTERNAL ONLY generic-summary compatibility field.
            conditionDiag(iTest).cond_evaluation_train = ...
                NaN;

            conditionDiag(iTest).cond_evaluation_test = ...
                NaN;

            conditionDiag(iTest).rel_residual = ...
                ER.maxCondition.rel_residual;

            conditionDiag(iTest).backward_error = ...
                ER.maxCondition.backward_error;

            conditionDiag(iTest).reg_param = ...
                NaN;

            conditionDiag(iTest).solver_used = ...
                'ensemble';

            conditionDiag(iTest).success = ...
                true;

        end


        continue

    end


    % =========================================================
    % 14.2 Query neighbourhood
    %
    % No exclusion for an independent test/query point.
    % =========================================================

    distAll = ...
        D_query_train(iTest,:).';


    try

        Nbh = ...
            localrbf.getNeighborhood( ...
                Xtrain, ...
                XtestScaled(iTest,:), ...
                radius_i, ...
                min_neighbors, ...
                [], ...
                distAll);

        neighborhoodMessage = '';

    catch ME

        Nbh = [];
        neighborhoodMessage = ME.message;

    end


    if isempty(Nbh) || ...
       ~isfield(Nbh,'success') || ...
       ~Nbh.success

        if isempty(neighborhoodMessage)

            if ~isempty(Nbh) && ...
               isfield(Nbh,'message')

                neighborhoodMessage = ...
                    Nbh.message;

            else

                neighborhoodMessage = ...
                    'Query neighbourhood construction failed.';

            end

        end


        nNeighborhoodFailures = ...
            nNeighborhoodFailures + 1;


        if saveStage2All

            Pi.failureStage = 'neighbourhood';
            Pi.message = neighborhoodMessage;
            Pi.totalWallTime = toc(tPoint);

            TestingResult.Point(iTest) = Pi;
            TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
            TestingResult.pointMessage{iTest} = Pi.message;

        end

        continue

    end


    idxFit = ...
        Nbh.idx(:);

    Xfit = ...
        Xtrain(idxFit,:);

    yfit = ...
        ytrain(idxFit);

    nFit = ...
        numel(idxFit);


    if saveStage2All

        Pi.neighborhood = Nbh;

        Pi.neighbor_idx = idxFit;
        Pi.neighbor_distances = Nbh.distances(:);
        Pi.neighbor_count = Nbh.count;

        Pi.requestedRadius = Nbh.requestedRadius;
        Pi.effectiveRadius = Nbh.effectiveRadius;
        Pi.selectedMaxDistance = Nbh.selectedMaxDistance;
        Pi.nWithinRadius = Nbh.nWithinRadius;
        Pi.nAddedForMinimum = Nbh.nAddedForMinimum;
        Pi.usedMinNeighborFallback = Nbh.usedMinNeighborFallback;

        Pi.effectiveRadius_raw = ...
            Pi.effectiveRadius .* S_global;

        Pi.selectedMaxDistance_raw = ...
            Pi.selectedMaxDistance .* S_global;

    end


    % =========================================================
    % 14.3 Polynomial admissibility
    % =========================================================

    if saveStage2All

        [isOK,polyInfo] = ...
            common.checkPolynomialAdmissibility( ...
                Xfit);

    else

        [isOK,~] = ...
            common.checkPolynomialAdmissibility( ...
                Xfit);

    end


    if ~isOK

        nPolynomialFailures = ...
            nPolynomialFailures + 1;


        if saveStage2All

            Pi.polynomialAdmissible = false;
            Pi.polynomialInfo = polyInfo;
            Pi.failureStage = 'polynomial admissibility';
            Pi.message = ...
                'Query neighbourhood is polynomially inadmissible.';
            Pi.totalWallTime = toc(tPoint);

            TestingResult.Point(iTest) = Pi;
            TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
            TestingResult.pointMessage{iTest} = Pi.message;

        end

        continue

    end


    if saveStage2All

        Pi.polynomialAdmissible = true;
        Pi.polynomialInfo = polyInfo;

    end


    % =========================================================
    % 14.4 Query-specific geometry scale and epsilon
    % =========================================================

    if useShape

        [~,d_local] = ...
            common.buildEpsilon( ...
                Xfit, ...
                1);


        if ~isscalar(d_local) || ...
           ~isreal(d_local) || ...
           ~isfinite(d_local) || ...
           d_local <= 0

            if saveStage2All

                Pi.failureStage = 'query geometry scale';
                Pi.message = 'Query local geometry scale is invalid.';
                Pi.totalWallTime = toc(tPoint);

                TestingResult.Point(iTest) = Pi;
                TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
                TestingResult.pointMessage{iTest} = Pi.message;

            end

            continue

        end


        epsilon = ...
            max( ...
                C_i .* d_local, ...
                1e-12);

    else

        d_local = NaN;
        epsilon = 1;

    end


    if saveStage2All

        Pi.d_local = d_local;
        Pi.epsilon = epsilon;

    end


    % =========================================================
    % 14.5 Fixed local geometry
    %
    % No pdist2 call is required inside the point loop.
    % =========================================================

    geometry = struct();

    geometry.R = ...
        D_train_train(idxFit,idxFit);

    geometry.P = ...
        P_train(idxFit,:);

    geometry.R_eval = ...
        reshape( ...
            D_query_train(iTest,idxFit), ...
            1,[]);

    geometry.P_eval = ...
        P_query(iTest,:);


    if saveStage2All

        Pi.system_size = ...
            nFit + p + 1;

    end


    % =========================================================
    % 14.6 Build / solve / predict / diagnose
    % =========================================================

    try

        S = ...
            common.buildRBFSystem( ...
                Xfit, ...
                yfit, ...
                kernel, ...
                epsilon, ...
                reg_i, ...
                geometry);


        TestingResult.nRBFSolves = ...
            TestingResult.nRBFSolves + 1;


        [coeffs,solveInfo] = ...
            common.solveLinearSystem( ...
                S, ...
                numericalOptions);


        if saveStage2All

            Pi.solveInfo = solveInfo;

        end


        if ~solveInfo.success || ...
           isempty(coeffs) || ...
           any(~isfinite(coeffs(:)))

            nFailedRBFSolves = ...
                nFailedRBFSolves + 1;


            if saveStage2All

                Pi.failureStage = 'local system solve';
                Pi.message = ...
                    'Test local RBF system solve was unsuccessful.';
                Pi.totalWallTime = toc(tPoint);

                TestingResult.Point(iTest) = Pi;
                TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
                TestingResult.pointMessage{iTest} = Pi.message;

            end

            continue

        end


        nSuccessfulRBFSolves = ...
            nSuccessfulRBFSolves + 1;


        [yhat,B_eval] = ...
            common.predictRBF( ...
                XtestScaled(iTest,:), ...
                Xfit, ...
                coeffs, ...
                kernel, ...
                epsilon, ...
                geometry);


        if ~isscalar(yhat) || ...
           ~isreal(yhat) || ...
           ~isfinite(yhat)

            nPredictionFailures = ...
                nPredictionFailures + 1;


            if saveStage2All

                Pi.failureStage = 'prediction';
                Pi.message = 'Test prediction is invalid.';
                Pi.totalWallTime = toc(tPoint);

                TestingResult.Point(iTest) = Pi;
                TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
                TestingResult.pointMessage{iTest} = Pi.message;

            end

            continue

        end


        Di = ...
            common.calculateConditionDiagnostics( ...
                S, ...
                coeffs, ...
                solveInfo, ...
                B_eval);


    catch ME

        if saveStage2All

            Pi.failureStage = 'test evaluation';
            Pi.message = ME.message;
            Pi.totalWallTime = toc(tPoint);

            TestingResult.Point(iTest) = Pi;
            TestingResult.pointWallTime(iTest) = Pi.totalWallTime;
            TestingResult.pointMessage{iTest} = Pi.message;

        end

        continue

    end


    % =========================================================
    % 14.7 Prediction / residual
    % =========================================================

    y_diff_i = NaN;

    if hasObservations

        y_diff_i = ...
            ytest(iTest) - yhat;

    end


    % =========================================================
    % 14.8 Compulsory retained output
    % =========================================================

    pointElapsed = ...
        toc(tPoint);


    TestingResult.point_success(iTest) = true;

    y_pred_test(iTest) = yhat;

    if hasObservations

        test_diff(iTest) = y_diff_i;

    end

    TestingResult.neighbor_count(iTest) = Nbh.count;

    TestingResult.cond_raw(iTest) = Di.cond_raw;
    TestingResult.cond_reg(iTest) = Di.cond_reg;
    TestingResult.cond_effective(iTest) = Di.cond_effective;
    TestingResult.cond_evaluation_test(iTest) = ...
        Di.cond_evaluation_test;
    TestingResult.rel_residual(iTest) = Di.rel_residual;
    TestingResult.backward_error(iTest) = Di.backward_error;
    TestingResult.solver_used{iTest} = Di.solver_used;


    % =========================================================
    % 14.9 DETAIL retained only when requested
    % =========================================================

    if saveStage2All

        coeffs = coeffs(:);

        Pi.coeffs = coeffs;
        Pi.lambda = coeffs(1:nFit);
        Pi.beta = coeffs(nFit+1:end);

        Pi.condition = ...
            testOnlyConditionDiagnosticsLocal( ...
                Di);

        Pi.cond_raw = Di.cond_raw;
        Pi.cond_reg = Di.cond_reg;
        Pi.cond_ratio = Di.cond_ratio;
        Pi.delta_log10_cond = Di.delta_log10_cond;
        Pi.cond_effective = Di.cond_effective;

        Pi.rank_eff = Di.rank_eff;
        Pi.rank_tol = Di.rank_tol;
        Pi.sigma_min_retained = Di.sigma_min_retained;

        Pi.cond_evaluation_test = Di.cond_evaluation_test;

        Pi.rel_residual = Di.rel_residual;
        Pi.backward_error = Di.backward_error;
        Pi.solver_used = Di.solver_used;

        Pi.yhat = yhat;

        if hasObservations

            Pi.y_diff = y_diff_i;
            Pi.residual = y_diff_i;
            Pi.AE = abs(y_diff_i);
            Pi.squaredResidual = y_diff_i.^2;

        end

        Pi.success = true;
        Pi.failureStage = '';
        Pi.message = '';
        Pi.totalWallTime = pointElapsed;

        TestingResult.Point(iTest) = Pi;
        TestingResult.pointWallTime(iTest) = pointElapsed;
        TestingResult.pointMessage{iTest} = '';

        TestingResult.d_local(iTest) = d_local;
        TestingResult.epsilon(iTest) = epsilon;

        TestingResult.neighbor_idx{iTest} = idxFit;
        TestingResult.neighbor_distances{iTest} = Nbh.distances(:);
        TestingResult.requestedRadius(iTest) = Nbh.requestedRadius;
        TestingResult.effectiveRadius(iTest) = Nbh.effectiveRadius;
        TestingResult.selectedMaxDistance(iTest) = Nbh.selectedMaxDistance;
        TestingResult.nWithinRadius(iTest) = Nbh.nWithinRadius;
        TestingResult.nAddedForMinimum(iTest) = Nbh.nAddedForMinimum;
        TestingResult.usedMinNeighborFallback(iTest) = ...
            Nbh.usedMinNeighborFallback;

        TestingResult.effectiveRadius_raw(iTest) = ...
            Nbh.effectiveRadius .* S_global;

        TestingResult.selectedMaxDistance_raw(iTest) = ...
            Nbh.selectedMaxDistance .* S_global;

        TestingResult.cond_ratio(iTest) = Di.cond_ratio;
        TestingResult.delta_log10_cond(iTest) = Di.delta_log10_cond;
        TestingResult.rank_eff(iTest) = Di.rank_eff;
        TestingResult.rank_tol(iTest) = Di.rank_tol;
        TestingResult.sigma_min_retained(iTest) = Di.sigma_min_retained;
        TestingResult.rel_residual(iTest) = Di.rel_residual;
        TestingResult.backward_error(iTest) = Di.backward_error;

        TestingResult.coeffs{iTest} = Pi.coeffs;
        TestingResult.lambda{iTest} = Pi.lambda;
        TestingResult.beta{iTest} = Pi.beta;

        conditionDiag(iTest).cond_raw = Di.cond_raw;
        conditionDiag(iTest).cond_reg = Di.cond_reg;
        conditionDiag(iTest).cond_ratio = Di.cond_ratio;
        conditionDiag(iTest).delta_log10_cond = Di.delta_log10_cond;
        conditionDiag(iTest).rank_eff = Di.rank_eff;
        conditionDiag(iTest).rank_tol = Di.rank_tol;
        conditionDiag(iTest).sigma_min_retained = Di.sigma_min_retained;
        conditionDiag(iTest).cond_effective = Di.cond_effective;
        % INTERNAL ONLY: retained for compatibility with the generic
        % condition summarizer; removed from all LocalRBFTesting exports.
        conditionDiag(iTest).cond_evaluation_train = ...
            Di.cond_evaluation_train;

        conditionDiag(iTest).cond_evaluation_test = ...
            Di.cond_evaluation_test;
        conditionDiag(iTest).rel_residual = Di.rel_residual;
        conditionDiag(iTest).backward_error = Di.backward_error;
        conditionDiag(iTest).reg_param = reg_i;
        conditionDiag(iTest).solver_used = Di.solver_used;
        conditionDiag(iTest).success = true;

    end

end


TestingResult.testingWallTime = ...
    toc(tTesting);


% Release heavy temporary geometry/transfer objects as soon as point
% evaluation is complete.  DETAIL, when requested, already owns the
% retained compact transfer information.
clear Transfer ...
      D_query_train ...
      D_train_train ...
      P_train ...
      P_query


% =============================================================
% 15. Canonical predictions
% =============================================================

% One canonical prediction and residual vector are compulsory.
TestingResult.yhat = y_pred_test;
TestingResult.residual = test_diff;

% Convenience aliases are DETAIL-only and are never constructed in the
% compulsory-only path.
if saveStage2All

    TestingResult.y_pred_test = y_pred_test;
    TestingResult.y_diff = test_diff;
    TestingResult.test_diff = test_diff;

end


% =============================================================
% 16. Prediction-success status
% =============================================================

TestingResult.nSuccessfulPoints = ...
    sum(TestingResult.point_success);

TestingResult.nFailedPoints = ...
    nTest - TestingResult.nSuccessfulPoints;

TestingResult.predictionSuccess = ...
    all(TestingResult.point_success) && ...
    all(isfinite(y_pred_test));


% =============================================================
% 17. Test error metrics
%
% Official RMSE requires successful prediction at EVERY test point.
% Partial metrics are retained only as DETAIL when requested, but the
% compulsory canonical AE/RMSE/MAE are always retained.
% =============================================================

TestingResult.AE = nan(nTest,1);
TestingResult.RMSE = NaN;
TestingResult.MAE = NaN;

if saveStage2All

    TestingResult.squaredResidual = nan(nTest,1);
    TestingResult.validResidualMask = false(nTest,1);
    TestingResult.nValidResiduals = 0;

    TestingResult.SSE_available = NaN;
    TestingResult.MSE_available = NaN;
    TestingResult.RMSE_available = NaN;
    TestingResult.MAE_available = NaN;

    TestingResult.SSE = NaN;
    TestingResult.MSE = NaN;

end


if hasObservations

    TestingResult.AE = ...
        abs(test_diff);


    squaredResidual = ...
        test_diff.^2;


    validResidual = ...
        TestingResult.point_success & ...
        isfinite(test_diff) & ...
        isfinite(squaredResidual);


    nValidResiduals = ...
        sum(validResidual);


    SSE_available = NaN;
    MSE_available = NaN;
    RMSE_available = NaN;
    MAE_available = NaN;


    if nValidResiduals > 0

        SSE_available = ...
            sum(squaredResidual(validResidual));

        MSE_available = ...
            SSE_available ./ nValidResiduals;

        RMSE_available = ...
            sqrt(MSE_available);

        MAE_available = ...
            mean(TestingResult.AE(validResidual));

    end


    if saveStage2All

        TestingResult.squaredResidual = squaredResidual;
        TestingResult.validResidualMask = validResidual;
        TestingResult.nValidResiduals = nValidResiduals;

        TestingResult.SSE_available = SSE_available;
        TestingResult.MSE_available = MSE_available;
        TestingResult.RMSE_available = RMSE_available;
        TestingResult.MAE_available = MAE_available;

    end


    if nValidResiduals == nTest

        TestingResult.RMSE = RMSE_available;
        TestingResult.MAE = MAE_available;

        test_rmse = ...
            TestingResult.RMSE;

        TestingResult.metricSuccess = ...
            true;


        if saveStage2All

            TestingResult.SSE = SSE_available;
            TestingResult.MSE = MSE_available;

        end

    end

end


% =============================================================
% 18. Shape-free DETAIL canonical values
% =============================================================

if ~useShape

    % Canonical shape-free parameter value is compulsory.
    TestingResult.C = nan(nTest,1);

    if saveStage2All

        TestingResult.d_local = nan(nTest,1);
        TestingResult.epsilon = ones(nTest,1);

    end

end


% =============================================================
% 19. DETAIL-only counters and summaries
% =============================================================

if saveStage2All

    TestingResult.nSuccessfulRBFSolves = ...
        nSuccessfulRBFSolves;

    TestingResult.nFailedRBFSolves = ...
        nFailedRBFSolves;

    TestingResult.nPolynomialFailures = ...
        nPolynomialFailures;

    TestingResult.nNeighborhoodFailures = ...
        nNeighborhoodFailures;

    TestingResult.nPredictionFailures = ...
        nPredictionFailures;


    TestingResult.nBackslash = ...
        sum(strcmp(TestingResult.solver_used,'backslash'));

    TestingResult.nLsqminnorm = ...
        sum(strcmp(TestingResult.solver_used,'lsqminnorm'));


    [ ...
        TestingResult.summary, ...
        TestingResult.series, ...
        TestingResult.mapData] = ...
        common.summarizeConditionDiagnostics( ...
            conditionDiag, ...
            XtestScaled);


    % LocalRBFTesting exports TEST/query diagnostics only.  The generic
    % summarizer may expose cond_evaluation_train because its schema is
    % shared with the training stage; strip only that field here.
    TestingResult.summary = ...
        removeFieldIfPresentLocal( ...
            TestingResult.summary, ...
            'cond_evaluation_train');

    TestingResult.series = ...
        removeFieldIfPresentLocal( ...
            TestingResult.series, ...
            'cond_evaluation_train');

    TestingResult.mapData = ...
        removeFieldIfPresentLocal( ...
            TestingResult.mapData, ...
            'cond_evaluation_train');


    % ---------------------------------------------------------
    % Additional result series
    % ---------------------------------------------------------

    TestingResult.series.radius = TestingResult.radius;
    TestingResult.series.radius_raw = TestingResult.radius_raw;
    TestingResult.series.C = TestingResult.C;
    TestingResult.series.reg_param = TestingResult.reg_param;
    TestingResult.series.d_local = TestingResult.d_local;
    TestingResult.series.epsilon = TestingResult.epsilon;
    TestingResult.series.neighbor_count = TestingResult.neighbor_count;
    TestingResult.series.yhat = TestingResult.yhat;
    TestingResult.series.y_diff = TestingResult.y_diff;
    TestingResult.series.point_success = TestingResult.point_success;
    TestingResult.series.rel_residual = TestingResult.rel_residual;
    TestingResult.series.backward_error = TestingResult.backward_error;

    if hasObservations

        TestingResult.series.AE = TestingResult.AE;

    end


    % ---------------------------------------------------------
    % Additional map-ready values
    % ---------------------------------------------------------

    TestingResult.mapData.radius = TestingResult.radius;
    TestingResult.mapData.radius_raw = TestingResult.radius_raw;
    TestingResult.mapData.C = TestingResult.C;
    TestingResult.mapData.reg_param = TestingResult.reg_param;
    TestingResult.mapData.epsilon = TestingResult.epsilon;
    TestingResult.mapData.neighbor_count = TestingResult.neighbor_count;
    TestingResult.mapData.yhat = TestingResult.yhat;
    TestingResult.mapData.y_diff = TestingResult.y_diff;
    TestingResult.mapData.rel_residual = TestingResult.rel_residual;
    TestingResult.mapData.backward_error = TestingResult.backward_error;

end


% =============================================================
% 20. Final status
% =============================================================

TestingResult.success = ...
    TestingResult.predictionSuccess;


if TestingResult.success

    TestingResult.message = '';

else

    TestingResult.message = ...
        sprintf( ...
            ['Test prediction succeeded for %d of %d ' ...
             'query points.'], ...
            TestingResult.nSuccessfulPoints, ...
            nTest);

end


% =============================================================
% 21. Total runtime
% =============================================================

TestingResult.totalWallTime = ...
    TestingResult.geometryWallTime + ...
    TestingResult.transferWallTime + ...
    TestingResult.testingWallTime;

end


% =============================================================
% Local helper
% =============================================================

function tf = getSaveStage2All(options,TrainingResult)
%GETSAVESTAGE2ALL Stage-2 DETAIL switch. Default false.
%
% Explicit testing options take precedence.  If execution controls were
% deliberately stripped by the batch numerical-options path, recover the
% run-level request from TrainingResult.options.

tf = false;

[found,value] = ...
    findSaveStage2All(options);

if ~found && ...
   isstruct(TrainingResult) && ...
   isfield(TrainingResult,'options') && ...
   isstruct(TrainingResult.options)

    [found,value] = ...
        findSaveStage2All(TrainingResult.options);

end

if found

    if ~(isscalar(value) && ...
         (islogical(value) || ...
          (isnumeric(value) && ...
           isreal(value) && ...
           isfinite(value) && ...
           any(value == [0 1]))))

        error( ...
            ['options.execution.saveStage2All must be true/false ' ...
             '(or numeric 1/0).']);

    end

    tf = logical(value);

end

end


% =============================================================
% Local helper
% =============================================================

function [found,value] = findSaveStage2All(options)
%FINDSAVESTAGE2ALL Read execution.saveStage2All without defaulting.

found = false;
value = [];

if isstruct(options) && ...
   isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'saveStage2All') && ...
   ~isempty(options.execution.saveStage2All)

    found = true;
    value = options.execution.saveStage2All;

end

end


% =============================================================
% Local helper
% =============================================================

function tf = getSaveEnsemblePointDiagnostics(options,TrainingResult)
%GETSAVEENSEMBLEPOINTDIAGNOSTICS
% Lightweight per-unique-tuple ensemble numerical-diagnostic storage.
% Default = false. It is used only when full Stage-2 DETAIL is off.

tf = false;


[found,value] = ...
    findSaveEnsemblePointDiagnostics( ...
        options);


if ~found && ...
   isstruct(TrainingResult) && ...
   isfield(TrainingResult,'options') && ...
   isstruct(TrainingResult.options)

    [found,value] = ...
        findSaveEnsemblePointDiagnostics( ...
            TrainingResult.options);

end


if found

    if ~(isscalar(value) && ...
         (islogical(value) || ...
          (isnumeric(value) && ...
           isreal(value) && ...
           isfinite(value) && ...
           any(value == [0 1]))))

        error( ...
            ['options.execution.saveEnsemblePointDiagnostics must be ' ...
             'true/false (or numeric 1/0).']);

    end


    tf = ...
        logical(value);

end

end


% =============================================================
% Local helper
% =============================================================

function [found,value] = findSaveEnsemblePointDiagnostics(options)
%FINDSAVEENSEMBLEPOINTDIAGNOSTICS Read option without defaulting.

found = ...
    false;


value = ...
    [];


if isstruct(options) && ...
   isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'saveEnsemblePointDiagnostics') && ...
   ~isempty(options.execution.saveEnsemblePointDiagnostics)

    found = ...
        true;


    value = ...
        options.execution.saveEnsemblePointDiagnostics;

end

end


% =============================================================
% Local helper
% =============================================================

function optionsOut = canonicalNumericalOptions(optionsIn)
%CANONICALNUMERICALOPTIONS Remove non-numerical controls.

optionsOut = optionsIn;


if isstruct(optionsOut) && ...
   isfield(optionsOut,'execution')

    optionsOut = ...
        rmfield(optionsOut,'execution');

end


if isstruct(optionsOut) && ...
   isfield(optionsOut,'transfer')

    optionsOut = ...
        rmfield(optionsOut,'transfer');

end

end

% =============================================================
% Local helper
% =============================================================

function ER = evaluateEnsembleTestPointLocal( ...
    iTest, ...
    Xtrain,ytrain,XtestScaled,kernel,useShape,min_neighbors, ...
    D_query_train,D_train_train,P_train,P_query,numericalOptions, ...
    wUnique,radiusUnique,regUnique,CUnique,S_global, ...
    retainDetail,retainDiagnosticDetail)
%EVALUATEENSEMBLETESTPOINTLOCAL
%
% Evaluate ONE far-ensemble query.
%
% Every UNIQUE complete tuple is solved exactly once:
%
%       (radius, RegParam, C) -> yhat_j
%
% and the final prediction is:
%
%       yhat = sum_j w_j * yhat_j
%
% Strict failure rule:
%   every tuple must succeed. Failed tuples are never dropped and the
%   original ensemble weights are never silently renormalized.
%
% Time saving:
%   tuples sharing the exact same radius reuse one neighbourhood,
%   polynomial-admissibility result, local geometry, and d_local.

wUnique = ...
    wUnique(:);


radiusUnique = ...
    radiusUnique(:);


regUnique = ...
    regUnique(:);


CUnique = ...
    CUnique(:);


nTuple = ...
    numel(wUnique);


ER = ...
    struct();


ER.success = ...
    false;


ER.message = ...
    '';


ER.failureStage = ...
    '';


ER.yhat = ...
    NaN;


ER.nRBFSolves = ...
    0;


ER.nSuccessfulRBFSolves = ...
    0;


ER.nFailedRBFSolves = ...
    0;


ER.nPolynomialFailures = ...
    0;


ER.nNeighborhoodFailures = ...
    0;


ER.nPredictionFailures = ...
    0;


ER.maxNeighborCount = ...
    NaN;


ER.condRawMaxTupleIndex = ...
    NaN;


ER.condRegMaxTupleIndex = ...
    NaN;


ER.condEffectiveMaxTupleIndex = ...
    NaN;


ER.relResidualMaxTupleIndex = ...
    NaN;


ER.backwardErrorMaxTupleIndex = ...
    NaN;


ER.maxCondition = ...
    emptyConditionDiagnostic();


ER.TupleDetail = ...
    [];


ER.TupleDiagnostic = ...
    [];


% =============================================================
% A. Validate tuple arrays / weights
% =============================================================

if nTuple < 1 || ...
   numel(radiusUnique) ~= nTuple || ...
   numel(regUnique) ~= nTuple || ...
   numel(CUnique) ~= nTuple

    ER.failureStage = ...
        'ensemble transfer';


    ER.message = ...
        'Ensemble tuple arrays are empty or inconsistent.';


    return

end


if any(~isfinite(wUnique)) || ...
   any(wUnique < 0) || ...
   abs(sum(wUnique)-1) > 1e-12

    ER.failureStage = ...
        'ensemble weights';


    ER.message = ...
        'Ensemble weights are invalid or do not sum to one.';


    return

end


if any(~isfinite(radiusUnique)) || ...
   any(radiusUnique <= 0) || ...
   any(~isfinite(regUnique)) || ...
   any(regUnique <= 0)

    ER.failureStage = ...
        'ensemble settings';


    ER.message = ...
        'Ensemble radius/RegParam tuple values are invalid.';


    return

end


if useShape && ...
   (any(~isfinite(CUnique)) || ...
    any(CUnique <= 0))

    ER.failureStage = ...
        'ensemble settings';


    ER.message = ...
        'Ensemble C tuple values are invalid.';


    return

end


% =============================================================
% B. DETAIL allocation only when requested
% =============================================================

if retainDetail

    emptyTuple = ...
        struct( ...
            'weight',NaN, ...
            'radius',NaN, ...
            'radius_raw',NaN, ...
            'C',NaN, ...
            'reg_param',NaN, ...
            'neighbor_idx',[], ...
            'neighbor_distances',[], ...
            'neighbor_count',NaN, ...
            'requestedRadius',NaN, ...
            'effectiveRadius',NaN, ...
            'effectiveRadius_raw',NaN, ...
            'selectedMaxDistance',NaN, ...
            'selectedMaxDistance_raw',NaN, ...
            'nWithinRadius',NaN, ...
            'nAddedForMinimum',NaN, ...
            'usedMinNeighborFallback',false, ...
            'd_local',NaN, ...
            'epsilon',NaN, ...
            'yhat',NaN, ...
            'condition',[], ...
            'solver_used','', ...
            'success',false, ...
            'failureStage','', ...
            'message','');


    TupleDetail = ...
        repmat( ...
            emptyTuple, ...
            nTuple,1);

else

    TupleDetail = ...
        [];

end


if retainDiagnosticDetail

    emptyTupleDiagnostic = ...
        struct( ...
            'condition',[], ...
            'solver_used','', ...
            'success',false, ...
            'failureStage','', ...
            'message','');


    TupleDiagnostic = ...
        repmat( ...
            emptyTupleDiagnostic, ...
            nTuple,1);

else

    TupleDiagnostic = ...
        [];

end


% =============================================================
% C. Small tuple-level working state
% =============================================================

yhatTuple = ...
    nan(nTuple,1);


neighborCountTuple = ...
    nan(nTuple,1);


condRawTuple = ...
    nan(nTuple,1);


condRegTuple = ...
    nan(nTuple,1);


condEffectiveTuple = ...
    nan(nTuple,1);


relResidualTuple = ...
    nan(nTuple,1);


backwardErrorTuple = ...
    nan(nTuple,1);


% Radius cache: one geometry construction per unique radius.
radiusCache = ...
    zeros(0,1);


radiusDataCache = ...
    cell(0,1);


distAll = ...
    D_query_train(iTest,:).';


% =============================================================
% D. Solve every unique complete tuple exactly once
% =============================================================

for j = 1:nTuple

    radius_j = ...
        radiusUnique(j);


    reg_j = ...
        regUnique(j);


    C_j = ...
        CUnique(j);


    if retainDetail

        TupleDetail(j).weight = ...
            wUnique(j);


        TupleDetail(j).radius = ...
            radius_j;


        TupleDetail(j).radius_raw = ...
            radius_j .* ...
            S_global;


        TupleDetail(j).reg_param = ...
            reg_j;


        TupleDetail(j).C = ...
            C_j;

    end


    % ---------------------------------------------------------
    % Reuse query-local geometry for repeated exact tuple radii.
    % ---------------------------------------------------------

    iCache = ...
        find( ...
            radiusCache == radius_j, ...
            1, ...
            'first');


    if isempty(iCache)

        RadiusData = ...
            struct();


        try

            Nbh = ...
                localrbf.getNeighborhood( ...
                    Xtrain, ...
                    XtestScaled(iTest,:), ...
                    radius_j, ...
                    min_neighbors, ...
                    [], ...
                    distAll);

        catch ME

            Nbh = ...
                [];


            RadiusData.message = ...
                ME.message;

        end


        if isempty(Nbh) || ...
           ~isfield(Nbh,'success') || ...
           ~Nbh.success

            ER.nNeighborhoodFailures = ...
                ER.nNeighborhoodFailures + 1;


            ER.failureStage = ...
                'ensemble neighbourhood';


            if isfield(RadiusData,'message') && ...
               ~isempty(RadiusData.message)

                ER.message = ...
                    RadiusData.message;


            elseif ~isempty(Nbh) && ...
                   isfield(Nbh,'message') && ...
                   ~isempty(Nbh.message)

                ER.message = ...
                    Nbh.message;


            else

                ER.message = ...
                    'Ensemble query neighbourhood construction failed.';

            end


            if retainDetail

                TupleDetail(j).failureStage = ...
                    ER.failureStage;


                TupleDetail(j).message = ...
                    ER.message;


                ER.TupleDetail = ...
                    TupleDetail;

            end


            if retainDiagnosticDetail

                TupleDiagnostic(j).failureStage = ...
                    ER.failureStage;


                TupleDiagnostic(j).message = ...
                    ER.message;


                ER.TupleDiagnostic = ...
                    TupleDiagnostic;

            end


            return

        end


        idxFit = ...
            Nbh.idx(:);


        Xfit = ...
            Xtrain(idxFit,:);


        yfit = ...
            ytrain(idxFit);


        [isOK,~] = ...
            common.checkPolynomialAdmissibility( ...
                Xfit);


        if ~isOK

            ER.nPolynomialFailures = ...
                ER.nPolynomialFailures + 1;


            ER.failureStage = ...
                'ensemble polynomial admissibility';


            ER.message = ...
                'An ensemble tuple neighbourhood is polynomially inadmissible.';


            if retainDetail

                TupleDetail(j).failureStage = ...
                    ER.failureStage;


                TupleDetail(j).message = ...
                    ER.message;


                ER.TupleDetail = ...
                    TupleDetail;

            end


            if retainDiagnosticDetail

                TupleDiagnostic(j).failureStage = ...
                    ER.failureStage;


                TupleDiagnostic(j).message = ...
                    ER.message;


                ER.TupleDiagnostic = ...
                    TupleDiagnostic;

            end


            return

        end


        if useShape

            [~,d_local] = ...
                common.buildEpsilon( ...
                    Xfit, ...
                    1);


            if ~isscalar(d_local) || ...
               ~isreal(d_local) || ...
               ~isfinite(d_local) || ...
               d_local <= 0

                ER.failureStage = ...
                    'ensemble query geometry scale';


                ER.message = ...
                    'An ensemble tuple local geometry scale is invalid.';


                if retainDetail

                    TupleDetail(j).failureStage = ...
                        ER.failureStage;


                    TupleDetail(j).message = ...
                        ER.message;


                    ER.TupleDetail = ...
                        TupleDetail;

                end


                if retainDiagnosticDetail

                    TupleDiagnostic(j).failureStage = ...
                        ER.failureStage;


                    TupleDiagnostic(j).message = ...
                        ER.message;


                    ER.TupleDiagnostic = ...
                        TupleDiagnostic;

                end


                return

            end

        else

            d_local = ...
                NaN;

        end


        geometry = ...
            struct();


        geometry.R = ...
            D_train_train(idxFit,idxFit);


        geometry.P = ...
            P_train(idxFit,:);


        geometry.R_eval = ...
            reshape( ...
                D_query_train(iTest,idxFit), ...
                1,[]);


        geometry.P_eval = ...
            P_query(iTest,:);


        RadiusData.Nbh = ...
            Nbh;


        RadiusData.idxFit = ...
            idxFit;


        RadiusData.Xfit = ...
            Xfit;


        RadiusData.yfit = ...
            yfit;


        RadiusData.d_local = ...
            d_local;


        RadiusData.geometry = ...
            geometry;


        radiusCache(end+1,1) = ...
            radius_j; %#ok<AGROW>


        radiusDataCache{end+1,1} = ...
            RadiusData; %#ok<AGROW>


    else

        RadiusData = ...
            radiusDataCache{iCache};


        Nbh = ...
            RadiusData.Nbh;


        idxFit = ...
            RadiusData.idxFit;


        Xfit = ...
            RadiusData.Xfit;


        yfit = ...
            RadiusData.yfit;


        d_local = ...
            RadiusData.d_local;


        geometry = ...
            RadiusData.geometry;

    end


    nFit = ...
        numel(idxFit);


    neighborCountTuple(j) = ...
        nFit;


    if useShape

        epsilon = ...
            max( ...
                C_j .* ...
                d_local, ...
                1e-12);

    else

        epsilon = ...
            1;

    end


    if retainDetail

        TupleDetail(j).neighbor_idx = ...
            idxFit;


        TupleDetail(j).neighbor_distances = ...
            Nbh.distances(:);


        TupleDetail(j).neighbor_count = ...
            Nbh.count;


        TupleDetail(j).requestedRadius = ...
            Nbh.requestedRadius;


        TupleDetail(j).effectiveRadius = ...
            Nbh.effectiveRadius;


        TupleDetail(j).effectiveRadius_raw = ...
            Nbh.effectiveRadius .* ...
            S_global;


        TupleDetail(j).selectedMaxDistance = ...
            Nbh.selectedMaxDistance;


        TupleDetail(j).selectedMaxDistance_raw = ...
            Nbh.selectedMaxDistance .* ...
            S_global;


        TupleDetail(j).nWithinRadius = ...
            Nbh.nWithinRadius;


        TupleDetail(j).nAddedForMinimum = ...
            Nbh.nAddedForMinimum;


        TupleDetail(j).usedMinNeighborFallback = ...
            Nbh.usedMinNeighborFallback;


        TupleDetail(j).d_local = ...
            d_local;


        TupleDetail(j).epsilon = ...
            epsilon;

    end


    % ---------------------------------------------------------
    % Build, solve, predict, diagnose this COMPLETE tuple.
    % ---------------------------------------------------------

    try

        S = ...
            common.buildRBFSystem( ...
                Xfit, ...
                yfit, ...
                kernel, ...
                epsilon, ...
                reg_j, ...
                geometry);


        ER.nRBFSolves = ...
            ER.nRBFSolves + 1;


        [coeffs,solveInfo] = ...
            common.solveLinearSystem( ...
                S, ...
                numericalOptions);


        if ~solveInfo.success || ...
           isempty(coeffs) || ...
           any(~isfinite(coeffs(:)))

            ER.nFailedRBFSolves = ...
                ER.nFailedRBFSolves + 1;


            ER.failureStage = ...
                'ensemble local system solve';


            ER.message = ...
                'An ensemble tuple RBF solve was unsuccessful.';


            if retainDetail

                TupleDetail(j).failureStage = ...
                    ER.failureStage;


                TupleDetail(j).message = ...
                    ER.message;


                ER.TupleDetail = ...
                    TupleDetail;

            end


            if retainDiagnosticDetail

                TupleDiagnostic(j).failureStage = ...
                    ER.failureStage;


                TupleDiagnostic(j).message = ...
                    ER.message;


                ER.TupleDiagnostic = ...
                    TupleDiagnostic;

            end


            return

        end


        ER.nSuccessfulRBFSolves = ...
            ER.nSuccessfulRBFSolves + 1;


        [yhat_j,B_eval] = ...
            common.predictRBF( ...
                XtestScaled(iTest,:), ...
                Xfit, ...
                coeffs, ...
                kernel, ...
                epsilon, ...
                geometry);


        if ~isscalar(yhat_j) || ...
           ~isreal(yhat_j) || ...
           ~isfinite(yhat_j)

            ER.nPredictionFailures = ...
                ER.nPredictionFailures + 1;


            ER.failureStage = ...
                'ensemble prediction';


            ER.message = ...
                'An ensemble tuple prediction is invalid.';


            if retainDetail

                TupleDetail(j).failureStage = ...
                    ER.failureStage;


                TupleDetail(j).message = ...
                    ER.message;


                ER.TupleDetail = ...
                    TupleDetail;

            end


            if retainDiagnosticDetail

                TupleDiagnostic(j).failureStage = ...
                    ER.failureStage;


                TupleDiagnostic(j).message = ...
                    ER.message;


                ER.TupleDiagnostic = ...
                    TupleDiagnostic;

            end


            return

        end


        Di = ...
            common.calculateConditionDiagnostics( ...
                S, ...
                coeffs, ...
                solveInfo, ...
                B_eval);


    catch ME

        ER.failureStage = ...
            'ensemble test evaluation';


        ER.message = ...
            ME.message;


        if retainDetail

            TupleDetail(j).failureStage = ...
                ER.failureStage;


            TupleDetail(j).message = ...
                ER.message;


            ER.TupleDetail = ...
                TupleDetail;

        end


        if retainDiagnosticDetail

            TupleDiagnostic(j).failureStage = ...
                ER.failureStage;


            TupleDiagnostic(j).message = ...
                ER.message;


            ER.TupleDiagnostic = ...
                TupleDiagnostic;

        end


        return

    end


    yhatTuple(j) = ...
        yhat_j;


    condRawTuple(j) = ...
        Di.cond_raw;


    condRegTuple(j) = ...
        Di.cond_reg;


    condEffectiveTuple(j) = ...
        Di.cond_effective;


    relResidualTuple(j) = ...
        Di.rel_residual;


    backwardErrorTuple(j) = ...
        Di.backward_error;


    if retainDetail

        TupleDetail(j).yhat = ...
            yhat_j;


        TupleDetail(j).condition = ...
            testOnlyConditionDiagnosticsLocal( ...
                Di);


        TupleDetail(j).solver_used = ...
            Di.solver_used;


        TupleDetail(j).success = ...
            true;

    end


    if retainDiagnosticDetail

        TupleDiagnostic(j).condition = ...
            testOnlyConditionDiagnosticsLocal( ...
                Di);


        TupleDiagnostic(j).solver_used = ...
            Di.solver_used;


        TupleDiagnostic(j).success = ...
            true;

    end

end


% =============================================================
% E. Strict ensemble aggregation
% =============================================================

if any(~isfinite(yhatTuple))

    ER.failureStage = ...
        'ensemble prediction';


    ER.message = ...
        'Not every unique ensemble tuple produced a finite prediction.';


    ER.TupleDetail = ...
        TupleDetail;


    ER.TupleDiagnostic = ...
        TupleDiagnostic;


    return

end


ER.yhat = ...
    sum( ...
        wUnique .* ...
        yhatTuple);


if ~isscalar(ER.yhat) || ...
   ~isreal(ER.yhat) || ...
   ~isfinite(ER.yhat)

    ER.failureStage = ...
        'ensemble prediction aggregation';


    ER.message = ...
        'Weighted ensemble prediction is invalid.';


    ER.TupleDetail = ...
        TupleDetail;


    ER.TupleDiagnostic = ...
        TupleDiagnostic;


    return

end


% =============================================================
% F. Conservative ensemble condition reporting
%
% There is no single ensemble RBF matrix.  For each canonical condition
% metric, retain the maximum value across the unique constituent systems:
%
%       cond_raw       = max_j cond_raw(j)
%       cond_reg       = max_j cond_reg(j)
%       cond_effective = max_j cond_effective(j)
%
% The maxima are calculated independently because the worst tuple can
% differ between metrics.  The same conservative rule is used for
% rel_residual and backward_error.  No mean/median/min or synthetic
% representative condition record is used.
% =============================================================

[ ...
    ER.maxCondition.cond_raw, ...
    ER.condRawMaxTupleIndex ...
] = ...
    finiteMaxWithIndexLocal( ...
        condRawTuple);


[ ...
    ER.maxCondition.cond_reg, ...
    ER.condRegMaxTupleIndex ...
] = ...
    finiteMaxWithIndexLocal( ...
        condRegTuple);


[ ...
    ER.maxCondition.cond_effective, ...
    ER.condEffectiveMaxTupleIndex ...
] = ...
    finiteMaxWithIndexLocal( ...
        condEffectiveTuple);


[ ...
    ER.maxCondition.rel_residual, ...
    ER.relResidualMaxTupleIndex ...
] = ...
    finiteMaxWithIndexLocal( ...
        relResidualTuple);


[ ...
    ER.maxCondition.backward_error, ...
    ER.backwardErrorMaxTupleIndex ...
] = ...
    finiteMaxWithIndexLocal( ...
        backwardErrorTuple);


ER.maxNeighborCount = ...
    finiteMaxOrNaNLocal( ...
        neighborCountTuple);


ER.maxCondition.solver_used = ...
    'ensemble';


ER.maxCondition.success = ...
    true;


ER.TupleDetail = ...
    TupleDetail;


ER.TupleDiagnostic = ...
    TupleDiagnostic;


ER.success = ...
    true;

end


% =============================================================
% Local helper
% =============================================================

function [value,j] = finiteMaxWithIndexLocal(x)
%FINITEMAXWITHINDEXLOCAL Maximum finite value and its original index.

valid = ...
    isfinite(x);


if ~any(valid)

    value = ...
        NaN;


    j = ...
        NaN;


    return

end


idx = ...
    find(valid);


[value,k] = ...
    max( ...
        x(valid));


j = ...
    idx(k);

end


% =============================================================
% Local helper
% =============================================================

function value = finiteMaxOrNaNLocal(x)
%FINITEMAXORNANLOCAL Maximum finite value.

x = ...
    x(isfinite(x));


if isempty(x)

    value = ...
        NaN;

else

    value = ...
        max(x);

end

end


% =============================================================
% Local helper
% =============================================================

function D = testOnlyConditionDiagnosticsLocal(Din)
%TESTONLYCONDITIONDIAGNOSTICSLOCAL
% Remove the generic training-side evaluation-condition field from a
% diagnostic structure exported by LocalRBFTesting.
%
% No diagnostic calculation is changed.

D = ...
    removeFieldIfPresentLocal( ...
        Din, ...
        'cond_evaluation_train');

end


% =============================================================
% Local helper
% =============================================================

function S = removeFieldIfPresentLocal(S,fieldName)
%REMOVEFIELDIFPRESENTLOCAL Remove one field and preserve all others.

if isstruct(S) && ...
   isfield(S,fieldName)

    S = ...
        rmfield( ...
            S, ...
            fieldName);

end

end


% =============================================================
% Local helper
% =============================================================

function P = emptyTestPoint()
%EMPTYTESTPOINT
%
% Full DETAIL point record.  This helper is called only when
% saveStage2All=true.

P = struct();

% Point identity
P.iTest = NaN;

P.xTest_raw = [];
P.xTest = [];
P.xTest_scaled = [];

P.hasObservation = false;
P.observation = NaN;


% Transfer
P.transferSuccess = false;
P.transferMethod = '';

P.nearestTrainingIndex = NaN;
P.nearestDistance = NaN;
P.nearestDistance_raw = NaN;

P.sourceTrainingIndex = [];
P.sourceDistance = [];
P.sourceDistance_raw = [];
P.sourceWeight = [];
P.nTransferSources = 0;


% Ensemble transfer / prediction detail
P.ensembleNUniqueTuples = 0;
P.ensembleCondRawMaxTupleIndex = NaN;
P.ensembleCondRegMaxTupleIndex = NaN;
P.ensembleCondEffectiveMaxTupleIndex = NaN;
P.ensembleRelResidualMaxTupleIndex = NaN;
P.ensembleBackwardErrorMaxTupleIndex = NaN;
P.ensemble = [];


% Transferred settings
P.radius = NaN;
P.radius_scaled = NaN;
P.radius_raw = NaN;

P.C = NaN;
P.reg_param = NaN;

P.d_local = NaN;
P.epsilon = NaN;


% Neighbourhood
P.neighborhood = [];

P.neighbor_idx = [];
P.neighbor_distances = [];
P.neighbor_count = 0;

P.requestedRadius = NaN;

P.effectiveRadius = NaN;
P.effectiveRadius_raw = NaN;

P.selectedMaxDistance = NaN;
P.selectedMaxDistance_raw = NaN;

P.nWithinRadius = NaN;
P.nAddedForMinimum = NaN;

P.usedMinNeighborFallback = false;


% Polynomial
P.polynomialAdmissible = false;
P.polynomialInfo = [];


% System
P.system_size = NaN;
P.solveInfo = [];


% Coefficients
P.coeffs = [];
P.lambda = [];
P.beta = [];


% Complete diagnostic
P.condition = [];


% Condition diagnostics
P.cond_raw = NaN;
P.cond_reg = NaN;

P.cond_ratio = NaN;
P.delta_log10_cond = NaN;

P.cond_effective = NaN;

P.rank_eff = NaN;
P.rank_tol = NaN;
P.sigma_min_retained = NaN;

P.cond_evaluation_test = NaN;

P.rel_residual = NaN;
P.backward_error = NaN;

P.solver_used = '';


% Prediction
P.yhat = NaN;

P.y_diff = NaN;
P.residual = NaN;

P.AE = NaN;
P.squaredResidual = NaN;


% Status
P.success = false;
P.failureStage = '';
P.message = '';


% Runtime
P.totalWallTime = 0;

end


% =============================================================
% Local helper
% =============================================================

function D = emptyConditionDiagnostic()
%EMPTYCONDITIONDIAGNOSTIC
%
% DETAIL summary record.  This helper is called only when
% saveStage2All=true.

D = struct();

D.cond_raw = NaN;
D.cond_reg = NaN;

D.cond_ratio = NaN;
D.delta_log10_cond = NaN;

D.rank_eff = NaN;
D.rank_tol = NaN;

D.sigma_min_retained = NaN;
D.cond_effective = NaN;

D.cond_evaluation_train = NaN;
D.cond_evaluation_test = NaN;

D.rel_residual = NaN;
D.backward_error = NaN;

D.reg_param = NaN;
D.solver_used = '';

D.success = false;

end


% =============================================================
% Local helper
% =============================================================

function out = scaleCellVectors(in,scaleFactor)
%SCALECELLVECTORS
%
% Called only for DETAIL transfer aliases.

out = cell(size(in));

for k = 1:numel(in)

    if isempty(in{k})

        out{k} = [];

    else

        out{k} = ...
            in{k} .* scaleFactor;

    end

end

end

function [TrainingResult, geometryCache] = LocalRBFTraining( ...
    X, y, radii, kernel, ...
    min_neighbors, ...
    C_min, C_max, ...
    Reg_min, Reg_max, ...
    tau_loc, tau_cap, ...
    options, method, nGrid)
%LOCALRBFTRAINING
%
% Public local-RBF training workflow for ONE kernel.
%
% Two-stage execution (optional):
%   options.execution.stage = 'all'    : original one-pass workflow
%                           = 'stage1' : optimize all candidate radii only
%                           = 'stage2' : load Stage-1 PointOpt and continue
%
% Stage 1 ends immediately after
%   localrbf.optimizeAtPointForAllCandidateRadii(...)
% for every target point. Stage 2 starts with the existing
%   localrbf.selectBestRadiusWithToleranceAtEachRadiusCapOverAllCaps(...)
% and then executes all remaining original training/evaluation logic.
% LocalRBFTraining_New.m is not used by this implementation.
%
% Stage-1 TRACE storage switch:
%   options.execution.saveTrace = true   : save all three TRACE files
%   options.execution.saveTrace = false  : save CORE + DIAG only
% Default is true. This switch affects storage only, not optimization.
%
% Stage-2 detail switch:
%   options.execution.saveStage2All = true  : retain Stage-2 detail
%   options.execution.saveStage2All = false : retain compulsory outputs only
% Default is false. Optional payloads are not constructed when disabled.
%
% X and radii are supplied in the ORIGINAL coordinate units.
% A single global coordinate scale is calculated as:
%
%       S_global = common.MeanDistant(X)
%
% and the internal local training geometry uses:
%
%       X_scaled     = X / S_global
%       radii_scaled = radii / S_global
%
% For every training point:
%
%   1. optimize the local RBF parameters at ALL candidate radii;
%   2. apply the local radius-tolerance rule at EVERY radius cap;
%   3. retain prediction and target-excluded residual for every cap.
%
% Across the complete training set:
%
%   4. assemble residual(i,k);
%   5. calculate ORIGINAL training RMSE(k) for every radius cap;
%   6. select the cap-RMSE source through options.transfer.capRMSESource:
%
%          'training' : use the original complete-training-set RMSE;
%          'pseudo'   : calculate strict pseudo-training RMSE internally
%                       for the SAME transfer configuration.
%
%   7. select the smallest cap satisfying:
%
%          RMSE_source(k) <= min(RMSE_source) + tau_cap
%
%   8. extract the final point-specific radius and optimized
%      parameter settings associated with the selected cap.
%
% TransferOptions are carried in options.transfer and include:
%   .method
%   .nIDWSources
%   .nearThresholdRaw
%   .parameterTransferMode
%   .capRMSESource
%
% When capRMSESource='pseudo', pseudo-training RMSE is calculated inside
% LocalRBFTraining.  External pseudoRMSE_all / pseudo_cap_success inputs are
% no longer required for cap selection.
%
% The default capRMSESource is 'training', reproducing the historical
% production behavior exactly.
%
% Residual convention:
%
%       residual = observation - prediction
%
% IMPORTANT:
%   - minimum fitting observations default to 15;
%   - raw X and raw radii must use the same original coordinate units;
%   - S_global is calculated ONCE from the raw full training geometry;
%   - X and radii are scaled ONCE internally;
%   - the full scaled distance matrix is calculated ONCE and reused;
%   - one point-radius optimization is performed only once;
%   - optimized radius results are reused for every radius cap;
%   - no RBF optimization is repeated during cap selection;
%   - no full CN/ECN/EVN diagnostics are calculated here;
%   - Stage 1 stores compact scientific results in CORE and retains complete
%     optimization histories separately in TRACE_SEARCH / TRACE_OPT /
%     TRACE_NUMERIC; Stage 2 loads CORE only;
%   - large full distance matrices are temporary and are not
%     duplicated in TrainingResult;
%   - original training RMSE is always retained unchanged for diagnostics;
%   - cap selection can use either original training RMSE or a strict
%     pseudo-training RMSE calculated internally from the same transfer
%     configuration;
%   - either cap-selection RMSE source requires a valid complete-set result;
%   - partial-cap original training RMSE is retained separately for diagnostics;
%   - RNG is initialized ONCE here; default RNG_SEED = 1;
%   - inner GA parallelism remains disabled in optimizeParameters;
%   - evaluateLocal is called after successful training using the
%     SAME geometryCache, so pdist2(X_scaled,X_scaled) is not repeated;
%   - saveStage2All=true uses full selected-system diagnostics;
%   - saveStage2All=false diagnoses final selected systems only;
%   - repeated caps selecting the same radius at one point reuse one
%     selected-system diagnostic calculation in full evaluation mode.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 5 || isempty(min_neighbors)

    min_neighbors = ...
        15;

end


if nargin < 12 || isempty(options)

    options = ...
        struct();

end

if nargin < 13 || isempty(method)
    method = 'ga';
end

if nargin < 14 || isempty(nGrid)
    nGrid = 25;
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


if ~isfield(options.solver,'compute_rel_residual') || ...
   isempty(options.solver.compute_rel_residual)

    options.solver.compute_rel_residual = ...
        false;

end


if ~isfield(options,'RNG_SEED')

    options.RNG_SEED = ...
        1;

end


% =============================================================
% 1.05 Transfer / final-cap-selection options
%
% These options do NOT affect Stage-1 optimization.  They are removed by
% canonicalOptimizationOptions before Stage-1 compatibility comparison.
% =============================================================

TransferOptions = ...
    getTransferOptions( ...
        options);


% =============================================================
% 1.1 Optional two-stage execution control
% =============================================================

executionStage = ...
    getExecutionStage( ...
        options);

stage1CheckpointDir = ...
    getStage1CheckpointDir( ...
        options, ...
        executionStage);


saveTrace = ...
    getSaveTrace( ...
        options);


saveStage2All = ...
    getSaveStage2All( ...
        options);


% Stage 1 must retain Point(i).optimization in CORE. Full Stage-2 mode
% retains per-point detail. Compact Stage-2/all mode never allocates the
% accumulated Point array; Pi remains point-local working state only.
retainPointDetail = ...
    strcmp(executionStage,'stage1') || ...
    saveStage2All;


% Keep execution controls outside the original numerical workflow.
% The existing optimization/evaluation functions therefore receive the
% same numerical options structure as in the one-stage implementation.
coreOptions = ...
    canonicalOptimizationOptions( ...
        options);


% =============================================================
% 2. Basic input checks
% =============================================================

if ~isnumeric(X) || ...
   isempty(X) || ...
   ~ismatrix(X) || ...
   ~isreal(X) || ...
   any(~isfinite(X(:)))

    error( ...
        'X must be a nonempty finite real numeric matrix.');

end


if ~isnumeric(y) || ...
   isempty(y) || ...
   ~isvector(y) || ...
   ~isreal(y) || ...
   any(~isfinite(y(:)))

    error( ...
        'y must be a nonempty finite real numeric vector.');

end


y = ...
    y(:);


[n,p] = ...
    size(X);


if numel(y) ~= n

    error( ...
        'X and y must contain the same number of observations.');

end


if ~isnumeric(radii) || ...
   isempty(radii) || ...
   ~isvector(radii) || ...
   ~isreal(radii) || ...
   any(~isfinite(radii(:))) || ...
   any(radii(:) <= 0)

    error( ...
        'radii must be a nonempty finite positive real vector.');

end


radii = ...
    radii(:);


K = ...
    numel(radii);


if any(diff(radii) <= 0)

    error( ...
        'Candidate radii must be strictly increasing.');

end


if ~isnumeric(min_neighbors) || ...
   ~isscalar(min_neighbors) || ...
   ~isreal(min_neighbors) || ...
   ~isfinite(min_neighbors) || ...
   min_neighbors < 1 || ...
   min_neighbors ~= round(min_neighbors)

    error( ...
        'min_neighbors must be a positive integer.');

end


if n - 1 < min_neighbors

    error( ...
        ['At least %d observations must remain after target ' ...
         'exclusion, but only %d are available.'], ...
        min_neighbors, ...
        n - 1);

end


if ~isnumeric(Reg_min) || ...
   ~isscalar(Reg_min) || ...
   ~isreal(Reg_min) || ...
   ~isfinite(Reg_min) || ...
   Reg_min <= 0 || ...
   ~isnumeric(Reg_max) || ...
   ~isscalar(Reg_max) || ...
   ~isreal(Reg_max) || ...
   ~isfinite(Reg_max) || ...
   Reg_max <= 0 || ...
   Reg_min > Reg_max

    error( ...
        ['Reg_min and Reg_max must be finite positive scalars ' ...
         'with Reg_min <= Reg_max.']);

end


if ~isnumeric(tau_loc) || ...
   ~isscalar(tau_loc) || ...
   ~isreal(tau_loc) || ...
   ~isfinite(tau_loc) || ...
   tau_loc < 0

    error( ...
        'tau_loc must be a finite nonnegative real scalar.');

end


if ~isnumeric(tau_cap) || ...
   ~isscalar(tau_cap) || ...
   ~isreal(tau_cap) || ...
   ~isfinite(tau_cap) || ...
   tau_cap < 0

    error( ...
        'tau_cap must be a finite nonnegative real scalar.');

end


% =============================================================
% 3. Kernel-specific parameter checks
% =============================================================

useShape = ...
    common.kernelHasShapeParameter(kernel);


if useShape

    if ~isnumeric(C_min) || ...
       ~isscalar(C_min) || ...
       ~isreal(C_min) || ...
       ~isfinite(C_min) || ...
       C_min <= 0 || ...
       ~isnumeric(C_max) || ...
       ~isscalar(C_max) || ...
       ~isreal(C_max) || ...
       ~isfinite(C_max) || ...
       C_max <= 0 || ...
       C_min > C_max

        error( ...
            ['C_min and C_max must be finite positive scalars ' ...
             'with C_min <= C_max for shaped kernels.']);

    end

else

    C_min = ...
        NaN;


    C_max = ...
        NaN;

end


% =============================================================
% 4. Global coordinate scaling
%
% S_global is calculated from the ORIGINAL full training geometry.
% The complete local training workflow then uses scaled coordinates
% and scaled candidate radii.
% =============================================================

X_raw = ...
    X;


radii_raw = ...
    radii;


S_global = ...
    common.MeanDistant( ...
        X_raw);


if ~isscalar(S_global) || ...
   ~isreal(S_global) || ...
   ~isfinite(S_global) || ...
   S_global <= 0

    error( ...
        ['common.MeanDistant(X) must return a finite ' ...
         'positive coordinate scale.']);

end


X = ...
    X_raw ./ ...
    S_global;


radii = ...
    radii_raw ./ ...
    S_global;


if any(~isfinite(X(:))) || ...
   any(~isfinite(radii)) || ...
   any(radii <= 0) || ...
   any(diff(radii) <= 0)

    error( ...
        'Coordinate/radius scaling produced invalid values.');

end


% =============================================================
% 5. RNG initialization
%
% RNG is initialized ONCE at this training level.
% No lower-level local function should reset it.
% =============================================================

RNG_SEED = ...
    options.RNG_SEED;


if ~isempty(RNG_SEED)

    if ~isnumeric(RNG_SEED) || ...
       ~isscalar(RNG_SEED) || ...
       ~isreal(RNG_SEED) || ...
       ~isfinite(RNG_SEED) || ...
       RNG_SEED < 0 || ...
       RNG_SEED ~= round(RNG_SEED)

        error( ...
            ['options.RNG_SEED must be empty or a finite ' ...
             'nonnegative integer scalar.']);

    end


    if ~strcmp( ...
            executionStage, ...
            'stage2')

        rng( ...
            RNG_SEED, ...
            'twister');

    end

end


% =============================================================
% 6. Initialize training result
% =============================================================

TrainingResult = ...
    struct();


TrainingResult.success = ...
    false;


TrainingResult.kernel = ...
    kernel;


TrainingResult.useShape = ...
    useShape;


% Internal canonical coordinates used by all local-RBF calculations.
TrainingResult.X = ...
    X;


TrainingResult.X_scaled = ...
    X;


TrainingResult.X_raw = ...
    X_raw;


TrainingResult.S_global = ...
    S_global;


TrainingResult.y = ...
    y;


TrainingResult.nPoints = ...
    n;


TrainingResult.nDimensions = ...
    p;


% Internal canonical candidate radii are scaled.
TrainingResult.radii = ...
    radii;


TrainingResult.radii_scaled = ...
    radii;


TrainingResult.radii_raw = ...
    radii_raw;


TrainingResult.nCaps = ...
    K;


TrainingResult.minNeighbors = ...
    min_neighbors;


TrainingResult.C_min = ...
    C_min;


TrainingResult.C_max = ...
    C_max;


TrainingResult.Reg_min = ...
    Reg_min;


TrainingResult.Reg_max = ...
    Reg_max;

TrainingResult.tuningMethod = ...
    method;

TrainingResult.nGrid = ...
    nGrid;

TrainingResult.tau_loc = ...
    tau_loc;


TrainingResult.tau_cap = ...
    tau_cap;


TrainingResult.TransferOptions = ...
    TransferOptions;


TrainingResult.capRMSESource = ...
    TransferOptions.capRMSESource;


TrainingResult.RNG_SEED = ...
    RNG_SEED;


TrainingResult.options = ...
    options;


TrainingResult.coordinateScaleApplied = ...
    true;


TrainingResult.message = ...
    '';


TrainingResult.executionStage = ...
    executionStage;


TrainingResult.stage1CheckpointDir = ...
    stage1CheckpointDir;


TrainingResult.stage1CheckpointFile = ...
    '';


TrainingResult.stage1Loaded = ...
    false;


TrainingResult.stage1Complete = ...
    false;


TrainingResult.stage1TraceSaved = ...
    saveTrace;


TrainingResult.stage1WallTime = ...
    0;


TrainingResult.stage2WallTime = ...
    0;


TrainingResult.stage1Summary = ...
    struct();


% =============================================================
% 6.1 Stage-1 storage state
%
% Stage-1 DIAG/TRACE/checkpoint objects are intentionally created only
% inside the Stage-1 branch below. When TRACE is disabled, no TRACE
% metadata, writer, MAT-file handle, temporary-file list, or cleanup
% object is constructed.
% =============================================================


% =============================================================
% 7. Initialize timing
% =============================================================

tTraining = ...
    tic;


% =============================================================
% 7.1 Reuse a completed Stage-1 checkpoint when available
% =============================================================

if strcmp(executionStage,'stage1') && ...
   reuseCompletedStage1(options)

  candidateCheckpointFile = ...
    getStage1CheckpointFile( ...
        options, ...
        stage1CheckpointDir, ...
        kernel, ...
        y, ...
        radii_raw, ...
        method, ...
        nGrid);


    candidateCheckpointPaths = ...
        localrbf.stage1CheckpointPaths( ...
            candidateCheckpointFile);


    if allStage1SplitFilesExist( ...
            candidateCheckpointPaths, ...
            saveTrace)

        try

            [ExistingStage1,existingCheckpointFile] = ...
                loadStage1Source( ...
                    stage1CheckpointDir, ...
                    kernel, ...
                    X_raw, ...
                    X, ...
                    y, ...
                    radii_raw, ...
                    radii, ...
                    min_neighbors, ...
                    C_min, ...
                    C_max, ...
                    Reg_min, ...
                    Reg_max, ...
                    RNG_SEED, ...
                    options, ...
                    method, ...
                    nGrid);


            TrainingResult = ...
                ExistingStage1;


            TrainingResult.executionStage = ...
                'stage1';


            TrainingResult.stage1CheckpointDir = ...
                stage1CheckpointDir;


            TrainingResult.stage1CheckpointFile = ...
                existingCheckpointFile;


            TrainingResult.stage1Loaded = ...
                true;


            TrainingResult.message = ...
                '';


            geometryCache = ...
                [];


            return

        catch
            % Existing file is incomplete or incompatible. Recompute
            % Stage 1 and overwrite it; never reuse it silently.
        end

    end

end


% =============================================================
% 7.2 Initialize split Stage-1 storage
%
% saveTrace = true:
%   CORE + DIAG + TRACE_SEARCH + TRACE_OPT + TRACE_NUMERIC
%   TRACE payloads are written progressively to temporary v7.3 files.
%
% saveTrace = false:
%   CORE + DIAG only. No TRACE structure is constructed or written.
%
% CORE is always the final commit marker.
% =============================================================

if strcmp( ...
        executionStage, ...
        'stage1')

    checkpointBaseFile = ...
        getStage1CheckpointFile( ...
            options, ...
            stage1CheckpointDir, ...
            kernel, ...
            y, ...
            radii_raw, ...
            method, ...
            nGrid);


    checkpointPaths = ...
        localrbf.stage1CheckpointPaths( ...
            checkpointBaseFile);


    Stage1Diag = ...
        initializeStage1StoragePart( ...
            'DIAG', ...
            kernel, ...
            n, ...
            K);


    if saveTrace

        % Track temporary TRACE files only when TRACE persistence is enabled.
        % Preallocate so cleanup is safe even if the first writer fails.
        traceTempFiles = ...
            {'','',''};


        Stage1TraceSearchMeta = ...
            initializeStage1StorageMetadata( ...
                'TRACE_SEARCH', ...
                kernel, ...
                n, ...
                K);


        Stage1TraceOptMeta = ...
            initializeStage1StorageMetadata( ...
                'TRACE_OPT', ...
                kernel, ...
                n, ...
                K);


        Stage1TraceNumericMeta = ...
            initializeStage1StorageMetadata( ...
                'TRACE_NUMERIC', ...
                kernel, ...
                n, ...
                K);


        try

            TraceSearchWriter = ...
                localrbf.openStage1ProgressivePart( ...
                    checkpointPaths.traceSearch, ...
                    'Stage1TraceSearch', ...
                    Stage1TraceSearchMeta, ...
                    n);


            traceTempFiles{1} = ...
                TraceSearchWriter.tempFile;


            TraceOptWriter = ...
                localrbf.openStage1ProgressivePart( ...
                    checkpointPaths.traceOpt, ...
                    'Stage1TraceOpt', ...
                    Stage1TraceOptMeta, ...
                    n);


            traceTempFiles{2} = ...
                TraceOptWriter.tempFile;


            TraceNumericWriter = ...
                localrbf.openStage1ProgressivePart( ...
                    checkpointPaths.traceNumeric, ...
                    'Stage1TraceNumeric', ...
                    Stage1TraceNumericMeta, ...
                    n);


            traceTempFiles{3} = ...
                TraceNumericWriter.tempFile;

        catch ME

            cleanupTemporaryStage1Files( ...
                traceTempFiles);


            rethrow(ME)

        end


        traceTempCleanup = ...
            onCleanup( ...
                @() cleanupTemporaryStage1Files( ...
                    traceTempFiles)); %#ok<NASGU>


        TraceSearchMat = ...
            matfile( ...
                TraceSearchWriter.tempFile, ...
                'Writable', ...
                true);


        TraceOptMat = ...
            matfile( ...
                TraceOptWriter.tempFile, ...
                'Writable', ...
                true);


        TraceNumericMat = ...
            matfile( ...
                TraceNumericWriter.tempFile, ...
                'Writable', ...
                true);

    else

        % Prevent stale TRACE files from an earlier TRACE-ON run from
        % appearing to belong to this new TRACE-OFF checkpoint.
        deleteStage1TraceFiles( ...
            checkpointPaths);

    end


    % Invalidate any previous completed checkpoint before computation.
    if isfile(checkpointPaths.core)

        delete(checkpointPaths.core);

    end

end


% =============================================================
% 8. Build shared training geometry ONCE
%
% D and P are reused by every point and every candidate radius.
% They are intentionally not copied into TrainingResult because
% they are exactly reconstructable from X.
% =============================================================

tGeometry = ...
    tic;


geometryCache = ...
    struct();


geometryCache.D = ...
    pdist2( ...
        X, ...
        X);


geometryCache.P = ...
    [ones(n,1), ...
     X];


TrainingResult.geometryWallTime = ...
    toc(tGeometry);


TrainingResult.geometryCacheInfo = ...
    struct();


TrainingResult.geometryCacheInfo.used = ...
    true;


TrainingResult.geometryCacheInfo.distanceMatrixSize = ...
    size(geometryCache.D);


TrainingResult.geometryCacheInfo.polynomialMatrixSize = ...
    size(geometryCache.P);


TrainingResult.geometryCacheInfo.distanceMatrixStored = ...
    false;


TrainingResult.geometryCacheInfo.polynomialMatrixStored = ...
    false;


TrainingResult.geometryCacheInfo.reconstructableFromX = ...
    true;


TrainingResult.geometryCacheInfo.reconstructableFromX_scaled = ...
    true;


TrainingResult.geometryCacheInfo.S_global = ...
    S_global;


% =============================================================
% 9. Detailed per-point results
% =============================================================

if retainPointDetail

    TrainingResult.Point = ...
        repmat( ...
            emptyPointTrainingResult(), ...
            n,1);

end


% =============================================================
% 10. Initialize point x cap matrices
%
% Rows    = training points
% Columns = candidate radius caps
%
% Stage 1 stops before cap selection, so none of these matrices are
% allocated in Stage 1. The matrices retained here for Stage 2/all are
% numerical working state required for final cap selection/evaluation.
% =============================================================

if ~strcmp(executionStage,'stage1')

TrainingResult.cap_success_by_point = ...
    false(n,K);


% -------------------------------------------------------------
% Radius selection
% -------------------------------------------------------------

TrainingResult.selectedRadiusIndex_by_point_cap = ...
    nan(n,K);


TrainingResult.selectedRadius_by_point_cap = ...
    nan(n,K);


TrainingResult.selectedAE_by_point_cap = ...
    nan(n,K);


if saveStage2All

    TrainingResult.minAE_by_point_cap = ...
        nan(n,K);


    TrainingResult.errorThreshold_by_point_cap = ...
        nan(n,K);

end


% -------------------------------------------------------------
% Optimized parameters
% -------------------------------------------------------------

TrainingResult.C_by_point_cap = ...
    nan(n,K);


TrainingResult.Reg_by_point_cap = ...
    nan(n,K);


TrainingResult.epsilon_by_point_cap = ...
    nan(n,K);


TrainingResult.d_local_by_point_cap = ...
    nan(n,K);


% -------------------------------------------------------------
% Predictions and residuals
%
% residual = observation - prediction
% -------------------------------------------------------------

TrainingResult.yhat_by_point_cap = ...
    nan(n,K);


TrainingResult.residual_by_point_cap = ...
    nan(n,K);


TrainingResult.y_diff_by_point_cap = ...
    nan(n,K);


TrainingResult.squaredResidual_by_point_cap = ...
    nan(n,K);


% -------------------------------------------------------------
% Numerical information
% -------------------------------------------------------------

TrainingResult.cond_reg_by_point_cap = ...
    nan(n,K);


TrainingResult.rel_residual_by_point_cap = ...
    nan(n,K);


TrainingResult.solver_used_by_point_cap = ...
    cell(n,K);


% -------------------------------------------------------------
% Neighbourhood information
% -------------------------------------------------------------

% Exact neighbour count is compact numeric state and is retained in both
% Stage-2 modes. Variable-length neighbourhood histories and secondary
% neighbourhood diagnostics are DETAIL-only and are never allocated when
% saveStage2All=false.
TrainingResult.neighbor_count_by_point_cap = ...
    nan(n,K);


if saveStage2All

    TrainingResult.neighbor_idx_by_point_cap = ...
        cell(n,K);


    TrainingResult.neighbor_distances_by_point_cap = ...
        cell(n,K);


    TrainingResult.effective_radius_by_point_cap = ...
        nan(n,K);


    TrainingResult.selected_max_distance_by_point_cap = ...
        nan(n,K);


    TrainingResult.n_within_radius_by_point_cap = ...
        nan(n,K);


    TrainingResult.n_added_for_minimum_by_point_cap = ...
        nan(n,K);


    TrainingResult.usedMinNeighborFallback_by_point_cap = ...
        false(n,K);

end

end


% =============================================================
% 11. Point-level status/timing arrays
% =============================================================

TrainingResult.point_success = ...
    false(n,1);


TrainingResult.pointOptimizationSuccess = ...
    false(n,1);


TrainingResult.pointOptimizationWallTime = ...
    zeros(n,1);


if ~strcmp(executionStage,'stage1')

    TrainingResult.pointCapSelectionSuccess = ...
        false(n,1);


    TrainingResult.pointCapSelectionWallTime = ...
        zeros(n,1);

end


TrainingResult.pointTotalWallTime = ...
    zeros(n,1);


TrainingResult.pointMessage = ...
    cell(n,1);


% =============================================================
% 12. Aggregate search counters
% =============================================================

TrainingResult.nEvaluations = ...
    0;


TrainingResult.nGAEvaluations = ...
    0;


TrainingResult.nFallbackEvaluations = ...
    0;


TrainingResult.nGridEvaluations = ...
    0;


TrainingResult.nRefinementEvaluations = ...
    0;


TrainingResult.nValidEvaluations = ...
    0;

TrainingResult.nFailedEvaluations = ...
    0;


TrainingResult.nRBFSolves = ...
    0;


TrainingResult.nPointOptimizationExceptions = ...
    0;


if ~strcmp(executionStage,'stage1')

    TrainingResult.nPointCapSelectionExceptions = ...
        0;

end


% =============================================================
% 12.1 Load Stage-1 source when requested
% =============================================================

Stage1Source = ...
    [];


if strcmp( ...
        executionStage, ...
        'stage2')

    [Stage1Source,stage1CheckpointFile] = ...
        loadStage1Source( ...
            stage1CheckpointDir, ...
            kernel, ...
            X_raw, ...
            X, ...
            y, ...
            radii_raw, ...
            radii, ...
            min_neighbors, ...
            C_min, ...
            C_max, ...
            Reg_min, ...
            Reg_max, ...
            RNG_SEED, ...
            options, ...
            method, ...
            nGrid);


    TrainingResult.stage1CheckpointFile = ...
        stage1CheckpointFile;


    TrainingResult.stage1Loaded = ...
        true;


    TrainingResult.stage1Complete = ...
        true;


    TrainingResult.stage1TraceSaved = ...
        logical( ...
            getFieldOrDefault( ...
                Stage1Source, ...
                'stage1TraceSaved', ...
                true));


    TrainingResult.stage1WallTime = ...
        getNumericScalarOrDefault( ...
            Stage1Source, ...
            'stage1WallTime', ...
            getNumericScalarOrDefault( ...
                Stage1Source, ...
                'totalWallTime', ...
                0));


    TrainingResult.stage1Summary.nStage1PointsComplete = ...
        getNumericScalarOrDefault( ...
            Stage1Source, ...
            'nStage1PointsComplete', ...
            n);


    TrainingResult.stage1Summary.nStage1PointsMissing = ...
        getNumericScalarOrDefault( ...
            Stage1Source, ...
            'nStage1PointsMissing', ...
            0);


    TrainingResult.stage1Summary.nSuccessfulPointOptimizations = ...
        getNumericScalarOrDefault( ...
            Stage1Source, ...
            'nSuccessfulPointOptimizations', ...
            NaN);


    TrainingResult.stage1Summary.nUnsuccessfulPointOptimizations = ...
        getNumericScalarOrDefault( ...
            Stage1Source, ...
            'nUnsuccessfulPointOptimizations', ...
            NaN);


    TrainingResult.nPointOptimizationExceptions = ...
        getNumericScalarOrDefault( ...
            Stage1Source, ...
            'nPointOptimizationExceptions', ...
            0);

end


% =============================================================
% 13. Process every training point
%
% Serial here by design.
% Parallelism is intended at the outer model x kernel level.
% =============================================================

for iTarget = 1:n

    tPoint = ...
        tic;


    Pi = ...
        emptyPointTrainingResult();


    Pi.iTarget = ...
        iTarget;


    Pi.xTarget = ...
        X(iTarget,:);


    Pi.yTarget = ...
        y(iTarget);


    % =========================================================
    % 13.1 Obtain all-radius optimization for this point
    %
    % Stage 1 / all: execute the existing optimizer.
    % Stage 2      : load the retained PointOpt only.
    % =========================================================

    if strcmp( ...
            executionStage, ...
            'stage2')

        PointOpt = ...
            Stage1Source.Point(iTarget).optimization;


        % Stage-1 CORE remains the authoritative PointOpt record.
        % Do not duplicate it inside the Stage-2 TrainingResult.
        Pi.optimizationSuccess = ...
            Stage1Source.Point(iTarget).optimizationSuccess;


        Pi.optimizationWallTime = ...
            Stage1Source.Point(iTarget).optimizationWallTime;


        TrainingResult.pointOptimizationWallTime(iTarget) = ...
            Pi.optimizationWallTime;

    else

        tPointOptimization = ...
            tic;


        try

            PointOpt = ...
                localrbf.optimizeAtPointForAllCandidateRadii( ...
                    X, ...
                    y, ...
                    iTarget, ...
                    radii, ...
                    kernel, ...
                    min_neighbors, ...
                    C_min, ...
                    C_max, ...
                    Reg_min, ...
                    Reg_max, ...
                    coreOptions, ...
                    geometryCache, ...
                    method, ...
                    nGrid);


        catch ME

            PointOpt = ...
                [];


            Pi.optimization = ...
                [];


            Pi.optimizationSuccess = ...
                false;


            Pi.failureStage = ...
                'point optimization';


            Pi.message = ...
                ME.message;


            TrainingResult.nPointOptimizationExceptions = ...
                TrainingResult.nPointOptimizationExceptions + 1;

        end


        % Optimization timing measures only the numerical/search work.
        % Storage splitting is deliberately excluded from this timing.
        Pi.optimizationWallTime = ...
            toc(tPointOptimization);


        TrainingResult.pointOptimizationWallTime(iTarget) = ...
            Pi.optimizationWallTime;


        % -----------------------------------------------------
        % Detach Stage-1 storage payload only after a successful
        % optimizer return. A storage-layout failure is fatal and
        % must never be misreported as a numerical optimization
        % failure or silently converted into a missing point.
        % -----------------------------------------------------

        if ~isempty(PointOpt)

            if strcmp( ...
                    executionStage, ...
                    'stage1')

                try

                    if saveTrace

                        [PointCore, ...
                         PointDiag, ...
                         PointTraceSearch, ...
                         PointTraceOpt, ...
                         PointTraceNumeric] = ...
                            localrbf.splitStage1PointOpt( ...
                                PointOpt, ...
                                true);

                    else

                        [PointCore,PointDiag] = ...
                            localrbf.splitStage1PointOpt( ...
                                PointOpt, ...
                                false);

                    end

                catch ME

                    error( ...
                        'localrbf:Stage1StorageSplitFailed', ...
                        ['Stage-1 storage split failed at target %d: %s'], ...
                        iTarget, ...
                        ME.message);

                end


                % ---------------------------------------------
                % CORE is the authoritative Stage-2 source.
                % Keep only compact PointOpt information in
                % TrainingResult; store heavy histories separately.
                % ---------------------------------------------

                Pi.optimization = ...
                    PointCore;


                Stage1Diag.Point{iTarget} = ...
                    PointDiag;


                % ---------------------------------------------
                % Write the three potentially large traces NOW.
                % They are never accumulated across training points
                % in MATLAB workspace memory.
                % ---------------------------------------------

                if saveTrace

                    TraceSearchMat.Point(iTarget,1) = ...
                        {PointTraceSearch};


                    TraceOptMat.Point(iTarget,1) = ...
                        {PointTraceOpt};


                    TraceNumericMat.Point(iTarget,1) = ...
                        {PointTraceNumeric};

                end


                % ---------------------------------------------
                % All subsequent Stage-1 bookkeeping uses only
                % CORE fields/counters. Rebind PointOpt so the
                % monolithic optimizer payload is no longer needed.
                % ---------------------------------------------

                PointOpt = ...
                    PointCore;


                clear PointCore PointDiag

                if saveTrace

                    clear PointTraceSearch ...
                          PointTraceOpt ...
                          PointTraceNumeric

                end

            else

                % In one-pass mode the full PointOpt history is optional
                % Stage-2 DETAIL. Do not retain it unless requested.
                if saveStage2All

                    Pi.optimization = ...
                        PointOpt;

                end

            end


            Pi.optimizationSuccess = ...
                PointOpt.success;

        end

    end


    % =========================================================
    % 12.2 Point optimization exception
    % =========================================================

    if isempty(PointOpt)

        Pi.totalWallTime = ...
            pointTotalElapsed( ...
                executionStage, ...
                Pi.optimizationWallTime, ...
                tPoint);


        if retainPointDetail

            TrainingResult.Point(iTarget) = ...
                Pi;

        end


        TrainingResult.pointTotalWallTime(iTarget) = ...
            Pi.totalWallTime;


        TrainingResult.pointMessage{iTarget} = ...
            Pi.message;


        continue

    end


    % =========================================================
    % 12.3 Aggregate optimization counts
    % =========================================================

    TrainingResult.nEvaluations = ...
        TrainingResult.nEvaluations + ...
        PointOpt.nEvaluations;


    TrainingResult.nGAEvaluations = ...
        TrainingResult.nGAEvaluations + ...
        PointOpt.nGAEvaluations;


TrainingResult.nFallbackEvaluations = ...
    TrainingResult.nFallbackEvaluations + ...
    PointOpt.nFallbackEvaluations;


TrainingResult.nGridEvaluations = ...
    TrainingResult.nGridEvaluations + ...
    PointOpt.nGridEvaluations;


TrainingResult.nRefinementEvaluations = ...
    TrainingResult.nRefinementEvaluations + ...
    PointOpt.nRefinementEvaluations;

    TrainingResult.nValidEvaluations = ...
        TrainingResult.nValidEvaluations + ...
        PointOpt.nValidEvaluations;


    TrainingResult.nFailedEvaluations = ...
        TrainingResult.nFailedEvaluations + ...
        PointOpt.nFailedEvaluations;


    TrainingResult.nRBFSolves = ...
        TrainingResult.nRBFSolves + ...
        PointOpt.nRBFSolves;


    % =========================================================
    % 13.4 Stage 1 stops here
    % =========================================================

    if strcmp( ...
            executionStage, ...
            'stage1')

        Pi.success = ...
            Pi.optimizationSuccess;


        if Pi.success

            Pi.failureStage = ...
                '';


            Pi.message = ...
                '';

        elseif isempty(Pi.message) && ...
               isfield(PointOpt,'message')

            Pi.message = ...
                PointOpt.message;

        end


        Pi.totalWallTime = ...
            pointTotalElapsed( ...
                executionStage, ...
                Pi.optimizationWallTime, ...
                tPoint);


        if retainPointDetail

            TrainingResult.Point(iTarget) = ...
                Pi;

        end


        TrainingResult.point_success(iTarget) = ...
            Pi.success;


        TrainingResult.pointOptimizationSuccess(iTarget) = ...
            Pi.optimizationSuccess;


        TrainingResult.pointTotalWallTime(iTarget) = ...
            Pi.totalWallTime;


        TrainingResult.pointMessage{iTarget} = ...
            Pi.message;


        continue

    end


    % =========================================================
    % 12.4 Select best radius under EVERY candidate cap
    %
    % This reuses the already optimized radius results.
    % No GA or RBF solve occurs here.
    % =========================================================

    tCapSelection = ...
        tic;


    try

        PointCaps = ...
            localrbf. ...
            selectBestRadiusWithToleranceAtEachRadiusCapOverAllCaps( ...
                PointOpt, ...
                tau_loc, ...
                saveStage2All);


        if saveStage2All

            Pi.capSelection = ...
                PointCaps;

        end


        Pi.capSelectionSuccess = ...
            PointCaps.success;


    catch ME

        PointCaps = ...
            [];


        Pi.capSelectionSuccess = ...
            false;


        Pi.failureStage = ...
            'point cap selection';


        Pi.message = ...
            ME.message;


        TrainingResult.nPointCapSelectionExceptions = ...
            TrainingResult.nPointCapSelectionExceptions + 1;

    end


    Pi.capSelectionWallTime = ...
        toc(tCapSelection);


    TrainingResult.pointCapSelectionWallTime(iTarget) = ...
        Pi.capSelectionWallTime;


    % =========================================================
    % 12.5 Cap-selection exception
    % =========================================================

    if isempty(PointCaps)

        Pi.totalWallTime = ...
            pointTotalElapsed( ...
                executionStage, ...
                Pi.optimizationWallTime, ...
                tPoint);


        if retainPointDetail

            TrainingResult.Point(iTarget) = ...
                Pi;

        end


        TrainingResult.pointOptimizationSuccess(iTarget) = ...
            Pi.optimizationSuccess;


        TrainingResult.pointTotalWallTime(iTarget) = ...
            Pi.totalWallTime;


        TrainingResult.pointMessage{iTarget} = ...
            Pi.message;


        continue

    end


    % =========================================================
    % 12.6 Copy compact point-cap information
    %
    % Detailed optimization and selection structures are retained in
    % TrainingResult.Point(iTarget) only when requested.
    % =========================================================

    TrainingResult.cap_success_by_point(iTarget,:) = ...
        reshape( ...
            PointCaps.cap_success, ...
            1,K);


    % ---------------------------------------------------------
    % Radius
    % ---------------------------------------------------------

    TrainingResult.selectedRadiusIndex_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.selectedRadiusIndex_all, ...
            1,K);


    TrainingResult.selectedRadius_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.selectedRadius_all, ...
            1,K);


    TrainingResult.selectedAE_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.selectedAE_all, ...
            1,K);


    if saveStage2All

        TrainingResult.minAE_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.minAE_all, ...
                1,K);


        TrainingResult.errorThreshold_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.errorThreshold_all, ...
                1,K);

    end


    % ---------------------------------------------------------
    % Parameters
    % ---------------------------------------------------------

    TrainingResult.C_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.C_selected_all, ...
            1,K);


    TrainingResult.Reg_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.Reg_selected_all, ...
            1,K);


    TrainingResult.epsilon_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.epsilon_selected_all, ...
            1,K);


    TrainingResult.d_local_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.d_local_selected_all, ...
            1,K);


    % ---------------------------------------------------------
    % Prediction and residual
    % ---------------------------------------------------------

    TrainingResult.yhat_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.yhat_selected_all, ...
            1,K);


    TrainingResult.residual_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.residual_selected_all, ...
            1,K);


    TrainingResult.y_diff_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.y_diff_selected_all, ...
            1,K);


    TrainingResult.squaredResidual_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.squaredResidual_selected_all, ...
            1,K);


    % ---------------------------------------------------------
    % Numerical information
    % ---------------------------------------------------------

    TrainingResult.cond_reg_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.cond_reg_selected_all, ...
            1,K);


    TrainingResult.rel_residual_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.rel_residual_selected_all, ...
            1,K);


    TrainingResult.solver_used_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.solver_used_selected_all, ...
            1,K);


    % ---------------------------------------------------------
    % Neighbourhood information
    % ---------------------------------------------------------

    TrainingResult.neighbor_count_by_point_cap(iTarget,:) = ...
        reshape( ...
            PointCaps.neighbor_count_selected_all, ...
            1,K);


    if saveStage2All

        TrainingResult.neighbor_idx_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.neighbor_idx_selected_all, ...
                1,K);


        TrainingResult.neighbor_distances_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.neighbor_distances_selected_all, ...
                1,K);


        TrainingResult.effective_radius_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.effective_radius_selected_all, ...
                1,K);


        TrainingResult.selected_max_distance_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.selected_max_distance_all, ...
                1,K);


        TrainingResult.n_within_radius_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.n_within_radius_selected_all, ...
                1,K);


        TrainingResult.n_added_for_minimum_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.n_added_for_minimum_selected_all, ...
                1,K);


        TrainingResult.usedMinNeighborFallback_by_point_cap(iTarget,:) = ...
            reshape( ...
                PointCaps.usedMinNeighborFallback_selected_all, ...
                1,K);

    end


    % =========================================================
    % 12.7 Final point status
    % =========================================================

    Pi.success = ...
        PointCaps.success;


    if Pi.success

        Pi.failureStage = ...
            '';


        Pi.message = ...
            '';

    elseif isempty(Pi.message)

        Pi.failureStage = ...
            'point cap selection';


        Pi.message = ...
            PointCaps.message;

    end


    Pi.totalWallTime = ...
        pointTotalElapsed( ...
            executionStage, ...
            Pi.optimizationWallTime, ...
            tPoint);


    if retainPointDetail

        TrainingResult.Point(iTarget) = ...
            Pi;

    end


    TrainingResult.point_success(iTarget) = ...
        Pi.success;


    TrainingResult.pointOptimizationSuccess(iTarget) = ...
        Pi.optimizationSuccess;


    TrainingResult.pointCapSelectionSuccess(iTarget) = ...
        Pi.capSelectionSuccess;


    TrainingResult.pointTotalWallTime(iTarget) = ...
        Pi.totalWallTime;


    TrainingResult.pointMessage{iTarget} = ...
        Pi.message;

end


% Stage 2 has now consumed all loaded PointOpt records. Keep the
% authoritative Stage-1 CORE on disk and release its in-memory copy before
% full-set cap selection/evaluation.
if strcmp(executionStage,'stage2')

    clear Stage1Source PointOpt PointCaps

end


% =============================================================
% 13.9 Stage-1 completion, checkpoint, and return
% =============================================================

if strcmp( ...
        executionStage, ...
        'stage1')

    havePointOpt = ...
        false(n,1);


    for iTarget = 1:n

        havePointOpt(iTarget) = ...
            isfield( ...
                TrainingResult.Point(iTarget), ...
                'optimization') && ...
            ~isempty( ...
                TrainingResult.Point(iTarget).optimization);

    end


    TrainingResult.stage1Complete = ...
        all(havePointOpt);


    TrainingResult.nStage1PointsComplete = ...
        nnz(havePointOpt);


    TrainingResult.nStage1PointsMissing = ...
        n - ...
        TrainingResult.nStage1PointsComplete;


    TrainingResult.nSuccessfulPointOptimizations = ...
        nnz( ...
            TrainingResult.pointOptimizationSuccess);


    TrainingResult.nUnsuccessfulPointOptimizations = ...
        n - ...
        TrainingResult.nSuccessfulPointOptimizations;


    TrainingResult.success = ...
        TrainingResult.stage1Complete;


    if TrainingResult.success

        TrainingResult.message = ...
            '';

    else

        TrainingResult.message = ...
            sprintf( ...
                ['Stage 1 is incomplete: %d of %d PointOpt ' ...
                 'records are missing.'], ...
                TrainingResult.nStage1PointsMissing, ...
                n);

    end


    TrainingResult.stage1WallTime = ...
        toc(tTraining);


    TrainingResult.coreTrainingWallTime = ...
        TrainingResult.stage1WallTime;


    TrainingResult.totalWallTime = ...
        TrainingResult.stage1WallTime;


    TrainingResult.EvaluationResult = ...
        [];


    TrainingResult.evaluationSuccess = ...
        false;


    TrainingResult.evaluationWallTime = ...
        0;


    TrainingResult.nRBFSolvesTraining = ...
        TrainingResult.nRBFSolves;


    TrainingResult.nRBFSolvesEvaluation = ...
        0;


    TrainingResult.nRBFSolvesTotal = ...
        TrainingResult.nRBFSolves;

    TrainingResult.stage1CheckpointBaseFile = ...
        checkpointBaseFile;


    TrainingResult.stage1CheckpointFile = ...
        checkpointPaths.core;


    TrainingResult.stage1CheckpointPaths = ...
        checkpointPaths;


    TrainingResult.stage1TraceSaved = ...
        saveTrace;


    if saveTrace

        TrainingResult.stage1StorageLayout = ...
            'split-v3-progressive-trace';

    else

        TrainingResult.stage1StorageLayout = ...
            'split-v3-core-diag';

    end


    TrainingResult.stage1Summary.checkpointPaths = ...
        checkpointPaths;


    TrainingResult.stage1Summary.traceSaved = ...
        saveTrace;


    % ---------------------------------------------------------
    % Finalize DIAG metadata.
    % ---------------------------------------------------------

    Stage1Diag = ...
        finalizeStage1StoragePart( ...
            Stage1Diag, ...
            TrainingResult, ...
            method, ...
            nGrid);


    % ---------------------------------------------------------
    % TRACE ON: finalize already-written progressive trace files.
    % TRACE OFF: this entire block is skipped; no trace I/O occurs.
    % ---------------------------------------------------------

    if saveTrace

        Stage1TraceSearchMeta = ...
            finalizeStage1StoragePart( ...
                Stage1TraceSearchMeta, ...
                TrainingResult, ...
                method, ...
                nGrid);


        Stage1TraceOptMeta = ...
            finalizeStage1StoragePart( ...
                Stage1TraceOptMeta, ...
                TrainingResult, ...
                method, ...
                nGrid);


        Stage1TraceNumericMeta = ...
            finalizeStage1StoragePart( ...
                Stage1TraceNumericMeta, ...
                TrainingResult, ...
                method, ...
                nGrid);


        clear TraceSearchMat ...
              TraceOptMat ...
              TraceNumericMat


        localrbf.finalizeStage1ProgressivePart( ...
            TraceSearchWriter, ...
            Stage1TraceSearchMeta);


        localrbf.finalizeStage1ProgressivePart( ...
            TraceOptWriter, ...
            Stage1TraceOptMeta);


        localrbf.finalizeStage1ProgressivePart( ...
            TraceNumericWriter, ...
            Stage1TraceNumericMeta);


        clear Stage1TraceSearchMeta ...
              Stage1TraceOptMeta ...
              Stage1TraceNumericMeta ...
              TraceSearchWriter ...
              TraceOptWriter ...
              TraceNumericWriter


        % Destroying the cleanup object now is safe: the temporary files
        % have already been atomically moved to their final TRACE names.
        traceTempCleanup = ...
            [];

    end


    % ---------------------------------------------------------
    % DIAG is always saved.
    % ---------------------------------------------------------

    localrbf.saveStage1PartAtomic( ...
        checkpointPaths.diag, ...
        'Stage1Diag', ...
        Stage1Diag);


    clear Stage1Diag


    % ---------------------------------------------------------
    % Publish CORE LAST and only for a complete Stage 1.
    % ---------------------------------------------------------

    if TrainingResult.stage1Complete

        Stage1Core = ...
            struct();


        Stage1Core.storageSplitVersion = ...
            2;


        Stage1Core.storageRole = ...
            'CORE';


        Stage1Core.complete = ...
            true;


        Stage1Core.traceSaved = ...
            saveTrace;


        if saveTrace

            Stage1Core.storageMode = ...
                'CORE+DIAG+TRACE';

        else

            Stage1Core.storageMode = ...
                'CORE+DIAG';

        end


        Stage1Core.TrainingResult = ...
            TrainingResult;


        localrbf.saveStage1PartAtomic( ...
            checkpointPaths.core, ...
            'Stage1Core', ...
            Stage1Core);


        clear Stage1Core

    else

        if isfile(checkpointPaths.core)

            delete(checkpointPaths.core);

        end

    end


    return

end


% =============================================================
% 13. Training-set residual availability for every cap
%
% A point contributes only when:
%
%   1. its cap selection succeeded;
%   2. its residual is finite;
%   3. its squared residual is finite.
% =============================================================

validResidualMask = ...
    TrainingResult.cap_success_by_point & ...
    isfinite(TrainingResult.residual_by_point_cap) & ...
    isfinite(TrainingResult.squaredResidual_by_point_cap);


TrainingResult.validResidualMask = ...
    validResidualMask;


TrainingResult.nValidPointsPerCap = ...
    sum( ...
        validResidualMask, ...
        1);


TrainingResult.nFailedPointsPerCap = ...
    n - ...
    TrainingResult.nValidPointsPerCap;


% =============================================================
% 14. RMSE for every cap
%
% RMSE_available_all:
%   diagnostic RMSE using all AVAILABLE valid points.
%
% RMSE_all:
%   official cap-selection RMSE, populated ONLY when every
%   training point is valid for that cap.
%
% This retains incomplete-cap information without allowing
% incomparable sample sizes to affect final cap selection.
% =============================================================

TrainingResult.SSE_available_all = ...
    nan(1,K);


TrainingResult.MSE_available_all = ...
    nan(1,K);


TrainingResult.RMSE_available_all = ...
    nan(1,K);


TrainingResult.SSE_all = ...
    nan(1,K);


TrainingResult.MSE_all = ...
    nan(1,K);


TrainingResult.RMSE_all = ...
    nan(1,K);


for kCap = 1:K

    valid = ...
        validResidualMask(:,kCap);


    nValid = ...
        TrainingResult.nValidPointsPerCap(kCap);


    if nValid > 0

        squaredResidual = ...
            TrainingResult. ...
            squaredResidual_by_point_cap( ...
                valid, ...
                kCap);


        SSE_available = ...
            sum( ...
                squaredResidual);


        MSE_available = ...
            SSE_available / ...
            nValid;


        RMSE_available = ...
            sqrt( ...
                MSE_available);


        TrainingResult.SSE_available_all(kCap) = ...
            SSE_available;


        TrainingResult.MSE_available_all(kCap) = ...
            MSE_available;


        TrainingResult.RMSE_available_all(kCap) = ...
            RMSE_available;

    end


    % ---------------------------------------------------------
    % Official comparable RMSE requires the complete train set.
    % ---------------------------------------------------------

    if nValid == n

        TrainingResult.SSE_all(kCap) = ...
            TrainingResult.SSE_available_all(kCap);


        TrainingResult.MSE_all(kCap) = ...
            TrainingResult.MSE_available_all(kCap);


        TrainingResult.RMSE_all(kCap) = ...
            TrainingResult.RMSE_available_all(kCap);

    end

end


% =============================================================
% 15. Valid cap status over complete training set
% =============================================================

TrainingResult.cap_success = ...
    TrainingResult.nValidPointsPerCap == n;


TrainingResult.nSuccessfulCaps = ...
    sum( ...
        TrainingResult.cap_success);


TrainingResult.nFailedCaps = ...
    K - ...
    TrainingResult.nSuccessfulCaps;


% =============================================================
% 15.1 Resolve RMSE source used ONLY for final cap selection
%
% Original training RMSE is ALWAYS preserved in:
%
%       TrainingResult.RMSE_all
%
% capRMSESource='training':
%       historical production behavior.
%
% capRMSESource='pseudo':
%       calculate STRICT pseudo-training RMSE internally for every
%       candidate radius cap using the SAME transfer configuration:
%
%           method
%           nIDWSources
%           nearThresholdRaw
%           parameterTransferMode
%
% Strict pseudo rule for target i:
%   - target i is excluded from transfer donors;
%   - target i is excluded from the fitting observations;
%   - settings are transferred from the other training points only;
%   - prediction is made at target i as an unseen query;
%   - ensemble uses the transferred unique complete tuples and requires
%     every tuple solve/prediction to succeed;
%   - official pseudo RMSE for a cap is available only when EVERY
%     training target has a valid strict pseudo prediction.
%
% No parameter optimization is repeated.
% =============================================================

TrainingResult.pseudoTrainingWallTime = ...
    0;


% Scientific provenance of the efficient pseudo-unseen definition.
% Stage-1 donor settings are reused; this is NOT a nested leave-one-out
% re-optimization.  For each pseudo target, the target itself is excluded
% from both the setting-transfer donor set and the local fitting set.
TrainingResult.pseudoTrainingComputed = ...
    false;


TrainingResult.pseudoTrainingDefinition = ...
    'target-excluded-transfer-and-fit-with-fixed-stage1-donor-settings';


TrainingResult.pseudoTrainingTargetExcludedFromTransfer = ...
    true;


TrainingResult.pseudoTrainingTargetExcludedFromFit = ...
    true;


TrainingResult.pseudoTrainingNestedReoptimization = ...
    false;


TrainingResult.nRBFSolvesPseudo = ...
    0;


TrainingResult.PseudoSSE_available_all = ...
    nan(1,K);


TrainingResult.PseudoMSE_available_all = ...
    nan(1,K);


TrainingResult.PseudoRMSE_available_all = ...
    nan(1,K);


TrainingResult.PseudoRMSE_all = ...
    nan(1,K);


TrainingResult.pseudo_cap_success = ...
    false(1,K);


TrainingResult.nPseudoValidPointsPerCap = ...
    zeros(1,K);


TrainingResult.nPseudoFailedPointsPerCap = ...
    n .* ones(1,K);


TrainingResult.nPseudoTransferFailures = ...
    0;


TrainingResult.nPseudoNeighborhoodFailures = ...
    0;


TrainingResult.nPseudoPolynomialFailures = ...
    0;


TrainingResult.nPseudoSolveFailures = ...
    0;


TrainingResult.nPseudoPredictionFailures = ...
    0;


% Final selected-cap strict pseudo-unseen pointwise state.
% These compact vectors are populated only when capRMSESource='pseudo'.
% They are retained so bootstrap/calibration can reuse the exact pseudo
% settings and residuals without repeating transfer or fitting.
TrainingResult.PseudoFinal = ...
    struct();


switch TransferOptions.capRMSESource

    case 'training'

        RMSE_for_cap_selection = ...
            TrainingResult.RMSE_all;


        success_for_cap_selection = ...
            TrainingResult.cap_success;


    case 'pseudo'

        tPseudoTraining = ...
            tic;


        PseudoResult = ...
            calculateStrictPseudoTrainingRMSELocal( ...
                TrainingResult, ...
                X, ...
                y, ...
                kernel, ...
                useShape, ...
                min_neighbors, ...
                TransferOptions, ...
                geometryCache, ...
                coreOptions, ...
                S_global);


        TrainingResult.pseudoTrainingWallTime = ...
            toc(tPseudoTraining);


        TrainingResult.pseudoTrainingComputed = ...
            true;


        TrainingResult.PseudoSSE_available_all = ...
            PseudoResult.SSE_available_all;


        TrainingResult.PseudoMSE_available_all = ...
            PseudoResult.MSE_available_all;


        TrainingResult.PseudoRMSE_available_all = ...
            PseudoResult.RMSE_available_all;


        TrainingResult.PseudoRMSE_all = ...
            PseudoResult.RMSE_all;


        TrainingResult.pseudo_cap_success = ...
            PseudoResult.cap_success;


        TrainingResult.nPseudoValidPointsPerCap = ...
            PseudoResult.nValidPointsPerCap;


        TrainingResult.nPseudoFailedPointsPerCap = ...
            PseudoResult.nFailedPointsPerCap;


        TrainingResult.nRBFSolvesPseudo = ...
            PseudoResult.nRBFSolves;


        TrainingResult.nPseudoTransferFailures = ...
            PseudoResult.nTransferFailures;


        TrainingResult.nPseudoNeighborhoodFailures = ...
            PseudoResult.nNeighborhoodFailures;


        TrainingResult.nPseudoPolynomialFailures = ...
            PseudoResult.nPolynomialFailures;


        TrainingResult.nPseudoSolveFailures = ...
            PseudoResult.nSolveFailures;


        TrainingResult.nPseudoPredictionFailures = ...
            PseudoResult.nPredictionFailures;


        RMSE_for_cap_selection = ...
            TrainingResult.PseudoRMSE_all;


        % A pseudo-valid cap must also contain complete valid original
        % point-specific settings because those settings are what Final
        % subsequently extracts.
        success_for_cap_selection = ...
            TrainingResult.pseudo_cap_success & ...
            TrainingResult.cap_success;


    otherwise

        error( ...
            'Unsupported capRMSESource: %s', ...
            TransferOptions.capRMSESource);

end


TrainingResult.capRMSE_all = ...
    reshape( ...
        RMSE_for_cap_selection, ...
        1,K);


TrainingResult.capRMSE_success = ...
    reshape( ...
        logical(success_for_cap_selection), ...
        1,K);


if strcmp( ...
        TransferOptions.capRMSESource, ...
        'pseudo')

    TrainingResult.pseudo_cap_success_effective = ...
        TrainingResult.capRMSE_success;

else

    TrainingResult.pseudo_cap_success_effective = ...
        false(1,K);

end


% =============================================================
% 16. Select final radius cap using the requested RMSE source
%
% The generic tau_cap selector itself is unchanged.
% =============================================================

tFinalCapSelection = ...
    tic;


FinalCapSelection = ...
    localrbf.selectBestRadiusCapWithToleranceOverTrainSet( ...
        radii, ...
        TrainingResult.capRMSE_all, ...
        TrainingResult.capRMSE_success, ...
        tau_cap);


TrainingResult.finalCapSelectionWallTime = ...
    toc(tFinalCapSelection);


TrainingResult.capSelection = ...
    FinalCapSelection;


% =============================================================
% 17. Initialize final selected settings
% =============================================================

Final = ...
    struct();


Final.success = ...
    false;


Final.capIndex = ...
    NaN;


Final.capRadius = ...
    NaN;


Final.RMSE = ...
    NaN;


Final.minRMSE = ...
    NaN;


Final.RMSEThreshold = ...
    NaN;


Final.tau_cap = ...
    tau_cap;


Final.capRMSESource = ...
    TransferOptions.capRMSESource;


Final.transferMethodForCapRMSE = ...
    TransferOptions.method;


Final.nIDWSourcesForCapRMSE = ...
    TransferOptions.nIDWSources;


Final.nearThresholdRawForCapRMSE = ...
    TransferOptions.nearThresholdRaw;


if any(strcmp( ...
        TransferOptions.method, ...
        {'nearest','ensemble'}))

    Final.parameterTransferModeForCapRMSE = ...
        'not-applicable';

else

    Final.parameterTransferModeForCapRMSE = ...
        TransferOptions.parameterTransferMode;

end


Final.capSelectionRMSE = ...
    NaN;


Final.capSelectionMinRMSE = ...
    NaN;


Final.capSelectionRMSEThreshold = ...
    NaN;


% Final.RMSE remains the ORIGINAL training RMSE at the selected cap.
% This preserves the historical meaning used by training/evaluation
% consistency checks even when cap selection itself uses pseudo RMSE.
Final.point_success = ...
    false(n,1);


Final.radiusIndex = ...
    nan(n,1);


Final.radius = ...
    nan(n,1);


Final.C = ...
    nan(n,1);


Final.reg_param = ...
    nan(n,1);


Final.epsilon = ...
    nan(n,1);


Final.d_local = ...
    nan(n,1);


Final.AE = ...
    nan(n,1);


Final.observation = ...
    y;


Final.yhat = ...
    nan(n,1);


Final.residual = ...
    nan(n,1);


Final.y_diff = ...
    nan(n,1);


Final.squaredResidual = ...
    nan(n,1);


Final.cond_reg = ...
    nan(n,1);


Final.rel_residual = ...
    nan(n,1);


Final.solver_used = ...
    cell(n,1);


% Exact final neighbour count is compulsory Stage-2 state.
Final.neighbor_count = ...
    nan(n,1);


% Variable-length and secondary final-neighbourhood detail is retained only
% in full Stage-2 mode. Compact mode reconstructs the final neighbourhood
% transiently inside evaluateLocal(...,'final') and never stores it here.
if saveStage2All

    Final.neighbor_idx = ...
        cell(n,1);


    Final.neighbor_distances = ...
        cell(n,1);


    Final.effectiveRadius = ...
        nan(n,1);


    Final.selectedMaxDistance = ...
        nan(n,1);


    Final.nWithinRadius = ...
        nan(n,1);


    Final.nAddedForMinimum = ...
        nan(n,1);


    Final.usedMinNeighborFallback = ...
        false(n,1);

end


Final.RMSE_recomputed = ...
    NaN;


Final.RMSEDifference = ...
    NaN;


Final.RMSEConsistencyTolerance = ...
    NaN;


Final.RMSEConsistent = ...
    false;


Final.message = ...
    '';


% =============================================================
% 18. Extract final point-specific settings
% =============================================================

if FinalCapSelection.success

    kStar = ...
        FinalCapSelection.capIndex;


    Final.capIndex = ...
        kStar;


    % ---------------------------------------------------------
    % Preserve exact strict pseudo-unseen pointwise state at the
    % finally selected cap.  This is the authoritative source for
    % subsequent pseudo bootstrap/calibration; no transfer is repeated.
    % ---------------------------------------------------------
    if strcmp(TransferOptions.capRMSESource,'pseudo') && ...
       exist('PseudoResult','var') && isstruct(PseudoResult)

        PF = struct();
        PF.PointSuccess = PseudoResult.point_success_by_point_cap(:,kStar);
        PF.Radius = PseudoResult.radius_by_point_cap(:,kStar);
        PF.RegParam = PseudoResult.reg_by_point_cap(:,kStar);
        PF.C = PseudoResult.C_by_point_cap(:,kStar);
        PF.NeighborCount = PseudoResult.neighbor_count_by_point_cap(:,kStar);
        PF.Prediction = PseudoResult.prediction_by_point_cap(:,kStar);
        PF.Residual = PseudoResult.residual_by_point_cap(:,kStar);
        PF.cond_raw = PseudoResult.cond_raw_by_point_cap(:,kStar);
        PF.cond_reg = PseudoResult.cond_reg_by_point_cap(:,kStar);
        PF.cond_effective = PseudoResult.cond_effective_by_point_cap(:,kStar);
        PF.rel_residual = PseudoResult.rel_residual_by_point_cap(:,kStar);
        PF.backward_error = PseudoResult.backward_error_by_point_cap(:,kStar);
        PF.solver_used = PseudoResult.solver_used_by_point_cap(:,kStar);
        PF.ScalarPointwiseParametersAvailable = ...
            all(PF.PointSuccess) && ...
            all(isfinite(PF.Radius)) && ...
            all(isfinite(PF.RegParam)) && ...
            (~useShape || all(isfinite(PF.C)));
        PF.TransferMethod = TransferOptions.method;
        PF.ParameterTransferMode = TransferOptions.parameterTransferMode;
        PF.CapIndex = kStar;
        PF.CapRadiusRaw = radii_raw(kStar);
        TrainingResult.PseudoFinal = PF;

        % Full nTrain-by-nCaps pseudo state is no longer needed after the
        % selected-cap vectors above have been copied into PseudoFinal.
        % Release it early to reduce peak memory during the remainder of
        % Stage 2 without losing any retained result.
        clear PseudoResult

    end


    Final.capRadius = ...
        FinalCapSelection.capRadius;


    % ---------------------------------------------------------
    % Cap-selection metric
    % ---------------------------------------------------------

    Final.capSelectionRMSE = ...
        FinalCapSelection.RMSE;


    Final.capSelectionMinRMSE = ...
        FinalCapSelection.minRMSE;


    Final.capSelectionRMSEThreshold = ...
        FinalCapSelection.RMSEThreshold;


    % Historical fields minRMSE/RMSEThreshold continue to describe the
    % metric used by tau_cap.  For pseudo mode these therefore refer to
    % pseudo-training RMSE.
    Final.minRMSE = ...
        FinalCapSelection.minRMSE;


    Final.RMSEThreshold = ...
        FinalCapSelection.RMSEThreshold;


    % IMPORTANT:
    % Final.RMSE keeps its historical meaning: ORIGINAL training RMSE
    % at the selected cap.  This avoids corrupting later consistency
    % checks when capRMSESource='pseudo'.
    Final.RMSE = ...
        TrainingResult.RMSE_all(kStar);


    Final.point_success = ...
        TrainingResult. ...
        cap_success_by_point(:,kStar);


    % ---------------------------------------------------------
    % Final radius
    % ---------------------------------------------------------

    Final.radiusIndex = ...
        TrainingResult. ...
        selectedRadiusIndex_by_point_cap(:,kStar);


    Final.radius = ...
        TrainingResult. ...
        selectedRadius_by_point_cap(:,kStar);


    % ---------------------------------------------------------
    % Final optimized parameters
    % ---------------------------------------------------------

    Final.C = ...
        TrainingResult. ...
        C_by_point_cap(:,kStar);


    Final.reg_param = ...
        TrainingResult. ...
        Reg_by_point_cap(:,kStar);


    Final.epsilon = ...
        TrainingResult. ...
        epsilon_by_point_cap(:,kStar);


    Final.d_local = ...
        TrainingResult. ...
        d_local_by_point_cap(:,kStar);


    % ---------------------------------------------------------
    % Final prediction and residual
    % ---------------------------------------------------------

    Final.AE = ...
        TrainingResult. ...
        selectedAE_by_point_cap(:,kStar);


    Final.yhat = ...
        TrainingResult. ...
        yhat_by_point_cap(:,kStar);


    Final.residual = ...
        TrainingResult. ...
        residual_by_point_cap(:,kStar);


    Final.y_diff = ...
        TrainingResult. ...
        y_diff_by_point_cap(:,kStar);


    Final.squaredResidual = ...
        TrainingResult. ...
        squaredResidual_by_point_cap(:,kStar);


    % ---------------------------------------------------------
    % Final numerical information
    % ---------------------------------------------------------

    Final.cond_reg = ...
        TrainingResult. ...
        cond_reg_by_point_cap(:,kStar);


    Final.rel_residual = ...
        TrainingResult. ...
        rel_residual_by_point_cap(:,kStar);


    Final.solver_used = ...
        TrainingResult. ...
        solver_used_by_point_cap(:,kStar);


    % ---------------------------------------------------------
    % Final neighbourhood information
    % ---------------------------------------------------------

    if saveStage2All

        % Full mode reuses the historical retained point-by-cap
        % neighbourhood information exactly.
        Final.neighbor_idx = ...
            TrainingResult. ...
            neighbor_idx_by_point_cap(:,kStar);


        Final.neighbor_distances = ...
            TrainingResult. ...
            neighbor_distances_by_point_cap(:,kStar);


        Final.neighbor_count = ...
            TrainingResult. ...
            neighbor_count_by_point_cap(:,kStar);


        Final.effectiveRadius = ...
            TrainingResult. ...
            effective_radius_by_point_cap(:,kStar);


        Final.selectedMaxDistance = ...
            TrainingResult. ...
            selected_max_distance_by_point_cap(:,kStar);


        Final.nWithinRadius = ...
            TrainingResult. ...
            n_within_radius_by_point_cap(:,kStar);


        Final.nAddedForMinimum = ...
            TrainingResult. ...
            n_added_for_minimum_by_point_cap(:,kStar);


        Final.usedMinNeighborFallback = ...
            TrainingResult. ...
            usedMinNeighborFallback_by_point_cap(:,kStar);

    else

        % Compact mode retains only the compulsory exact neighbour count.
        % The final neighbourhood itself is reconstructed transiently inside
        % evaluateLocal(...,'final'), so no variable-length cell payload is
        % accumulated in TrainingResult.Final.
        Final.neighbor_count = ...
            TrainingResult. ...
            neighbor_count_by_point_cap(:,kStar);

    end


    % =========================================================
    % 18.1 Final RMSE consistency check
    % =========================================================

    if all(Final.point_success) && ...
       all(isfinite(Final.residual))

        Final.RMSE_recomputed = ...
            sqrt( ...
                mean( ...
                    Final.residual.^2));


        Final.RMSEDifference = ...
            abs( ...
                Final.RMSE_recomputed - ...
                Final.RMSE);


        Final.RMSEConsistencyTolerance = ...
            1e-10 * ...
            max( ...
                1, ...
                abs(Final.RMSE));


        Final.RMSEConsistent = ...
            Final.RMSEDifference <= ...
            Final.RMSEConsistencyTolerance;


        Final.success = ...
            true;


        Final.message = ...
            '';

    else

        Final.success = ...
            false;


        Final.message = ...
            ['The selected radius cap does not contain valid ' ...
             'results for every training point.'];

    end


else

    Final.message = ...
        FinalCapSelection.message;

end


% -------------------------------------------------------------
% Preserve both scaled and original-unit radius information.
% -------------------------------------------------------------

Final.capRadius_scaled = ...
    Final.capRadius;


Final.capRadius_raw = ...
    Final.capRadius .* ...
    S_global;


Final.radius_scaled = ...
    Final.radius;


Final.radius_raw = ...
    Final.radius .* ...
    S_global;


if saveStage2All

    Final.effectiveRadius_scaled = ...
        Final.effectiveRadius;


    Final.effectiveRadius_raw = ...
        Final.effectiveRadius .* ...
        S_global;


    Final.selectedMaxDistance_scaled = ...
        Final.selectedMaxDistance;


    Final.selectedMaxDistance_raw = ...
        Final.selectedMaxDistance .* ...
        S_global;

end


TrainingResult.Final = ...
    Final;


% =============================================================
% 19. Final training status
% =============================================================

TrainingResult.nSuccessfulPointsAnyCap = ...
    sum( ...
        TrainingResult.point_success);


TrainingResult.nFailedPointsAnyCap = ...
    n - ...
    TrainingResult.nSuccessfulPointsAnyCap;


TrainingResult.success = ...
    Final.success;


if TrainingResult.success

    TrainingResult.message = ...
        '';

else

    if ~isempty(Final.message)

        TrainingResult.message = ...
            Final.message;

    else

        TrainingResult.message = ...
            'Local RBF training did not produce a valid final model.';

    end

end


% =============================================================
% 20. Core-training runtime
% =============================================================

TrainingResult.coreTrainingWallTime = ...
    toc(tTraining);


% =============================================================
% 21. Evaluate selected local systems
%
% The SAME geometryCache is passed to evaluateLocal.
%
% At each point, repeated caps selecting the same radius reuse one
% unique selected-system evaluation/diagnostic calculation.
% =============================================================

TrainingResult.EvaluationResult = ...
    [];


TrainingResult.evaluationSuccess = ...
    false;


TrainingResult.evaluationWallTime = ...
    0;


TrainingResult.nRBFSolvesTraining = ...
    TrainingResult.nRBFSolves;


TrainingResult.nRBFSolvesEvaluation = ...
    0;


TrainingResult.nRBFSolvesTotal = ...
    TrainingResult.nRBFSolvesTraining + ...
    TrainingResult.nRBFSolvesPseudo;


if TrainingResult.success

    tEvaluation = ...
        tic;


    if saveStage2All

        evaluationMode = ...
            'full';

    else

        evaluationMode = ...
            'final';

    end


    try

        EvaluationResult = ...
            localrbf.evaluateLocal( ...
                TrainingResult, ...
                coreOptions, ...
                geometryCache, ...
                evaluationMode);


    catch ME

        EvaluationResult = ...
            struct();


        EvaluationResult.success = ...
            false;


        EvaluationResult.message = ...
            ME.message;


        EvaluationResult.nRBFSolves = ...
            0;

    end


    TrainingResult.evaluationWallTime = ...
        toc(tEvaluation);


    TrainingResult.EvaluationResult = ...
        EvaluationResult;


    TrainingResult.evaluationSuccess = ...
        isfield(EvaluationResult,'success') && ...
        EvaluationResult.success;


    if isfield(EvaluationResult,'nRBFSolves') && ...
       isnumeric(EvaluationResult.nRBFSolves) && ...
       isscalar(EvaluationResult.nRBFSolves) && ...
       isfinite(EvaluationResult.nRBFSolves)

        TrainingResult.nRBFSolvesEvaluation = ...
            EvaluationResult.nRBFSolves;

    end


    TrainingResult.nRBFSolvesTotal = ...
        TrainingResult.nRBFSolvesTraining + ...
        TrainingResult.nRBFSolvesPseudo + ...
        TrainingResult.nRBFSolvesEvaluation;

end


% =============================================================
% 22. Canonical selected-model outputs
%
% These top-level aliases provide direct access analogous to the
% canonical global evaluation outputs while retaining the complete
% local training/evaluation structures.
%
% Residual convention:
%
%       y_diff = observation - prediction
% =============================================================

TrainingResult.yhat = ...
    nan(n,1);


TrainingResult.y_diff = ...
    nan(n,1);


TrainingResult.residual = ...
    nan(n,1);


TrainingResult.RMSE = ...
    NaN;


TrainingResult.C = ...
    Final.C;


TrainingResult.reg_param = ...
    Final.reg_param;


TrainingResult.epsilon = ...
    Final.epsilon;


TrainingResult.radius = ...
    Final.radius;


TrainingResult.radius_scaled = ...
    Final.radius_scaled;


TrainingResult.radius_raw = ...
    Final.radius_raw;


TrainingResult.finalCapIndex = ...
    Final.capIndex;


TrainingResult.finalCapRadius = ...
    Final.capRadius;


TrainingResult.finalCapRadius_scaled = ...
    Final.capRadius_scaled;


TrainingResult.finalCapRadius_raw = ...
    Final.capRadius_raw;


TrainingResult.finalCapRMSESource = ...
    Final.capRMSESource;


TrainingResult.finalCapTransferMethod = ...
    Final.transferMethodForCapRMSE;


TrainingResult.finalCapNIDWSources = ...
    Final.nIDWSourcesForCapRMSE;


TrainingResult.finalCapNearThresholdRaw = ...
    Final.nearThresholdRawForCapRMSE;


TrainingResult.finalCapParameterTransferMode = ...
    Final.parameterTransferModeForCapRMSE;


TrainingResult.finalCapSelectionRMSE = ...
    Final.capSelectionRMSE;


TrainingResult.finalCapSelectionMinRMSE = ...
    Final.capSelectionMinRMSE;


TrainingResult.finalCapSelectionRMSEThreshold = ...
    Final.capSelectionRMSEThreshold;


if TrainingResult.success

    % ---------------------------------------------------------
    % Always retain the valid selected training result.
    % ---------------------------------------------------------

    TrainingResult.yhat = ...
        Final.yhat;


    TrainingResult.y_diff = ...
        Final.y_diff;


    TrainingResult.residual = ...
        Final.residual;


    TrainingResult.RMSE = ...
        Final.RMSE;


    % ---------------------------------------------------------
    % If full local evaluation succeeds, use its recomputed
    % canonical values.
    % ---------------------------------------------------------

    if TrainingResult.evaluationSuccess

        EvaluationResult = ...
            TrainingResult.EvaluationResult;


        if isfield(EvaluationResult,'yhat') && ...
           isfield(EvaluationResult,'y_diff') && ...
           isfield(EvaluationResult,'RMSE') && ...
           numel(EvaluationResult.yhat) == n && ...
           numel(EvaluationResult.y_diff) == n && ...
           all(isfinite(EvaluationResult.yhat(:))) && ...
           all(isfinite(EvaluationResult.y_diff(:))) && ...
           isscalar(EvaluationResult.RMSE) && ...
           isfinite(EvaluationResult.RMSE)

            TrainingResult.yhat = ...
                EvaluationResult.yhat(:);


            TrainingResult.y_diff = ...
                EvaluationResult.y_diff(:);


            TrainingResult.residual = ...
                TrainingResult.y_diff;


            TrainingResult.RMSE = ...
                EvaluationResult.RMSE;

        end

    end

end


TrainingResult.y_train_pred = ...
    TrainingResult.yhat;


TrainingResult.y_diff_train = ...
    TrainingResult.y_diff;


TrainingResult.train_rmse = ...
    TrainingResult.RMSE;


TrainingResult.AE_train = ...
    abs( ...
        TrainingResult.y_diff);


TrainingResult.squaredResidual_train = ...
    TrainingResult.y_diff.^2;


% =============================================================
% 23. Training-versus-evaluation consistency
% =============================================================

TrainingResult.evaluationRMSEDifference = ...
    NaN;


TrainingResult.evaluationRMSEConsistencyTolerance = ...
    NaN;


TrainingResult.evaluationRMSEConsistent = ...
    false;


if TrainingResult.success && ...
   isfinite(Final.RMSE) && ...
   isfinite(TrainingResult.RMSE)

    TrainingResult.evaluationRMSEDifference = ...
        abs( ...
            TrainingResult.RMSE - ...
            Final.RMSE);


    TrainingResult.evaluationRMSEConsistencyTolerance = ...
        1e-10 * ...
        max( ...
            [1, ...
             abs(TrainingResult.RMSE), ...
             abs(Final.RMSE)]);


    TrainingResult.evaluationRMSEConsistent = ...
        TrainingResult.evaluationRMSEDifference <= ...
        TrainingResult.evaluationRMSEConsistencyTolerance;

end


% =============================================================
% 24. Evaluation reuse information
% =============================================================

TrainingResult.nSelectedPointCapPairs = ...
    getFieldOrDefault( ...
        TrainingResult.EvaluationResult, ...
        'nSelectedPointCapPairs', ...
        NaN);


TrainingResult.nUniqueSelectedSystemsTotal = ...
    getFieldOrDefault( ...
        TrainingResult.EvaluationResult, ...
        'nUniqueSelectedSystemsTotal', ...
        NaN);


TrainingResult.nDiagnosticCalculationsAvoided = ...
    getFieldOrDefault( ...
        TrainingResult.EvaluationResult, ...
        'nDiagnosticCalculationsAvoided', ...
        NaN);


TrainingResult.diagnosticReuseFraction = ...
    getFieldOrDefault( ...
        TrainingResult.EvaluationResult, ...
        'diagnosticReuseFraction', ...
        NaN);


% =============================================================
% 25. Geometry-cache reporting
% =============================================================

TrainingResult.geometryCacheInfo.reusedForEvaluation = ...
    TrainingResult.success;


TrainingResult.geometryCacheInfo.storedInTrainingResult = ...
    false;


% =============================================================
% 25.1 Release nonrequested Stage-2 detail
%
% All fields removed here have already served their numerical purpose.
% The final model, canonical accuracy outputs, EvaluationResult, aggregate
% counters, and timing remain available for testing and compulsory storage.
% =============================================================

if ~saveStage2All

    TrainingResult = ...
        stripNonrequestedStage2Detail( ...
            TrainingResult);

end


% =============================================================
% 26. Total public-training runtime
% =============================================================

if strcmp( ...
        executionStage, ...
        'stage2')

    TrainingResult.stage2WallTime = ...
        toc(tTraining);


    TrainingResult.totalWallTime = ...
        TrainingResult.stage1WallTime + ...
        TrainingResult.stage2WallTime;

else

    TrainingResult.totalWallTime = ...
        toc(tTraining);

end

end


% =============================================================
% Local helper
% =============================================================

function T = getTransferOptions(options)
%GETTRANSFEROPTIONS Resolve transfer/cap-selection controls.
%
% Defaults reproduce current production behavior.
%
% Stage 1 remains independent of transfer/cap-selection evaluation.
% Strict pseudo-training RMSE, when requested, is generated only in
% Stage 2/all after point-by-cap settings have been assembled.

T = ...
    struct();


T.method = ...
    'hybrid';


T.nIDWSources = ...
    7;


T.nearThresholdRaw = ...
    [];


T.parameterTransferMode = ...
    'log_log';


T.capRMSESource = ...
    'training';


if isfield(options,'transfer') && ...
   ~isempty(options.transfer)

    if ~isstruct(options.transfer) || ...
       ~isscalar(options.transfer)

        error( ...
            'options.transfer must be a scalar structure.');

    end


    U = ...
        options.transfer;


    fields = ...
        fieldnames(T);


    for iField = 1:numel(fields)

        f = ...
            fields{iField};


        if isfield(U,f) && ...
           ~isempty(U.(f))

            T.(f) = ...
                U.(f);

        end

    end

end


% -------------------------------------------------------------
% Transfer method
% -------------------------------------------------------------

if ~(ischar(T.method) || ...
     (isstring(T.method) && isscalar(T.method)))

    error( ...
        'options.transfer.method must be a character vector/string scalar.');

end


T.method = ...
    lower( ...
        strtrim( ...
            char(T.method)));


if ~any(strcmp( ...
        T.method, ...
        {'nearest','idw','hybrid','ensemble'}))

    error( ...
        ['options.transfer.method must be ''nearest'', ''idw'', ' ...
         '''hybrid'', or ''ensemble''.']);

end


% -------------------------------------------------------------
% Number of N-nearest transfer sources
% -------------------------------------------------------------

if ~isnumeric(T.nIDWSources) || ...
   ~isscalar(T.nIDWSources) || ...
   ~isreal(T.nIDWSources) || ...
   ~isfinite(T.nIDWSources) || ...
   T.nIDWSources < 1 || ...
   T.nIDWSources ~= round(T.nIDWSources)

    error( ...
        'options.transfer.nIDWSources must be a positive integer.');

end


T.nIDWSources = ...
    double(T.nIDWSources);


% -------------------------------------------------------------
% Near threshold
%
% It is active only for hybrid/ensemble.  Preserve [] for methods where
% it is irrelevant.  When supplied, validate it in raw coordinate units.
% -------------------------------------------------------------

if ~isempty(T.nearThresholdRaw)

    if ~isnumeric(T.nearThresholdRaw) || ...
       ~isscalar(T.nearThresholdRaw) || ...
       ~isreal(T.nearThresholdRaw) || ...
       ~isfinite(T.nearThresholdRaw) || ...
       T.nearThresholdRaw <= 0

        error( ...
            ['options.transfer.nearThresholdRaw must be empty or a ' ...
             'finite positive real scalar.']);

    end

    T.nearThresholdRaw = ...
        double(T.nearThresholdRaw);

end


% -------------------------------------------------------------
% Parameter aggregation mode
%
% Retained for provenance.  It is inactive for nearest and ensemble.
% -------------------------------------------------------------

if ~(ischar(T.parameterTransferMode) || ...
     (isstring(T.parameterTransferMode) && ...
      isscalar(T.parameterTransferMode)))

    error( ...
        ['options.transfer.parameterTransferMode must be a ' ...
         'character vector/string scalar.']);

end


T.parameterTransferMode = ...
    lower( ...
        strtrim( ...
            char(T.parameterTransferMode)));


validParameterModes = ...
    { ...
    'arithmetic_arithmetic', ...
    'arithmetic_log', ...
    'log_arithmetic', ...
    'log_log', ...
    'arith_arith', ...
    'arith_log', ...
    'log_arith', ...
    'm21', ...
    'm22', ...
    'm23', ...
    'm24' ...
    };


if ~any(strcmp( ...
        T.parameterTransferMode, ...
        validParameterModes))

    error( ...
        'Unsupported options.transfer.parameterTransferMode.');

end


% -------------------------------------------------------------
% RMSE source for tau_cap
% -------------------------------------------------------------

if ~(ischar(T.capRMSESource) || ...
     (isstring(T.capRMSESource) && isscalar(T.capRMSESource)))

    error( ...
        ['options.transfer.capRMSESource must be ''training'' ' ...
         'or ''pseudo''.']);

end


T.capRMSESource = ...
    lower( ...
        strtrim( ...
            char(T.capRMSESource)));


switch T.capRMSESource

    case {'training','original','origin'}

        T.capRMSESource = ...
            'training';


    case {'pseudo','pseudo-training','pseudotraining'}

        T.capRMSESource = ...
            'pseudo';


    otherwise

        error( ...
            ['options.transfer.capRMSESource must be ''training'' ' ...
             'or ''pseudo''.']);

end

end


% =============================================================
% Local helper
% =============================================================

function P = calculateStrictPseudoTrainingRMSELocal( ...
    TrainingResult,X,y,kernel,useShape,min_neighbors, ...
    TransferOptions,geometryCache,numericalOptions,S_global)
%CALCULATESTRICTPSEUDOTRAININGRMSELOCAL
%
% Strict target-excluded pseudo-unseen evaluation for every radius cap.
%
% IMPORTANT:
%   Stage-1 donor settings are reused.  This helper does not perform
%   nested leave-one-out parameter re-optimization.
%
% For target i:
%   1. remove i from all transfer donors;
%   2. remove i from all fitting observations;
%   3. transfer cap-specific settings from the remaining points;
%   4. solve/predict at i exactly as an unseen query;
%   5. accumulate pseudo residual = y(i) - yhat_pseudo(i).
%
% Official RMSE_all(k) is populated only when ALL n targets succeed.
%
% This helper performs no parameter optimization and reuses the existing
% full training distance/polynomial geometry.

[n,~] = ...
    size(X);


K = ...
    size( ...
        TrainingResult.selectedRadius_by_point_cap, ...
        2);


P = ...
    struct();


P.SSE_available_all = ...
    nan(1,K);


P.MSE_available_all = ...
    nan(1,K);


P.RMSE_available_all = ...
    nan(1,K);


P.RMSE_all = ...
    nan(1,K);


% Transient pointwise pseudo state. Only the finally selected cap is
% copied into TrainingResult.PseudoFinal, keeping Stage-2 files compact.
P.point_success_by_point_cap = false(n,K);
P.radius_by_point_cap = nan(n,K);
P.reg_by_point_cap = nan(n,K);
P.C_by_point_cap = nan(n,K);
P.neighbor_count_by_point_cap = nan(n,K);
P.prediction_by_point_cap = nan(n,K);
P.residual_by_point_cap = nan(n,K);
P.cond_raw_by_point_cap = nan(n,K);
P.cond_reg_by_point_cap = nan(n,K);
P.cond_effective_by_point_cap = nan(n,K);
P.rel_residual_by_point_cap = nan(n,K);
P.backward_error_by_point_cap = nan(n,K);
P.solver_used_by_point_cap = cell(n,K);


P.cap_success = ...
    false(1,K);


P.nValidPointsPerCap = ...
    zeros(1,K);


P.nFailedPointsPerCap = ...
    n .* ones(1,K);


P.nRBFSolves = ...
    0;


P.nTransferFailures = ...
    0;


P.nNeighborhoodFailures = ...
    0;


P.nPolynomialFailures = ...
    0;


P.nSolveFailures = ...
    0;


P.nPredictionFailures = ...
    0;


if n < 2

    return

end


% -------------------------------------------------------------
% Transfer threshold is required only by hybrid/ensemble.
% Public TransferOptions threshold is in ORIGINAL coordinate units.
% -------------------------------------------------------------

if any(strcmp( ...
        TransferOptions.method, ...
        {'hybrid','ensemble'}))

    if isempty(TransferOptions.nearThresholdRaw)

        error( ...
            ['options.transfer.nearThresholdRaw is required when ' ...
             'capRMSESource=''pseudo'' and transfer method is ''%s''.'], ...
            TransferOptions.method);

    end


    nearThresholdScaled = ...
        TransferOptions.nearThresholdRaw ./ ...
        S_global;

else

    % transferSettingsFromTrainToTest validates this argument even when
    % the selected method does not use a near/far threshold.
    nearThresholdScaled = ...
        1;

end


if ~isfinite(nearThresholdScaled) || ...
   nearThresholdScaled <= 0

    error( ...
        'Scaled pseudo-training transfer threshold is invalid.');

end


% idw_radius is a backward-compatibility argument in the current transfer
% function and is not used for donor selection. Supply one valid scalar.
idwRadiusCompatibility = ...
    1;


for kCap = 1:K

    % A cap that cannot provide complete original point-specific settings
    % cannot become the final cap. Skip it entirely to save pseudo solves.
    if ~TrainingResult.cap_success(kCap)

        continue

    end


    sumSquaredResidual = ...
        0;


    nValid = ...
        0;


    for iTarget = 1:n

        donorMask = ...
            true(n,1);


        donorMask(iTarget) = ...
            false;


        donorIdx = ...
            find(donorMask);


        FinalDonor = ...
            struct();


        FinalDonor.success = ...
            true;


        FinalDonor.radius = ...
            TrainingResult. ...
            selectedRadius_by_point_cap( ...
                donorIdx, ...
                kCap);


        FinalDonor.reg_param = ...
            TrainingResult. ...
            Reg_by_point_cap( ...
                donorIdx, ...
                kCap);


        if useShape

            FinalDonor.C = ...
                TrainingResult. ...
                C_by_point_cap( ...
                    donorIdx, ...
                    kCap);

        end


        D_query_donor = ...
            geometryCache.D( ...
                iTarget, ...
                donorIdx);


        try

            Transfer = ...
                localrbf.transferSettingsFromTrainToTest( ...
                    X(donorIdx,:), ...
                    X(iTarget,:), ...
                    FinalDonor, ...
                    kernel, ...
                    nearThresholdScaled, ...
                    idwRadiusCompatibility, ...
                    D_query_donor, ...
                    false, ...
                    TransferOptions.method, ...
                    TransferOptions.nIDWSources, ...
                    TransferOptions.parameterTransferMode);

        catch

            P.nTransferFailures = ...
                P.nTransferFailures + 1;


            continue

        end


        if ~isfield(Transfer,'querySuccess') || ...
           ~Transfer.querySuccess(1)

            P.nTransferFailures = ...
                P.nTransferFailures + 1;


            continue

        end


        transferMethod = ...
            Transfer.transferMethod{1};


        if strcmp(transferMethod,'ensemble')

            [ ...
                pointSuccess, ...
                yhatPseudo, ...
                Counts, ...
                Di ...
            ] = ...
                evaluateStrictPseudoEnsembleQueryLocal( ...
                    iTarget, ...
                    donorIdx, ...
                    X, ...
                    y, ...
                    kernel, ...
                    useShape, ...
                    min_neighbors, ...
                    geometryCache, ...
                    numericalOptions, ...
                    Transfer.ensembleUniqueWeight{1}, ...
                    Transfer.ensembleUniqueRadius{1}, ...
                    Transfer.ensembleUniqueRegParam{1}, ...
                    Transfer.ensembleUniqueC{1});

        else

            [ ...
                pointSuccess, ...
                yhatPseudo, ...
                Counts, ...
                Di ...
            ] = ...
                evaluateStrictPseudoScalarQueryLocal( ...
                    iTarget, ...
                    donorIdx, ...
                    X, ...
                    y, ...
                    kernel, ...
                    useShape, ...
                    min_neighbors, ...
                    geometryCache, ...
                    numericalOptions, ...
                    Transfer.radius(1), ...
                    Transfer.reg_param(1), ...
                    Transfer.C(1));

        end


        P.nRBFSolves = ...
            P.nRBFSolves + ...
            Counts.nRBFSolves;


        P.nNeighborhoodFailures = ...
            P.nNeighborhoodFailures + ...
            Counts.nNeighborhoodFailures;


        P.nPolynomialFailures = ...
            P.nPolynomialFailures + ...
            Counts.nPolynomialFailures;


        P.nSolveFailures = ...
            P.nSolveFailures + ...
            Counts.nSolveFailures;


        P.nPredictionFailures = ...
            P.nPredictionFailures + ...
            Counts.nPredictionFailures;


        if ~pointSuccess || ...
           ~isfinite(yhatPseudo)

            continue

        end


        residual = ...
            y(iTarget) - ...
            yhatPseudo;


        squaredResidual = ...
            residual.^2;


        if ~isfinite(residual) || ...
           ~isfinite(squaredResidual)

            P.nPredictionFailures = ...
                P.nPredictionFailures + 1;


            continue

        end


        P.point_success_by_point_cap(iTarget,kCap) = true;
        P.prediction_by_point_cap(iTarget,kCap) = yhatPseudo;
        P.residual_by_point_cap(iTarget,kCap) = residual;

        % Bootstrap uses one fixed Radius/Reg/C tuple per pseudo target.
        % The bootstrap package intentionally has no ensemble mode; when
        % the upstream pseudo transfer is ensemble, scalar parameters are
        % therefore left unavailable instead of inventing a surrogate.
        if ~strcmp(transferMethod,'ensemble')
            P.radius_by_point_cap(iTarget,kCap) = Transfer.radius(1);
            P.reg_by_point_cap(iTarget,kCap) = Transfer.reg_param(1);
            P.C_by_point_cap(iTarget,kCap) = Transfer.C(1);
            if isfield(Di,'neighbor_count'), P.neighbor_count_by_point_cap(iTarget,kCap) = Di.neighbor_count; end
            if isfield(Di,'cond_raw'), P.cond_raw_by_point_cap(iTarget,kCap) = Di.cond_raw; end
            if isfield(Di,'cond_reg'), P.cond_reg_by_point_cap(iTarget,kCap) = Di.cond_reg; end
            if isfield(Di,'cond_effective'), P.cond_effective_by_point_cap(iTarget,kCap) = Di.cond_effective; end
            if isfield(Di,'rel_residual'), P.rel_residual_by_point_cap(iTarget,kCap) = Di.rel_residual; end
            if isfield(Di,'backward_error'), P.backward_error_by_point_cap(iTarget,kCap) = Di.backward_error; end
            if isfield(Di,'solver_used'), P.solver_used_by_point_cap{iTarget,kCap} = Di.solver_used; end
        end

        sumSquaredResidual = ...
            sumSquaredResidual + ...
            squaredResidual;


        nValid = ...
            nValid + 1;

    end


    P.nValidPointsPerCap(kCap) = ...
        nValid;


    P.nFailedPointsPerCap(kCap) = ...
        n - ...
        nValid;


    if nValid > 0

        P.SSE_available_all(kCap) = ...
            sumSquaredResidual;


        P.MSE_available_all(kCap) = ...
            sumSquaredResidual ./ ...
            nValid;


        P.RMSE_available_all(kCap) = ...
            sqrt( ...
                P.MSE_available_all(kCap));

    end


    if nValid == n

        P.RMSE_all(kCap) = ...
            P.RMSE_available_all(kCap);


        P.cap_success(kCap) = ...
            true;

    end

end

end


% =============================================================
% Local helper
% =============================================================

function [success,yhat,Counts,Di] = ...
    evaluateStrictPseudoScalarQueryLocal( ...
        iTarget,donorIdx,X,y,kernel,useShape,min_neighbors, ...
        geometryCache,numericalOptions,radius_i,reg_i,C_i)
%EVALUATESTRICTPSEUDOSCALARQUERYLOCAL
% Solve one strict pseudo-unseen scalar transferred system.

[success,yhat,Counts,Di] = ...
    solveStrictPseudoTupleLocal( ...
        iTarget, ...
        donorIdx, ...
        X, ...
        y, ...
        kernel, ...
        useShape, ...
        min_neighbors, ...
        geometryCache, ...
        numericalOptions, ...
        radius_i, ...
        reg_i, ...
        C_i);

end


% =============================================================
% Local helper
% =============================================================

function [success,yhat,Counts,Di] = ...
    evaluateStrictPseudoEnsembleQueryLocal( ...
        iTarget,donorIdx,X,y,kernel,useShape,min_neighbors, ...
        geometryCache,numericalOptions, ...
        wUnique,radiusUnique,regUnique,CUnique)
%EVALUATESTRICTPSEUDOENSEMBLEQUERYLOCAL
%
% Strict ensemble pseudo prediction.
% Every unique complete tuple must succeed. Failed tuples are not dropped
% and weights are never silently renormalized.

success = ...
    false;


yhat = ...
    NaN;


Di = struct();


Counts = ...
    emptyPseudoCountsLocal();


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


if nTuple < 1 || ...
   numel(radiusUnique) ~= nTuple || ...
   numel(regUnique) ~= nTuple || ...
   numel(CUnique) ~= nTuple || ...
   any(~isfinite(wUnique)) || ...
   any(wUnique < 0) || ...
   abs(sum(wUnique)-1) > 1e-12

    Counts.nPredictionFailures = ...
        1;


    return

end


yhatTuple = ...
    nan(nTuple,1);


for j = 1:nTuple

    [ ...
        tupleSuccess, ...
        yhatTuple(j), ...
        TupleCounts ...
    ] = ...
        solveStrictPseudoTupleLocal( ...
            iTarget, ...
            donorIdx, ...
            X, ...
            y, ...
            kernel, ...
            useShape, ...
            min_neighbors, ...
            geometryCache, ...
            numericalOptions, ...
            radiusUnique(j), ...
            regUnique(j), ...
            CUnique(j));


    Counts = ...
        addPseudoCountsLocal( ...
            Counts, ...
            TupleCounts);


    if ~tupleSuccess

        return

    end

end


yhat = ...
    sum( ...
        wUnique .* ...
        yhatTuple);


if ~isscalar(yhat) || ...
   ~isreal(yhat) || ...
   ~isfinite(yhat)

    Counts.nPredictionFailures = ...
        Counts.nPredictionFailures + 1;


    yhat = ...
        NaN;


    return

end


success = ...
    true;

end


% =============================================================
% Local helper
% =============================================================

function [success,yhat,Counts,Di] = ...
    solveStrictPseudoTupleLocal( ...
        iTarget,donorIdx,X,y,kernel,useShape,min_neighbors, ...
        geometryCache,numericalOptions,radius_i,reg_i,C_i)
%SOLVESTRICTPSEUDOTUPLELOCAL
%
% Build/solve one pseudo-unseen local system using ONLY donorIdx.
% iTarget is excluded from donorIdx by the caller.

success = ...
    false;


yhat = ...
    NaN;


Di = struct();


Counts = ...
    emptyPseudoCountsLocal();


if ~isscalar(radius_i) || ...
   ~isreal(radius_i) || ...
   ~isfinite(radius_i) || ...
   radius_i <= 0 || ...
   ~isscalar(reg_i) || ...
   ~isreal(reg_i) || ...
   ~isfinite(reg_i) || ...
   reg_i <= 0

    Counts.nPredictionFailures = ...
        1;


    return

end


if useShape && ...
   (~isscalar(C_i) || ...
    ~isreal(C_i) || ...
    ~isfinite(C_i) || ...
    C_i <= 0)

    Counts.nPredictionFailures = ...
        1;


    return

end


distAll = ...
    geometryCache.D( ...
        iTarget, ...
        donorIdx).';


try

    Nbh = ...
        localrbf.getNeighborhood( ...
            X(donorIdx,:), ...
            X(iTarget,:), ...
            radius_i, ...
            min_neighbors, ...
            [], ...
            distAll);

catch

    Counts.nNeighborhoodFailures = ...
        1;


    return

end


if isempty(Nbh) || ...
   ~isfield(Nbh,'success') || ...
   ~Nbh.success

    Counts.nNeighborhoodFailures = ...
        1;


    return

end


idxFitLocal = ...
    Nbh.idx(:);


idxFitGlobal = ...
    donorIdx(idxFitLocal);


Xfit = ...
    X(idxFitGlobal,:);


yfit = ...
    y(idxFitGlobal);


[isOK,~] = ...
    common.checkPolynomialAdmissibility( ...
        Xfit);


if ~isOK

    Counts.nPolynomialFailures = ...
        1;


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

        Counts.nPredictionFailures = ...
            1;


        return

    end


    epsilon = ...
        max( ...
            C_i .* ...
            d_local, ...
            1e-12);

else

    epsilon = ...
        1;

end


geometry = ...
    struct();


geometry.R = ...
    geometryCache.D( ...
        idxFitGlobal, ...
        idxFitGlobal);


geometry.P = ...
    geometryCache.P( ...
        idxFitGlobal, ...
        :);


geometry.R_eval = ...
    reshape( ...
        geometryCache.D( ...
            iTarget, ...
            idxFitGlobal), ...
        1,[]);


geometry.P_eval = ...
    geometryCache.P( ...
        iTarget, ...
        :);


try

    S = ...
        common.buildRBFSystem( ...
            Xfit, ...
            yfit, ...
            kernel, ...
            epsilon, ...
            reg_i, ...
            geometry);


    Counts.nRBFSolves = ...
        Counts.nRBFSolves + 1;


    [coeffs,solveInfo] = ...
        common.solveLinearSystem( ...
            S, ...
            numericalOptions);


    if ~solveInfo.success || ...
       isempty(coeffs) || ...
       any(~isfinite(coeffs(:)))

        Counts.nSolveFailures = ...
            Counts.nSolveFailures + 1;


        return

    end


    [yhat,B_target] = ...
        common.predictRBF( ...
            X(iTarget,:), ...
            Xfit, ...
            coeffs, ...
            kernel, ...
            epsilon, ...
            geometry);

    if isscalar(yhat) && isreal(yhat) && isfinite(yhat)
        % Numerical diagnostics must not alter the already-established
        % pseudo prediction/RMSE success decision.
        try
            Di = common.calculateConditionDiagnostics( ...
                S, coeffs, solveInfo, B_target);
            Di.neighbor_count = Nbh.count;
        catch
            Di = struct('neighbor_count',Nbh.count);
        end
    end

catch

    Counts.nSolveFailures = ...
        Counts.nSolveFailures + 1;


    yhat = ...
        NaN;


    return

end


if ~isscalar(yhat) || ...
   ~isreal(yhat) || ...
   ~isfinite(yhat)

    Counts.nPredictionFailures = ...
        Counts.nPredictionFailures + 1;


    yhat = ...
        NaN;


    return

end


success = ...
    true;

end


% =============================================================
% Local helper
% =============================================================

function C = emptyPseudoCountsLocal()
%EMPTYPSEUDOCOUNTSLOCAL Small scalar failure/solve counter record.

C = ...
    struct();


C.nRBFSolves = ...
    0;


C.nNeighborhoodFailures = ...
    0;


C.nPolynomialFailures = ...
    0;


C.nSolveFailures = ...
    0;


C.nPredictionFailures = ...
    0;

end


% =============================================================
% Local helper
% =============================================================

function A = addPseudoCountsLocal(A,B)
%ADDPSEUDOCOUNTSLOCAL Add small scalar pseudo-evaluation counters.

A.nRBFSolves = ...
    A.nRBFSolves + ...
    B.nRBFSolves;


A.nNeighborhoodFailures = ...
    A.nNeighborhoodFailures + ...
    B.nNeighborhoodFailures;


A.nPolynomialFailures = ...
    A.nPolynomialFailures + ...
    B.nPolynomialFailures;


A.nSolveFailures = ...
    A.nSolveFailures + ...
    B.nSolveFailures;


A.nPredictionFailures = ...
    A.nPredictionFailures + ...
    B.nPredictionFailures;

end


% =============================================================
% Local helper
% =============================================================

function checkpointFile = getStage1CheckpointFile( ...
    options,checkpointDir,kernel,y,radii_raw,method,nGrid)
%GETSTAGE1CHECKPOINTFILE Resolve explicit per-run file when supplied.

checkpointFile = ...
    '';


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'stage1CheckpointFile') && ...
   ~isempty(options.execution.stage1CheckpointFile)

    checkpointFile = ...
        char(options.execution.stage1CheckpointFile);

end


if isempty(checkpointFile)

    error( ...
        ['Two-stage LocalRBFTraining requires an explicit deterministic ' ...
         'options.execution.stage1CheckpointFile. The batch manager ' ...
         'localrbf.localRBFBatchSummaryParallel_TuneReg supplies this ' ...
         'automatically.']);

end

checkpointParent = ...
    fileparts(checkpointFile);


if ~isempty(checkpointParent) && ...
   ~exist(checkpointParent,'dir')

    error( ...
        'Stage-1 checkpoint directory does not exist: %s', ...
        checkpointParent);

end

end


% =============================================================
% Local helper
% =============================================================

function tf = reuseCompletedStage1(options)
%REUSECOMPLETEDSTAGE1 Default true; fresh runs delete checkpoints outside.

tf = ...
    true;


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'reuseCompletedStage1') && ...
   ~isempty(options.execution.reuseCompletedStage1)

    value = ...
        options.execution.reuseCompletedStage1;


    if ~(isscalar(value) && ...
         (islogical(value) || isnumeric(value)))

        error( ...
            ['options.execution.reuseCompletedStage1 must be a ' ...
             'scalar logical/numeric value.']);

    end


    tf = ...
        logical(value);

end

end


% =============================================================
% Local helper
% =============================================================

function tf = getSaveTrace(options)
%GETSAVETRACE Master Stage-1 TRACE storage switch. Default true.

% This is an execution/storage option only. It is removed by
% canonicalOptimizationOptions before numerical optimization functions are
% called, so changing it cannot alter GA/grid/RBF calculations.

tf = ...
    true;


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'saveTrace') && ...
   ~isempty(options.execution.saveTrace)

    value = ...
        options.execution.saveTrace;


    if ~(isscalar(value) && ...
         (islogical(value) || ...
          (isnumeric(value) && ...
           isreal(value) && ...
           isfinite(value) && ...
           any(value == [0 1]))))

        error( ...
            ['options.execution.saveTrace must be true/false ' ...
             '(or numeric 1/0).']);

    end


    tf = ...
        logical(value);

end

end



% =============================================================
% Local helper
% =============================================================

function tf = getSaveStage2All(options)
%GETSAVESTAGE2ALL Optional Stage-2 DETAIL retention. Default false.
%
% This is execution/storage-only and is removed from coreOptions together
% with the rest of options.execution before numerical routines are called.

tf = ...
    false;


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'saveStage2All') && ...
   ~isempty(options.execution.saveStage2All)

    value = ...
        options.execution.saveStage2All;


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


    tf = ...
        logical(value);

end


end



% =============================================================
% Local helper
% =============================================================

function stage = getExecutionStage(options)
%GETEXECUTIONSTAGE Resolve optional two-stage execution mode.

stage = ...
    'all';


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'stage') && ...
   ~isempty(options.execution.stage)

    stage = ...
        lower( ...
            char(options.execution.stage));

end


if ~any( ...
        strcmp( ...
            stage, ...
            {'all','stage1','stage2'}))

    error( ...
        ['options.execution.stage must be ''all'', ''stage1'', ' ...
         'or ''stage2''.']);

end

end


% =============================================================
% Local helper
% =============================================================

function checkpointDir = getStage1CheckpointDir(options,executionStage)
%GETSTAGE1CHECKPOINTDIR Validate Stage-1 checkpoint directory.

checkpointDir = ...
    '';


if strcmp(executionStage,'all')
    return
end


if ~isfield(options,'execution') || ...
   ~isstruct(options.execution) || ...
   ~isfield(options.execution,'stage1CheckpointDir') || ...
   isempty(options.execution.stage1CheckpointDir)

    error( ...
        ['Two-stage execution requires ' ...
         'options.execution.stage1CheckpointDir.']);

end


checkpointDir = ...
    char(options.execution.stage1CheckpointDir);


if strcmp(executionStage,'stage1') && ...
   ~exist(checkpointDir,'dir')

    mkdir(checkpointDir);

end


if ~exist(checkpointDir,'dir')

    error( ...
        'Stage-1 checkpoint directory does not exist: %s', ...
        checkpointDir);

end

end


% =============================================================
% Local helper
% =============================================================

function [T,checkpointFile] = loadStage1Source( ...
    checkpointDir, kernel, ...
    X_raw, X_scaled, y, ...
    radii_raw, radii_scaled, ...
    min_neighbors, C_min, C_max, ...
    Reg_min, Reg_max, RNG_SEED, options, method, nGrid)
%LOADSTAGE1SOURCE Strictly load an exact Stage-1 optimization result.

checkpointBaseFile = ...
    getStage1CheckpointFile( ...
        options, ...
        checkpointDir, ...
        kernel, ...
        y, ...
        radii_raw, ...
        method, ...
        nGrid);


checkpointPaths = ...
    localrbf.stage1CheckpointPaths( ...
        checkpointBaseFile);


checkpointFile = ...
    checkpointPaths.core;


% Validate the checkpoint record before loading CORE. CORE+DIAG are
% always required; TRACE files are required only when CORE says they were
% saved. Stage 2 never falls back to automatic re-optimization.
checkpointTraceSaved = ...
    validateCompleteStage1SplitRecord( ...
        checkpointPaths, ...
        kernel, ...
        size(X_raw,1), ...
        numel(radii_raw), ...
        RNG_SEED, ...
        method, ...
        nGrid);


% For Stage-1 reuse, honor an explicit request to save TRACE.
% Stage 2 itself never requires TRACE for scientific computation.
if strcmp(getExecutionStage(options),'stage1') && ...
   getSaveTrace(options) && ...
   ~checkpointTraceSaved

    error( ...
        ['Existing Stage-1 checkpoint was created with saveTrace=false, ' ...
         'but the current Stage-1 run requests saveTrace=true.']);

end


S = ...
    load( ...
        checkpointFile, ...
        'Stage1Core');


if ~isfield(S,'Stage1Core') || ...
   ~isstruct(S.Stage1Core) || ...
   ~isscalar(S.Stage1Core) || ...
   ~isfield(S.Stage1Core,'TrainingResult') || ...
   ~isstruct(S.Stage1Core.TrainingResult)

    error( ...
        'Invalid Stage-1 CORE checkpoint: %s', ...
        checkpointFile);

end



if ~isfield(S.Stage1Core,'complete') || ...
   ~isscalar(S.Stage1Core.complete) || ...
   ~logical(S.Stage1Core.complete)

    error( ...
        'Stage-1 CORE checkpoint is not marked complete: %s', ...
        checkpointFile);

end



T = ...
    S.Stage1Core.TrainingResult;


validateExactStage1( ...
    T, ...
    kernel, ...
    X_raw, ...
    X_scaled, ...
    y, ...
    radii_raw, ...
    radii_scaled, ...
    min_neighbors, ...
    C_min, ...
    C_max, ...
    Reg_min, ...
    Reg_max, ...
    RNG_SEED, ...
    options, ...
    method, nGrid, checkpointFile);



if ~isfield(T,'stage1Complete') || ...
   ~T.stage1Complete

    error( ...
        'Stage-1 checkpoint is incomplete: %s', ...
        checkpointFile);

end


if ~isfield(T,'Point') || ...
   numel(T.Point) ~= size(X_raw,1)

    error( ...
        'Stage-1 checkpoint has an invalid Point array: %s', ...
        checkpointFile);

end


for ii = 1:numel(T.Point)

    if ~isfield(T.Point(ii),'optimization') || ...
       isempty(T.Point(ii).optimization)

        error( ...
            'Stage-1 PointOpt %d is missing in %s.', ...
            ii, ...
            checkpointFile);

    end

end

end


% =============================================================
% Local helper
% =============================================================

function validateExactStage1( ...
    T, kernel, X_raw, X_scaled, y, ...
    radii_raw, radii_scaled, min_neighbors, ...
    C_min, C_max, Reg_min, Reg_max, RNG_SEED, ...
    options, method, nGrid, checkpointFile)
%VALIDATEEXACTSTAGE1 Prevent incompatible Stage-1 reuse.

required = ...
    {'kernel','X','X_raw','S_global','y', ...
     'radii','radii_raw','minNeighbors', ...
     'C_min','C_max','Reg_min','Reg_max', ...
     'RNG_SEED','options', ...
     'tuningMethod','nGrid'};


for ii = 1:numel(required)

    if ~isfield(T,required{ii})

        error( ...
            'Stage-1 checkpoint lacks field %s: %s', ...
            required{ii}, ...
            checkpointFile);

    end

end


if ~strcmpi(char(T.tuningMethod),char(method))

    error( ...
        'Stage-1 tuning-method mismatch for %s.', ...
        checkpointFile);

end


if ~isequaln( ...
        double(T.nGrid(:)), ...
        double(nGrid(:)))

    error( ...
        'Stage-1 nGrid mismatch for %s.', ...
        checkpointFile);

end




if ~strcmpi(char(T.kernel),char(kernel))
    error('Stage-1 kernel mismatch for %s.',checkpointFile);
end


if ~isequaln(double(T.X_raw),double(X_raw))
    error('Stage-1 raw-coordinate mismatch for %s.',checkpointFile);
end


if ~isequaln(double(T.X),double(X_scaled))
    error('Stage-1 scaled-coordinate mismatch for %s.',checkpointFile);
end


if ~isequaln(double(T.y(:)),double(y(:)))
    error('Stage-1 response mismatch for %s.',checkpointFile);
end


if ~isequaln(double(T.radii_raw(:)),double(radii_raw(:)))
    error('Stage-1 raw-radius mismatch for %s.',checkpointFile);
end


if ~isequaln(double(T.radii(:)),double(radii_scaled(:)))
    error('Stage-1 scaled-radius mismatch for %s.',checkpointFile);
end


if ~isequaln(T.minNeighbors,min_neighbors)
    error('Stage-1 min_neighbors mismatch for %s.',checkpointFile);
end


if ~isequaln(T.C_min,C_min) || ...
   ~isequaln(T.C_max,C_max)
    error('Stage-1 C-bound mismatch for %s.',checkpointFile);
end


if ~isequaln(T.Reg_min,Reg_min) || ...
   ~isequaln(T.Reg_max,Reg_max)
    error('Stage-1 regularization-bound mismatch for %s.',checkpointFile);
end


if ~isequaln(T.RNG_SEED,RNG_SEED)
    error('Stage-1 RNG_SEED mismatch for %s.',checkpointFile);
end


stage1Options = ...
    canonicalOptimizationOptions( ...
        T.options);


currentOptions = ...
    canonicalOptimizationOptions( ...
        options);


if ~isequaln(stage1Options,currentOptions)

    error( ...
        ['Stage-1 optimization/numerical options mismatch for %s. ' ...
         'Only non-optimization execution/transfer controls may differ between stages.'], ...
        checkpointFile);

end

end


% =============================================================
% Local helper
% =============================================================
function optionsOut = canonicalOptimizationOptions(optionsIn)
%CANONICALOPTIMIZATIONOPTIONS
% Remove controls that do not affect Stage-1 optimization.
%
% Stage-1 compatibility must depend only on optimization/numerical
% settings. Execution and train-to-test transfer controls may change
% between Stage 1 and Stage 2.

optionsOut = ...
    optionsIn;


% Execution controls do not affect Stage-1 numerical optimization.
if isstruct(optionsOut) && ...
   isfield(optionsOut,'execution')

    optionsOut = ...
        rmfield( ...
            optionsOut, ...
            'execution');

end


% Transfer controls are used only during Stage-2 testing and therefore
% must not invalidate an existing Stage-1 optimization checkpoint.
if isstruct(optionsOut) && ...
   isfield(optionsOut,'transfer')

    optionsOut = ...
        rmfield( ...
            optionsOut, ...
            'transfer');

end

end


% =============================================================
% Local helper
% =============================================================

function t = pointTotalElapsed(stage,optimizationWallTime,tPoint)
%POINTTOTALELAPSED Preserve cumulative per-point cost in Stage 2.

currentElapsed = ...
    toc(tPoint);


if strcmp(stage,'stage2')

    t = ...
        optimizationWallTime + ...
        currentElapsed;

else

    t = ...
        currentElapsed;

end

end


% =============================================================
% Local helper
% =============================================================

function value = getNumericScalarOrDefault(S,fieldName,defaultValue)
%GETNUMERICSCALARORDEFAULT Read a finite scalar when available.

value = ...
    defaultValue;


if isstruct(S) && ...
   isfield(S,fieldName) && ...
   isnumeric(S.(fieldName)) && ...
   isscalar(S.(fieldName)) && ...
   isfinite(S.(fieldName))

    value = ...
        S.(fieldName);

end

end


% =============================================================
% Local helper
% =============================================================

function tf = allStage1SplitFilesExist(Paths,requireTrace)
%ALLSTAGE1SPLITFILESEXIST Check required Stage-1 files for storage mode.

if nargin < 2 || isempty(requireTrace)

    requireTrace = ...
        true;

end


requireTrace = ...
    logical(requireTrace);


requiredFields = ...
    {'core','diag'};


if requireTrace

    requiredFields = ...
        [requiredFields, ...
         {'traceSearch','traceOpt','traceNumeric'}];

end


tf = ...
    isstruct(Paths) && ...
    isscalar(Paths);


if ~tf
    return
end


for ii = 1:numel(requiredFields)

    f = ...
        requiredFields{ii};


    if ~isfield(Paths,f) || ...
       ~(ischar(Paths.(f)) || ...
         (isstring(Paths.(f)) && isscalar(Paths.(f)))) || ...
       ~isfile(char(Paths.(f)))

        tf = ...
            false;

        return

    end

end

end



% =============================================================
% Local helper
% =============================================================

function traceSaved = validateCompleteStage1SplitRecord( ...
    Paths,kernel,nPoints,nRadii,RNG_SEED,method,nGrid)
%VALIDATECOMPLETESTAGE1SPLITRECORD Validate required Stage-1 files.
%
% CORE records whether TRACE was saved. CORE+DIAG are always required.
% The three TRACE files are required only when CORE.traceSaved is true.

if ~allStage1SplitFilesExist(Paths,false)

    error( ...
        ['Complete CORE+DIAG Stage-1 checkpoint was not found. ' ...
         'Stage 2 never re-optimizes automatically.']);

end


SCore = ...
    load( ...
        Paths.core, ...
        'Stage1Core');


if ~isfield(SCore,'Stage1Core') || ...
   ~isstruct(SCore.Stage1Core) || ...
   ~isscalar(SCore.Stage1Core)

    error( ...
        'Invalid Stage-1 CORE metadata: %s', ...
        Paths.core);

end


if ~isfield(SCore.Stage1Core,'storageSplitVersion') || ...
   ~isnumeric(SCore.Stage1Core.storageSplitVersion) || ...
   ~isscalar(SCore.Stage1Core.storageSplitVersion) || ...
   ~isfinite(SCore.Stage1Core.storageSplitVersion) || ...
   SCore.Stage1Core.storageSplitVersion ~= 2

    error( ...
        'Unsupported Stage-1 CORE storageSplitVersion in %s.', ...
        Paths.core);

end


if ~isfield(SCore.Stage1Core,'storageRole') || ...
   ~strcmp(char(SCore.Stage1Core.storageRole),'CORE')

    error( ...
        'Stage-1 CORE storageRole mismatch in %s.', ...
        Paths.core);

end


% Backward compatibility: split-v2 always saved all TRACE files.
if isfield(SCore.Stage1Core,'traceSaved')

    value = ...
        SCore.Stage1Core.traceSaved;


    if ~(isscalar(value) && ...
         (islogical(value) || ...
          (isnumeric(value) && ...
           isreal(value) && ...
           isfinite(value) && ...
           any(value == [0 1]))))

        error( ...
            'Invalid Stage-1 CORE traceSaved metadata: %s', ...
            Paths.core);

    end


    traceSaved = ...
        logical(value);

else

    traceSaved = ...
        true;

end


DiagMeta = ...
    validateStage1PartMetadata( ...
        Paths.diag, ...
        'Stage1Diag', ...
        'DIAG', ...
        kernel, ...
        nPoints, ...
        nRadii, ...
        RNG_SEED, ...
        method, ...
        nGrid, ...
        false);


if isfield(DiagMeta,'traceSaved') && ...
   logical(DiagMeta.traceSaved) ~= traceSaved

    error( ...
        'Stage-1 CORE/DIAG traceSaved metadata mismatch.');

end


if traceSaved

    if ~allStage1SplitFilesExist(Paths,true)

        error( ...
            ['Stage-1 CORE reports traceSaved=true, but one or more ' ...
             'TRACE files are missing.']);

    end


    SearchMeta = ...
        validateStage1PartMetadata( ...
            Paths.traceSearch, ...
            'Stage1TraceSearch', ...
            'TRACE_SEARCH', ...
            kernel, ...
            nPoints, ...
            nRadii, ...
            RNG_SEED, ...
            method, ...
            nGrid, ...
            true);


    if isfield(SearchMeta,'traceSaved') && ...
       ~logical(SearchMeta.traceSaved)

        error( ...
            'Stage-1 TRACE_SEARCH reports traceSaved=false.');

    end


    OptMeta = ...
        validateStage1PartMetadata( ...
            Paths.traceOpt, ...
            'Stage1TraceOpt', ...
            'TRACE_OPT', ...
            kernel, ...
            nPoints, ...
            nRadii, ...
            RNG_SEED, ...
            method, ...
            nGrid, ...
            true);


    if isfield(OptMeta,'traceSaved') && ...
       ~logical(OptMeta.traceSaved)

        error( ...
            'Stage-1 TRACE_OPT reports traceSaved=false.');

    end


    NumericMeta = ...
        validateStage1PartMetadata( ...
            Paths.traceNumeric, ...
            'Stage1TraceNumeric', ...
            'TRACE_NUMERIC', ...
            kernel, ...
            nPoints, ...
            nRadii, ...
            RNG_SEED, ...
            method, ...
            nGrid, ...
            true);


    if isfield(NumericMeta,'traceSaved') && ...
       ~logical(NumericMeta.traceSaved)

        error( ...
            'Stage-1 TRACE_NUMERIC reports traceSaved=false.');

    end

end

end



% =============================================================
% Local helper
% =============================================================

function Meta = validateStage1PartMetadata( ...
    fileName,variableName,expectedRole, ...
    kernel,nPoints,nRadii,RNG_SEED,method,nGrid, ...
    requireTopLevelPoint)
%VALIDATESTAGE1PARTMETADATA Validate auxiliary split-file metadata.
%
% For progressive TRACE files, the large Point payload is a separate
% top-level variable. Therefore loading VARIABLEName reads only compact
% metadata and never materializes the trace payload.

W = ...
    whos( ...
        '-file', ...
        fileName);


names = ...
    {W.name};


if ~any(strcmp(names,variableName))

    error( ...
        'Stage-1 file lacks variable %s: %s', ...
        variableName, ...
        fileName);

end


if requireTopLevelPoint && ...
   ~any(strcmp(names,'Point'))

    error( ...
        ['Stage-1 trace file is not in progressive split-v3 layout ' ...
         '(top-level Point missing): %s'], ...
        fileName);

end


S = ...
    load( ...
        fileName, ...
        variableName);


Meta = ...
    S.(variableName);


if ~isstruct(Meta) || ...
   ~isscalar(Meta)

    error( ...
        'Invalid Stage-1 metadata variable %s in %s.', ...
        variableName, ...
        fileName);

end


required = ...
    { ...
    'storageSplitVersion', ...
    'storageRole', ...
    'kernel', ...
    'nPoints', ...
    'nRadii', ...
    'complete', ...
    'RNG_SEED', ...
    'tuningMethod', ...
    'nGrid' ...
    };


for k = 1:numel(required)

    if ~isfield(Meta,required{k})

        error( ...
            'Stage-1 metadata lacks field %s: %s', ...
            required{k}, ...
            fileName);

    end

end


if ~isnumeric(Meta.storageSplitVersion) || ...
   ~isscalar(Meta.storageSplitVersion) || ...
   ~isfinite(Meta.storageSplitVersion) || ...
   Meta.storageSplitVersion ~= 2

    error( ...
        'Unsupported Stage-1 storageSplitVersion for %s.', ...
        fileName);

end


if ~strcmp(char(Meta.storageRole),expectedRole)

    error( ...
        'Stage-1 storage-role mismatch for %s.', ...
        fileName);

end


if ~strcmpi(char(Meta.kernel),char(kernel))

    error( ...
        'Stage-1 kernel mismatch for %s.', ...
        fileName);

end


if ~isequaln(double(Meta.nPoints),double(nPoints)) || ...
   ~isequaln(double(Meta.nRadii),double(nRadii))

    error( ...
        'Stage-1 point/radius count mismatch for %s.', ...
        fileName);

end


if ~isscalar(Meta.complete) || ...
   ~logical(Meta.complete)

    error( ...
        'Stage-1 auxiliary part is not marked complete: %s', ...
        fileName);

end


if ~isequaln(Meta.RNG_SEED,RNG_SEED)

    error( ...
        'Stage-1 RNG_SEED mismatch for %s.', ...
        fileName);

end


if ~strcmpi(char(Meta.tuningMethod),char(method))

    error( ...
        'Stage-1 tuning-method mismatch for %s.', ...
        fileName);

end


if ~isequaln( ...
        double(Meta.nGrid(:)), ...
        double(nGrid(:)))

    error( ...
        'Stage-1 nGrid mismatch for %s.', ...
        fileName);

end


if isfield(Meta,'nStage1PointsMissing') && ...
   (~isscalar(Meta.nStage1PointsMissing) || ...
    Meta.nStage1PointsMissing ~= 0)

    error( ...
        'Stage-1 auxiliary part reports missing points: %s', ...
        fileName);

end

end


% =============================================================
% Local helper
% =============================================================

function S = initializeStage1StorageMetadata(role,kernel,nPoints,nRadii)
%INITIALIZESTAGE1STORAGEMETADATA Metadata only; no Point payload in RAM.

S = ...
    struct();


S.storageSplitVersion = ...
    2;


S.storageRole = ...
    role;


S.storageFileLayout = ...
    'progressive-point-v1';


S.kernel = ...
    kernel;


S.nPoints = ...
    nPoints;


S.nRadii = ...
    nRadii;


S.complete = ...
    false;

end


% =============================================================
% Local helper
% =============================================================

function deleteStage1TraceFiles(Paths)
%DELETESTAGE1TRACEFILES Remove stale TRACE files for TRACE-OFF runs.

fields = ...
    {'traceSearch','traceOpt','traceNumeric'};


for k = 1:numel(fields)

    f = ...
        fields{k};


    if isfield(Paths,f) && ...
       isfile(Paths.(f))

        delete(Paths.(f));

    end

end

end



% =============================================================
% Local helper
% =============================================================

function cleanupTemporaryStage1Files(files)
%CLEANUPTEMPORARYSTAGE1FILES Best-effort cleanup of progressive temp files.

for k = 1:numel(files)

    fileName = ...
        files{k};


    if isempty(fileName)
        continue
    end


    if isfile(fileName)

        try

            delete(fileName);

        catch

            % Best-effort cleanup only.

        end

    end

end

end


% =============================================================
% Local helper
% =============================================================

function fileName = getWriterTempFile(Writer)
%GETWRITERTEMPFILE Safe temp-file extraction during partial initialization.

fileName = ...
    '';


if isstruct(Writer) && ...
   isscalar(Writer) && ...
   isfield(Writer,'tempFile') && ...
   ~isempty(Writer.tempFile)

    fileName = ...
        Writer.tempFile;

end

end


% =============================================================
% Local helper
% =============================================================

function S = initializeStage1StoragePart(role,kernel,nPoints,nRadii)
%INITIALIZESTAGE1STORAGEPART Preallocate one split Stage-1 trace part.

S = ...
    struct();


S.storageSplitVersion = ...
    2;


S.storageRole = ...
    role;


S.kernel = ...
    kernel;


S.nPoints = ...
    nPoints;


S.nRadii = ...
    nRadii;


% Cell storage avoids repeated struct-array reshaping while PointOpt
% contents are detached one point at a time.
S.Point = ...
    cell(nPoints,1);


S.complete = ...
    false;

end



% =============================================================
% Local helper
% =============================================================

function S = finalizeStage1StoragePart(S,T,method,nGrid)
%FINALIZESTAGE1STORAGEPART Add compact cross-file provenance metadata.

S.complete = ...
    T.stage1Complete;


S.nStage1PointsComplete = ...
    T.nStage1PointsComplete;


S.nStage1PointsMissing = ...
    T.nStage1PointsMissing;


S.nSuccessfulPointOptimizations = ...
    T.nSuccessfulPointOptimizations;


S.nUnsuccessfulPointOptimizations = ...
    T.nUnsuccessfulPointOptimizations;


S.stage1WallTime = ...
    T.stage1WallTime;


S.RNG_SEED = ...
    T.RNG_SEED;


S.tuningMethod = ...
    method;


S.nGrid = ...
    nGrid;


S.traceSaved = ...
    T.stage1TraceSaved;

end



% =============================================================
% Local helper
% =============================================================

function P = emptyPointTrainingResult()
%EMPTYPOINTTRAININGRESULT
% Uniform training record for one target observation.


P = ...
    struct();


% =============================================================
% Point identity
% =============================================================

P.iTarget = ...
    NaN;


P.xTarget = ...
    [];


P.yTarget = ...
    NaN;


% =============================================================
% Detailed retained results
% =============================================================

P.optimization = ...
    [];


P.capSelection = ...
    [];


% =============================================================
% Status
% =============================================================

P.optimizationSuccess = ...
    false;


P.capSelectionSuccess = ...
    false;


P.success = ...
    false;


P.failureStage = ...
    '';


P.message = ...
    '';


% =============================================================
% Timing
% =============================================================

P.optimizationWallTime = ...
    0;


P.capSelectionWallTime = ...
    0;


P.totalWallTime = ...
    0;

end



% =============================================================
% Local helper
% =============================================================

function T = stripNonrequestedStage2Detail(T)
%STRIPNONREQUESTEDSTAGE2DETAIL Release completed Stage-2 working/detail data.
%
% Called only when saveStage2All=false and only after evaluateLocal and all
% canonical selected-model outputs have been completed.

fields = { ...
    'Point', ...
    'cap_success_by_point', ...
    'selectedRadiusIndex_by_point_cap', ...
    'selectedRadius_by_point_cap', ...
    'selectedAE_by_point_cap', ...
    'minAE_by_point_cap', ...
    'errorThreshold_by_point_cap', ...
    'C_by_point_cap', ...
    'Reg_by_point_cap', ...
    'epsilon_by_point_cap', ...
    'd_local_by_point_cap', ...
    'yhat_by_point_cap', ...
    'residual_by_point_cap', ...
    'y_diff_by_point_cap', ...
    'squaredResidual_by_point_cap', ...
    'cond_reg_by_point_cap', ...
    'rel_residual_by_point_cap', ...
    'solver_used_by_point_cap', ...
    'neighbor_idx_by_point_cap', ...
    'neighbor_distances_by_point_cap', ...
    'neighbor_count_by_point_cap', ...
    'effective_radius_by_point_cap', ...
    'selected_max_distance_by_point_cap', ...
    'n_within_radius_by_point_cap', ...
    'n_added_for_minimum_by_point_cap', ...
    'usedMinNeighborFallback_by_point_cap', ...
    'validResidualMask', ...
    'nValidPointsPerCap', ...
    'nFailedPointsPerCap', ...
    'SSE_available_all', ...
    'MSE_available_all', ...
    'RMSE_available_all', ...
    'SSE_all', ...
    'MSE_all', ...
    'RMSE_all', ...
    'cap_success', ...
    'capSelection' ...
    };


existing = ...
    intersect( ...
        fields, ...
        fieldnames(T), ...
        'stable');


if ~isempty(existing)

    T = ...
        rmfield( ...
            T, ...
            existing);

end

end



% =============================================================
% Local helper
% =============================================================

function value = getFieldOrDefault( ...
    S, fieldName, defaultValue)

if isstruct(S) && ...
   isfield(S,fieldName)

    value = ...
        S.(fieldName);

else

    value = ...
        defaultValue;

end

end

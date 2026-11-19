function Results = localRBFBatchSummaryParallel_TuneReg( ...
    InputData, CaseName, ListOfModel, NameFunction_RBF, ...
    Radii, min_neighbors, ...
    CMin, CMax, RegMin, RegMax, ...
    tau_loc, tau_cap, ...
    near_threshold, idw_radius, ...
    verbose, RNG_SEED, options, OutputFile, ...
    Method, nGrid)
%LOCALRBFBATCHSUMMARYPARALLEL_TUNEREG
%
% Parallel batch manager for local RBF training and independent testing.
%
% Optional two-stage execution is controlled by options.execution.stage:
%   'all'    : original training + testing workflow;
%   'stage1' : optimize all candidate radii only; no testing;
%   'stage2' : load Stage-1 PointOpt, finish training, then test.
%
% One PARFOR iteration = one Model x Kernel run.
% Inner GA execution remains serial inside localrbf.optimizeParameters.
%
% Expected input structure:
%
%   InputData.Train.XYZ
%   InputData.Test.XYZ
%   InputData.Train.DeltaZeta.<model>
%   InputData.Test.DeltaZeta.<model>
%
% Public coordinate inputs use ORIGINAL coordinate units.
% LocalRBFTraining performs the common S_global scaling internally.
% LocalRBFTesting reuses the same training-derived S_global.
%
% Stage-1 authoritative data are stored as CORE + DIAG and optional TRACE.
% Stage-2/all scientific output uses one self-contained compulsory CORE
% plus optional DETAIL. CORE contains the collector-equivalent final
% scientific result, all three pointwise numerical-condition diagnostics,
% runtime, and the data/final settings required by fixed-setting bootstrap.
% Complete TrainingResult and TestingResult objects are never retained
% inside batch Results during normal successful execution.
%
% IMPORTANT:
%   - min_neighbors is configurable; default = 15;
%   - Radii may be a kernel-specific structure or one common vector;
%   - all candidate radii use the same original units as Train.XYZ;
%   - near_threshold and idw_radius are supplied in the same original
%     units as Train.XYZ/Test.XYZ;
%   - options.transfer.method may be one common text value or an
%     nModel x nKernel cell/string matrix; allowed values are:
%         'nearest', 'idw', 'hybrid', or 'ensemble';
%   - options.transfer.nIDWSources may be one common positive integer or
%     an nModel x nKernel numeric matrix for IDW/hybrid/ensemble;
%   - options.transfer.parameterTransferMode may be one common text value
%     or an nModel x nKernel cell/string matrix;
%     M21-M24 apply only to IDW/hybrid. For nearest/ensemble the normalized
%     stored value is 'not-applicable' because aggregation is inactive;
%   - options.transfer.capRMSESource selects final-cap RMSE:
%         'training' or 'pseudo';
%   - near_threshold may be one common positive scalar or an
%     nModel x nKernel numeric matrix in raw coordinate units;
%   - each run's near_threshold is copied to optionsRun.transfer.
%     nearThresholdRaw so pseudo-training cap selection and independent
%     testing use exactly the same per-combination threshold;
%   - near_threshold is used for source selection only in hybrid/ensemble;
%   - idw_radius is retained for backward compatibility/metadata only and
%     is not used for source selection;
%   - radius and RegParam are transferred for all kernels;
%   - C is additionally transferred for shaped kernels;
%   - shaped-kernel epsilon is recomputed from test neighbourhood geometry;
%   - each training run returns geometryCache and the SAME cache is reused
%     immediately by LocalRBFTesting;
%   - geometryCache is not retained in Results because it is reconstructable;
%   - Stage-1 authoritative data are stored by LocalRBFTraining as
%     CORE + DIAG and optionally TRACE_SEARCH/TRACE_OPT/TRACE_NUMERIC;
%   - options.execution.saveTrace controls only Stage-1 TRACE persistence;
%   - ordinary per-run batch checkpoints are skipped in Stage 1 because
%     the split Stage-1 files are already the authoritative checkpoint;
%   - Stage-2/all writes one self-contained compulsory CORE per run;
%   - options.execution.saveStage2All controls optional full Stage-2 DETAIL;
%   - when saveStage2All=false, options.execution.saveEnsemblePointDiagnostics
%     optionally retains lightweight per-tuple ensemble test diagnostics in
%     CORE; false retains only the compulsory conservative maxima;
%   - tau_loc and tau_cap may be:
%       (a) one common nonnegative scalar; or
%       (b) an nModel x nKernel numeric matrix;
%     rows follow ListOfModel and columns follow NameFunction_RBF;
%     the scalar form is expanded internally to the full matrix;
%   - options.execution.saveStage1Summary controls only the extra Stage-1 batch summary file;
%   - RNG_SEED is fixed for each independent Model x Kernel run so PARFOR
%     scheduling does not change the numerical search sequence.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 5 || isempty(Radii)

    Radii.gaussian = (30:10:200).';
    Radii.multiquadric = (30:10:200).';

    Radii.thinplate = (30:10:200).';
    Radii.cubic = (30:10:200).';
    Radii.linear = (30:10:200).';

end

if nargin < 6 || isempty(min_neighbors)
    min_neighbors = 15;
end

if nargin < 7 || isempty(CMin)

    CMin.gaussian = 0.01;
    CMin.multiquadric = 0.001;

    CMin.thinplate = 1;
    CMin.cubic = 1;
    CMin.linear = 1;

end

if nargin < 8 || isempty(CMax)

    CMax.gaussian = 2;
    CMax.multiquadric = 2;

    CMax.thinplate = 1;
    CMax.cubic = 1;
    CMax.linear = 1;

end

if nargin < 9 || isempty(RegMin)

    RegMin.gaussian = 1e-8;
    RegMin.multiquadric = 1e-8;

    RegMin.thinplate = 1e-8;
    RegMin.cubic = 1e-8;
    RegMin.linear = 1e-8;

end

if nargin < 10 || isempty(RegMax)

    RegMax.gaussian = 1e-2;
    RegMax.multiquadric = 1e-2;

    RegMax.thinplate = 1e-1;
    RegMax.cubic = 1e-1;
    RegMax.linear = 1e-1;

end

if nargin < 11 || isempty(tau_loc)
    tau_loc = 0;
end

if nargin < 12 || isempty(tau_cap)
    tau_cap = 0;
end

if nargin < 13 || isempty(near_threshold)
    near_threshold = 10;
end

if nargin < 14 || isempty(idw_radius)
    idw_radius = 30;
end

if nargin < 15 || isempty(verbose)
    verbose = true;
end

if nargin < 16
    RNG_SEED = 1;
end

if nargin < 17 || isempty(options)
    options = struct();
end

if nargin < 18
    OutputFile = '';
end


if nargin < 19 || isempty(Method)
    Method = struct();
end

if nargin < 20 || isempty(nGrid)
    nGrid = struct();
end
% =============================================================
% 2. Basic option validation
% =============================================================

if ~isstruct(options)
    error('options must be a structure.');
end

if ~isfield(options,'solver') || ~isstruct(options.solver)
    options.solver = struct();
end

if ~isfield(options.solver,'cond_lsq') || ...
   isempty(options.solver.cond_lsq)

    options.solver.cond_lsq = 1e12;

end

if ~isnumeric(options.solver.cond_lsq) || ...
   ~isscalar(options.solver.cond_lsq) || ...
   ~isreal(options.solver.cond_lsq) || ...
   ~isfinite(options.solver.cond_lsq) || ...
   options.solver.cond_lsq < 1

    error( ...
        'options.solver.cond_lsq must be a finite real scalar >= 1.');

end

% Parameter-search evaluations should not calculate the extra relative
% residual. Selected training/test systems enable it when required.
if ~isfield(options.solver,'compute_rel_residual') || ...
   isempty(options.solver.compute_rel_residual)

    options.solver.compute_rel_residual = false;

end

if ~(isscalar(options.solver.compute_rel_residual) && ...
     (islogical(options.solver.compute_rel_residual) || ...
      isnumeric(options.solver.compute_rel_residual)))

    error( ...
        ['options.solver.compute_rel_residual must be a scalar ' ...
         'logical or numeric value.']);

end

options.solver.compute_rel_residual = ...
    logical(options.solver.compute_rel_residual);


% =============================================================
% 2.1 Optional two-stage execution control
% =============================================================

executionStage = ...
    getExecutionStageBatch( ...
        options);


saveTrace = ...
    getSaveTraceBatch( ...
        options);

% Optional Stage-1 batch-summary persistence. This does NOT control the
% authoritative Stage-1 CORE/DIAG/(TRACE) checkpoints.
saveStage1Summary = ...
    getExecutionLogicalOption( ...
        options, ...
        'saveStage1Summary', ...
        false);

% Optional Stage-2 DETAIL persistence. The self-contained Stage-2 CORE is
% always constructed for Stage-2/all. DETAIL is not constructed at all
% when this flag is false.
saveStage2All = ...
    getExecutionLogicalOption( ...
        options, ...
        'saveStage2All', ...
        false);


saveEnsemblePointDiagnosticsRequested = ...
    getExecutionLogicalOption( ...
        options, ...
        'saveEnsemblePointDiagnostics', ...
        false);


saveEnsemblePointDiagnostics = ...
    ~saveStage2All && ...
    saveEnsemblePointDiagnosticsRequested;


% =============================================================
% 2.2 Transfer / cap-selection options
% =============================================================

if ~isfield(options,'transfer') || ...
   isempty(options.transfer)

    options.transfer = ...
        struct();

end


if ~isstruct(options.transfer)

    error( ...
        'options.transfer must be a structure.');

end


% -------------------------------------------------------------
% Transfer method setting
%
% Exact GGM x Kernel normalization is deferred until ListOfModel and
% NameFunction_RBF are known.
% -------------------------------------------------------------

if ~isfield(options.transfer,'method') || ...
   isempty(options.transfer.method)

    options.transfer.method = ...
        'hybrid';

end


transferMethodSetting = ...
    options.transfer.method;


% -------------------------------------------------------------
% N-nearest donor count setting
%
% Exact GGM x Kernel normalization is deferred until dimensions are known.
% -------------------------------------------------------------

if ~isfield(options.transfer,'nIDWSources') || ...
   isempty(options.transfer.nIDWSources)

    options.transfer.nIDWSources = ...
        7;

end


nIDWSourcesSetting = ...
    options.transfer.nIDWSources;


% -------------------------------------------------------------
% RegParam/C transfer-mode setting
%
% M21 = arithmetic_arithmetic
% M22 = log_arithmetic
% M23 = arithmetic_log
% M24 = log_log
%
% Exact GGM x Kernel normalization is deferred until dimensions and the
% corresponding transfer method are known.
% -------------------------------------------------------------

if ~isfield(options.transfer,'parameterTransferMode') || ...
   isempty(options.transfer.parameterTransferMode)

    options.transfer.parameterTransferMode = ...
        'log_log';

end


parameterTransferModeSetting = ...
    options.transfer.parameterTransferMode;


% -------------------------------------------------------------
% Final-cap RMSE source
% -------------------------------------------------------------

if ~isfield(options.transfer,'capRMSESource') || ...
   isempty(options.transfer.capRMSESource)

    options.transfer.capRMSESource = ...
        'training';

end


capRMSESource = ...
    normalizeCapRMSESourceBatch( ...
        options.transfer.capRMSESource);


% -------------------------------------------------------------
% Canonical synchronized transfer options
%
% CAP_RMSE_SOURCE is common. N_IDW_SOURCES and near_threshold are
% normalized to GGM x Kernel matrices after nModel/nKernel are known.
% -------------------------------------------------------------

options.transfer.capRMSESource = ...
    capRMSESource;


nearThresholdSetting = ...
    near_threshold;


% =============================================================
% 3. Reproducibility for parallel execution
% =============================================================

if isempty(RNG_SEED)

    error( ...
        ['RNG_SEED must be nonempty for parallel batch execution ' ...
         'to make each Model x Kernel run reproducible.']);

end

if ~isnumeric(RNG_SEED) || ...
   ~isscalar(RNG_SEED) || ...
   ~isreal(RNG_SEED) || ...
   ~isfinite(RNG_SEED) || ...
   RNG_SEED < 0 || ...
   RNG_SEED ~= round(RNG_SEED)

    error('RNG_SEED must be a finite nonnegative integer scalar.');

end


% =============================================================
% 4. Scalar-setting validation
% =============================================================

if ~isnumeric(min_neighbors) || ...
   ~isscalar(min_neighbors) || ...
   ~isreal(min_neighbors) || ...
   ~isfinite(min_neighbors) || ...
   min_neighbors < 1 || ...
   min_neighbors ~= round(min_neighbors)

    error('min_neighbors must be a positive integer.');

end

% tau_loc and tau_cap are numeric:
%   - one common scalar; or
%   - one nModel x nKernel matrix.
% Exact dimensions are validated after ListOfModel/NameFunction_RBF are
% normalized.
if ~isnumeric(tau_loc)
    error('tau_loc must be a numeric scalar or nModel x nKernel matrix.');
end

if ~isnumeric(tau_cap)
    error('tau_cap must be a numeric scalar or nModel x nKernel matrix.');
end

if ~isnumeric(near_threshold) || ...
   ~isreal(near_threshold)

    error( ...
        ['near_threshold must be a positive numeric scalar or ' ...
         'nModel x nKernel matrix.']);

end

if ~isnumeric(idw_radius) || ...
   ~isscalar(idw_radius) || ...
   ~isreal(idw_radius) || ...
   ~isfinite(idw_radius) || ...
   idw_radius <= 0

    error( ...
        'idw_radius must be a finite positive real scalar.');

end

if ~(isscalar(verbose) && ...
     (islogical(verbose) || isnumeric(verbose)))

    error('verbose must be a scalar logical or numeric value.');

end

verbose = logical(verbose);


% =============================================================
% 5. Input data
% =============================================================

CaseName = char(CaseName);

ListOfModel = ...
    cellstr(string(ListOfModel(:)));

NameFunction_RBF = ...
    cellstr(string(NameFunction_RBF(:)));

nModel = numel(ListOfModel);
nKernel = numel(NameFunction_RBF);


if nModel < 1 || nKernel < 1
    error('At least one model and one RBF kernel are required.');
end


% -------------------------------------------------------------
% Normalize Stage-2 tolerance matrices once on the client.
%
% Rows    = ListOfModel
% Columns = NameFunction_RBF
%
% A scalar remains backward compatible and is expanded to every
% Model x Kernel combination. PARFOR then uses direct matrix indexing.
% -------------------------------------------------------------

tau_loc = ...
    normalizeToleranceMatrix( ...
        tau_loc, ...
        nModel, ...
        nKernel, ...
        'tau_loc');


tau_cap = ...
    normalizeToleranceMatrix( ...
        tau_cap, ...
        nModel, ...
        nKernel, ...
        'tau_cap');


% -------------------------------------------------------------
% Normalize GGM x Kernel transfer settings.
%
% Rows    = ListOfModel
% Columns = NameFunction_RBF
%
% A common scalar text value remains backward compatible and is expanded
% to all combinations.
% -------------------------------------------------------------

transferModeByRun = ...
    normalizeTransferMethodMatrixBatch( ...
        transferMethodSetting, ...
        nModel, ...
        nKernel);


parameterTransferModeByRun = ...
    normalizeParameterTransferModeMatrixBatch( ...
        parameterTransferModeSetting, ...
        transferModeByRun, ...
        nModel, ...
        nKernel);


nIDWSourcesByRun = ...
    normalizePositiveIntegerMatrixBatch( ...
        nIDWSourcesSetting, ...
        nModel, ...
        nKernel, ...
        'options.transfer.nIDWSources');


nearThresholdByRun = ...
    normalizePositiveRealMatrixBatch( ...
        nearThresholdSetting, ...
        nModel, ...
        nKernel, ...
        'near_threshold');


options.transfer.method = ...
    transferModeByRun;


options.transfer.parameterTransferMode = ...
    parameterTransferModeByRun;


options.transfer.nIDWSources = ...
    nIDWSourcesByRun;


options.transfer.nearThresholdRaw = ...
    nearThresholdByRun;


if ~isstruct(Method)
    error('Method must be a kernel-specific structure.');
end

if ~isstruct(nGrid)
    error('nGrid must be a kernel-specific structure.');
end


for j = 1:nKernel

    kernelKey = ...
        lower(NameFunction_RBF{j});

    if ~isfield(Method,kernelKey)
        Method.(kernelKey) = 'ga';
    end

    if ~isfield(nGrid,kernelKey)
        nGrid.(kernelKey) = 25;
    end

end

if ~isstruct(InputData) || ...
   ~isfield(InputData,'Train') || ...
   ~isfield(InputData,'Test') || ...
   ~isfield(InputData.Train,'XYZ') || ...
   ~isfield(InputData.Test,'XYZ') || ...
   ~isfield(InputData.Train,'DeltaZeta') || ...
   ~isfield(InputData.Test,'DeltaZeta')

    error('InputData does not contain the required Train/Test fields.');

end

X = InputData.Train.XYZ;
X_test = InputData.Test.XYZ;

if ~isnumeric(X) || ...
   isempty(X) || ...
   ~ismatrix(X) || ...
   ~isreal(X) || ...
   any(~isfinite(X(:)))

    error('InputData.Train.XYZ must be a finite real numeric matrix.');

end

if ~isnumeric(X_test) || ...
   isempty(X_test) || ...
   ~ismatrix(X_test) || ...
   ~isreal(X_test) || ...
   any(~isfinite(X_test(:)))

    error('InputData.Test.XYZ must be a finite real numeric matrix.');

end

if size(X,2) ~= size(X_test,2)

    error( ...
        'Training and testing coordinates must have the same dimensions.');

end

if size(X,1) - 1 < min_neighbors

    error( ...
        ['At least %d observations must remain after target exclusion, ' ...
         'but only %d are available.'], ...
        min_neighbors, ...
        size(X,1)-1);

end


% =============================================================
% 6. Validate and pre-extract model response data before workers
%
% Keeping only the required response vectors avoids broadcasting the
% complete InputData structure to PARFOR workers.
% =============================================================

YTrainByModel = ...
    cell(nModel,1);


YTestByModel = ...
    cell(nModel,1);


for i = 1:nModel

    modelName = ListOfModel{i};

    if ~isfield(InputData.Train.DeltaZeta,modelName)

        error( ...
            'Training data for model "%s" were not found.', ...
            modelName);

    end

    if ~isfield(InputData.Test.DeltaZeta,modelName)

        error( ...
            'Testing data for model "%s" were not found.', ...
            modelName);

    end

    y = InputData.Train.DeltaZeta.(modelName);
    y_test = InputData.Test.DeltaZeta.(modelName);

    if ~isnumeric(y) || ...
       ~isvector(y) || ...
       ~isreal(y) || ...
       numel(y) ~= size(X,1) || ...
       any(~isfinite(y(:)))

        error( ...
            'Invalid training response data for model "%s".', ...
            modelName);

    end

    if ~isnumeric(y_test) || ...
       ~isvector(y_test) || ...
       ~isreal(y_test) || ...
       numel(y_test) ~= size(X_test,1) || ...
       any(~isfinite(y_test(:)))

        error( ...
            'Invalid testing response data for model "%s".', ...
            modelName);

    end


    YTrainByModel{i} = ...
        y(:);


    YTestByModel{i} = ...
        y_test(:);

end


% =============================================================
% 7. Validate kernel-specific search bounds
% =============================================================

for j = 1:nKernel

    kernel = ...
        NameFunction_RBF{j};

    kernelKey = ...
        lower(kernel);


    % ---------------------------------------------------------
    % Kernel-specific tuning method
    % ---------------------------------------------------------

    methodKernel = ...
        lower( ...
            strtrim( ...
                char(Method.(kernelKey))));


    if ~ismember( ...
            methodKernel, ...
            {'ga','grid'})

        error( ...
            'Invalid tuning Method for kernel "%s".', ...
            kernel);

    end


    Method.(kernelKey) = ...
        methodKernel;


    % ---------------------------------------------------------
    % Kernel-specific grid setting
    % ---------------------------------------------------------

    gridKernel = ...
        nGrid.(kernelKey);


    if ~isnumeric(gridKernel) || ...
       isempty(gridKernel) || ...
       ~isvector(gridKernel) || ...
       numel(gridKernel) > 2 || ...
       ~isreal(gridKernel) || ...
       any(~isfinite(gridKernel(:))) || ...
       any(gridKernel(:) < 1) || ...
       any(gridKernel(:) ~= round(gridKernel(:)))

        error( ...
            'Invalid nGrid for kernel "%s".', ...
            kernel);

    end


    nGrid.(kernelKey) = ...
        gridKernel(:).';


    % ---------------------------------------------------------
    % Candidate radii
    % ---------------------------------------------------------

    radiiKernel = ...
        getKernelRadii( ...
            Radii, ...
            kernelKey);

    useShape = ...
        common.kernelHasShapeParameter(kernel);

    if useShape

        if ~isfield(CMin,kernelKey) || ...
           ~isfield(CMax,kernelKey)

            error( ...
                'C bounds were not provided for kernel "%s".', ...
                kernel);

        end

        cmin = CMin.(kernelKey);
        cmax = CMax.(kernelKey);

        if ~isnumeric(cmin) || ...
           ~isscalar(cmin) || ...
           ~isreal(cmin) || ...
           ~isfinite(cmin) || ...
           cmin <= 0 || ...
           ~isnumeric(cmax) || ...
           ~isscalar(cmax) || ...
           ~isreal(cmax) || ...
           ~isfinite(cmax) || ...
           cmax < cmin

            error('Invalid C bounds for kernel "%s".',kernel);

        end

    end

    if ~isfield(RegMin,kernelKey) || ...
       ~isfield(RegMax,kernelKey)

        error( ...
            'RegParam bounds were not provided for kernel "%s".', ...
            kernel);

    end

    rmin = RegMin.(kernelKey);
    rmax = RegMax.(kernelKey);

    if ~isnumeric(rmin) || ...
       ~isscalar(rmin) || ...
       ~isreal(rmin) || ...
       ~isfinite(rmin) || ...
       rmin <= 0 || ...
       ~isnumeric(rmax) || ...
       ~isscalar(rmax) || ...
       ~isreal(rmax) || ...
       ~isfinite(rmax) || ...
       rmax < rmin

        error('Invalid RegParam bounds for kernel "%s".',kernel);

    end

if isscalar(gridKernel)

    nCgridKernel = ...
        gridKernel;

    nRgridKernel = ...
        gridKernel;

else

    nCgridKernel = ...
        gridKernel(1);

    nRgridKernel = ...
        gridKernel(2);

end


if useShape && ...
   cmax > cmin && ...
   nCgridKernel < 2

    error( ...
        'nGrid for C must be at least 2 for kernel "%s".', ...
        kernel);

end


if rmax > rmin && ...
   nRgridKernel < 2

    error( ...
        'nGrid for RegParam must be at least 2 for kernel "%s".', ...
        kernel);

end

end


% =============================================================
% 8. Output/checkpoint preparation
% =============================================================

if ~isempty(OutputFile)

    outputDir = fileparts(OutputFile);

    if ~isempty(outputDir) && ...
       ~exist(outputDir,'dir')

        mkdir(outputDir);

    end

end

checkpointDir = '';

% Stage-2 self-contained CORE files are written per Model x Kernel run.
% Prefer an explicit directory; otherwise derive it from OutputFile.
% Stage 1 never creates Stage-2 files.
if ~strcmp(executionStage,'stage1')

    if isfield(options,'execution') && ...
       isstruct(options.execution) && ...
       isfield(options.execution,'stage2CheckpointDir') && ...
       ~isempty(options.execution.stage2CheckpointDir)

        checkpointDir = ...
            char(options.execution.stage2CheckpointDir);

    elseif ~isempty(OutputFile)

        checkpointDir = ...
            fullfile( ...
                fileparts(OutputFile), ...
                [CaseName '_local_stage2_checkpoints']);

    end

    if ~isempty(checkpointDir) && ...
       ~exist(checkpointDir,'dir')

        mkdir(checkpointDir);

    end

end


% -------------------------------------------------------------
% Stage-1 optimization checkpoint directory
% -------------------------------------------------------------

stage1CheckpointDir = ...
    '';


if ~strcmp(executionStage,'all')

    if isfield(options,'execution') && ...
       isstruct(options.execution) && ...
       isfield(options.execution,'stage1CheckpointDir') && ...
       ~isempty(options.execution.stage1CheckpointDir)

        stage1CheckpointDir = ...
            char(options.execution.stage1CheckpointDir);

    elseif ~isempty(OutputFile)

        stage1CheckpointDir = ...
            fullfile( ...
                fileparts(OutputFile), ...
                [CaseName '_local_stage1_checkpoints']);

    else

        error( ...
            ['Two-stage execution requires either OutputFile or ' ...
             'options.execution.stage1CheckpointDir.']);

    end


    if strcmp(executionStage,'stage1') && ...
       ~exist(stage1CheckpointDir,'dir')

        mkdir(stage1CheckpointDir);

    end


    if ~exist(stage1CheckpointDir,'dir')

        error( ...
            'Stage-1 checkpoint directory does not exist: %s', ...
            stage1CheckpointDir);

    end

end


% Standalone Stage 2 may be launched with only an existing Stage-1
% checkpoint directory and no OutputFile.  In that case, derive a sibling
% Stage-2 checkpoint directory so compulsory Stage-2 output is still saved.
if strcmp(executionStage,'stage2') && ...
   isempty(checkpointDir) && ...
   ~isempty(stage1CheckpointDir)

    checkpointDir = ...
        fullfile( ...
            fileparts(stage1CheckpointDir), ...
            [CaseName '_local_stage2_checkpoints']);

    if ~exist(checkpointDir,'dir')
        mkdir(checkpointDir);
    end

end


% =============================================================
% 9. Initialize results
% =============================================================

Results = struct();
Results.Run = struct();

Results.CaseName = CaseName;
Results.ListOfModel = ListOfModel;
Results.NameFunction_RBF = NameFunction_RBF;

Results.Settings.Method = ...
    Method;

Results.Settings.method = ...
    Method;     % retain existing field as compatibility alias

Results.Settings.nGrid = ...
    nGrid;
Results.Settings.ExecutionStage = executionStage;
Results.Settings.SaveTraceRequested = saveTrace;
Results.Settings.SaveStage1Summary = saveStage1Summary;
Results.Settings.SaveStage2All = saveStage2All;
Results.Settings.SaveEnsemblePointDiagnosticsRequested = ...
    saveEnsemblePointDiagnosticsRequested;
Results.Settings.SaveEnsemblePointDiagnostics = ...
    saveEnsemblePointDiagnostics;
Results.Settings.Stage2CoreSchema = 'LocalRBF.Stage2Core.v2';
Results.Settings.Stage1CheckpointDir = stage1CheckpointDir;
Results.Settings.Stage2CheckpointDir = checkpointDir;
Results.Settings.Radii = Radii;
Results.Settings.minNeighbors = min_neighbors;
Results.Settings.CMin = CMin;
Results.Settings.CMax = CMax;
Results.Settings.RegMin = RegMin;
Results.Settings.RegMax = RegMax;
Results.Settings.tau_loc = tau_loc;
Results.Settings.tau_cap = tau_cap;
Results.Settings.TauLoc = tau_loc;
Results.Settings.TauCap = tau_cap;
Results.Settings.near_threshold = nearThresholdByRun;
Results.Settings.nearThresholdByModelKernel = nearThresholdByRun;
Results.Settings.idw_radius = idw_radius;
Results.Settings.idw_power = 2;
Results.Settings.TransferMethod = transferModeByRun;
Results.Settings.TransferMethodByModelKernel = transferModeByRun;
Results.Settings.NIDWSources = nIDWSourcesByRun;
Results.Settings.NIDWSourcesByModelKernel = nIDWSourcesByRun;
Results.Settings.ParameterTransferMode = parameterTransferModeByRun;
Results.Settings.ParameterTransferModeByModelKernel = ...
    parameterTransferModeByRun;
Results.Settings.CapRMSESource = capRMSESource;
Results.Settings.NearThresholdRaw = nearThresholdByRun;
Results.Settings.NearThresholdRawByModelKernel = nearThresholdByRun;
Results.Settings.verbose = verbose;
Results.Settings.RNG_SEED = RNG_SEED;
Results.Settings.options = options;
Results.Settings.InnerGAParallel = false;

Results.DataInfo.nTrain = size(X,1);
Results.DataInfo.nTest = size(X_test,1);
Results.DataInfo.dimension = size(X,2);
Results.DataInfo.nModels = nModel;
Results.DataInfo.nKernels = nKernel;
Results.DataInfo.nRadiiByKernel = struct();

for j = 1:nKernel

    kernelKey = lower(NameFunction_RBF{j});

    Results.DataInfo.nRadiiByKernel.(kernelKey) = ...
        numel(getKernelRadii(Radii,kernelKey));

end
Results.DataInfo.S_global = common.MeanDistant(X);

Results.MatrixInfo.rows = ListOfModel;
Results.MatrixInfo.columns = NameFunction_RBF;
Results.MatrixInfo.description = ...
    'Rows = input models; columns = RBF kernels';

% -------------------------------------------------------------
% Accuracy / selected-cap summaries
% -------------------------------------------------------------

Results.train_rmse = nan(nModel,nKernel);
Results.test_rmse = nan(nModel,nKernel);
Results.nRadii = nan(nModel,nKernel);

% Exact neighbour count for every training point
% Each cell contains an nTrain x 1 vector.
Results.neighborCount = ...
    cell(nModel,nKernel);

Results.finalCapIndex = nan(nModel,nKernel);
Results.finalCapRadiusRaw = nan(nModel,nKernel);
Results.finalCapRadiusScaled = nan(nModel,nKernel);

Results.capRMSESource = cell(nModel,nKernel);
Results.capSelectionRMSE = nan(nModel,nKernel);
Results.capSelectionMinRMSE = nan(nModel,nKernel);
Results.capSelectionRMSEThreshold = nan(nModel,nKernel);

Results.radiusMedianRaw = nan(nModel,nKernel);
Results.radiusMinRaw = nan(nModel,nKernel);
Results.radiusMaxRaw = nan(nModel,nKernel);

Results.CMedian = nan(nModel,nKernel);
Results.CMinSelected = nan(nModel,nKernel);
Results.CMaxSelected = nan(nModel,nKernel);

Results.RegMedian = nan(nModel,nKernel);
Results.RegMinSelected = nan(nModel,nKernel);
Results.RegMaxSelected = nan(nModel,nKernel);

Results.epsilonMedian = nan(nModel,nKernel);
Results.epsilonMin = nan(nModel,nKernel);
Results.epsilonMax = nan(nModel,nKernel);

% -------------------------------------------------------------
% Search / cap-selection counts
% -------------------------------------------------------------

Results.nEvaluations = nan(nModel,nKernel);
Results.nGAEvaluations = nan(nModel,nKernel);
Results.nFallbackEvaluations = nan(nModel,nKernel);
Results.nGridEvaluations = nan(nModel,nKernel);
Results.nRefinementEvaluations = nan(nModel,nKernel);
Results.nValidEvaluations = nan(nModel,nKernel);
Results.nFailedEvaluations = nan(nModel,nKernel);

Results.nSuccessfulCaps = nan(nModel,nKernel);
Results.nFailedCaps = nan(nModel,nKernel);
Results.nSuccessfulTrainPointsAnyCap = nan(nModel,nKernel);
Results.nFailedTrainPointsAnyCap = nan(nModel,nKernel);

% -------------------------------------------------------------
% Selected-system diagnostic reuse
% -------------------------------------------------------------

Results.nSelectedPointCapPairs = nan(nModel,nKernel);
Results.nUniqueSelectedSystems = nan(nModel,nKernel);
Results.nDiagnosticCalculationsAvoided = nan(nModel,nKernel);
Results.diagnosticReuseFraction = nan(nModel,nKernel);

% -------------------------------------------------------------
% Test transfer summaries
% -------------------------------------------------------------

Results.nNearestTransfers = nan(nModel,nKernel);
Results.nIDWTransfers = nan(nModel,nKernel);
Results.nEnsembleTransfers = nan(nModel,nKernel);
Results.nTransferFailures = nan(nModel,nKernel);
Results.nSuccessfulTestPoints = nan(nModel,nKernel);
Results.nFailedTestPoints = nan(nModel,nKernel);

% -------------------------------------------------------------
% RBF solve counts
% -------------------------------------------------------------

Results.trainSearchRBFSolves = nan(nModel,nKernel);
Results.trainPseudoRBFSolves = nan(nModel,nKernel);
Results.trainDiagnosticRBFSolves = nan(nModel,nKernel);
Results.trainRBFSolves = nan(nModel,nKernel);
Results.testRBFSolves = nan(nModel,nKernel);
Results.totalRBFSolves = nan(nModel,nKernel);

% -------------------------------------------------------------
% Timing
% -------------------------------------------------------------

Results.trainWallTime = nan(nModel,nKernel);
Results.trainCPUTime = nan(nModel,nKernel);
Results.pseudoTrainingWallTime = nan(nModel,nKernel);
Results.testWallTime = nan(nModel,nKernel);
Results.testCPUTime = nan(nModel,nKernel);
Results.totalWallTime = nan(nModel,nKernel);
Results.totalCPUTime = nan(nModel,nKernel);

% -------------------------------------------------------------
% Status / messages
% -------------------------------------------------------------

Results.trainSuccess = false(nModel,nKernel);
Results.evaluationSuccess = false(nModel,nKernel);
Results.testSuccess = false(nModel,nKernel);
Results.runSuccess = false(nModel,nKernel);

Results.trainMessage = cell(nModel,nKernel);
Results.evaluationMessage = cell(nModel,nKernel);
Results.testMessage = cell(nModel,nKernel);


% =============================================================
% 10. Parallel preparation
%
% Parallel execution is controlled by:
%
%   options.execution.useParallel
%   options.execution.numWorkers
%   options.execution.allowSerialFallback
%
% Important:
%   - one PARFOR iteration = one Model x Kernel combination;
%   - nRun == 1 is executed serially on the client because a pool cannot
%     accelerate a single outer task;
%   - nRun > 1 uses a Processes pool when requested;
%   - pool startup failure either falls back to serial execution or throws,
%     according to allowSerialFallback.
% =============================================================

nRun = ...
    nModel * nKernel;


RunCell = ...
    cell(nRun,1);


useParallel = ...
    getExecutionLogicalOption( ...
        options, ...
        'useParallel', ...
        true);


allowSerialFallback = ...
    getExecutionLogicalOption( ...
        options, ...
        'allowSerialFallback', ...
        true);


numWorkersRequested = ...
    getExecutionWorkerCount( ...
        options);


maxWorkers = ...
    0;


pool = ...
    [];


parallelFallbackMessage = ...
    '';


if useParallel && ...
   nRun > 1

    pool = ...
        gcp('nocreate');


    if isempty(pool)

        try

            if isempty(numWorkersRequested)

                pool = ...
                    parpool('Processes');

            else

                pool = ...
                    parpool( ...
                        'Processes', ...
                        numWorkersRequested);

            end

        catch ME

            if allowSerialFallback

                pool = ...
                    [];


                parallelFallbackMessage = ...
                    ME.message;

            else

                rethrow(ME);

            end

        end

    end


    if ~isempty(pool)

        maxWorkers = ...
            min( ...
                pool.NumWorkers, ...
                nRun);

    end

end


Results.Settings.ParallelRequested = ...
    useParallel;


Results.Settings.Parallel = ...
    maxWorkers > 0;


Results.Settings.NumWorkersRequested = ...
    numWorkersRequested;


Results.Settings.NumWorkers = ...
    maxWorkers;


Results.Settings.ParallelFallbackAllowed = ...
    allowSerialFallback;


Results.Settings.ParallelFallbackUsed = ...
    useParallel && ...
    nRun > 1 && ...
    maxWorkers == 0;


Results.Settings.ParallelFallbackMessage = ...
    parallelFallbackMessage;


% Resolve small input-source metadata once on the client.  Do not pass
% the complete InputData structure into Stage-2 storage helpers inside
% PARFOR.
inputSourceFile = ...
    getInputSourceFile( ...
        InputData);


% =============================================================
% 11. Model x Kernel loop
%
% maxWorkers = 0:
%   PARFOR executes serially on the client and does not require a pool.
%
% maxWorkers > 0:
%   PARFOR uses at most maxWorkers process workers.
% =============================================================

parfor (irun = 1:nRun, maxWorkers)

    i = floor((irun-1) / nKernel) + 1;
    j = mod(irun-1,nKernel) + 1;

    modelName = ListOfModel{i};
    kernel = NameFunction_RBF{j};
    kernelKey = lower(kernel);

    tauLocRun = ...
        tau_loc(i,j);

    tauCapRun = ...
        tau_cap(i,j);

    methodRun = ...
    Method.(kernelKey);

nGridRun = ...
    nGrid.(kernelKey);

    radii = ...
        getKernelRadii( ...
            Radii, ...
            kernelKey);

    useShape = ...
        common.kernelHasShapeParameter(kernel);

    ModelAndFunction = ...
        [modelName '_' kernel];

    if isscalar(nGridRun)

    tuningTag = ...
        sprintf( ...
            '%s_G%d', ...
            methodRun, ...
            nGridRun);

else

    tuningTag = ...
        sprintf( ...
            '%s_G%d_%d', ...
            methodRun, ...
            nGridRun(1), ...
            nGridRun(2));

end

runField = ...
    matlab.lang.makeValidName( ...
        sprintf( ...
            'M%03d_K%03d_%s_%s', ...
            i, ...
            j, ...
            ModelAndFunction, ...
            tuningTag));
    y = ...
        YTrainByModel{i};


    y_test = ...
        YTestByModel{i};

    if useShape

        C_min = CMin.(kernelKey);
        C_max = CMax.(kernelKey);

    else

        C_min = NaN;
        C_max = NaN;

    end

    Reg_min = RegMin.(kernelKey);
    Reg_max = RegMax.(kernelKey);


    transferModeRun = ...
        transferModeByRun{i,j};


    parameterTransferModeRun = ...
        parameterTransferModeByRun{i,j};


    nIDWSourcesRun = ...
        nIDWSourcesByRun(i,j);


    nearThresholdRun = ...
        nearThresholdByRun(i,j);


    optionsRun = options;
    optionsRun.RNG_SEED = RNG_SEED;


    % Lower-level Training/Testing use one scalar configuration for this
    % exact GGM x Kernel combination.
    optionsRun.transfer.method = ...
        transferModeRun;


    optionsRun.transfer.parameterTransferMode = ...
        parameterTransferModeRun;


    optionsRun.transfer.nIDWSources = ...
        nIDWSourcesRun;


    % Same raw threshold for internal pseudo-training and testing.
    optionsRun.transfer.nearThresholdRaw = ...
        nearThresholdRun;

    % Resolve the storage policy ONCE at batch level and pass the exact
    % resolved values to Training/Testing.  Numerical routines remove the
    % execution structure internally, so these flags cannot affect the RBF
    % mathematics.  This also avoids relying on implicit/default recovery
    % inside lower-level functions.
    if ~isfield(optionsRun,'execution') || ...
       ~isstruct(optionsRun.execution)

        optionsRun.execution = struct();

    end

    optionsRun.execution.stage = executionStage;
    optionsRun.execution.saveTrace = saveTrace;
    optionsRun.execution.saveStage2All = saveStage2All;
    optionsRun.execution.saveEnsemblePointDiagnostics = ...
        saveEnsemblePointDiagnostics;

    if ~strcmp(executionStage,'all')

    optionsRun.execution.stage1CheckpointDir = ...
        stage1CheckpointDir;


    if strcmp(executionStage,'stage2')

        % Stage 2 reuses an existing Stage-1 checkpoint using the
        % stable Model + Kernel + tuning identity rather than the
        % current Mxxx/Kxxx batch-position prefix.
        optionsRun.execution.stage1CheckpointFile = ...
            resolveExistingStage1CheckpointFile( ...
                stage1CheckpointDir, ...
                modelName, ...
                kernel, ...
                tuningTag);

    else

        % Stage 1 creates its checkpoint using the current runField.
        optionsRun.execution.stage1CheckpointFile = ...
            fullfile( ...
                stage1CheckpointDir, ...
                [runField '.mat']);

    end

end

    Rrun = emptyRunResult();

    Rrun.i = i;
    Rrun.j = j;
    Rrun.model = modelName;
    Rrun.kernel = kernel;
    Rrun.ModelAndFunction = ModelAndFunction;
    Rrun.runField = runField;

    Rrun.nRadii = numel(radii);

    Rrun.C_min = C_min;
    Rrun.C_max = C_max;
    Rrun.Reg_min = Reg_min;
    Rrun.Reg_max = Reg_max;

    RunInfo = struct();
    RunInfo.model = modelName;
    RunInfo.kernel = kernel;
    RunInfo.ModelAndFunction = ModelAndFunction;
    RunInfo.useShape = useShape;

    RunInfo.Settings.radii = radii;
    RunInfo.Settings.minNeighbors = min_neighbors;
    RunInfo.Settings.C_min = C_min;
    RunInfo.Settings.C_max = C_max;
    RunInfo.Settings.Reg_min = Reg_min;
    RunInfo.Settings.Reg_max = Reg_max;
    RunInfo.Settings.tau_loc = tauLocRun;
 RunInfo.Settings.method = ...
    methodRun;

RunInfo.Settings.nGrid = ...
    nGridRun;
    RunInfo.Settings.tau_cap = tauCapRun;
    RunInfo.Settings.near_threshold = nearThresholdRun;
    RunInfo.Settings.idw_radius = idw_radius;

    RunInfo.Settings.transferMethod = ...
        transferModeRun;

    RunInfo.Settings.nIDWSources = ...
        nIDWSourcesRun;

    RunInfo.Settings.parameterTransferMode = ...
        parameterTransferModeRun;

    RunInfo.Settings.capRMSESource = ...
        capRMSESource;

    RunInfo.Settings.nearThresholdRaw = ...
        nearThresholdRun;

    RunInfo.Settings.RNG_SEED = RNG_SEED;
    RunInfo.Settings.executionStage = executionStage;
    RunInfo.Settings.saveTraceRequested = saveTrace;
    RunInfo.Settings.saveTraceActual = NaN;
    RunInfo.Settings.saveStage2All = saveStage2All;
    RunInfo.Settings.saveEnsemblePointDiagnosticsRequested = ...
        saveEnsemblePointDiagnosticsRequested;
    RunInfo.Settings.saveEnsemblePointDiagnostics = ...
        saveEnsemblePointDiagnostics;
    RunInfo.Settings.stage1CheckpointDir = stage1CheckpointDir;

    RunInfo.Train.success = false;
    RunInfo.Train.evaluationSuccess = false;
    RunInfo.Train.message = 'Not run yet.';
    RunInfo.Train.wallTime = NaN;
    RunInfo.Train.cpuTime = NaN;
    RunInfo.Train.Stage1CheckpointPaths = struct();

    RunInfo.Test.success = false;
    RunInfo.Test.message = 'Not run yet.';
    RunInfo.Test.wallTime = NaN;
    RunInfo.Test.cpuTime = NaN;

    % Stage-2 scientific payloads are saved as split parts and are never
    % embedded as complete TrainingResult/TestingResult copies in Results.
    % Do not even create Stage-2 storage fields during a Stage-1-only run.
    if ~strcmp(executionStage,'stage1')
        RunInfo.Stage2Paths = struct();
        RunInfo.Stage2StorageMode = '';
    end

    if verbose

        fprintf('\n====================================================\n');
        fprintf('Case: %s | Model: %s | Kernel: %s\n', ...
            CaseName,modelName,kernel);
        fprintf('====================================================\n');

    end

    % =========================================================
    % 11.1 Local training
    % =========================================================

    TrainingResult = [];
    geometryCache = [];

    tTrain = tic;
    cpuTrain0 = cputime;

    try

       [TrainingResult,geometryCache] = ...
    localrbf.LocalRBFTraining( ...
        X, ...
        y, ...
        radii, ...
        kernel, ...
        min_neighbors, ...
        C_min, ...
        C_max, ...
        Reg_min, ...
        Reg_max, ...
        tauLocRun, ...
        tauCapRun, ...
        optionsRun, ...
     methodRun, ...
        nGridRun);

        Rrun.trainWallTime = toc(tTrain);
        Rrun.trainCPUTime = cputime - cpuTrain0;

        RunInfo.Train.wallTime = Rrun.trainWallTime;
        RunInfo.Train.cpuTime = Rrun.trainCPUTime;

        % Two-stage runs always expose the exact split-checkpoint paths
        % and compact Stage-1 metadata. Stage 1 itself must not duplicate
        % the complete CORE PointOpt payload inside batch Results.
        if ~strcmp(executionStage,'all')

            stage1BaseFile = ...
                optionsRun.execution.stage1CheckpointFile;


            stage1PathsRun = ...
                localrbf.stage1CheckpointPaths( ...
                    stage1BaseFile);


            actualTraceSaved = ...
                getLogicalField( ...
                    TrainingResult, ...
                    'stage1TraceSaved', ...
                    saveTrace);


            RunInfo.Settings.saveTraceActual = ...
                actualTraceSaved;


            RunInfo.Train.Stage1CheckpointPaths = ...
                stage1PathsRun;


            % The optional Stage-1 batch summary is constructed only when
            % explicitly requested, and only during a Stage-1 run.
            if strcmp(executionStage,'stage1') && ...
               saveStage1Summary

                RunInfo.Train.Stage1Summary = ...
                    makeStage1BatchSummary( ...
                        TrainingResult, ...
                        stage1PathsRun, ...
                        actualTraceSaved);

            end

        end

        Rrun.trainSuccess = ...
            isfield(TrainingResult,'success') && ...
            TrainingResult.success;

        Rrun.evaluationSuccess = ...
            getLogicalField( ...
                TrainingResult, ...
                'evaluationSuccess', ...
                false);

        RunInfo.Train.success = Rrun.trainSuccess;
        RunInfo.Train.evaluationSuccess = Rrun.evaluationSuccess;

        if Rrun.trainSuccess

            Rrun.trainMessage = '';
            RunInfo.Train.message = '';

        else

            Rrun.trainMessage = ...
                getTextField( ...
                    TrainingResult, ...
                    'message', ...
                    'Local RBF training was unsuccessful.');

            RunInfo.Train.message = Rrun.trainMessage;

        end

        if strcmp(executionStage,'stage1')

            Rrun.evaluationMessage = ...
                'Not run in Stage 1.';

        elseif Rrun.evaluationSuccess

            Rrun.evaluationMessage = '';

        elseif isfield(TrainingResult,'EvaluationResult') && ...
               isstruct(TrainingResult.EvaluationResult)

            Rrun.evaluationMessage = ...
                getTextField( ...
                    TrainingResult.EvaluationResult, ...
                    'message', ...
                    'Selected-system evaluation was unsuccessful.');

        else

            Rrun.evaluationMessage = ...
                'Selected-system evaluation was not completed.';

        end

        % -----------------------------------------------------
        % Compact selected-training outputs
        % -----------------------------------------------------

        Rrun.train_rmse = ...
            getScalarField(TrainingResult,'RMSE',NaN);

        Rrun.finalCapIndex = ...
            getScalarField(TrainingResult,'finalCapIndex',NaN);

        Rrun.finalCapRadiusRaw = ...
            getScalarField(TrainingResult,'finalCapRadius_raw',NaN);

        Rrun.finalCapRadiusScaled = ...
            getScalarField(TrainingResult,'finalCapRadius_scaled',NaN);

        Rrun.capRMSESource = ...
            getTextField( ...
                TrainingResult, ...
                'finalCapRMSESource', ...
                capRMSESource);

        Rrun.capSelectionRMSE = ...
            getScalarField( ...
                TrainingResult, ...
                'finalCapSelectionRMSE', ...
                NaN);

        Rrun.capSelectionMinRMSE = ...
            getScalarField( ...
                TrainingResult, ...
                'finalCapSelectionMinRMSE', ...
                NaN);

        Rrun.capSelectionRMSEThreshold = ...
            getScalarField( ...
                TrainingResult, ...
                'finalCapSelectionRMSEThreshold', ...
                NaN);

        Rrun.pseudoTrainingWallTime = ...
            getScalarField( ...
                TrainingResult, ...
                'pseudoTrainingWallTime', ...
                0);

        if isfield(TrainingResult,'Final') && ...
           isstruct(TrainingResult.Final)

            Final = TrainingResult.Final;

                % Exact number of neighbours used at every training point
    Rrun.neighborCount = ...
        getVectorField( ...
            Final, ...
            'neighbor_count');

            radiusRaw = getVectorField(Final,'radius_raw');
            CSelected = getVectorField(Final,'C');
            RegSelected = getVectorField(Final,'reg_param');
            epsilonSelected = getVectorField(Final,'epsilon');

            [Rrun.radiusMedianRaw, ...
             Rrun.radiusMinRaw, ...
             Rrun.radiusMaxRaw] = ...
                finiteSummary(radiusRaw);

            [Rrun.CMedian, ...
             Rrun.CMinSelected, ...
             Rrun.CMaxSelected] = ...
                finiteSummary(CSelected);

            [Rrun.RegMedian, ...
             Rrun.RegMinSelected, ...
             Rrun.RegMaxSelected] = ...
                finiteSummary(RegSelected);

            [Rrun.epsilonMedian, ...
             Rrun.epsilonMin, ...
             Rrun.epsilonMax] = ...
                finiteSummary(epsilonSelected);

        end

        % -----------------------------------------------------
        % Search / cap counts
        % -----------------------------------------------------

        Rrun.nEvaluations = ...
            getScalarField(TrainingResult,'nEvaluations',NaN);

        Rrun.nGAEvaluations = ...
            getScalarField(TrainingResult,'nGAEvaluations',NaN);

Rrun.nFallbackEvaluations = ...
    getScalarField( ...
        TrainingResult, ...
        'nFallbackEvaluations', ...
        NaN);


Rrun.nGridEvaluations = ...
    getScalarField( ...
        TrainingResult, ...
        'nGridEvaluations', ...
        NaN);


Rrun.nRefinementEvaluations = ...
    getScalarField( ...
        TrainingResult, ...
        'nRefinementEvaluations', ...
        NaN);


Rrun.nValidEvaluations = ...
    getScalarField( ...
        TrainingResult, ...
        'nValidEvaluations', ...
        NaN);

        Rrun.nFailedEvaluations = ...
            getScalarField(TrainingResult,'nFailedEvaluations',NaN);

        Rrun.nSuccessfulCaps = ...
            getScalarField(TrainingResult,'nSuccessfulCaps',NaN);

        Rrun.nFailedCaps = ...
            getScalarField(TrainingResult,'nFailedCaps',NaN);

        Rrun.nSuccessfulTrainPointsAnyCap = ...
            getScalarField( ...
                TrainingResult, ...
                'nSuccessfulPointsAnyCap', ...
                NaN);

        Rrun.nFailedTrainPointsAnyCap = ...
            getScalarField( ...
                TrainingResult, ...
                'nFailedPointsAnyCap', ...
                NaN);

        % -----------------------------------------------------
        % Selected diagnostic reuse
        % -----------------------------------------------------

        Rrun.nSelectedPointCapPairs = ...
            getScalarField( ...
                TrainingResult, ...
                'nSelectedPointCapPairs', ...
                NaN);

        Rrun.nUniqueSelectedSystems = ...
            getScalarField( ...
                TrainingResult, ...
                'nUniqueSelectedSystemsTotal', ...
                NaN);

        Rrun.nDiagnosticCalculationsAvoided = ...
            getScalarField( ...
                TrainingResult, ...
                'nDiagnosticCalculationsAvoided', ...
                NaN);

        Rrun.diagnosticReuseFraction = ...
            getScalarField( ...
                TrainingResult, ...
                'diagnosticReuseFraction', ...
                NaN);

        % -----------------------------------------------------
        % Training RBF solve counts
        % -----------------------------------------------------

        Rrun.trainSearchRBFSolves = ...
            getScalarField( ...
                TrainingResult, ...
                'nRBFSolvesTraining', ...
                getScalarField(TrainingResult,'nRBFSolves',NaN));

        Rrun.trainPseudoRBFSolves = ...
            getScalarField( ...
                TrainingResult, ...
                'nRBFSolvesPseudo', ...
                0);

        Rrun.trainDiagnosticRBFSolves = ...
            getScalarField( ...
                TrainingResult, ...
                'nRBFSolvesEvaluation', ...
                0);

        Rrun.trainRBFSolves = ...
            getScalarField( ...
                TrainingResult, ...
                'nRBFSolvesTotal', ...
                sumFinitePair( ...
                    Rrun.trainSearchRBFSolves, ...
                    Rrun.trainDiagnosticRBFSolves));

        if verbose

            if strcmp(executionStage,'stage1')
                fprintf('Stage-1 complete = %d\n',Rrun.trainSuccess);
                fprintf('Optimization wall= %.3f s\n',Rrun.trainWallTime);
                fprintf('Optimization CPU = %.3f s\n',Rrun.trainCPUTime);
                fprintf('Search solves    = %g\n',Rrun.trainRBFSolves);
            else
                fprintf('Train RMSE       = %.6g\n',Rrun.train_rmse);
                fprintf('Selected cap     = %g\n',Rrun.finalCapIndex);
                fprintf('Training wall    = %.3f s\n',Rrun.trainWallTime);
                fprintf('Training CPU     = %.3f s\n',Rrun.trainCPUTime);
                fprintf('Training solves  = %g\n',Rrun.trainRBFSolves);
            end

        end

    catch ME

        Rrun.trainWallTime = toc(tTrain);
        Rrun.trainCPUTime = cputime - cpuTrain0;

        Rrun.trainSuccess = false;
        Rrun.evaluationSuccess = false;
        Rrun.trainMessage = ME.message;
        Rrun.evaluationMessage = ...
            'Not run because training failed.';

        RunInfo.Train.success = false;
        RunInfo.Train.evaluationSuccess = false;
        RunInfo.Train.message = ME.message;
        RunInfo.Train.wallTime = Rrun.trainWallTime;
        RunInfo.Train.cpuTime = Rrun.trainCPUTime;

        if ~strcmp(executionStage,'all')

            stage1BaseFile = ...
                optionsRun.execution.stage1CheckpointFile;


            stage1PathsRun = ...
                localrbf.stage1CheckpointPaths( ...
                    stage1BaseFile);


            actualTraceSaved = ...
                getLogicalField( ...
                    TrainingResult, ...
                    'stage1TraceSaved', ...
                    saveTrace);


            RunInfo.Settings.saveTraceActual = ...
                actualTraceSaved;


            RunInfo.Train.Stage1CheckpointPaths = ...
                stage1PathsRun;


            if strcmp(executionStage,'stage1') && ...
               saveStage1Summary

                RunInfo.Train.Stage1Summary = ...
                    makeStage1BatchSummary( ...
                        TrainingResult, ...
                        stage1PathsRun, ...
                        actualTraceSaved);

            end

        end

        if verbose

            fprintf('[TRAIN FAILED] %s | %s | %s\n', ...
                CaseName,modelName,kernel);
            fprintf('Reason: %s\n',ME.message);

        end

    end


    % =========================================================
    % 11.2 Independent testing
    % =========================================================

    TestingResult = [];
    y_pred_test = [];
    test_diff = [];
    test_rmse = NaN;

    if Rrun.trainSuccess && ...
       ~strcmp(executionStage,'stage1')

        tTest = tic;
        cpuTest0 = cputime;

        try

            [y_pred_test,test_diff,test_rmse,TestingResult] = ...
                localrbf.LocalRBFTesting( ...
                    TrainingResult, ...
                    X_test, ...
                    y_test, ...
                    nearThresholdRun, ...
                    idw_radius, ...
                    optionsRun, ...
                    geometryCache);

            Rrun.testWallTime = toc(tTest);
            Rrun.testCPUTime = cputime - cpuTest0;

            RunInfo.Test.wallTime = Rrun.testWallTime;
            RunInfo.Test.cpuTime = Rrun.testCPUTime;

            Rrun.test_rmse = test_rmse;

            Rrun.testSuccess = ...
                isfield(TestingResult,'success') && ...
                TestingResult.success && ...
                isscalar(test_rmse) && ...
                isfinite(test_rmse);

            RunInfo.Test.success = Rrun.testSuccess;

            if Rrun.testSuccess

                Rrun.testMessage = '';
                RunInfo.Test.message = '';

            else

                Rrun.testMessage = ...
                    getTextField( ...
                        TestingResult, ...
                        'message', ...
                        'Local RBF testing was unsuccessful.');

                RunInfo.Test.message = Rrun.testMessage;

            end

            Rrun.testRBFSolves = ...
                getScalarField(TestingResult,'nRBFSolves',NaN);

            Rrun.nNearestTransfers = ...
                getScalarField(TestingResult,'nNearestTransfers',NaN);

            Rrun.nIDWTransfers = ...
                getScalarField(TestingResult,'nIDWTransfers',NaN);

            Rrun.nEnsembleTransfers = ...
                getScalarField(TestingResult,'nEnsembleTransfers',0);

            Rrun.nTransferFailures = ...
                getScalarField(TestingResult,'nTransferFailures',NaN);

            Rrun.nSuccessfulTestPoints = ...
                getScalarField(TestingResult,'nSuccessfulPoints',NaN);

            Rrun.nFailedTestPoints = ...
                getScalarField(TestingResult,'nFailedPoints',NaN);

            if verbose

                fprintf('Test RMSE        = %.6g\n',test_rmse);
                fprintf('Testing wall     = %.3f s\n',Rrun.testWallTime);
                fprintf('Testing CPU      = %.3f s\n',Rrun.testCPUTime);
                fprintf('Testing solves   = %g\n',Rrun.testRBFSolves);

            end

        catch ME

            Rrun.testWallTime = toc(tTest);
            Rrun.testCPUTime = cputime - cpuTest0;

            Rrun.testSuccess = false;
            Rrun.testMessage = ME.message;

            RunInfo.Test.success = false;
            RunInfo.Test.message = ME.message;
            RunInfo.Test.wallTime = Rrun.testWallTime;
            RunInfo.Test.cpuTime = Rrun.testCPUTime;

            if verbose

                fprintf('[TEST FAILED] %s | %s | %s\n', ...
                    CaseName,modelName,kernel);
                fprintf('Reason: %s\n',ME.message);

            end

        end

    else

        Rrun.testSuccess = false;

        if strcmp(executionStage,'stage1') && ...
           Rrun.trainSuccess

            Rrun.testMessage = ...
                'Not run in Stage 1.';

        else

            Rrun.testMessage = ...
                'Not run because training failed.';

        end

        RunInfo.Test.success = false;
        RunInfo.Test.message = Rrun.testMessage;

    end


    % =========================================================
    % 11.3 Total run cost / status
    % =========================================================

    Rrun.totalWallTime = ...
        sumFinitePair( ...
            Rrun.trainWallTime, ...
            Rrun.testWallTime);

    Rrun.totalCPUTime = ...
        sumFinitePair( ...
            Rrun.trainCPUTime, ...
            Rrun.testCPUTime);

    Rrun.totalRBFSolves = ...
        sumFinitePair( ...
            Rrun.trainRBFSolves, ...
            Rrun.testRBFSolves);

    if strcmp(executionStage,'stage1')

        Rrun.runSuccess = ...
            Rrun.trainSuccess;

    else

        Rrun.runSuccess = ...
            Rrun.trainSuccess && ...
            Rrun.testSuccess;

    end

    RunInfo.success = Rrun.runSuccess;
    RunInfo.totalWallTime = Rrun.totalWallTime;
    RunInfo.totalCPUTime = Rrun.totalCPUTime;
    RunInfo.totalRBFSolves = Rrun.totalRBFSolves;

    % =========================================================
    % 11.4 Storage and release
    %
    % Stage 1:
    %   authoritative CORE/DIAG/(TRACE) files are already written by
    %   LocalRBFTraining. No duplicate TrainingResult is returned.
    %
    % Stage 2 / all:
    %   one self-contained CORE is always constructed. CORE contains the
    %   collector-equivalent final scientific result, all three numerical
    %   condition diagnostics, runtime, and bootstrap-ready data/settings.
    %   DETAIL is optional and contains only non-compulsory/heavy provenance.
    % =========================================================

    if strcmp(executionStage,'stage1')

        TrainingResult = [];
        TestingResult = [];
        geometryCache = [];

    else

        Stage2Paths = ...
            makeStage2CheckpointPaths( ...
                checkpointDir, ...
                runField, ...
                saveStage2All);


        StorageMeta = ...
            makeStage2StorageMeta( ...
                inputSourceFile, ...
                OutputFile, ...
                runField, ...
                CaseName, ...
                modelName, ...
                kernel, ...
                X, ...
                X_test, ...
                radii, ...
                tauLocRun, ...
                tauCapRun, ...
                nearThresholdRun, ...
                idw_radius, ...
                transferModeRun, ...
                nIDWSourcesRun, ...
                parameterTransferModeRun, ...
                capRMSESource, ...
                min_neighbors, ...
                methodRun, ...
                nGridRun, ...
                Reg_min, ...
                Reg_max, ...
                C_min, ...
                C_max, ...
                RNG_SEED, ...
                options.solver, ...
                stage1CheckpointDir, ...
                saveStage2All, ...
                saveEnsemblePointDiagnostics);


        Stage2Core = ...
            buildStage2Core( ...
                StorageMeta, ...
                TrainingResult, ...
                TestingResult, ...
                X, ...
                y, ...
                X_test, ...
                y_test, ...
                Rrun);


        % -----------------------------------------------------
        % CORE schema / storage contract
        % -----------------------------------------------------

        Stage2Core.Schema = ...
            'LocalRBF.Stage2Core.v2';


        Stage2Core.Storage = ...
            struct();

        Stage2Core.Storage.SchemaVersion = ...
            2;

        % Complete means the authoritative CORE file was committed.
        % DETAIL is optional and has its own completion flag.
        Stage2Core.Storage.Complete = ...
            false;

        Stage2Core.Storage.DetailRequested = ...
            saveStage2All;

        Stage2Core.Storage.DetailComplete = ...
            false;

        Stage2Core.Storage.RequiredParts = ...
            {'CORE'};

        Stage2Core.Storage.CoreFile = ...
            Stage2Paths.core;

        Stage2Core.Storage.DetailFile = ...
            Stage2Paths.detail;


        Stage2Core.Source = ...
            struct();

        Stage2Core.Source.Mode = ...
            'STAGE2_DIRECT';

        Stage2Core.Source.InputDataFile = ...
            inputSourceFile;

        Stage2Core.Source.BatchOutputFile = ...
            OutputFile;

        Stage2Core.Source.Stage1CheckpointDir = ...
            stage1CheckpointDir;

        Stage2Core.Source.Stage2CoreFile = ...
            Stage2Paths.core;

        % Collector-compatible path aliases. Numerical and Runtime are now
        % embedded directly in CORE rather than stored as separate files.
        Stage2Core.Source.Core = ...
            Stage2Paths.core;

        Stage2Core.Source.Numerical = ...
            '';

        Stage2Core.Source.Runtime = ...
            '';

        Stage2Core.Source.Stage2DetailFile = ...
            Stage2Paths.detail;


        % -----------------------------------------------------
        % Optional DETAIL: construct exactly once and only when requested.
        % DETAIL failure must never discard the compulsory CORE result.
        % -----------------------------------------------------

        detailBuilt = ...
            false;

        Stage2Detail = ...
            [];

        RunInfo.Stage2StorageError = ...
            '';


        if saveStage2All

            try

                Stage2Detail = ...
                    buildStage2Detail( ...
                        TrainingResult, ...
                        TestingResult, ...
                        Rrun);

                detailBuilt = ...
                    true;

            catch MEdetail

                warning( ...
                    'localRBF:Stage2DetailBuildFailed', ...
                    'Stage-2 DETAIL construction failed for %s: %s', ...
                    ModelAndFunction, ...
                    MEdetail.message);

                RunInfo.Stage2StorageError = ...
                    MEdetail.message;

                % Rare error-path information preservation only.
                RunInfo.Stage2DetailFallback = ...
                    struct( ...
                        'TrainingResult',TrainingResult, ...
                        'TestingResult',TestingResult);

            end

        end


        stage2FilesSaved = ...
            false;

        detailFileSaved = ...
            false;


        if ~isempty(checkpointDir)

            % Invalidate the previous CORE commit marker BEFORE writing any
            % optional DETAIL. This prevents an interrupted update from
            % leaving an old completed CORE beside a new DETAIL file.
            commitMarkerInvalidated = ...
                false;

            try

                invalidateStage2CommitMarker( ...
                    Stage2Paths.core);

                commitMarkerInvalidated = ...
                    true;

            catch MEinvalidate

                warning( ...
                    'localRBF:Stage2CommitInvalidationFailed', ...
                    'Could not invalidate previous Stage-2 CORE for %s: %s', ...
                    ModelAndFunction, ...
                    MEinvalidate.message);

                RunInfo.Stage2StorageError = ...
                    MEinvalidate.message;

                % Preserve the complete computed result in memory on this
                % exceptional path rather than risking a mixed checkpoint.
                RunInfo.Stage2Core = ...
                    Stage2Core;

                if saveStage2All && detailBuilt

                    RunInfo.Stage2Detail = ...
                        Stage2Detail;

                end

            end


            if commitMarkerInvalidated

                % -------------------------------------------------
                % DETAIL is optional. Attempt it first when requested.
                % A DETAIL failure does not invalidate the compulsory CORE.
                % -------------------------------------------------

                if saveStage2All && detailBuilt

                    try

                        saveNamedVariableAtomic( ...
                            Stage2Paths.detail, ...
                            'Stage2Detail', ...
                            Stage2Detail);

                        detailFileSaved = ...
                            true;

                    catch MEdetailSave

                        warning( ...
                            'localRBF:Stage2DetailCheckpointFailed', ...
                            'Stage-2 DETAIL checkpoint failed for %s: %s', ...
                            ModelAndFunction, ...
                            MEdetailSave.message);

                        RunInfo.Stage2StorageError = ...
                            MEdetailSave.message;

                        % Preserve requested optional information on the rare
                        % DETAIL I/O failure path.
                        RunInfo.Stage2Detail = ...
                            Stage2Detail;

                    end

                end


                % Never leave a stale DETAIL from an older run attached to
                % a newly committed CORE when current DETAIL was not saved.
                if saveStage2All && ~detailFileSaved

                    deleteIfExistsLocal( ...
                        Stage2Paths.detail);

                end


                Stage2Core.Storage.DetailComplete = ...
                    detailFileSaved;


                % -------------------------------------------------
                % CORE is authoritative. Publish it atomically LAST.
                % -------------------------------------------------

                try

                    Stage2Core.Storage.Complete = ...
                        true;


                    saveNamedVariableAtomic( ...
                        Stage2Paths.core, ...
                        'Stage2Core', ...
                        Stage2Core);


                    stage2FilesSaved = ...
                        true;


                    % DETAIL from an earlier DETAIL-ON run is not part of the
                    % current DETAIL-OFF record.
                    if ~saveStage2All

                        staleDetailFile = ...
                            fullfile( ...
                                checkpointDir, ...
                                [runField '_STAGE2_DETAIL.mat']);

                        deleteIfExistsLocal( ...
                            staleDetailFile);

                    end

                catch MEcheckpoint

                    deleteIfExistsLocal( ...
                        Stage2Paths.core);


                    Stage2Core.Storage.Complete = ...
                        false;


                    warning( ...
                        'localRBF:Stage2CheckpointFailed', ...
                        'Stage-2 CORE checkpoint failed for %s: %s', ...
                        ModelAndFunction, ...
                        MEcheckpoint.message);


                    RunInfo.Stage2StorageError = ...
                        MEcheckpoint.message;


                    % Do not discard computed scientific results after I/O
                    % failure. This large fallback exists only on error.
                    RunInfo.Stage2Core = ...
                        Stage2Core;

                end

            end


            RunInfo.Stage2Paths = ...
                Stage2Paths;

            RunInfo.Stage2FilesSaved = ...
                stage2FilesSaved;

            RunInfo.Stage2DetailSaved = ...
                detailFileSaved;

        else

            % No checkpoint directory: preserve the authoritative CORE in
            % the returned object rather than losing the scientific result.
            RunInfo.Stage2Core = ...
                Stage2Core;

            RunInfo.Stage2FilesSaved = ...
                false;

            RunInfo.Stage2DetailSaved = ...
                false;


            if saveStage2All && detailBuilt

                RunInfo.Stage2Detail = ...
                    Stage2Detail;

            end

        end


        if saveStage2All && detailFileSaved

            RunInfo.Stage2StorageMode = ...
                'CORE+DETAIL';

        elseif saveStage2All

            RunInfo.Stage2StorageMode = ...
                'CORE+DETAIL_REQUESTED';

        elseif saveEnsemblePointDiagnostics

            RunInfo.Stage2StorageMode = ...
                'CORE+ENSEMBLE_POINT_DIAGNOSTICS';

        else

            RunInfo.Stage2StorageMode = ...
                'CORE';

        end


        % Build the same compact comparison information that the standalone
        % collector previously derived by reloading checkpoint files. This
        % row is built directly from the authoritative CORE already in
        % memory, so no directory scan or MAT-file reload is required.
        Rrun.SummaryRow = ...
            makeStage2SummaryRow( ...
                Stage2Core, ...
                Stage2Paths.core);


        % Release large computation objects before PARFOR returns. Normal
        % successful execution returns only compact RunInfo + SummaryRow.
        Stage2Core = [];

        if detailBuilt
            Stage2Detail = [];
        end

        TrainingResult = [];
        TestingResult = [];
        geometryCache = [];

    end


    Rrun.RunInfo = RunInfo;


    RunCell{irun} = Rrun;

end


% =============================================================
% 12. Merge independent runs into Results
% =============================================================

SummaryRows = ...
    struct([]);

for irun = 1:nRun

    Rrun = RunCell{irun};

    i = Rrun.i;
    j = Rrun.j;

    Results.Run.(Rrun.runField) = ...
        Rrun.RunInfo;

    % ---------------------------------------------------------
    % Accuracy / selected settings
    % ---------------------------------------------------------

    Results.train_rmse(i,j) = Rrun.train_rmse;
    Results.test_rmse(i,j) = Rrun.test_rmse;
    Results.nRadii(i,j) = Rrun.nRadii;
    Results.neighborCount{i,j} = ...
    Rrun.neighborCount;

    Results.finalCapIndex(i,j) = Rrun.finalCapIndex;
    Results.finalCapRadiusRaw(i,j) = Rrun.finalCapRadiusRaw;
    Results.finalCapRadiusScaled(i,j) = Rrun.finalCapRadiusScaled;

    Results.capRMSESource{i,j} = Rrun.capRMSESource;
    Results.capSelectionRMSE(i,j) = Rrun.capSelectionRMSE;
    Results.capSelectionMinRMSE(i,j) = Rrun.capSelectionMinRMSE;
    Results.capSelectionRMSEThreshold(i,j) = ...
        Rrun.capSelectionRMSEThreshold;

    Results.radiusMedianRaw(i,j) = Rrun.radiusMedianRaw;
    Results.radiusMinRaw(i,j) = Rrun.radiusMinRaw;
    Results.radiusMaxRaw(i,j) = Rrun.radiusMaxRaw;

    Results.CMedian(i,j) = Rrun.CMedian;
    Results.CMinSelected(i,j) = Rrun.CMinSelected;
    Results.CMaxSelected(i,j) = Rrun.CMaxSelected;

    Results.RegMedian(i,j) = Rrun.RegMedian;
    Results.RegMinSelected(i,j) = Rrun.RegMinSelected;
    Results.RegMaxSelected(i,j) = Rrun.RegMaxSelected;

    Results.epsilonMedian(i,j) = Rrun.epsilonMedian;
    Results.epsilonMin(i,j) = Rrun.epsilonMin;
    Results.epsilonMax(i,j) = Rrun.epsilonMax;

    % ---------------------------------------------------------
    % Search / cap counts
    % ---------------------------------------------------------

    Results.nEvaluations(i,j) = Rrun.nEvaluations;
    Results.nGAEvaluations(i,j) = Rrun.nGAEvaluations;
    Results.nFallbackEvaluations(i,j) = Rrun.nFallbackEvaluations;
    Results.nGridEvaluations(i,j) = ...
    Rrun.nGridEvaluations;


Results.nRefinementEvaluations(i,j) = ...
    Rrun.nRefinementEvaluations;

    Results.nValidEvaluations(i,j) = Rrun.nValidEvaluations;
    Results.nFailedEvaluations(i,j) = Rrun.nFailedEvaluations;

    Results.nSuccessfulCaps(i,j) = Rrun.nSuccessfulCaps;
    Results.nFailedCaps(i,j) = Rrun.nFailedCaps;

    Results.nSuccessfulTrainPointsAnyCap(i,j) = ...
        Rrun.nSuccessfulTrainPointsAnyCap;

    Results.nFailedTrainPointsAnyCap(i,j) = ...
        Rrun.nFailedTrainPointsAnyCap;

    % ---------------------------------------------------------
    % Diagnostic reuse
    % ---------------------------------------------------------

    Results.nSelectedPointCapPairs(i,j) = ...
        Rrun.nSelectedPointCapPairs;

    Results.nUniqueSelectedSystems(i,j) = ...
        Rrun.nUniqueSelectedSystems;

    Results.nDiagnosticCalculationsAvoided(i,j) = ...
        Rrun.nDiagnosticCalculationsAvoided;

    Results.diagnosticReuseFraction(i,j) = ...
        Rrun.diagnosticReuseFraction;

    % ---------------------------------------------------------
    % Transfer / test points
    % ---------------------------------------------------------

    Results.nNearestTransfers(i,j) = Rrun.nNearestTransfers;
    Results.nIDWTransfers(i,j) = Rrun.nIDWTransfers;
    Results.nEnsembleTransfers(i,j) = Rrun.nEnsembleTransfers;
    Results.nTransferFailures(i,j) = Rrun.nTransferFailures;

    Results.nSuccessfulTestPoints(i,j) = ...
        Rrun.nSuccessfulTestPoints;

    Results.nFailedTestPoints(i,j) = ...
        Rrun.nFailedTestPoints;

    % ---------------------------------------------------------
    % RBF solves
    % ---------------------------------------------------------

    Results.trainSearchRBFSolves(i,j) = ...
        Rrun.trainSearchRBFSolves;

    Results.trainPseudoRBFSolves(i,j) = ...
        Rrun.trainPseudoRBFSolves;

    Results.trainDiagnosticRBFSolves(i,j) = ...
        Rrun.trainDiagnosticRBFSolves;

    Results.trainRBFSolves(i,j) = Rrun.trainRBFSolves;
    Results.testRBFSolves(i,j) = Rrun.testRBFSolves;
    Results.totalRBFSolves(i,j) = Rrun.totalRBFSolves;

    % ---------------------------------------------------------
    % Timing
    % ---------------------------------------------------------

    Results.trainWallTime(i,j) = Rrun.trainWallTime;
    Results.trainCPUTime(i,j) = Rrun.trainCPUTime;
    Results.pseudoTrainingWallTime(i,j) = Rrun.pseudoTrainingWallTime;

    Results.testWallTime(i,j) = Rrun.testWallTime;
    Results.testCPUTime(i,j) = Rrun.testCPUTime;

    Results.totalWallTime(i,j) = Rrun.totalWallTime;
    Results.totalCPUTime(i,j) = Rrun.totalCPUTime;

    % ---------------------------------------------------------
    % Status / messages
    % ---------------------------------------------------------

    Results.trainSuccess(i,j) = Rrun.trainSuccess;
    Results.evaluationSuccess(i,j) = Rrun.evaluationSuccess;
    Results.testSuccess(i,j) = Rrun.testSuccess;
    Results.runSuccess(i,j) = Rrun.runSuccess;

    Results.trainMessage{i,j} = Rrun.trainMessage;
    Results.evaluationMessage{i,j} = Rrun.evaluationMessage;
    Results.testMessage{i,j} = Rrun.testMessage;

    % ---------------------------------------------------------
    % Collector-equivalent compact Stage-2 row
    % ---------------------------------------------------------

    if ~strcmp(executionStage,'stage1') && ...
       isstruct(Rrun.SummaryRow) && ...
       ~isempty(fieldnames(Rrun.SummaryRow))

        row = ...
            Rrun.SummaryRow;

        row.RunID = ...
            string(sprintf('R%03d',irun));

        if isempty(SummaryRows)

            SummaryRows = ...
                row;

        else

            SummaryRows(end+1,1) = ...
                row; %#ok<AGROW>

        end

    end

end


if ~strcmp(executionStage,'stage1')

    if isempty(SummaryRows)

        Results.Summary = ...
            table();

    else

        Results.Summary = ...
            struct2table(SummaryRows);

        Results.Summary = ...
            sortrows( ...
                Results.Summary, ...
                {'Case','Model','Kernel','RunField'});

    end

end


% =============================================================
% 13. Batch-level summary
% =============================================================

Results.nRuns = nRun;
Results.nSuccessfulRuns = nnz(Results.runSuccess);
Results.nFailedRuns = nRun - Results.nSuccessfulRuns;

Results.nSuccessfulTrainingRuns = nnz(Results.trainSuccess);
Results.nSuccessfulEvaluationRuns = nnz(Results.evaluationSuccess);
Results.nSuccessfulTestingRuns = nnz(Results.testSuccess);

Results.CheckpointDir = checkpointDir;
Results.Stage1CheckpointDir = stage1CheckpointDir;
Results.Stage2CheckpointDir = checkpointDir;
Results.ExecutionStage = executionStage;


if strcmp(executionStage,'stage1')

    if saveTrace
        Results.Stage1StorageMode = 'CORE+DIAG+TRACE';
    else
        Results.Stage1StorageMode = 'CORE+DIAG';
    end

elseif strcmp(executionStage,'stage2')

    Results.Stage1StorageMode = ...
        'loaded split checkpoint; see per-run Stage-1 metadata';

else

    Results.Stage1StorageMode = ...
        'not used in one-pass execution';

end

if strcmp(executionStage,'stage1')

    if saveStage1Summary
        Results.BatchStorageMode = ...
            'Stage-1 compact batch summary saved';
    else
        Results.BatchStorageMode = ...
            'Stage-1 compact batch summary not saved';
    end

elseif saveStage2All

    Results.BatchStorageMode = ...
        'Stage-2 self-contained CORE + optional DETAIL';

elseif saveEnsemblePointDiagnostics

    Results.BatchStorageMode = ...
        'Stage-2 CORE + lightweight ensemble point diagnostics';

else

    Results.BatchStorageMode = ...
        'Stage-2 self-contained CORE';

end

Results.OutputFile = OutputFile;


% =============================================================
% 14. Final save
% =============================================================

% Stage 1 batch summary is genuinely optional.  Do not create a separate
% optional payload and discard it later; simply skip persistence entirely.
if ~isempty(OutputFile) && ...
   (~strcmp(executionStage,'stage1') || saveStage1Summary)

    save(OutputFile,'Results','-v7.3');

end

end


% =============================================================
% Local helper
% =============================================================

function M = normalizePositiveIntegerMatrixBatch( ...
    setting,nModel,nKernel,settingName)
%NORMALIZEPOSITIVEINTEGERMATRIXBATCH
% Accept one common positive integer or nModel x nKernel numeric matrix.

if ~isnumeric(setting) || ...
   ~isreal(setting)

    error('%s must be numeric and real.',settingName);

end


if isscalar(setting)

    M = ...
        repmat(double(setting),nModel,nKernel);

elseif isequal(size(setting),[nModel nKernel])

    M = ...
        double(setting);

else

    error( ...
        ['%s must be one scalar or an nModel x nKernel matrix ' ...
         'of size %d x %d.'], ...
        settingName,nModel,nKernel);

end


if any(~isfinite(M(:))) || ...
   any(M(:) < 1) || ...
   any(M(:) ~= round(M(:)))

    error( ...
        '%s entries must all be finite positive integers.', ...
        settingName);

end

end


% =============================================================
% Local helper
% =============================================================

function M = normalizePositiveRealMatrixBatch( ...
    setting,nModel,nKernel,settingName)
%NORMALIZEPOSITIVEREALMATRIXBATCH
% Accept one common positive scalar or nModel x nKernel numeric matrix.

if ~isnumeric(setting) || ...
   ~isreal(setting)

    error('%s must be numeric and real.',settingName);

end


if isscalar(setting)

    M = ...
        repmat(double(setting),nModel,nKernel);

elseif isequal(size(setting),[nModel nKernel])

    M = ...
        double(setting);

else

    error( ...
        ['%s must be one scalar or an nModel x nKernel matrix ' ...
         'of size %d x %d.'], ...
        settingName,nModel,nKernel);

end


if any(~isfinite(M(:))) || ...
   any(M(:) <= 0)

    error( ...
        '%s entries must all be finite and positive.', ...
        settingName);

end

end


% =============================================================
% Local helper
% =============================================================

function M = normalizeTransferMethodMatrixBatch(setting,nModel,nKernel)
%NORMALIZETRANSFERMETHODMATRIXBATCH

C = ...
    normalizeTextMatrixBatch( ...
        setting,nModel,nKernel,'options.transfer.method');


M = ...
    cell(nModel,nKernel);


for i = 1:nModel

    for j = 1:nKernel

        value = ...
            lower(strtrim(C{i,j}));


        if ~any(strcmp(value,{'nearest','idw','hybrid','ensemble'}))

            error( ...
                ['options.transfer.method{%d,%d} must be nearest, idw, ' ...
                 'hybrid, or ensemble.'], ...
                i,j);

        end


        M{i,j} = ...
            value;

    end

end

end


% =============================================================
% Local helper
% =============================================================

function M = normalizeParameterTransferModeMatrixBatch( ...
    setting,transferModeByRun,nModel,nKernel)
%NORMALIZEPARAMETERTRANSFERMODEMATRIXBATCH

C = ...
    normalizeTextMatrixBatch( ...
        setting,nModel,nKernel, ...
        'options.transfer.parameterTransferMode');


M = ...
    cell(nModel,nKernel);


for i = 1:nModel

    for j = 1:nKernel

        method = ...
            transferModeByRun{i,j};


    if any(strcmp(method,{'nearest','ensemble'}))

    % Parameter aggregation is inactive for nearest/ensemble.
    % Keep a valid internal placeholder because downstream
    % training/testing still validates this field.
    M{i,j} = ...
        'log_log';

    continue

end


        value = ...
            C{i,j};


        if strcmpi(strtrim(value),'not-applicable')

            error( ...
                ['options.transfer.parameterTransferMode{%d,%d} cannot be ' ...
                 '''not-applicable'' for transfer method ''%s''.'], ...
                i,j,method);

        end


        M{i,j} = ...
            normalizeParameterTransferModeBatch(value);

    end

end

end


% =============================================================
% Local helper
% =============================================================

function C = normalizeTextMatrixBatch(setting,nModel,nKernel,settingName)
%NORMALIZETEXTMATRIXBATCH

if ischar(setting)

    C = ...
        repmat({setting},nModel,nKernel);

    return

end


if isstring(setting)

    if isscalar(setting)

        C = ...
            repmat({char(setting)},nModel,nKernel);

        return

    end


    if ~isequal(size(setting),[nModel nKernel])

        error( ...
            '%s string matrix must have size %d x %d.', ...
            settingName,nModel,nKernel);

    end


    C = ...
        cellstr(setting);

    return

end


if ~iscell(setting) || ...
   ~isequal(size(setting),[nModel nKernel])

    error( ...
        ['%s must be common text or an nModel x nKernel cell/string ' ...
         'matrix of size %d x %d.'], ...
        settingName,nModel,nKernel);

end


C = ...
    cell(nModel,nKernel);


for i = 1:nModel

    for j = 1:nKernel

        value = ...
            setting{i,j};


        if ischar(value)

            C{i,j} = ...
                value;

        elseif isstring(value) && isscalar(value)

            C{i,j} = ...
                char(value);

        else

            error( ...
                '%s{%d,%d} must be char or scalar string.', ...
                settingName,i,j);

        end

    end

end

end


% =============================================================
% Local helper
% =============================================================

function mode = normalizeParameterTransferModeBatch(value)
%NORMALIZEPARAMETERTRANSFERMODEBATCH Canonicalize M21-M24 transfer modes.

if ~(ischar(value) || ...
     (isstring(value) && isscalar(value)))

    error( ...
        'options.transfer.parameterTransferMode must be text.');

end


mode = ...
    lower( ...
        strtrim( ...
            char(value)));


switch mode

    case {'arithmetic_arithmetic','arith_arith','m21'}

        mode = ...
            'arithmetic_arithmetic';


    case {'log_arithmetic','log_arith','m22'}

        mode = ...
            'log_arithmetic';


    case {'arithmetic_log','arith_log','m23'}

        mode = ...
            'arithmetic_log';


    case {'log_log','m24'}

        mode = ...
            'log_log';


    otherwise

        error( ...
            ['options.transfer.parameterTransferMode must be one of: ' ...
             'arithmetic_arithmetic, log_arithmetic, arithmetic_log, ' ...
             'log_log (or aliases M21-M24).']);

end

end


% =============================================================
% Local helper
% =============================================================

function source = normalizeCapRMSESourceBatch(value)
%NORMALIZECAPRMSESOURCEBATCH Canonicalize final-cap RMSE source.

if ~(ischar(value) || ...
     (isstring(value) && isscalar(value)))

    error( ...
        'options.transfer.capRMSESource must be text.');

end


source = ...
    lower( ...
        strtrim( ...
            char(value)));


switch source

    case {'training','original','origin'}

        source = ...
            'training';


    case {'pseudo','pseudo-training','pseudotraining'}

        source = ...
            'pseudo';


    otherwise

        error( ...
            ['options.transfer.capRMSESource must be ' ...
             '''training'' or ''pseudo''.']);

end

end


% =============================================================
% Local helper
% =============================================================

function value = getFinalTextFieldLocal(T,fieldName,defaultValue)
%GETFINALTEXTFIELDLOCAL Read text metadata from TrainingResult.Final.

value = ...
    defaultValue;


if ~isstruct(T) || ...
   ~isfield(T,'Final') || ...
   ~isstruct(T.Final)

    return

end


value = ...
    getTextField( ...
        T.Final, ...
        fieldName, ...
        defaultValue);

end


% =============================================================
% Local helper
% =============================================================

function stage = getExecutionStageBatch(options)
%GETEXECUTIONSTAGEBATCH Resolve optional batch execution stage.

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


if ~any(strcmp(stage,{'all','stage1','stage2'}))

    error( ...
        ['options.execution.stage must be ''all'', ''stage1'', ' ...
         'or ''stage2''.']);

end

end


function tf = getSaveTraceBatch(options)
%GETSAVETRACEBATCH Resolve Stage-1 TRACE persistence. Default = true.

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

function S = makeStage1BatchSummary(T,Paths,saveTrace)
%MAKESTAGE1BATCHSUMMARY Compact Stage-1 metadata for batch Results.
%
% The complete Stage-1 scientific data remain in the split checkpoint
% files. This summary intentionally contains only scalar/status metadata
% and deterministic file paths so batch aggregation does not duplicate
% the PointOpt arrays.

S = ...
    struct();


S.success = ...
    getLogicalField( ...
        T, ...
        'success', ...
        false);


S.stage1Complete = ...
    getLogicalField( ...
        T, ...
        'stage1Complete', ...
        false);


S.traceSaved = ...
    logical(saveTrace);


if isstruct(T) && ...
   isfield(T,'stage1Summary') && ...
   isstruct(T.stage1Summary) && ...
   isfield(T.stage1Summary,'traceSaved') && ...
   isscalar(T.stage1Summary.traceSaved)

    S.traceSaved = ...
        logical(T.stage1Summary.traceSaved);

end



S.nPoints = ...
    getScalarField(T,'nPoints',NaN);


S.nDimensions = ...
    getScalarField(T,'nDimensions',NaN);


S.nRadii = ...
    getScalarField(T,'nCaps',NaN);


S.S_global = ...
    getScalarField(T,'S_global',NaN);


S.nStage1PointsComplete = ...
    getStage1SummaryScalar( ...
        T, ...
        'nStage1PointsComplete', ...
        NaN);


S.nStage1PointsMissing = ...
    getStage1SummaryScalar( ...
        T, ...
        'nStage1PointsMissing', ...
        NaN);


S.nSuccessfulPointOptimizations = ...
    getStage1SummaryScalar( ...
        T, ...
        'nSuccessfulPointOptimizations', ...
        NaN);


S.nUnsuccessfulPointOptimizations = ...
    getStage1SummaryScalar( ...
        T, ...
        'nUnsuccessfulPointOptimizations', ...
        NaN);


S.nEvaluations = ...
    getScalarField(T,'nEvaluations',NaN);


S.nGAEvaluations = ...
    getScalarField(T,'nGAEvaluations',NaN);


S.nFallbackEvaluations = ...
    getScalarField(T,'nFallbackEvaluations',NaN);


S.nGridEvaluations = ...
    getScalarField(T,'nGridEvaluations',NaN);


S.nRefinementEvaluations = ...
    getScalarField(T,'nRefinementEvaluations',NaN);


S.nValidEvaluations = ...
    getScalarField(T,'nValidEvaluations',NaN);


S.nFailedEvaluations = ...
    getScalarField(T,'nFailedEvaluations',NaN);


S.nRBFSolves = ...
    getScalarField(T,'nRBFSolves',NaN);


S.geometryWallTime = ...
    getScalarField(T,'geometryWallTime',NaN);


S.stage1WallTime = ...
    getScalarField( ...
        T, ...
        'stage1WallTime', ...
        getScalarField(T,'totalWallTime',NaN));


S.coreFile = ...
    Paths.core;


S.diagFile = ...
    Paths.diag;


if S.traceSaved

    S.traceSearchFile = ...
        Paths.traceSearch;


    S.traceOptFile = ...
        Paths.traceOpt;


    S.traceNumericFile = ...
        Paths.traceNumeric;

else

    S.traceSearchFile = ...
        '';


    S.traceOptFile = ...
        '';


    S.traceNumericFile = ...
        '';

end


end



% =============================================================
% Local helper
% =============================================================

function value = getStage1SummaryScalar(T,fieldName,defaultValue)
%GETSTAGE1SUMMARYSCALAR Read Stage-1 scalar from top-level or nested summary.

value = ...
    getScalarField( ...
        T, ...
        fieldName, ...
        defaultValue);


if isequaln(value,defaultValue) && ...
   isstruct(T) && ...
   isfield(T,'stage1Summary') && ...
   isstruct(T.stage1Summary)

    value = ...
        getScalarField( ...
            T.stage1Summary, ...
            fieldName, ...
            defaultValue);

end

end


% =============================================================
% Local helper
% =============================================================

function value = getExecutionLogicalOption( ...
    options, fieldName, defaultValue)
%GETEXECUTIONLOGICALOPTION Read one logical execution option.

value = ...
    logical(defaultValue);


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,fieldName) && ...
   ~isempty(options.execution.(fieldName))

    candidate = ...
        options.execution.(fieldName);


    if ~(isscalar(candidate) && ...
         (islogical(candidate) || ...
          (isnumeric(candidate) && ...
           isreal(candidate) && ...
           isfinite(candidate) && ...
           any(candidate == [0 1]))))

        error( ...
            ['options.execution.%s must be true/false ' ...
             '(or numeric 1/0).'], ...
            fieldName);

    end


    value = ...
        logical(candidate);

end

end


% =============================================================
% Local helper
% =============================================================

function nWorkers = getExecutionWorkerCount(options)
%GETEXECUTIONWORKERCOUNT Read optional requested Processes worker count.

nWorkers = ...
    [];


if isfield(options,'execution') && ...
   isstruct(options.execution) && ...
   isfield(options.execution,'numWorkers') && ...
   ~isempty(options.execution.numWorkers)

    candidate = ...
        options.execution.numWorkers;


    if ~isnumeric(candidate) || ...
       ~isscalar(candidate) || ...
       ~isreal(candidate) || ...
       ~isfinite(candidate) || ...
       candidate < 1 || ...
       candidate ~= round(candidate)

        error( ...
            ['options.execution.numWorkers must be empty or a ' ...
             'positive integer scalar.']);

    end


    nWorkers = ...
        double(candidate);

end

end


% =============================================================
% Local helper
% =============================================================

function R = emptyRunResult()
%EMPTYRUNRESULT

R = struct();

R.i = NaN;
R.j = NaN;
R.model = '';
R.kernel = '';
R.ModelAndFunction = '';
R.runField = '';
R.nRadii = NaN;
R.neighborCount =[];

R.C_min = NaN;
R.C_max = NaN;
R.Reg_min = NaN;
R.Reg_max = NaN;

R.train_rmse = NaN;
R.test_rmse = NaN;

R.finalCapIndex = NaN;
R.finalCapRadiusRaw = NaN;
R.finalCapRadiusScaled = NaN;

R.capRMSESource = '';
R.capSelectionRMSE = NaN;
R.capSelectionMinRMSE = NaN;
R.capSelectionRMSEThreshold = NaN;

R.radiusMedianRaw = NaN;
R.radiusMinRaw = NaN;
R.radiusMaxRaw = NaN;

R.CMedian = NaN;
R.CMinSelected = NaN;
R.CMaxSelected = NaN;

R.RegMedian = NaN;
R.RegMinSelected = NaN;
R.RegMaxSelected = NaN;

R.epsilonMedian = NaN;
R.epsilonMin = NaN;
R.epsilonMax = NaN;

R.nEvaluations = NaN;
R.nGAEvaluations = NaN;
R.nFallbackEvaluations = NaN;
R.nGridEvaluations = NaN;
R.nRefinementEvaluations = NaN;
R.nValidEvaluations = NaN;
R.nFailedEvaluations = NaN;

R.nSuccessfulCaps = NaN;
R.nFailedCaps = NaN;
R.nSuccessfulTrainPointsAnyCap = NaN;
R.nFailedTrainPointsAnyCap = NaN;

R.nSelectedPointCapPairs = NaN;
R.nUniqueSelectedSystems = NaN;
R.nDiagnosticCalculationsAvoided = NaN;
R.diagnosticReuseFraction = NaN;

R.nNearestTransfers = NaN;
R.nIDWTransfers = NaN;
R.nEnsembleTransfers = NaN;
R.nTransferFailures = NaN;
R.nSuccessfulTestPoints = NaN;
R.nFailedTestPoints = NaN;

R.trainSearchRBFSolves = NaN;
R.trainPseudoRBFSolves = NaN;
R.trainDiagnosticRBFSolves = NaN;
R.trainRBFSolves = NaN;
R.testRBFSolves = NaN;
R.totalRBFSolves = NaN;

R.trainWallTime = NaN;
R.trainCPUTime = NaN;
R.pseudoTrainingWallTime = NaN;
R.testWallTime = NaN;
R.testCPUTime = NaN;
R.totalWallTime = NaN;
R.totalCPUTime = NaN;

R.trainSuccess = false;
R.evaluationSuccess = false;
R.testSuccess = false;
R.runSuccess = false;

R.trainMessage = '';
R.evaluationMessage = '';
R.testMessage = '';

R.RunInfo = struct();
R.SummaryRow = struct();

end


% =============================================================
% Local helper
% =============================================================

function value = getScalarField(S,fieldName,defaultValue)
%GETSCALARFIELD

value = defaultValue;

if ~isstruct(S) || ...
   ~isfield(S,fieldName)
    return
end

candidate = S.(fieldName);

if isnumeric(candidate) && ...
   isscalar(candidate) && ...
   isreal(candidate)

    value = candidate;

end

end


% =============================================================
% Local helper
% =============================================================

function value = getLogicalField(S,fieldName,defaultValue)
%GETLOGICALFIELD

value = defaultValue;

if ~isstruct(S) || ...
   ~isfield(S,fieldName)
    return
end

candidate = S.(fieldName);

if isscalar(candidate) && ...
   (islogical(candidate) || isnumeric(candidate))

    value = logical(candidate);

end

end


% =============================================================
% Local helper
% =============================================================

function value = getTextField(S,fieldName,defaultValue)
%GETTEXTFIELD

value = defaultValue;

if ~isstruct(S) || ...
   ~isfield(S,fieldName)
    return
end

candidate = S.(fieldName);

if ischar(candidate)

    value = candidate;

elseif isstring(candidate) && isscalar(candidate)

    value = char(candidate);

end

end


% =============================================================
% Local helper
% =============================================================

function value = getVectorField(S,fieldName)
%GETVECTORFIELD

value = [];

if ~isstruct(S) || ...
   ~isfield(S,fieldName)
    return
end

candidate = S.(fieldName);

if isnumeric(candidate) && ...
   isreal(candidate) && ...
   isvector(candidate)

    value = candidate(:);

end

end


% =============================================================
% Local helper
% =============================================================

function [medValue,minValue,maxValue] = finiteSummary(x)
%FINITESUMMARY

medValue = NaN;
minValue = NaN;
maxValue = NaN;

if isempty(x)
    return
end

x = x(:);
x = x(isfinite(x));

if isempty(x)
    return
end

medValue = median(x);
minValue = min(x);
maxValue = max(x);

end


% =============================================================
% Local helper
% =============================================================

function M = normalizeToleranceMatrix( ...
    setting,nModel,nKernel,settingName)
%NORMALIZETOLERANCEMATRIX Normalize one Stage-2 tolerance setting.
%
% Accepted forms:
%
%   Common scalar:
%       tau_loc = 0.001;
%
%   Model x Kernel matrix:
%       % rows    = ListOfModel
%       % columns = NameFunction_RBF
%       tau_loc = [
%           0.0010  0.0020
%           0.0015  0.0025
%       ];
%
% The output is always an nModel x nKernel double matrix.  This keeps the
% PARFOR path simple, avoids dynamic field-name restrictions on model
% names, and makes the Model x Kernel mapping identical to Results matrices.

if ~isnumeric(setting) || ...
   ~isreal(setting) || ...
   isempty(setting)

    error( ...
        '%s must be a nonempty real numeric scalar or matrix.', ...
        settingName);

end


if isscalar(setting)

    if ~isfinite(setting) || ...
       setting < 0

        error( ...
            '%s scalar must be finite and nonnegative.', ...
            settingName);

    end


    M = ...
        repmat( ...
            double(setting), ...
            nModel, ...
            nKernel);

    return

end


if ~ismatrix(setting) || ...
   ~isequal(size(setting),[nModel nKernel])

    error( ...
        ['%s matrix must have size nModel x nKernel = %d x %d. ' ...
         'Rows follow ListOfModel; columns follow NameFunction_RBF.'], ...
        settingName, ...
        nModel, ...
        nKernel);

end


if any(~isfinite(setting(:))) || ...
   any(setting(:) < 0)

    error( ...
        '%s matrix values must all be finite and nonnegative.', ...
        settingName);

end


M = ...
    double(setting);

end


% =============================================================
% Local helper
% =============================================================

function radii = getKernelRadii(Radii,kernelKey)
%GETKERNELRADII
%
% Accept either one common numeric radius vector or a structure with one
% radius vector per kernel.

if isnumeric(Radii)

    radii = Radii;

elseif isstruct(Radii) && ...
       isfield(Radii,kernelKey)

    radii = Radii.(kernelKey);

else

    error( ...
        'Candidate radii were not provided for kernel "%s".', ...
        kernelKey);

end

if ~isnumeric(radii) || ...
   isempty(radii) || ...
   ~isvector(radii) || ...
   ~isreal(radii) || ...
   any(~isfinite(radii(:))) || ...
   any(radii(:) <= 0)

    error( ...
        'Candidate radii for kernel "%s" must be finite and positive.', ...
        kernelKey);

end

radii = radii(:);

if any(diff(radii) <= 0)

    error( ...
        'Candidate radii for kernel "%s" must be strictly increasing.', ...
        kernelKey);

end

end


% =============================================================
% Local helper
% =============================================================

function value = sumFinitePair(a,b)
%SUMFINITEPAIR
%
% Sum two scalar costs/counts without converting a missing second stage
% into zero. If only one stage is finite, return the available stage.

if isfinite(a) && isfinite(b)

    value = a + b;

elseif isfinite(a)

    value = a;

elseif isfinite(b)

    value = b;

else

    value = NaN;

end

end

% =============================================================
% Local helper
% =============================================================

function Meta = makeStage2StorageMeta( ...
    inputSourceFile,OutputFile,runField,CaseName,modelName,kernel, ...
    X,Xtest,radii,tau_loc,tau_cap,near_threshold,idw_radius, ...
    transferMode,nIDWSources,parameterTransferMode,capRMSESource, ...
    min_neighbors,methodRun,nGridRun,Reg_min,Reg_max,C_min,C_max, ...
    RNG_SEED,solverOptions,stage1CheckpointDir, ...
    saveStage2All,saveEnsemblePointDiagnostics)
%MAKESTAGE2STORAGEMETA Small reproducibility metadata only.

Meta = struct();
Meta.SourceFile = inputSourceFile;
Meta.OutputFile = OutputFile;
Meta.RunField = runField;
Meta.Case = CaseName;
Meta.Model = modelName;
Meta.Kernel = kernel;
Meta.NTrain = size(X,1);
Meta.NTest = size(Xtest,1);
Meta.CapCandidatesRaw = radii(:);
Meta.CapCandidatesKm = radii(:)./1000;
Meta.TauLoc = tau_loc;
Meta.TauCap = tau_cap;
Meta.NearThresholdRaw = near_threshold;
Meta.NearThresholdKm = near_threshold./1000;
Meta.IDWRadiusRaw = idw_radius;
Meta.IDWRadiusKm = idw_radius./1000;



Meta.TransferMethod = ...
    transferMode;

Meta.NIDWSources = ...
    nIDWSources;

Meta.ParameterTransferMode = ...
    parameterTransferMode;

Meta.CapRMSESource = ...
    capRMSESource;

Meta.IDWPower = 2;

Meta.MinNeighbors = min_neighbors;
Meta.OptimizationMethod = methodRun;
Meta.NGrid = nGridRun;
Meta.RegSearchRange = [Reg_min Reg_max];
Meta.CSearchRange = [C_min C_max];
Meta.RNGSeed = RNG_SEED;
Meta.RNG_SEED = RNG_SEED;
Meta.SolverOptions = solverOptions;
Meta.SolverCondThreshold = getScalarField(solverOptions,'cond_lsq',NaN);
Meta.Stage1CheckpointDir = stage1CheckpointDir;
Meta.SaveStage2All = logical(saveStage2All);
Meta.SaveEnsemblePointDiagnostics = ...
    logical(saveEnsemblePointDiagnostics);
end


function Core = buildStage2Core( ...
    Meta,TrainingResult,TestingResult,X,y,Xtest,ytest,Rrun)
%BUILDSTAGE2CORE Build the authoritative self-contained Stage-2 result.
%
% CORE intentionally contains everything needed for:
%   - the information previously reconstructed by the revised collector;
%   - direct scientific inspection of final train/test results;
%   - fixed-selected-settings bootstrap without Stage-1 optimization;
%   - all three pointwise numerical-condition diagnostics.
%
% Heavy transfer-source provenance and other non-compulsory diagnostics
% remain in optional DETAIL.

nTrain = ...
    numel(y);

nTest = ...
    numel(ytest);


Core = ...
    struct();


% -------------------------------------------------------------
% Setup / reproducibility metadata
% -------------------------------------------------------------

Core.Setup = ...
    Meta;

Core.Setup.S_global = ...
    getScalarField( ...
        TrainingResult, ...
        'S_global', ...
        NaN);

% Compatibility alias retained for existing downstream code.
Core.Setup.SGlobal = ...
    Core.Setup.S_global;


% -------------------------------------------------------------
% Final-cap selection provenance
%
% Keep both original-training and pseudo-training cap metrics small and
% explicit.  These are K-length vectors, not nTrain x K matrices.
% -------------------------------------------------------------

nCaps = ...
    numel( ...
        Meta.CapCandidatesRaw);


Core.Selection = ...
    struct();


Core.Selection.CapRMSESource = ...
    getTextField( ...
        TrainingResult, ...
        'finalCapRMSESource', ...
        getTextField(Meta,'CapRMSESource','training'));


Core.Selection.FinalCapIndex = ...
    getScalarField( ...
        TrainingResult, ...
        'finalCapIndex', ...
        NaN);


Core.Selection.FinalCapRadiusRaw = ...
    getScalarField( ...
        TrainingResult, ...
        'finalCapRadius_raw', ...
        NaN);


Core.Selection.CapSelectionRMSE = ...
    getScalarField( ...
        TrainingResult, ...
        'finalCapSelectionRMSE', ...
        NaN);


Core.Selection.CapSelectionMinRMSE = ...
    getScalarField( ...
        TrainingResult, ...
        'finalCapSelectionMinRMSE', ...
        NaN);


Core.Selection.CapSelectionRMSEThreshold = ...
    getScalarField( ...
        TrainingResult, ...
        'finalCapSelectionRMSEThreshold', ...
        NaN);


Core.Selection.TrainingRMSEAll = ...
    vectorFieldOrNaN( ...
        TrainingResult, ...
        'RMSE_all', ...
        nCaps);


Core.Selection.PseudoRMSEAll = ...
    vectorFieldOrNaN( ...
        TrainingResult, ...
        'PseudoRMSE_all', ...
        nCaps);


Core.Selection.PseudoRMSEAvailableAll = ...
    vectorFieldOrNaN( ...
        TrainingResult, ...
        'PseudoRMSE_available_all', ...
        nCaps);


Core.Selection.CapRMSEAll = ...
    vectorFieldOrNaN( ...
        TrainingResult, ...
        'capRMSE_all', ...
        nCaps);


Core.Selection.TrainingCapSuccess = ...
    logicalFieldVectorOrFalse( ...
        TrainingResult, ...
        'cap_success', ...
        nCaps);


Core.Selection.PseudoCapSuccess = ...
    logicalFieldVectorOrFalse( ...
        TrainingResult, ...
        'pseudo_cap_success', ...
        nCaps);


Core.Selection.PseudoCapSuccessEffective = ...
    logicalFieldVectorOrFalse( ...
        TrainingResult, ...
        'pseudo_cap_success_effective', ...
        nCaps);


Core.Selection.PseudoTrainingComputed = ...
    getLogicalField( ...
        TrainingResult, ...
        'pseudoTrainingComputed', ...
        false);


Core.Selection.PseudoTrainingDefinition = ...
    getTextField( ...
        TrainingResult, ...
        'pseudoTrainingDefinition', ...
        '');


Core.Selection.TransferMethodForCapRMSE = ...
    getFinalTextFieldLocal( ...
        TrainingResult, ...
        'transferMethodForCapRMSE', ...
        getTextField( ...
            Meta, ...
            'TransferMethod', ...
            ''));


Core.Selection.ParameterTransferModeForCapRMSE = ...
    getFinalTextFieldLocal( ...
        TrainingResult, ...
        'parameterTransferModeForCapRMSE', ...
        getTextField( ...
            Meta, ...
            'ParameterTransferMode', ...
            ''));


% -------------------------------------------------------------
% Bootstrap-ready original data
% -------------------------------------------------------------

Core.Data = ...
    struct();

Core.Data.XTrain = ...
    X;

Core.Data.YTrain = ...
    y(:);

Core.Data.XTest = ...
    Xtest;

Core.Data.YTest = ...
    ytest(:);


% -------------------------------------------------------------
% Final selected / transferred parameters
% -------------------------------------------------------------

Core.Parameters = ...
    struct();

Core.Parameters.Train = ...
    struct();

Core.Parameters.Test = ...
    struct();


Final = ...
    struct();

if isstruct(TrainingResult) && ...
   isfield(TrainingResult,'Final') && ...
   isstruct(TrainingResult.Final)

    Final = ...
        TrainingResult.Final;

end


Core.Parameters.Train.Radius = ...
    vectorFieldOrNaN( ...
        Final, ...
        'radius_raw', ...
        nTrain);

Core.Parameters.Train.C = ...
    vectorFieldOrNaN( ...
        Final, ...
        'C', ...
        nTrain);

Core.Parameters.Train.Reg = ...
    vectorFieldOrNaN( ...
        Final, ...
        'reg_param', ...
        nTrain);

Core.Parameters.Train.NeighborCount = ...
    vectorFieldOrNaN( ...
        Final, ...
        'neighbor_count', ...
        nTrain);


Core.Parameters.Test.Radius = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'radius_raw', ...
        nTest);

Core.Parameters.Test.C = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'C', ...
        nTest);

Core.Parameters.Test.Reg = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'reg_param', ...
        nTest);

Core.Parameters.Test.NeighborCount = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'neighbor_count', ...
        nTest);

% Actual pointwise nearest/IDW method remains compulsory because it is
% scientifically meaningful and was preserved by the revised collector.
Core.Parameters.Test.TransferMethod = ...
    cellFieldOrEmpty( ...
        TestingResult, ...
        'transferMethod', ...
        nTest);


% Exact final selected-cap pseudo-unseen settings for bootstrap/calibration.
% Bootstrap intentionally uses one scalar Radius/Reg/C tuple per target
% and has no ensemble execution branch.
Core.Parameters.Pseudo = struct();
if isfield(TrainingResult,'PseudoFinal') && isstruct(TrainingResult.PseudoFinal)
    PF = TrainingResult.PseudoFinal;
    Core.Parameters.Pseudo.Radius = vectorFieldOrNaN(PF,'Radius',nTrain);
    Core.Parameters.Pseudo.Reg = vectorFieldOrNaN(PF,'RegParam',nTrain);
    Core.Parameters.Pseudo.C = vectorFieldOrNaN(PF,'C',nTrain);
    Core.Parameters.Pseudo.NeighborCount = vectorFieldOrNaN(PF,'NeighborCount',nTrain);
    Core.Parameters.Pseudo.PointSuccess = logicalFieldVectorOrFalse(PF,'PointSuccess',nTrain);
    Core.Parameters.Pseudo.ScalarPointwiseParametersAvailable = ...
        getLogicalField(PF,'ScalarPointwiseParametersAvailable',false);
    Core.Parameters.Pseudo.TransferMethod = getTextField(PF,'TransferMethod','');
    Core.Parameters.Pseudo.ParameterTransferMode = getTextField(PF,'ParameterTransferMode','');
    Core.Parameters.Pseudo.CapIndex = getScalarField(PF,'CapIndex',NaN);
    Core.Parameters.Pseudo.CapRadiusRaw = getScalarField(PF,'CapRadiusRaw',NaN);
else
    Core.Parameters.Pseudo.Radius = nan(nTrain,1);
    Core.Parameters.Pseudo.Reg = nan(nTrain,1);
    Core.Parameters.Pseudo.C = nan(nTrain,1);
    Core.Parameters.Pseudo.NeighborCount = nan(nTrain,1);
    Core.Parameters.Pseudo.PointSuccess = false(nTrain,1);
    Core.Parameters.Pseudo.ScalarPointwiseParametersAvailable = false;
    Core.Parameters.Pseudo.TransferMethod = '';
    Core.Parameters.Pseudo.ParameterTransferMode = '';
    Core.Parameters.Pseudo.CapIndex = NaN;
    Core.Parameters.Pseudo.CapRadiusRaw = NaN;
end


Core.Parameters.Test.EnsembleNUniqueTuples = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'ensemble_n_unique_tuples', ...
        nTest);


if strcmpi( ...
        getTextField(Meta,'TransferMethod',''), ...
        'ensemble')

    Core.Parameters.Test.EnsembleUniqueWeight = ...
        cellFieldOrEmpty( ...
            TestingResult, ...
            'ensembleUniqueWeight', ...
            nTest);

    Core.Parameters.Test.EnsembleUniqueRadiusRaw = ...
        cellFieldOrEmpty( ...
            TestingResult, ...
            'ensembleUniqueRadiusRaw', ...
            nTest);

    Core.Parameters.Test.EnsembleUniqueRegParam = ...
        cellFieldOrEmpty( ...
            TestingResult, ...
            'ensembleUniqueRegParam', ...
            nTest);

    Core.Parameters.Test.EnsembleUniqueC = ...
        cellFieldOrEmpty( ...
            TestingResult, ...
            'ensembleUniqueC', ...
            nTest);

end


% -------------------------------------------------------------
% Accuracy: full pointwise + scalar summaries
% -------------------------------------------------------------

Core.Accuracy = ...
    struct();

Core.Accuracy.Train = ...
    struct();

Core.Accuracy.Test = ...
    struct();


Core.Accuracy.Train.Observed = ...
    y(:);

Core.Accuracy.Train.Prediction = ...
    vectorFieldOrNaN( ...
        TrainingResult, ...
        'yhat', ...
        nTrain);

Core.Accuracy.Train.Residual = ...
    firstVectorFieldOrNaN( ...
        TrainingResult, ...
        {'residual','y_diff'}, ...
        nTrain);

Core.Accuracy.Train.AE = ...
    abs( ...
        Core.Accuracy.Train.Residual);

Core.Accuracy.Train.RMSE = ...
    getScalarField( ...
        TrainingResult, ...
        'RMSE', ...
        NaN);

Core.Accuracy.Train.MAE = ...
    finiteMean( ...
        Core.Accuracy.Train.AE);

Core.Accuracy.Train.MaxAE = ...
    finiteMax( ...
        Core.Accuracy.Train.AE);


Core.Accuracy.Test.Observed = ...
    ytest(:);

Core.Accuracy.Test.Prediction = ...
    firstVectorFieldOrNaN( ...
        TestingResult, ...
        {'yhat','y_pred_test'}, ...
        nTest);

Core.Accuracy.Test.Residual = ...
    firstVectorFieldOrNaN( ...
        TestingResult, ...
        {'residual','y_diff','test_diff'}, ...
        nTest);

Core.Accuracy.Test.AE = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'AE', ...
        nTest);

if all(~isfinite(Core.Accuracy.Test.AE))

    Core.Accuracy.Test.AE = ...
        abs( ...
            Core.Accuracy.Test.Residual);

end

Core.Accuracy.Test.RMSE = ...
    getScalarField( ...
        TestingResult, ...
        'RMSE', ...
        Rrun.test_rmse);

Core.Accuracy.Test.MAE = ...
    getScalarField( ...
        TestingResult, ...
        'MAE', ...
        finiteMean(Core.Accuracy.Test.AE));

Core.Accuracy.Test.MaxAE = ...
    finiteMax( ...
        Core.Accuracy.Test.AE);


Core.Accuracy.Pseudo = struct();
Core.Accuracy.Pseudo.Observed = y(:);
if isfield(TrainingResult,'PseudoFinal') && isstruct(TrainingResult.PseudoFinal)
    PF = TrainingResult.PseudoFinal;
    Core.Accuracy.Pseudo.Prediction = vectorFieldOrNaN(PF,'Prediction',nTrain);
    Core.Accuracy.Pseudo.Residual = vectorFieldOrNaN(PF,'Residual',nTrain);
    Core.Accuracy.Pseudo.Success = logicalFieldVectorOrFalse(PF,'PointSuccess',nTrain);
else
    Core.Accuracy.Pseudo.Prediction = nan(nTrain,1);
    Core.Accuracy.Pseudo.Residual = nan(nTrain,1);
    Core.Accuracy.Pseudo.Success = false(nTrain,1);
end
Core.Accuracy.Pseudo.AE = abs(Core.Accuracy.Pseudo.Residual);
Core.Accuracy.Pseudo.RMSE = sqrt(finiteMean(Core.Accuracy.Pseudo.Residual.^2));
Core.Accuracy.Pseudo.MAE = finiteMean(Core.Accuracy.Pseudo.AE);
Core.Accuracy.Pseudo.MaxAE = finiteMax(Core.Accuracy.Pseudo.AE);


% -------------------------------------------------------------
% Numerical diagnostics: all three pointwise condition types
% -------------------------------------------------------------

Core.Numerical = ...
    struct();

Core.Numerical.Train = ...
    struct();

Core.Numerical.Test = ...
    struct();

Core.Numerical.Pseudo = ...
    struct();


Eval = ...
    struct();

if isstruct(TrainingResult) && ...
   isfield(TrainingResult,'EvaluationResult') && ...
   isstruct(TrainingResult.EvaluationResult)

    Eval = ...
        TrainingResult.EvaluationResult;

end


[Core.Numerical.Train.Cond1,~] = ...
    evaluationVectorOrNaN( ...
        Eval, ...
        'cond_raw', ...
        nTrain);

[Core.Numerical.Train.Cond2,trainCond2Found] = ...
    evaluationVectorOrNaN( ...
        Eval, ...
        'cond_reg', ...
        nTrain);

if ~trainCond2Found

    Core.Numerical.Train.Cond2 = ...
        vectorFieldOrNaN( ...
            Final, ...
            'cond_reg', ...
            nTrain);

end

[Core.Numerical.Train.Cond3,~] = ...
    evaluationVectorOrNaN( ...
        Eval, ...
        'cond_effective', ...
        nTrain);

[Core.Numerical.Train.Solver,trainSolverFound] = ...
    evaluationCellOrEmpty( ...
        Eval, ...
        'solver_used', ...
        nTrain);

if ~trainSolverFound

    Core.Numerical.Train.Solver = ...
        cellFieldOrEmpty( ...
            Final, ...
            'solver_used', ...
            nTrain);

end

[Core.Numerical.Train.Success,trainNumericalSuccessFound] = ...
    evaluationSuccessVector( ...
        Eval, ...
        nTrain);

if ~trainNumericalSuccessFound

    Core.Numerical.Train.Success = ...
        logicalFieldVectorOrFalse( ...
            Final, ...
            'point_success', ...
            nTrain);

end


Core.Numerical.Test.Cond1 = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'cond_raw', ...
        nTest);

Core.Numerical.Test.Cond2 = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'cond_reg', ...
        nTest);

Core.Numerical.Test.Cond3 = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'cond_effective', ...
        nTest);

Core.Numerical.Test.Solver = ...
    cellFieldOrEmpty( ...
        TestingResult, ...
        'solver_used', ...
        nTest);

Core.Numerical.Test.Success = ...
    logicalFieldVectorOrFalse( ...
        TestingResult, ...
        'point_success', ...
        nTest);


[Core.Numerical.Train.CondEvaluation,~] = ...
    evaluationVectorOrNaN( ...
        Eval, ...
        'cond_evaluation_train', ...
        nTrain);


[Core.Numerical.Train.RelResidual,~] = ...
    evaluationVectorOrNaN( ...
        Eval, ...
        'rel_residual', ...
        nTrain);


[Core.Numerical.Train.BackwardError,~] = ...
    evaluationVectorOrNaN( ...
        Eval, ...
        'backward_error', ...
        nTrain);


Core.Numerical.Test.CondEvaluation = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'cond_evaluation_test', ...
        nTest);


Core.Numerical.Test.RelResidual = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'rel_residual', ...
        nTest);


Core.Numerical.Test.BackwardError = ...
    vectorFieldOrNaN( ...
        TestingResult, ...
        'backward_error', ...
        nTest);


if isfield(TrainingResult,'PseudoFinal') && isstruct(TrainingResult.PseudoFinal)
    PF = TrainingResult.PseudoFinal;
    Core.Numerical.Pseudo.Cond1 = vectorFieldOrNaN(PF,'cond_raw',nTrain);
    Core.Numerical.Pseudo.Cond2 = vectorFieldOrNaN(PF,'cond_reg',nTrain);
    Core.Numerical.Pseudo.Cond3 = vectorFieldOrNaN(PF,'cond_effective',nTrain);
    Core.Numerical.Pseudo.RelResidual = vectorFieldOrNaN(PF,'rel_residual',nTrain);
    Core.Numerical.Pseudo.BackwardError = vectorFieldOrNaN(PF,'backward_error',nTrain);
    Core.Numerical.Pseudo.Solver = cellFieldOrEmpty(PF,'solver_used',nTrain);
    Core.Numerical.Pseudo.Success = logicalFieldVectorOrFalse(PF,'PointSuccess',nTrain);
else
    Core.Numerical.Pseudo.Cond1 = nan(nTrain,1);
    Core.Numerical.Pseudo.Cond2 = nan(nTrain,1);
    Core.Numerical.Pseudo.Cond3 = nan(nTrain,1);
    Core.Numerical.Pseudo.RelResidual = nan(nTrain,1);
    Core.Numerical.Pseudo.BackwardError = nan(nTrain,1);
    Core.Numerical.Pseudo.Solver = cell(nTrain,1);
    Core.Numerical.Pseudo.Success = false(nTrain,1);
end


Core.Numerical.Test.EnsemblePointDiagnosticsSaved = ...
    false;


if isstruct(TestingResult) && ...
   isfield(TestingResult,'ensemblePointDiagnostics') && ...
   iscell(TestingResult.ensemblePointDiagnostics) && ...
   numel(TestingResult.ensemblePointDiagnostics) == nTest

    Core.Numerical.Test.EnsemblePointDiagnostics = ...
        TestingResult.ensemblePointDiagnostics(:);


    Core.Numerical.Test.EnsemblePointDiagnosticsSaved = ...
        true;

end


Core.Numerical.Definition = ...
    struct();

Core.Numerical.Definition.Cond1 = ...
    'cond_raw';

Core.Numerical.Definition.Cond2 = ...
    'cond_reg';

Core.Numerical.Definition.Cond3 = ...
    'cond_effective';

Core.Numerical.Definition.TrainCondEvaluation = ...
    'cond_evaluation_train';

Core.Numerical.Definition.TestCondEvaluation = ...
    'cond_evaluation_test';

Core.Numerical.Definition.RelResidual = ...
    'rel_residual';

Core.Numerical.Definition.BackwardError = ...
    'backward_error';


% -------------------------------------------------------------
% Runtime / computational cost
% -------------------------------------------------------------

Core.Runtime = ...
    struct();

Core.Runtime.TrainSec = ...
    Rrun.trainWallTime;

Core.Runtime.TestSec = ...
    Rrun.testWallTime;

Core.Runtime.TotalSec = ...
    Rrun.totalWallTime;

Core.Runtime.TrainCPUTime = ...
    Rrun.trainCPUTime;

Core.Runtime.TestCPUTime = ...
    Rrun.testCPUTime;

Core.Runtime.TotalCPUTime = ...
    Rrun.totalCPUTime;

Core.Runtime.PseudoTrainingSec = ...
    Rrun.pseudoTrainingWallTime;


Core.Computation = ...
    struct();

Core.Computation.TrainSearchRBFSolves = ...
    Rrun.trainSearchRBFSolves;

Core.Computation.TrainPseudoRBFSolves = ...
    Rrun.trainPseudoRBFSolves;

Core.Computation.TrainDiagnosticRBFSolves = ...
    Rrun.trainDiagnosticRBFSolves;

Core.Computation.TrainRBFSolves = ...
    Rrun.trainRBFSolves;

Core.Computation.TestRBFSolves = ...
    Rrun.testRBFSolves;

Core.Computation.TotalRBFSolves = ...
    Rrun.totalRBFSolves;

Core.Computation.NNearestTransfers = ...
    Rrun.nNearestTransfers;

Core.Computation.NIDWTransfers = ...
    Rrun.nIDWTransfers;

Core.Computation.NEnsembleTransfers = ...
    Rrun.nEnsembleTransfers;

Core.Computation.NTransferFailures = ...
    Rrun.nTransferFailures;


% -------------------------------------------------------------
% Computational / scientific status
% -------------------------------------------------------------

Core.Status = ...
    struct();

Core.Status.TrainSuccess = ...
    logical(Rrun.trainSuccess);

Core.Status.EvaluationSuccess = ...
    logical(Rrun.evaluationSuccess);

Core.Status.TestSuccess = ...
    logical(Rrun.testSuccess);

Core.Status.RunSuccess = ...
    logical(Rrun.runSuccess);

Core.Status.TrainMessage = ...
    Rrun.trainMessage;

Core.Status.EvaluationMessage = ...
    Rrun.evaluationMessage;

Core.Status.TestMessage = ...
    Rrun.testMessage;


% -------------------------------------------------------------
% Unit contract
% -------------------------------------------------------------

Core.Units = ...
    struct();

Core.Units.Coordinates = ...
    'original coordinate units';

Core.Units.Radius = ...
    'original coordinate units';

% Collector-compatible alias.
Core.Units.ParameterRadius = ...
    'original coordinate units';

Core.Units.Distance = ...
    'original coordinate units';

Core.Units.CapCandidates = ...
    'original coordinate units';

Core.Units.SummaryRadius = ...
    'km';

Core.Units.Response = ...
    'input response units';


% -------------------------------------------------------------
% Explicit fixed-setting bootstrap contract
% -------------------------------------------------------------

Core.Bootstrap = ...
    struct();

Core.Bootstrap.Mode = ...
    'fixed-selected-settings';

Core.Bootstrap.Ready = ...
    logical(Rrun.runSuccess);

Core.Bootstrap.FixedSettingsSource = ...
    'Stage2Core.Parameters';

Core.Bootstrap.RecomputeNeighborhood = ...
    true;

Core.Bootstrap.RecomputeEpsilon = ...
    common.kernelHasShapeParameter( ...
        Meta.Kernel);

Core.Bootstrap.RequiresStage1Optimization = ...
    false;


Core.Bootstrap.EnsemblePointDiagnosticsSaved = ...
    Core.Numerical.Test.EnsemblePointDiagnosticsSaved;


if strcmpi( ...
        getTextField(Meta,'TransferMethod',''), ...
        'ensemble')

    Core.Bootstrap.EnsembleTupleSettingsSource = ...
        'Stage2Core.Parameters.Test.EnsembleUnique*';

end

end


function S = makeStage2SummaryRow(Core,sourceFile)
%MAKESTAGE2SUMMARYROW Compact collector-equivalent Stage-2 summary row.
%
% Every value is derived from the finalized authoritative CORE so the
% summary cannot diverge from the saved scientific result.

S = ...
    struct();

S.RunID = ...
    "";

S.Case = ...
    string(getTextField(Core.Setup,'Case',''));

S.Model = ...
    string(getTextField(Core.Setup,'Model',''));

S.Kernel = ...
    string(getTextField(Core.Setup,'Kernel',''));

S.RunField = ...
    string(getTextField(Core.Setup,'RunField',''));

S.SourceMode = ...
    "STAGE2_DIRECT";

S.NTrain = ...
    getScalarField(Core.Setup,'NTrain',NaN);

S.NTest = ...
    getScalarField(Core.Setup,'NTest',NaN);

S.OptimizationMethod = ...
    string(getTextField(Core.Setup,'OptimizationMethod',''));

S.TauLoc = ...
    getScalarField(Core.Setup,'TauLoc',NaN);

S.TauCap = ...
    getScalarField(Core.Setup,'TauCap',NaN);

S.TransferMode = ...
    string(getTextField(Core.Setup,'TransferMethod',''));

S.NIDWSources = ...
    getScalarField(Core.Setup,'NIDWSources',NaN);

S.ParameterTransferMode = ...
    string(getTextField(Core.Setup,'ParameterTransferMode',''));

S.CapRMSESource = ...
    string(getTextField(Core.Selection,'CapRMSESource',''));

S.CapSelectionRMSE = ...
    getScalarField(Core.Selection,'CapSelectionRMSE',NaN);

S.EnsemblePointDiagnosticsSaved = ...
    double( ...
        getLogicalField( ...
            Core.Numerical.Test, ...
            'EnsemblePointDiagnosticsSaved', ...
            false));

S.IDWPower = ...
    getScalarField(Core.Setup,'IDWPower',NaN);

S.MinNeighbors = ...
    getScalarField(Core.Setup,'MinNeighbors',NaN);

S.TrainRMSE = ...
    getScalarField(Core.Accuracy.Train,'RMSE',NaN);

S.TrainMAE = ...
    getScalarField(Core.Accuracy.Train,'MAE',NaN);

S.TrainMaxAE = ...
    getScalarField(Core.Accuracy.Train,'MaxAE',NaN);

S.TestRMSE = ...
    getScalarField(Core.Accuracy.Test,'RMSE',NaN);

S.TestMAE = ...
    getScalarField(Core.Accuracy.Test,'MAE',NaN);

S.TestMaxAE = ...
    getScalarField(Core.Accuracy.Test,'MaxAE',NaN);

S.TrainRadiusMedianKm = ...
    medfiniteLocal(Core.Parameters.Train.Radius ./ 1000);

S.TestRadiusMedianKm = ...
    medfiniteLocal(Core.Parameters.Test.Radius ./ 1000);

S.TrainCMedian = ...
    medfiniteLocal(Core.Parameters.Train.C);

S.TestCMedian = ...
    medfiniteLocal(Core.Parameters.Test.C);

S.TrainRegMedian = ...
    medfiniteLocal(Core.Parameters.Train.Reg);

S.TestRegMedian = ...
    medfiniteLocal(Core.Parameters.Test.Reg);

S.TrainNeighborMedian = ...
    medfiniteLocal(Core.Parameters.Train.NeighborCount);

S.TestNeighborMedian = ...
    medfiniteLocal(Core.Parameters.Test.NeighborCount);

transferMethodPointwise = ...
    cellTextToStringSafe( ...
        Core.Parameters.Test.TransferMethod);

S.NNearestTransfers = ...
    sum(strcmpi(transferMethodPointwise,'nearest'));

S.NIDWTransfers = ...
    sum(strcmpi(transferMethodPointwise,'idw'));

S.NEnsembleTransfers = ...
    sum(strcmpi(transferMethodPointwise,'ensemble'));

S.NFailedTransfers = ...
    sum(strcmpi(transferMethodPointwise,'failed'));

S.TrainCond1Name = ...
    string(getTextField(Core.Numerical.Definition,'Cond1',''));

S.TrainCond1Median = ...
    medfiniteLocal(Core.Numerical.Train.Cond1);

S.TrainCond2Name = ...
    string(getTextField(Core.Numerical.Definition,'Cond2',''));

S.TrainCond2Median = ...
    medfiniteLocal(Core.Numerical.Train.Cond2);

S.TrainCond3Name = ...
    string(getTextField(Core.Numerical.Definition,'Cond3',''));

S.TrainCond3Median = ...
    medfiniteLocal(Core.Numerical.Train.Cond3);

S.TestCond1Name = ...
    string(getTextField(Core.Numerical.Definition,'Cond1',''));

S.TestCond1Median = ...
    medfiniteLocal(Core.Numerical.Test.Cond1);

S.TestCond2Name = ...
    string(getTextField(Core.Numerical.Definition,'Cond2',''));

S.TestCond2Median = ...
    medfiniteLocal(Core.Numerical.Test.Cond2);

S.TestCond3Name = ...
    string(getTextField(Core.Numerical.Definition,'Cond3',''));

S.TestCond3Median = ...
    medfiniteLocal(Core.Numerical.Test.Cond3);

S.TrainCondEvaluationMedian = ...
    medfiniteLocal(Core.Numerical.Train.CondEvaluation);

S.TestCondEvaluationMedian = ...
    medfiniteLocal(Core.Numerical.Test.CondEvaluation);

S.TrainRelResidualMax = ...
    finiteMax(Core.Numerical.Train.RelResidual);

S.TestRelResidualMax = ...
    finiteMax(Core.Numerical.Test.RelResidual);

S.TrainBackwardErrorMax = ...
    finiteMax(Core.Numerical.Train.BackwardError);

S.TestBackwardErrorMax = ...
    finiteMax(Core.Numerical.Test.BackwardError);

S.TrainPseudoRBFSolves = ...
    getScalarField(Core.Computation,'TrainPseudoRBFSolves',NaN);

S.PseudoTrainingSec = ...
    getScalarField(Core.Runtime,'PseudoTrainingSec',NaN);

S.TrainSec = ...
    getScalarField(Core.Runtime,'TrainSec',NaN);

S.TestSec = ...
    getScalarField(Core.Runtime,'TestSec',NaN);

S.TotalSec = ...
    getScalarField(Core.Runtime,'TotalSec',NaN);

S.TrainCPUTime = ...
    getScalarField(Core.Runtime,'TrainCPUTime',NaN);

S.TestCPUTime = ...
    getScalarField(Core.Runtime,'TestCPUTime',NaN);

S.TotalCPUTime = ...
    getScalarField(Core.Runtime,'TotalCPUTime',NaN);

S.TrainSuccess = ...
    double(getLogicalField(Core.Status,'TrainSuccess',false));

S.EvaluationSuccess = ...
    double(getLogicalField(Core.Status,'EvaluationSuccess',false));

S.TestSuccess = ...
    double(getLogicalField(Core.Status,'TestSuccess',false));

S.RunSuccess = ...
    double(getLogicalField(Core.Status,'RunSuccess',false));

S.SourceFile = ...
    string(sourceFile);

end


function m = medfiniteLocal(v)
%MEDFINITELOCAL Median of finite values only.

m = ...
    NaN;

if isempty(v)
    return
end

v = ...
    v(:);

q = ...
    isfinite(v);

if any(q)

    m = ...
        median(v(q));

end

end


function Detail = buildStage2Detail(TrainingResult,TestingResult,Rrun)
%BUILDSTAGE2DETAIL Optional Stage-2-only remainder.
%
% This function is called ONLY when saveStage2All=true.  It starts from
% the complete computed Stage-2 results and removes fields already retained
% in compulsory CORE.  Thus compulsory and optional
% storage do not intentionally duplicate the same canonical quantities.

Detail = struct();
Detail.Train = stripTrainingCompulsoryAliases(TrainingResult);
Detail.Test = stripTestingCompulsoryAliases(TestingResult);

% Keep heavy transfer-source provenance in one explicit DETAIL branch.
% The pointwise actual TransferMethod remains in CORE; source indices,
% distances, weights and related local-geometry provenance remain optional.
Detail.Transfer = struct();
if isstruct(TestingResult) && ...
   isfield(TestingResult,'Transfer') && ...
   isstruct(TestingResult.Transfer)

    Detail.Transfer = ...
        stripTransferCoreAliases( ...
            TestingResult.Transfer);

end

% Avoid storing the same Transfer structure twice inside DETAIL.
if isfield(Detail.Test,'Transfer')
    Detail.Test = rmfield(Detail.Test,'Transfer');
end

% Preserve every non-compulsory batch diagnostic by subtraction rather
% than by a hand-maintained allow-list. This protects future fields while
% still excluding quantities already present in CORE.
Detail.Batch = stripRrunCompulsoryAliases(Rrun);
end


function T = stripTransferCoreAliases(T)
%STRIPTRANSFERCOREALIASES Keep only optional/heavy transfer provenance.

if ~isstruct(T)
    T = struct();
    return
end

T = removeFieldsIfPresentLocal(T,{ ...
    'success','kernel','useShape','nTrain','nQuery','nDimensions', ...
    'transferMode','near_threshold','near_threshold_scaled','near_threshold_raw', ...
    'near_threshold_used_for_source_selection', ...
    'idw_radius','idw_radius_scaled','idw_radius_raw', ...
    'idw_radius_used_for_source_selection', ...
    'idw_n_nearest','idw_n_nearest_effective','idw_source_selection', ...
    'idw_power','radiusTransferScale','regParamTransferScale','CTransferScale', ...
    'distanceMatrixProvided','distanceMatrixSize','distanceMatrixStored', ...
    'querySuccess','transferMethod','messageByQuery', ...
    'nSuccessfulQueries','nFailedQueries','nNearestTransfers','nIDWTransfers', ...
    'nEnsembleTransfers','nClassifiedQueries','classificationCountConsistent', ...
    'ensembleNUniqueTuples','ensembleUniqueWeight','ensembleUniqueRadius', ...
    'ensembleUniqueRegParam','ensembleUniqueC', ...
    'radius','radius_scaled','radius_raw','reg_param','C'});
end


function B = stripRrunCompulsoryAliases(B)
%STRIPRRUNCOMPULSORYALIASES Remove fields already represented elsewhere.

if ~isstruct(B)
    B = struct();
    return
end

B = removeFieldsIfPresentLocal(B,{ ...
    'model','kernel','ModelAndFunction','runField','nRadii', ...
    'neighborCount','C_min','C_max','Reg_min','Reg_max', ...
    'train_rmse','test_rmse', ...
    'radiusMedianRaw','radiusMinRaw','radiusMaxRaw', ...
    'CMedian','CMinSelected','CMaxSelected', ...
    'RegMedian','RegMinSelected','RegMaxSelected', ...
    'trainSuccess','evaluationSuccess','testSuccess','runSuccess', ...
    'trainMessage','evaluationMessage','testMessage', ...
    'trainWallTime','testWallTime','totalWallTime', ...
    'trainCPUTime','testCPUTime','totalCPUTime','RunInfo'});
end


function T = stripTrainingCompulsoryAliases(T)
%STRIPTRAININGCOMPULSORYALIASES Remove Stage-2 compulsory aliases.

if ~isstruct(T)
    T = struct();
    return
end

T = removeFieldsIfPresentLocal(T,{ ...
    'kernel','useShape','y','nPoints','nDimensions','radii_raw', ...
    'minNeighbors','C_min','C_max','Reg_min','Reg_max', ...
    'tuningMethod','nGrid','tau_loc','tau_cap','RNG_SEED','S_global', ...
    'C','reg_param','radius','radius_raw','radius_scaled', ...
    'yhat','y_train_pred','y_diff','residual','y_diff_train','AE_train', ...
    'squaredResidual_train','RMSE','train_rmse', ...
    'RMSE_all','RMSE_available_all','PseudoRMSE_all', ...
    'PseudoRMSE_available_all','capRMSE_all','cap_success', ...
    'pseudo_cap_success','pseudo_cap_success_effective', ...
    'pseudoTrainingWallTime','pseudoTrainingComputed', ...
    'pseudoTrainingDefinition','nRBFSolvesPseudo', ...
    'coreTrainingWallTime','evaluationWallTime', ...
    'stage1WallTime','stage2WallTime','totalWallTime'});

if isfield(T,'Final') && isstruct(T.Final)
    % Remove only exact compulsory aliases. Final.yhat/residual/RMSE and
    % Final.cond_reg/solver_used/point_success are the values selected from
    % the optimization/cap-selection path, whereas CORE may use
    % the independently recomputed EvaluationResult values. They are
    % therefore distinct provenance and must remain in DETAIL when the user
    % requests full Stage-2 information.
    T.Final = removeFieldsIfPresentLocal(T.Final,{ ...
        'C','reg_param','radius','radius_raw','radius_scaled', ...
        'observation','neighbor_count'});
end

if isfield(T,'EvaluationResult') && isstruct(T.EvaluationResult)
    T.EvaluationResult = stripEvaluationCompulsoryAliases(T.EvaluationResult);
end

% In an independent Stage-2 run, Point(i).optimization is the already
% authoritative Stage-1 PointOpt loaded from checkpoint.  Do not duplicate
% that Stage-1 payload inside Stage-2 DETAIL.  One-pass 'all' mode keeps it
% because no separate Stage-1 checkpoint is guaranteed to exist.
if isfield(T,'executionStage') && ...
   (ischar(T.executionStage) || (isstring(T.executionStage) && isscalar(T.executionStage))) && ...
   strcmpi(char(T.executionStage),'stage2') && ...
   isfield(T,'Point') && isstruct(T.Point)

    T.Point = ...
        stripStructArrayFields( ...
            T.Point, ...
            {'optimization'});

end

end


function E = stripEvaluationCompulsoryAliases(E)
if ~isstruct(E)
    E = struct();
    return
end

E = removeFieldsIfPresentLocal(E,{ ...
    'yhat','y_pred','prediction','y_diff','residual','AE','squaredResidual', ...
    'RMSE','MAE','cond_raw','cond_reg','cond_effective','solver_used', ...
    'cond_evaluation_train','rel_residual','backward_error', ...
    'point_success','success_by_point'});

% Preserve non-compulsory information that may exist only in compact
% series/mapData containers.  Remove only exact compulsory aliases.
if isfield(E,'series') && isstruct(E.series)
    E.series = removeFieldsIfPresentLocal(E.series,{ ...
        'yhat','y_pred','prediction','y_diff','residual','AE','squaredResidual', ...
        'RMSE','MAE','cond_raw','cond_reg','cond_effective','solver_used', ...
        'cond_evaluation_train','rel_residual','backward_error', ...
        'point_success','success_by_point','radius','radius_raw','C','reg_param', ...
        'neighbor_count'});
end

if isfield(E,'mapData') && isstruct(E.mapData)
    E.mapData = removeFieldsIfPresentLocal(E.mapData,{ ...
        'yhat','y_pred','prediction','y_diff','residual','AE','squaredResidual', ...
        'RMSE','MAE','cond_raw','cond_reg','cond_effective','solver_used', ...
        'cond_evaluation_train','rel_residual','backward_error', ...
        'point_success','success_by_point','radius','radius_raw','C','reg_param', ...
        'neighbor_count'});
end

if isfield(E,'Point') && isstruct(E.Point)
    E.Point = stripStructArrayFields(E.Point,{ ...
        'observation','yhat','prediction','y_diff','residual','AE','squaredResidual', ...
        'radius','radius_raw','C','reg_param','neighbor_count', ...
        'cond_raw','cond_reg','cond_effective','solver_used', ...
        'cond_evaluation_train','rel_residual','backward_error','success'});
end
end


function T = stripTestingCompulsoryAliases(T)
%STRIPTESTINGCOMPULSORYALIASES Remove Stage-2 compulsory aliases.

if ~isstruct(T)
    T = struct();
    return
end

T = removeFieldsIfPresentLocal(T,{ ...
    'kernel','useShape','nTrain','nTest','S_global','ytest','minNeighbors', ...
    'near_threshold_raw','idw_radius_raw', ...
    'transferMode','nIDWSources', ...
    'ensemble_n_unique_tuples','ensembleUniqueWeight', ...
    'ensembleUniqueRadiusRaw','ensembleUniqueRegParam','ensembleUniqueC', ...
    'radius','radius_raw','radius_scaled','C','reg_param','neighbor_count', ...
    'transferMethod','yhat','y_pred_test','y_diff','test_diff','residual', ...
    'AE','squaredResidual','validResidualMask','nValidResiduals', ...
    'SSE_available','MSE_available','RMSE_available','MAE_available', ...
    'SSE','MSE','RMSE','MAE','cond_raw','cond_reg','cond_effective', ...
    'cond_evaluation_test','rel_residual','backward_error', ...
    'solver_used','point_success', ...
    'geometryWallTime','transferWallTime','testingWallTime','totalWallTime'});

% Keep only non-compulsory content from compact convenience containers.
% Exact copies of CORE arrays, including compulsory numerical-reliability
% diagnostics, are removed; optional local-geometry provenance is retained.
if isfield(T,'series') && isstruct(T.series)
    T.series = removeFieldsIfPresentLocal(T.series,{ ...
        'radius','radius_raw','radius_scaled','C','reg_param','neighbor_count', ...
        'transferMethod','yhat','y_pred_test','y_diff','test_diff','residual', ...
        'AE','squaredResidual','RMSE','MAE','cond_raw','cond_reg','cond_effective', ...
        'cond_evaluation_test','rel_residual','backward_error', ...
        'solver_used','point_success'});
end

if isfield(T,'mapData') && isstruct(T.mapData)
    T.mapData = removeFieldsIfPresentLocal(T.mapData,{ ...
        'radius','radius_raw','radius_scaled','C','reg_param','neighbor_count', ...
        'transferMethod','yhat','y_pred_test','y_diff','test_diff','residual', ...
        'AE','squaredResidual','RMSE','MAE','cond_raw','cond_reg','cond_effective', ...
        'cond_evaluation_test','rel_residual','backward_error', ...
        'solver_used','point_success'});
end

if isfield(T,'Transfer') && isstruct(T.Transfer)
    T.Transfer = removeFieldsIfPresentLocal(T.Transfer,{ ...
        'transferMethod','radius','radius_raw','radius_scaled','C','reg_param'});
end

if isfield(T,'Point') && isstruct(T.Point)
    T.Point = stripStructArrayFields(T.Point,{ ...
        'observation','transferMethod','radius','radius_raw','radius_scaled', ...
        'C','reg_param','neighbor_count','yhat','prediction','y_diff', ...
        'residual','AE','squaredResidual','cond_raw','cond_reg','cond_effective', ...
        'cond_evaluation_test','rel_residual','backward_error', ...
        'solver_used','success'});
end

end


function Paths = makeStage2CheckpointPaths(checkpointDir,runField,saveDetail)
%MAKESTAGE2CHECKPOINTPATHS Deterministic Stage-2 filenames.
%
% The revised Stage-2 architecture has one compulsory self-contained CORE
% and one optional DETAIL file.

Paths = ...
    struct( ...
        'core','', ...
        'detail','');

if isempty(checkpointDir)
    return
end

Paths.core = ...
    fullfile( ...
        checkpointDir, ...
        [runField '_STAGE2_CORE.mat']);

if saveDetail

    Paths.detail = ...
        fullfile( ...
            checkpointDir, ...
            [runField '_STAGE2_DETAIL.mat']);

end

end


function saveNamedVariableAtomic(fileName,variableName,value)
%SAVENAMEDVARIABLEATOMIC PARFOR-safe atomic MAT-file writer.

if ~(ischar(fileName) || (isstring(fileName) && isscalar(fileName)))
    error('Stage-2 checkpoint fileName must be text.');
end
fileName = char(fileName);
if isempty(strtrim(fileName))
    error('Stage-2 checkpoint fileName must be nonempty.');
end

if ~(ischar(variableName) || ...
     (isstring(variableName) && isscalar(variableName)))
    error('Stage-2 checkpoint variableName must be text.');
end
variableName = char(variableName);
if ~isvarname(variableName)
    error('Invalid Stage-2 checkpoint variable name: %s',variableName);
end

parentDir = fileparts(fileName);
if isempty(parentDir)
    parentDir = pwd;
end
if ~exist(parentDir,'dir')
    mkdir(parentDir);
end

temporaryFile = [tempname(parentDir) '.mat'];
cleanupObj = onCleanup(@() deleteIfExistsLocal(temporaryFile)); %#ok<NASGU>
S = struct();
S.(variableName) = value;
save(temporaryFile,'-struct','S',variableName,'-v7.3');
clear S value

fileInfo = dir(temporaryFile);
if isempty(fileInfo) || ~isscalar(fileInfo) || fileInfo.bytes <= 0
    error('Temporary Stage-2 checkpoint is missing or empty: %s',temporaryFile);
end

[ok,msg] = movefile(temporaryFile,fileName,'f');
if ~ok
    error('Could not finalize Stage-2 checkpoint: %s',msg);
end

finalInfo = dir(fileName);
if isempty(finalInfo) || ~isscalar(finalInfo) || finalInfo.bytes <= 0
    error('Final Stage-2 checkpoint is missing or empty: %s',fileName);
end
end


function value = getInputSourceFile(InputData)
value = '';
if ~isstruct(InputData)
    return
end
candidates = {'SourceFile','sourceFile','InputFile','inputFile'};
for k = 1:numel(candidates)
    f = candidates{k};
    if isfield(InputData,f)
        v = InputData.(f);
        if ischar(v)
            value = v;
            return
        elseif isstring(v) && isscalar(v)
            value = char(v);
            return
        end
    end
end
end


function [v,found] = evaluationVectorOrNaN(E,fieldName,n)
%EVALUATIONVECTORORNAN Read canonical vector without treating Inf as missing.
v = nan(n,1);
found = false;

if isstruct(E) && isfield(E,fieldName)
    x = E.(fieldName);
    if isnumeric(x) && isreal(x) && isvector(x) && numel(x)==n
        v = x(:);
        found = true;
        return
    end
end

if isstruct(E) && isfield(E,'series') && isstruct(E.series) && ...
   isfield(E.series,fieldName)
    x = E.series.(fieldName);
    if isnumeric(x) && isreal(x) && isvector(x) && numel(x)==n
        v = x(:);
        found = true;
        return
    end
end

if isstruct(E) && isfield(E,'Point') && isstruct(E.Point) && ...
   numel(E.Point)==n && isfield(E.Point,fieldName)
    v = structArrayNumericFieldOrNaN(E.Point,fieldName,n);
    found = true;
end
end


function [c,found] = evaluationCellOrEmpty(E,fieldName,n)
%EVALUATIONCELLOREMPTY Read canonical cell vector preserving explicit empties.
c = cell(n,1);
found = false;

if isstruct(E) && isfield(E,fieldName)
    x = E.(fieldName);
    if iscell(x) && numel(x)==n
        c = x(:);
        found = true;
        return
    elseif isstring(x) && numel(x)==n
        c = cellstr(x(:));
        found = true;
        return
    end
end

if isstruct(E) && isfield(E,'series') && isstruct(E.series) && ...
   isfield(E.series,fieldName)
    x = E.series.(fieldName);
    if iscell(x) && numel(x)==n
        c = x(:);
        found = true;
        return
    elseif isstring(x) && numel(x)==n
        c = cellstr(x(:));
        found = true;
        return
    end
end

if isstruct(E) && isfield(E,'Point') && isstruct(E.Point) && ...
   numel(E.Point)==n && isfield(E.Point,fieldName)
    c = structArrayCellFieldOrEmpty(E.Point,fieldName,n);
    found = true;
end
end


function [v,found] = evaluationSuccessVector(E,n)
%EVALUATIONSUCCESSVECTOR Preserve an explicitly stored all-false vector.
% Fallback to Final success is allowed only when EvaluationResult contains
% no valid point-level success information at all.

v = false(n,1);
found = false;

if ~isstruct(E)
    return
end

for fieldName = {'point_success','success_by_point'}
    fn = fieldName{1};
    if isfield(E,fn)
        x = E.(fn);
        if (islogical(x) || isnumeric(x)) && isvector(x) && numel(x)==n
            if islogical(x) || all(isfinite(x(:)) & (x(:)==0 | x(:)==1))
                v = logical(x(:));
                found = true;
                return
            end
        end
    end
end

if isfield(E,'series') && isstruct(E.series)
    for fieldName = {'point_success','success_by_point'}
        fn = fieldName{1};
        if isfield(E.series,fn)
            x = E.series.(fn);
            if (islogical(x) || isnumeric(x)) && isvector(x) && numel(x)==n
                if islogical(x) || all(isfinite(x(:)) & (x(:)==0 | x(:)==1))
                    v = logical(x(:));
                    found = true;
                    return
                end
            end
        end
    end
end

if isfield(E,'Point') && isstruct(E.Point) && numel(E.Point)==n
    for fieldName = {'success','point_success'}
        fn = fieldName{1};
        if isfield(E.Point,fn)
            v = structArrayLogicalFieldOrFalse(E.Point,fn,n);
            found = true;
            return
        end
    end
end
end


function v = structArrayNumericFieldOrNaN(S,fieldName,n)
v = nan(n,1);
if ~isstruct(S) || numel(S)~=n || ~isfield(S,fieldName)
    return
end
try
    x = reshape([S.(fieldName)],[],1);
    if isnumeric(x) && isreal(x) && numel(x)==n
        v = x;
    end
catch
end
end


function c = structArrayCellFieldOrEmpty(S,fieldName,n)
c = repmat({''},n,1);
if ~isstruct(S) || numel(S)~=n || ~isfield(S,fieldName)
    return
end
for k = 1:n
    x = S(k).(fieldName);
    if ischar(x)
        c{k} = x;
    elseif isstring(x) && isscalar(x)
        c{k} = char(x);
    end
end
end


function v = structArrayLogicalFieldOrFalse(S,fieldName,n)
v = false(n,1);
if ~isstruct(S) || numel(S)~=n || ~isfield(S,fieldName)
    return
end
for k = 1:n
    x = S(k).(fieldName);
    if islogical(x) && isscalar(x)
        v(k) = x;
    elseif isnumeric(x) && isscalar(x) && isfinite(x) && any(x==[0 1])
        v(k) = logical(x);
    end
end
end


function v = vectorFieldOrNaN(S,fieldName,n)
v = nan(n,1);
if ~isstruct(S) || ~isfield(S,fieldName)
    return
end
x = S.(fieldName);
if isnumeric(x) && isreal(x) && isvector(x) && numel(x)==n
    v = x(:);
end
end


function v = firstVectorFieldOrNaN(S,fieldNames,n)
v = nan(n,1);
for k = 1:numel(fieldNames)
    candidate = vectorFieldOrNaN(S,fieldNames{k},n);
    if any(isfinite(candidate)) || ...
       (isstruct(S) && isfield(S,fieldNames{k}) && numel(S.(fieldNames{k}))==n)
        v = candidate;
        return
    end
end
end


function c = cellFieldOrEmpty(S,fieldName,n)
c = cell(n,1);
if ~isstruct(S) || ~isfield(S,fieldName)
    return
end
x = S.(fieldName);
if iscell(x) && numel(x)==n
    c = x(:);
elseif isstring(x) && numel(x)==n
    c = cellstr(x(:));
end
end


function c = firstCellFieldOrEmpty(S1,S2,fieldNames,n)
c = cell(n,1);
for k = 1:numel(fieldNames)
    c1 = cellFieldOrEmpty(S1,fieldNames{k},n);
    if any(~cellfun(@isempty,c1))
        c = c1;
        return
    end
    c2 = cellFieldOrEmpty(S2,fieldNames{k},n);
    if any(~cellfun(@isempty,c2))
        c = c2;
        return
    end
end
end


function v = logicalFieldVectorOrFalse(S,fieldName,n)
v = false(n,1);
if ~isstruct(S) || ~isfield(S,fieldName)
    return
end
x = S.(fieldName);
if isvector(x) && numel(x)==n && (islogical(x) || isnumeric(x))
    v = logical(x(:));
end
end


function v = firstLogicalFieldOrFalse(S1,S2,fieldNames,n)
v = false(n,1);
for k = 1:numel(fieldNames)
    if isstruct(S1) && isfield(S1,fieldNames{k})
        x = S1.(fieldNames{k});
        if isvector(x) && numel(x)==n && (islogical(x) || isnumeric(x))
            v = logical(x(:));
            return
        end
    end
    if isstruct(S2) && isfield(S2,fieldNames{k})
        x = S2.(fieldNames{k});
        if isvector(x) && numel(x)==n && (islogical(x) || isnumeric(x))
            v = logical(x(:));
            return
        end
    end
end
end


function value = finiteMean(x)
value = NaN;
if isempty(x)
    return
end
x = x(:);
x = x(isfinite(x));
if ~isempty(x)
    value = mean(x);
end
end


function value = finiteMax(x)
value = NaN;
if isempty(x)
    return
end
x = x(:);
x = x(isfinite(x));
if ~isempty(x)
    value = max(x);
end
end


function S = removeFieldsIfPresentLocal(S,names)
if ~isstruct(S)
    return
end
present = names(isfield(S,names));
if ~isempty(present)
    S = rmfield(S,present);
end
end


function A = stripStructArrayFields(A,names)
if ~isstruct(A) || isempty(A)
    return
end
present = names(isfield(A,names));
if ~isempty(present)
    A = rmfield(A,present);
end
end


function invalidateStage2CommitMarker(fileName)
%INVALIDATESTAGE2COMMITMARKER Remove prior CORE before split-file update.
%
% CORE is the Stage-2 commit marker.  Failure to invalidate an existing
% CORE must stop publication before any auxiliary part is overwritten;
% otherwise a later I/O failure could leave a mixed old/new record that
% appears complete.

if ~isfile(fileName)
    return
end

delete(fileName);

if isfile(fileName)
    error('Could not invalidate previous Stage-2 CORE checkpoint: %s',fileName);
end
end


function deleteIfExistsLocal(fileName)
%DELETEIFEXISTSLOCAL Best-effort cleanup after interrupted checkpoint save.

if isfile(fileName)
    try
        delete(fileName);
    catch
        % Best effort only.
    end
end

end


% =============================================================
% Local helper
% =============================================================

function s = cellTextToStringSafe(c)
%CELLTEXTTOSTRINGSAFE Convert cell/string/char text safely.
%
% Empty or invalid entries become "" rather than causing the batch
% summary construction to fail.

if isstring(c)

    s = ...
        c(:);

    return

end


if ischar(c)

    s = ...
        string(c);

    return

end


if ~iscell(c)

    s = ...
        strings(0,1);

    return

end


s = ...
    strings(numel(c),1);


for k = 1:numel(c)

    x = ...
        c{k};


    if ischar(x)

        s(k) = ...
            string(x);

    elseif isstring(x) && ...
           isscalar(x)

        s(k) = ...
            x;

    else

        % [], invalid values, etc.
        s(k) = ...
            "";

    end

end

end

% =============================================================
% Local helper
% =============================================================

function checkpointBaseFile = resolveExistingStage1CheckpointFile( ...
    checkpointDir,modelName,kernel,tuningTag)
%RESOLVEEXISTINGSTAGE1CHECKPOINTFILE
%
% Find an existing Stage-1 checkpoint using the stable identity:
%
%   Model + Kernel + tuning method/grid
%
% Mxxx/Kxxx prefixes are deliberately ignored because they depend on
% model/kernel positions in a particular batch run.

if ~isfolder(checkpointDir)

    error( ...
        'Stage-1 checkpoint directory does not exist: %s', ...
        checkpointDir);

end


Fcore = ...
    dir( ...
        fullfile( ...
            checkpointDir, ...
            '*_CORE.mat'));


if isempty(Fcore)

    error( ...
        'No Stage-1 CORE checkpoints were found in: %s', ...
        checkpointDir);

end


expectedSuffix = ...
    "_" + ...
    string(modelName) + ...
    "_" + ...
    string(kernel) + ...
    "_" + ...
    string(tuningTag) + ...
    "_CORE.mat";


fileNames = ...
    string({Fcore.name});


isMatch = ...
    endsWith( ...
        fileNames, ...
        expectedSuffix, ...
        'IgnoreCase',true);


idx = ...
    find(isMatch);


if isempty(idx)

    error( ...
        ['No Stage-1 checkpoint was found for:\n' ...
         'Model  = %s\n' ...
         'Kernel = %s\n' ...
         'Tuning = %s\n' ...
         'Directory = %s'], ...
        modelName, ...
        kernel, ...
        tuningTag, ...
        checkpointDir);

end


if numel(idx) > 1

    error( ...
        ['Multiple Stage-1 checkpoints were found for:\n' ...
         'Model  = %s\n' ...
         'Kernel = %s\n' ...
         'Tuning = %s'], ...
        modelName, ...
        kernel, ...
        tuningTag);

end


coreName = ...
    string(Fcore(idx).name);


baseStem = ...
    extractBefore( ...
        coreName, ...
        "_CORE.mat");


checkpointBaseFile = ...
    fullfile( ...
        checkpointDir, ...
        char(baseStem + ".mat"));


% Verify complete CORE + DIAG pair.
Paths = ...
    localrbf.stage1CheckpointPaths( ...
        checkpointBaseFile);


if ~isfile(Paths.core)

    error( ...
        'Stage-1 CORE checkpoint was not found: %s', ...
        Paths.core);

end


if ~isfile(Paths.diag)

    error( ...
        'Stage-1 DIAG checkpoint was not found: %s', ...
        Paths.diag);

end

end
function Results = globalRBFBootstrapBatchSummaryParallel( ...
    InputData, FitResults, ...
    nBootstrap, verbose, RNG_SEED, ...
    bootstrapOptions, OutputFile)

%GLOBALRBFBOOTSTRAPBATCHSUMMARYPARALLEL
%
% Parallel batch bootstrap-variance calculation for GLOBAL RBF.
%
% One function call processes ONE case.
%
% Case name, GGM list, kernel list, selected C_global, selected
% RegParam, S_global, k_used, solver settings, and the ORIGINAL
% fixed CV partition are obtained from FitResults.
%
% Parallel architecture is identical to:
%
%       globalRBFBatchSummaryParallel_TuneReg
%
% i.e.
%
%       one parfor iteration = one GGM x Kernel combination.
%
% For every GGM x Kernel:
%
%   TRAINING uncertainty:
%       globalrbf.globalRBFBootstrapForTraining
%
%       - reuses EXACTLY the original cvp_fixed;
%       - creates NO new cvpartition;
%       - with NumSpatialSegments<=1, uses the exact current ordinary
%         bootstrap on each original CV fitting set;
%       - with NumSpatialSegments>1, partitions each CV fitting set once
%         into adaptive radial annuli about its spatial centroid and applies
%         the same resampling-with-replacement operation within each annulus;
%       - preserves the fitting-set sample size;
%       - keeps selected C_global and RegParam fixed;
%       - refits bootstrap RBF coefficients;
%       - internally predicts the SAME CV validation locations;
%       - applies the sequential >80% dominance safeguard, capped at 3% of B;
%       - calculates Eq. (18) bootstrap variance from retained statistics;
%       - applies Eq. (A.1) finite-sample adjustment.
%
%   TEST uncertainty:
%       globalrbf.globalRBFBootstrapForTesting
%
%       - with NumSpatialSegments<=1, uses the exact current ordinary
%         bootstrap on the complete n-observation fitting set;
%       - with NumSpatialSegments>1, partitions the complete fitting set
%         once into adaptive radial annuli about its spatial centroid and
%         applies the same resampling-with-replacement operation in each;
%       - preserves bootstrap sample size n;
%       - keeps selected C_global and RegParam fixed;
%       - internally predicts the SAME test locations;
%       - applies the sequential >80% dominance safeguard, capped at 3% of B;
%       - calculates Eq. (18) from retained statistics;
%       - applies Eq. (A.1).
%
% Bootstrap replicates remain SERIAL inside each worker.
% Only independent GGM x Kernel combinations are parallelized.
%
% A checkpoint is saved immediately after one complete
% GGM x Kernel calculation finishes.
%
% Required existing functions:
%
%   common.bootstrapRBFReplicate
%   globalrbf.buildAdaptiveRadialSegments
%   globalrbf.resampleRadialSegments
%   globalrbf.globalRBFBootstrapForTraining
%   globalrbf.globalRBFBootstrapForTesting


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 3 || isempty(nBootstrap)
    nBootstrap = 200;
end


if nargin < 4 || isempty(verbose)
    verbose = true;
end


if nargin < 5
    RNG_SEED = 1;
end


if nargin < 6 || isempty(bootstrapOptions)
    bootstrapOptions = struct();
end


if nargin < 7
    OutputFile = '';
end


% -------------------------------------------------------------
% Inner bootstrap printing is OFF by default.
%
% The outer batch still reports one summary per GGM x Kernel
% when verbose=true. This substantially reduces parallel I/O.
% -------------------------------------------------------------

if ~isfield(bootstrapOptions,'Verbose') || ...
        isempty(bootstrapOptions.Verbose)

    bootstrapOptions.Verbose = false;

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


% =============================================================
% 2. Validate bootstrap settings
% =============================================================

if ~isnumeric(nBootstrap) || ...
   ~isscalar(nBootstrap) || ...
   ~isfinite(nBootstrap) || ...
   nBootstrap < 2 || ...
   nBootstrap ~= floor(nBootstrap)

    error('nBootstrap must be an integer >= 2.');

end


if isempty(RNG_SEED)

    error(['RNG_SEED must be nonempty for reproducible ' ...
           'parallel bootstrap execution.']);

end


if ~isnumeric(RNG_SEED) || ...
   ~isscalar(RNG_SEED) || ...
   ~isfinite(RNG_SEED) || ...
   RNG_SEED < 0 || ...
   RNG_SEED > (2^32-1) || ...
   RNG_SEED ~= floor(RNG_SEED)

    error(['RNG_SEED must be a finite nonnegative integer ' ...
           'not exceeding 2^32-1.']);

end


if ~isscalar(verbose)

    error('verbose must be scalar.');

end


verbose = logical(verbose);


if ~isscalar(bootstrapOptions.Verbose)

    error('bootstrapOptions.Verbose must be scalar.');

end


bootstrapOptions.Verbose = ...
    logical(bootstrapOptions.Verbose);


if ~isscalar(bootstrapOptions.StoreIndices)

    error('bootstrapOptions.StoreIndices must be scalar.');

end


bootstrapOptions.StoreIndices = ...
    logical(bootstrapOptions.StoreIndices);


if ~isnumeric(bootstrapOptions.MaxAttemptsFactor) || ...
   ~isscalar(bootstrapOptions.MaxAttemptsFactor) || ...
   ~isfinite(bootstrapOptions.MaxAttemptsFactor) || ...
   bootstrapOptions.MaxAttemptsFactor < 1

    error(['bootstrapOptions.MaxAttemptsFactor must be ' ...
           'a finite scalar >= 1.']);

end


if ~isnumeric(bootstrapOptions.NumSpatialSegments) || ...
   ~isscalar(bootstrapOptions.NumSpatialSegments) || ...
   ~isfinite(bootstrapOptions.NumSpatialSegments) || ...
   bootstrapOptions.NumSpatialSegments < 0 || ...
   bootstrapOptions.NumSpatialSegments ~= ...
        round(bootstrapOptions.NumSpatialSegments)

    error(['bootstrapOptions.NumSpatialSegments must be a ' ...
           'nonnegative integer. Values 0 and 1 use the exact ' ...
           'current ordinary bootstrap; values >1 enable annular resampling.']);

end


if ~isnumeric(bootstrapOptions.MinPointsPerAnnulus) || ...
   ~isscalar(bootstrapOptions.MinPointsPerAnnulus) || ...
   ~isfinite(bootstrapOptions.MinPointsPerAnnulus) || ...
   bootstrapOptions.MinPointsPerAnnulus < 1 || ...
   bootstrapOptions.MinPointsPerAnnulus ~= ...
        round(bootstrapOptions.MinPointsPerAnnulus)

    error(['bootstrapOptions.MinPointsPerAnnulus must be a ' ...
           'positive integer.']);

end


if ~isnumeric(bootstrapOptions.DominanceLimit) || ...
   ~isscalar(bootstrapOptions.DominanceLimit) || ...
   ~isfinite(bootstrapOptions.DominanceLimit) || ...
   bootstrapOptions.DominanceLimit <= 0 || ...
   bootstrapOptions.DominanceLimit >= 1

    error(['bootstrapOptions.DominanceLimit must be a finite ' ...
           'scalar strictly between 0 and 1.']);

end


if ~isnumeric(bootstrapOptions.MaxDominantFraction) || ...
   ~isscalar(bootstrapOptions.MaxDominantFraction) || ...
   ~isfinite(bootstrapOptions.MaxDominantFraction) || ...
   bootstrapOptions.MaxDominantFraction < 0 || ...
   bootstrapOptions.MaxDominantFraction >= 1

    error(['bootstrapOptions.MaxDominantFraction must be a finite ' ...
           'scalar in [0,1).']);

end


% -------------------------------------------------------------
% Use an independent deterministic RNG stream for TEST bootstrap.
% -------------------------------------------------------------

TestRNG_SEED = ...
    mod(double(RNG_SEED) + 1000003,2^32);


% =============================================================
% 3. Validate FitResults identity
%
% Case/model/kernel definitions come ONLY from FitResults.
% This prevents accidental model/kernel ordering mismatch.
% =============================================================

requiredFitFields = { ...
    'CaseName', ...
    'ListOfModel', ...
    'NameFunction_RBF', ...
    'reg_param', ...
    'C_global', ...
    'CVInfo', ...
    'Run'};


for k = 1:numel(requiredFitFields)

    if ~isfield(FitResults,requiredFitFields{k})

        error( ...
            'FitResults.%s is required.', ...
            requiredFitFields{k});

    end

end


CaseName = ...
    char(FitResults.CaseName);


ListOfModel = ...
    FitResults.ListOfModel;


NameFunction_RBF = ...
    FitResults.NameFunction_RBF;


nModel = ...
    numel(ListOfModel);


nKernel = ...
    numel(NameFunction_RBF);


if isempty(CaseName)

    error('FitResults.CaseName must be nonempty.');

end


if nModel < 1

    error('FitResults.ListOfModel is empty.');

end


if nKernel < 1

    error('FitResults.NameFunction_RBF is empty.');

end


if ~isequal( ...
        size(FitResults.reg_param), ...
        [nModel,nKernel])

    error(['FitResults.reg_param dimensions are inconsistent ' ...
           'with model/kernel lists.']);

end


if ~isequal( ...
        size(FitResults.C_global), ...
        [nModel,nKernel])

    error(['FitResults.C_global dimensions are inconsistent ' ...
           'with model/kernel lists.']);

end


% =============================================================
% 4. Input coordinates
% =============================================================

if ~isfield(InputData,'Train') || ...
   ~isfield(InputData.Train,'XYZ') || ...
   ~isfield(InputData.Train,'DeltaZeta')

    error('InputData.Train.XYZ/DeltaZeta are required.');

end


if ~isfield(InputData,'Test') || ...
   ~isfield(InputData.Test,'XYZ')

    error('InputData.Test.XYZ is required.');

end


X = ...
    InputData.Train.XYZ;


X_test = ...
    InputData.Test.XYZ;


if ~isnumeric(X) || ...
   isempty(X) || ...
   ~ismatrix(X) || ...
   ~isreal(X) || ...
   any(~isfinite(X(:)))

    error(['Training coordinates must be a nonempty finite ' ...
           'real numeric matrix.']);

end


if ~isnumeric(X_test) || ...
   isempty(X_test) || ...
   ~ismatrix(X_test) || ...
   ~isreal(X_test) || ...
   any(~isfinite(X_test(:)))

    error(['Testing coordinates must be a nonempty finite ' ...
           'real numeric matrix.']);

end


if size(X,2) ~= size(X_test,2)

    error(['Training and testing coordinates must have ' ...
           'the same dimensions.']);

end


nTrain = ...
    size(X,1);


nTest = ...
    size(X_test,1);


% =============================================================
% 5. ORIGINAL fixed CV partition
%
% Absolutely NO new cvpartition is generated here.
% =============================================================

if ~isfield(FitResults.CVInfo,'cvp_fixed') || ...
        isempty(FitResults.CVInfo.cvp_fixed)

    error([ ...
        'FitResults.CVInfo.cvp_fixed is required. ' ...
        'Bootstrap training must reuse the original fixed ' ...
        'CV partition.']);

end


cvp_fixed = ...
    FitResults.CVInfo.cvp_fixed;


kFixed = ...
    cvp_fixed.NumTestSets;


if kFixed < 2

    error('The stored fixed CV partition is invalid.');

end


% Confirm dataset size.
idxCheck = ...
    test(cvp_fixed,1);


if numel(idxCheck) ~= nTrain

    error(['FitResults.CVInfo.cvp_fixed does not correspond ' ...
           'to the supplied training dataset.']);

end


% -------------------------------------------------------------
% Check k_used against the stored partition BEFORE workers start.
% -------------------------------------------------------------

if isfield(FitResults,'k_used')

    kValues = ...
        FitResults.k_used(isfinite(FitResults.k_used));


    if ~isempty(kValues) && ...
       any(kValues ~= kFixed)

        error(['FitResults.k_used is inconsistent with the ' ...
               'stored cvp_fixed partition.']);

    end

end


if isfield(FitResults.CVInfo,'k_used') && ...
        ~isempty(FitResults.CVInfo.k_used)

    if FitResults.CVInfo.k_used ~= kFixed

        error(['FitResults.CVInfo.k_used is inconsistent with ' ...
               'FitResults.CVInfo.cvp_fixed.']);

    end

end


% -------------------------------------------------------------
% Numeric fold IDs.
% -------------------------------------------------------------

foldID = ...
    zeros(nTrain,1);


validationCount = ...
    zeros(nTrain,1);


for f = 1:kFixed

    idxVal = ...
        test(cvp_fixed,f);


    foldID(idxVal) = ...
        f;


    validationCount(idxVal) = ...
        validationCount(idxVal) + 1;

end


if any(validationCount ~= 1)

    error(['Each training observation must occur exactly once ' ...
           'in the fixed CV validation sets.']);

end


% If FitResults already contains foldID, verify it.
if isfield(FitResults.CVInfo,'foldID') && ...
        ~isempty(FitResults.CVInfo.foldID)

    storedFoldID = ...
        FitResults.CVInfo.foldID(:);


    if ~isequal(storedFoldID,foldID)

        error(['FitResults.CVInfo.foldID is inconsistent with ' ...
               'FitResults.CVInfo.cvp_fixed.']);

    end

end


% =============================================================
% 6. Validate GGM data
% =============================================================

for i = 1:nModel

    modelName = ...
        char(ListOfModel{i});


    if ~isfield( ...
            InputData.Train.DeltaZeta, ...
            modelName)

        error( ...
            'Training data for model "%s" were not found.', ...
            modelName);

    end


    y = ...
        InputData.Train.DeltaZeta.(modelName);


    if numel(y) ~= nTrain

        error( ...
            ['Training coordinate/data size mismatch for ' ...
             'model "%s".'], ...
            modelName);

    end


    if any(~isfinite(y(:)))

        error( ...
            'Training data for model "%s" contain NaN or Inf.', ...
            modelName);

    end

end


% =============================================================
% 7. Validate kernels
% =============================================================

for j = 1:nKernel

    kernel = ...
        char(NameFunction_RBF{j});


    % This also validates whether the kernel is supported.
    common.kernelHasShapeParameter(kernel);

end


% =============================================================
% 8. Prepare complete trainingInfo for every GGM x Kernel
%
% Do this BEFORE parfor:
%
%   - catches structural problems immediately;
%   - avoids repeated client-side reconstruction on workers;
%   - restores only information intentionally removed from the
%     stored per-run trainingInfo.
% =============================================================

nRun = ...
    nModel*nKernel;


TrainingInfoCell = ...
    cell(nRun,1);


RunNameCell = ...
    cell(nRun,1);


OriginalTestOK = ...
    false(nRun,1);


for irun = 1:nRun

    i = ...
        floor((irun-1)/nKernel) + 1;


    j = ...
        mod(irun-1,nKernel) + 1;


    modelName = ...
        char(ListOfModel{i});


    kernel = ...
        char(NameFunction_RBF{j});


    useShape = ...
        common.kernelHasShapeParameter(kernel);


    ModelAndFunction = ...
        [modelName '_' kernel];


    RunNameCell{irun} = ...
        ModelAndFunction;


    % ---------------------------------------------------------
    % Original training status.
    % ---------------------------------------------------------

    originalTrainOK = ...
        true;


    if isfield(FitResults,'trainSuccess')

        originalTrainOK = ...
            FitResults.trainSuccess(i,j);

    end


    if ~originalTrainOK

        TrainingInfoCell{irun} = ...
            [];


        OriginalTestOK(irun) = ...
            false;


        continue

    end


    % ---------------------------------------------------------
    % Stored detailed fitting result.
    % ---------------------------------------------------------

    if ~isfield( ...
            FitResults.Run, ...
            ModelAndFunction)

        error( ...
            'FitResults.Run.%s was not found.', ...
            ModelAndFunction);

    end


    SourceRun = ...
        FitResults.Run.(ModelAndFunction);


    if ~isfield(SourceRun,'Train') || ...
       ~isfield(SourceRun.Train,'trainingInfo')

        error( ...
            ['FitResults.Run.%s.Train.trainingInfo ' ...
             'was not found.'], ...
            ModelAndFunction);

    end


    trainingInfo = ...
        SourceRun.Train.trainingInfo;


    % ---------------------------------------------------------
    % Do NOT overwrite an explicit failure flag.
    % ---------------------------------------------------------

    if isfield(trainingInfo,'success') && ...
            ~trainingInfo.success

        error( ...
            ['Stored trainingInfo indicates failure for ' ...
             '"%s".'], ...
            ModelAndFunction);

    end


    % ---------------------------------------------------------
    % Restore EXACT original fixed partition.
    %
    % globalRBFBatchSummaryParallel_TuneReg intentionally removed
    % cvp_fixed from per-run trainingInfo storage.
    % ---------------------------------------------------------

    trainingInfo.cvp_fixed = ...
        cvp_fixed;


    % ---------------------------------------------------------
    % Verify kernel identity before assigning compact results.
    % ---------------------------------------------------------

    if isfield(trainingInfo,'kernel') && ...
            ~isempty(trainingInfo.kernel)

        if ~strcmpi( ...
                char(trainingInfo.kernel), ...
                kernel)

            error( ...
                ['Stored trainingInfo.kernel is inconsistent ' ...
                 'for "%s".'], ...
                ModelAndFunction);

        end

    end


    trainingInfo.kernel = ...
        kernel;


    trainingInfo.useShape = ...
        useShape;


    % ---------------------------------------------------------
    % Selected RegParam.
    % ---------------------------------------------------------

    reg_param = ...
        FitResults.reg_param(i,j);


    if ~isscalar(reg_param) || ...
       ~isfinite(reg_param) || ...
       reg_param <= 0

        error( ...
            'Invalid selected RegParam for "%s".', ...
            ModelAndFunction);

    end


    % Check stored value when available.
    if isfield(trainingInfo,'reg_param') && ...
       ~isempty(trainingInfo.reg_param) && ...
       isfinite(trainingInfo.reg_param)

        if trainingInfo.reg_param ~= reg_param

            error( ...
                ['Stored RegParam is inconsistent with compact ' ...
                 'FitResults.reg_param for "%s".'], ...
                ModelAndFunction);

        end

    end


    trainingInfo.reg_param = ...
        reg_param;


    % ---------------------------------------------------------
    % Selected C_global.
    % ---------------------------------------------------------

    if useShape

        C_global = ...
            FitResults.C_global(i,j);


        if ~isscalar(C_global) || ...
           ~isfinite(C_global) || ...
           C_global <= 0

            error( ...
                'Invalid selected C_global for "%s".', ...
                ModelAndFunction);

        end


        if isfield(trainingInfo,'C_global') && ...
           ~isempty(trainingInfo.C_global) && ...
           isfinite(trainingInfo.C_global)

            if trainingInfo.C_global ~= C_global

                error( ...
                    ['Stored C_global is inconsistent with compact ' ...
                     'FitResults.C_global for "%s".'], ...
                    ModelAndFunction);

            end

        end


        trainingInfo.C_global = ...
            C_global;

    else

        % Shape-free kernels use epsilon=1 internally.
        % C is only a dummy implementation value.
        trainingInfo.C_global = ...
            1;

    end


    % ---------------------------------------------------------
    % k_used.
    % ---------------------------------------------------------

    if isfield(FitResults,'k_used') && ...
            isfinite(FitResults.k_used(i,j))

        k_used = ...
            FitResults.k_used(i,j);

    elseif isfield(trainingInfo,'k_used') && ...
            ~isempty(trainingInfo.k_used)

        k_used = ...
            trainingInfo.k_used;

    else

        k_used = ...
            kFixed;

    end


    if k_used ~= kFixed

        error( ...
            ['k_used for "%s" differs from the stored fixed ' ...
             'CV partition.'], ...
            ModelAndFunction);

    end


    trainingInfo.k_used = ...
        k_used;


    % ---------------------------------------------------------
    % S_global.
    %
    % CRITICAL:
    % Reuse the ORIGINAL training scale.
    % Never calculate a new S_global in bootstrap.
    % ---------------------------------------------------------

    if ~isfield(trainingInfo,'S_global') || ...
       ~isscalar(trainingInfo.S_global) || ...
       ~isfinite(trainingInfo.S_global) || ...
       trainingInfo.S_global <= 0

        error( ...
            'Valid S_global was not found for "%s".', ...
            ModelAndFunction);

    end


    % ---------------------------------------------------------
    % Solver options.
    % ---------------------------------------------------------

    if ~isfield(trainingInfo,'options') || ...
            isempty(trainingInfo.options)

        if isfield(FitResults,'Settings') && ...
           isfield(FitResults.Settings,'options')

            trainingInfo.options = ...
                FitResults.Settings.options;

        else

            error( ...
                'Solver options were not found for "%s".', ...
                ModelAndFunction);

        end

    end


    if ~isfield(trainingInfo.options,'solver') || ...
       ~isstruct(trainingInfo.options.solver)

        trainingInfo.options.solver = ...
            struct();

    end


    if ~isfield( ...
            trainingInfo.options.solver, ...
            'cond_lsq') || ...
       isempty(trainingInfo.options.solver.cond_lsq)

        trainingInfo.options.solver.cond_lsq = ...
            1e12;

    end


    if ~isnumeric(trainingInfo.options.solver.cond_lsq) || ...
       ~isscalar(trainingInfo.options.solver.cond_lsq) || ...
       ~isreal(trainingInfo.options.solver.cond_lsq) || ...
       ~isfinite(trainingInfo.options.solver.cond_lsq) || ...
       trainingInfo.options.solver.cond_lsq < 1

        error( ...
            'Invalid options.solver.cond_lsq for "%s".', ...
            ModelAndFunction);

    end


    TrainingInfoCell{irun} = ...
        trainingInfo;


    % ---------------------------------------------------------
    % Original final-test status.
    % ---------------------------------------------------------

    if isfield(FitResults,'testSuccess')

        OriginalTestOK(irun) = ...
            FitResults.testSuccess(i,j);

    else

        OriginalTestOK(irun) = ...
            true;

    end

end


% =============================================================
% 9. Initialize Results
% =============================================================

Results = ...
    struct();


Results.Run = ...
    struct();


Results.CaseName = ...
    CaseName;


Results.ListOfModel = ...
    ListOfModel;


Results.NameFunction_RBF = ...
    NameFunction_RBF;


% -------------------------------------------------------------
% Settings
% -------------------------------------------------------------

Results.Settings.nBootstrap = ...
    nBootstrap;


Results.Settings.RNG_SEED_Train = ...
    RNG_SEED;


Results.Settings.RNG_SEED_Test = ...
    TestRNG_SEED;


Results.Settings.verbose = ...
    verbose;


Results.Settings.DominanceLimit = ...
    bootstrapOptions.DominanceLimit;


Results.Settings.MaxDominantFraction = ...
    bootstrapOptions.MaxDominantFraction;


Results.Settings.DominanceControlMethod = ...
    'sequential_largest_variance_contributor';


Results.Settings.bootstrapOptions = ...
    bootstrapOptions;


% -------------------------------------------------------------
% Original fitted model information.
% -------------------------------------------------------------

Results.Source.reg_param = ...
    FitResults.reg_param;


Results.Source.C_global = ...
    FitResults.C_global;


if isfield(FitResults,'epsilon_global')

    Results.Source.epsilon_global = ...
        FitResults.epsilon_global;

end


if isfield(FitResults,'cv_rmse')

    Results.Source.cv_rmse = ...
        FitResults.cv_rmse;

end


if isfield(FitResults,'test_rmse')

    Results.Source.test_rmse = ...
        FitResults.test_rmse;

end


if isfield(FitResults,'k_used')

    Results.Source.k_used = ...
        FitResults.k_used;

end


% -------------------------------------------------------------
% Problem size.
% -------------------------------------------------------------

Results.DataInfo.nTrain = ...
    nTrain;


Results.DataInfo.nTest = ...
    nTest;


Results.DataInfo.dimension = ...
    size(X,2);


Results.DataInfo.nModels = ...
    nModel;


Results.DataInfo.nKernels = ...
    nKernel;


% -------------------------------------------------------------
% Matrix interpretation.
% -------------------------------------------------------------

Results.MatrixInfo.rows = ...
    ListOfModel;


Results.MatrixInfo.columns = ...
    NameFunction_RBF;


Results.MatrixInfo.description = ...
    'Rows = input models; columns = RBF kernels';


% -------------------------------------------------------------
% Store fixed CV partition ONCE in final Results.
% -------------------------------------------------------------

Results.CVInfo.cvp_fixed = ...
    cvp_fixed;


Results.CVInfo.foldID = ...
    foldID;


if isfield(FitResults.CVInfo,'k_requested')

    Results.CVInfo.k_requested = ...
        FitResults.CVInfo.k_requested;

end


Results.CVInfo.k_used = ...
    kFixed;


% =============================================================
% 10. Compact variance summaries
%
% Eq. (25) = BootstrapVariance.
% Eq. (26) = AdjustedVariance = final Variance.
% =============================================================

Results.trainMeanBootstrapVariance = ...
    nan(nModel,nKernel);


Results.trainMedianBootstrapVariance = ...
    nan(nModel,nKernel);


Results.trainMeanVariance = ...
    nan(nModel,nKernel);


Results.trainMedianVariance = ...
    nan(nModel,nKernel);


Results.testMeanBootstrapVariance = ...
    nan(nModel,nKernel);


Results.testMedianBootstrapVariance = ...
    nan(nModel,nKernel);


Results.testMeanVariance = ...
    nan(nModel,nKernel);


Results.testMedianVariance = ...
    nan(nModel,nKernel);


% -------------------------------------------------------------
% Dominance-safeguard summaries.
% -------------------------------------------------------------

Results.trainDominantPointsCount = ...
    nan(nModel,nKernel);


Results.trainDominantCount = ...
    nan(nModel,nKernel);


Results.trainDominantPoints = ...
    cell(nModel,nKernel);


Results.testDominantPointsCount = ...
    nan(nModel,nKernel);


Results.testDominantCount = ...
    nan(nModel,nKernel);


Results.testDominantPoints = ...
    cell(nModel,nKernel);


% =============================================================
% 11. Timing
%
% Same matrix convention as TuneReg batch.
% =============================================================

Results.trainWallTime = ...
    nan(nModel,nKernel);


Results.trainCPUTime = ...
    nan(nModel,nKernel);


Results.testWallTime = ...
    nan(nModel,nKernel);


Results.testCPUTime = ...
    nan(nModel,nKernel);


Results.totalWallTime = ...
    nan(nModel,nKernel);


Results.totalCPUTime = ...
    nan(nModel,nKernel);


% =============================================================
% 12. RBF solve counts
% =============================================================

Results.trainRBFSolves = ...
    nan(nModel,nKernel);


Results.testRBFSolves = ...
    nan(nModel,nKernel);


Results.totalRBFSolves = ...
    nan(nModel,nKernel);


% =============================================================
% 13. Bootstrap rejection counts
% =============================================================

Results.trainRejected = ...
    nan(nModel,nKernel);


Results.testRejected = ...
    nan(nModel,nKernel);


Results.totalRejected = ...
    nan(nModel,nKernel);


% =============================================================
% 14. Run status
% =============================================================

Results.trainSuccess = ...
    false(nModel,nKernel);


Results.testSuccess = ...
    false(nModel,nKernel);


Results.trainMessage = ...
    cell(nModel,nKernel);


Results.testMessage = ...
    cell(nModel,nKernel);


% =============================================================
% 15. Output/checkpoint directories
% =============================================================

checkpointDir = '';


if ~isempty(OutputFile)

    outputDir = ...
        fileparts(OutputFile);


    if ~isempty(outputDir) && ...
            ~exist(outputDir,'dir')

        mkdir(outputDir);

    end


    checkpointDir = ...
        fullfile( ...
            outputDir, ...
            [CaseName '_bootstrap_checkpoints']);


    if ~exist(checkpointDir,'dir')

        mkdir(checkpointDir);

    end

end


% =============================================================
% 16. Parallel pool
% =============================================================

pool = ...
    gcp('nocreate');


if isempty(pool)

    pool = ...
        parpool;

end


Results.Settings.Parallel = ...
    true;


Results.Settings.NumWorkers = ...
    pool.NumWorkers;


% =============================================================
% 17. Prepare parallel Model x Kernel execution
% =============================================================

RunCell = ...
    cell(nRun,1);


tParallel = ...
    tic;


% =============================================================
% 18. Parallel Model x Kernel loop
% =============================================================

parfor irun = 1:nRun

    i = ...
        floor((irun-1)/nKernel) + 1;


    j = ...
        mod(irun-1,nKernel) + 1;


    modelName = ...
        char(ListOfModel{i});


    kernel = ...
        char(NameFunction_RBF{j});


    ModelAndFunction = ...
        RunNameCell{irun};


    trainingInfo = ...
        TrainingInfoCell{irun};


    y = ...
        InputData.Train.DeltaZeta.(modelName);


    y = ...
        y(:);


    % ---------------------------------------------------------
    % Independent run result.
    % ---------------------------------------------------------

    Rrun = ...
        struct();


    Rrun.i = ...
        i;


    Rrun.j = ...
        j;


    Rrun.CaseName = ...
        CaseName;


    Rrun.model = ...
        modelName;


    Rrun.kernel = ...
        kernel;


    Rrun.ModelAndFunction = ...
        ModelAndFunction;


    % ---------------------------------------------------------
    % Each checkpoint remains self-contained.
    %
    % This duplicated cvp_fixed is kept only in the checkpoint
    % Rrun, not in final Results.Run.
    % ---------------------------------------------------------

    Rrun.cvp_fixed = ...
        cvp_fixed;


    Rrun.foldID = ...
        foldID;


    % ---------------------------------------------------------
    % Timing.
    % ---------------------------------------------------------

    Rrun.trainWallTime = NaN;
    Rrun.trainCPUTime = NaN;

    Rrun.testWallTime = NaN;
    Rrun.testCPUTime = NaN;

    Rrun.totalWallTime = NaN;
    Rrun.totalCPUTime = NaN;


    % ---------------------------------------------------------
    % Computational counts.
    % ---------------------------------------------------------

    Rrun.trainRBFSolves = NaN;
    Rrun.testRBFSolves = NaN;
    Rrun.totalRBFSolves = NaN;

    Rrun.trainRejected = NaN;
    Rrun.testRejected = NaN;
    Rrun.totalRejected = NaN;


    % ---------------------------------------------------------
    % Status.
    % ---------------------------------------------------------

    Rrun.trainSuccess = false;
    Rrun.testSuccess = false;

    Rrun.trainMessage = '';
    Rrun.testMessage = '';


    % ---------------------------------------------------------
    % Detailed output.
    % ---------------------------------------------------------

    RunInfo = ...
        struct();


    RunInfo.model = ...
        modelName;


    RunInfo.kernel = ...
        kernel;


    RunInfo.Train = ...
        [];


    RunInfo.Test = ...
        [];


    % ---------------------------------------------------------
    % Skip if original fitting failed.
    % ---------------------------------------------------------

    if isempty(trainingInfo)

        Rrun.trainMessage = ...
            'Not run because original RBF training failed.';


        Rrun.testMessage = ...
            'Not run because original RBF training failed.';


        Rrun.RunInfo = ...
            RunInfo;


        RunCell{irun} = ...
            Rrun;


        % Save status checkpoint as well.
        if ~isempty(checkpointDir)

            checkpointFile = ...
                fullfile( ...
                    checkpointDir, ...
                    sprintf( ...
                        '%s_%s.mat', ...
                        modelName, ...
                        kernel));


            checkpointData = ...
                struct('Rrun',Rrun);


            try

                save( ...
                    checkpointFile, ...
                    '-fromstruct', ...
                    checkpointData, ...
                    '-v7.3');

            catch MEcheckpoint

                warning( ...
                    'Checkpoint failed for %s: %s', ...
                    ModelAndFunction, ...
                    MEcheckpoint.message);

            end

        end


        continue

    end


    if verbose

        fprintf('\n====================================================\n');

        fprintf( ...
            'Bootstrap | Case: %s | Model: %s | Kernel: %s\n', ...
            CaseName, ...
            modelName, ...
            kernel);

        fprintf('====================================================\n');

    end


    % =========================================================
    % 19. TRAINING bootstrap variance
    % =========================================================

    tTrain = ...
        tic;


    cpuTrain0 = ...
        cputime;


    try

        Btrain = ...
            globalrbf.globalRBFBootstrapForTraining( ...
                X, ...
                y, ...
                trainingInfo, ...
                cvp_fixed, ...
                nBootstrap, ...
                RNG_SEED, ...
                bootstrapOptions);


        trainWallTime = ...
            toc(tTrain);


        trainCPUTime = ...
            cputime - cpuTrain0;


        Rrun.trainWallTime = ...
            trainWallTime;


        Rrun.trainCPUTime = ...
            trainCPUTime;


        Rrun.trainSuccess = ...
            Btrain.success;


        Rrun.trainMessage = ...
            '';


        Rrun.trainRBFSolves = ...
            Btrain.Computation.totalRBFSolves;


        Rrun.trainRejected = ...
            Btrain.Computation.totalRejected;


        % -----------------------------------------------------
        % cvp_fixed is already stored at Results.CVInfo and
        % Rrun.cvp_fixed for checkpoint self-containment.
        % Remove only this duplicated copy from detailed result.
        % -----------------------------------------------------

        if isfield(Btrain,'CV') && ...
           isfield(Btrain.CV,'cvp_fixed')

            Btrain.CV = ...
                rmfield( ...
                    Btrain.CV, ...
                    'cvp_fixed');

        end


        RunInfo.Train = ...
            Btrain;


        if verbose

            fprintf( ...
                'Training bootstrap wall = %.3f s\n', ...
                trainWallTime);


            fprintf( ...
                'Training bootstrap CPU  = %.3f s\n', ...
                trainCPUTime);


            fprintf( ...
                'Training RBF solves     = %d\n', ...
                Rrun.trainRBFSolves);


            fprintf( ...
                'Training rejected       = %d\n', ...
                Rrun.trainRejected);

        end


    catch ME

        trainWallTime = ...
            toc(tTrain);


        trainCPUTime = ...
            cputime - cpuTrain0;


        Rrun.trainWallTime = ...
            trainWallTime;


        Rrun.trainCPUTime = ...
            trainCPUTime;


        Rrun.trainSuccess = ...
            false;


        Rrun.trainMessage = ...
            ME.message;


        RunInfo.Train = ...
            [];


        if verbose

            fprintf( ...
                '[TRAIN BOOTSTRAP FAILED] %s | %s | %s\n', ...
                CaseName, ...
                modelName, ...
                kernel);


            fprintf( ...
                'Reason: %s\n', ...
                ME.message);

        end

    end


    % =========================================================
    % 20. TESTING bootstrap variance
    %
    % This depends on the ORIGINAL selected RBF configuration,
    % not on successful completion of the training-bootstrap
    % variance calculation.
    % =========================================================

    if OriginalTestOK(irun)

        tTest = ...
            tic;


        cpuTest0 = ...
            cputime;


        try

            Btest = ...
                globalrbf.globalRBFBootstrapForTesting( ...
                    X, ...
                    y, ...
                    X_test, ...
                    trainingInfo, ...
                    nBootstrap, ...
                    TestRNG_SEED, ...
                    bootstrapOptions);


            testWallTime = ...
                toc(tTest);


            testCPUTime = ...
                cputime - cpuTest0;


            Rrun.testWallTime = ...
                testWallTime;


            Rrun.testCPUTime = ...
                testCPUTime;


            Rrun.testSuccess = ...
                Btest.success;


            Rrun.testMessage = ...
                '';


            Rrun.testRBFSolves = ...
                Btest.Computation.nRBFSolves;


            Rrun.testRejected = ...
                Btest.Computation.nRejected;


            RunInfo.Test = ...
                Btest;


            if verbose

                fprintf( ...
                    'Testing bootstrap wall  = %.3f s\n', ...
                    testWallTime);


                fprintf( ...
                    'Testing bootstrap CPU   = %.3f s\n', ...
                    testCPUTime);


                fprintf( ...
                    'Testing RBF solves      = %d\n', ...
                    Rrun.testRBFSolves);


                fprintf( ...
                    'Testing rejected        = %d\n', ...
                    Rrun.testRejected);

            end


        catch ME

            testWallTime = ...
                toc(tTest);


            testCPUTime = ...
                cputime - cpuTest0;


            Rrun.testWallTime = ...
                testWallTime;


            Rrun.testCPUTime = ...
                testCPUTime;


            Rrun.testSuccess = ...
                false;


            Rrun.testMessage = ...
                ME.message;


            RunInfo.Test = ...
                [];


            if verbose

                fprintf( ...
                    '[TEST BOOTSTRAP FAILED] %s | %s | %s\n', ...
                    CaseName, ...
                    modelName, ...
                    kernel);


                fprintf( ...
                    'Reason: %s\n', ...
                    ME.message);

            end

        end


    else

        Rrun.testSuccess = ...
            false;


        Rrun.testMessage = ...
            ['Not run because the original final test ' ...
             'prediction failed.'];


        RunInfo.Test = ...
            [];

    end


    % =========================================================
    % 21. Total timing
    % =========================================================

    if isfinite(Rrun.trainWallTime) && ...
       isfinite(Rrun.testWallTime)

        Rrun.totalWallTime = ...
            Rrun.trainWallTime + ...
            Rrun.testWallTime;

    elseif isfinite(Rrun.trainWallTime)

        Rrun.totalWallTime = ...
            Rrun.trainWallTime;

    elseif isfinite(Rrun.testWallTime)

        Rrun.totalWallTime = ...
            Rrun.testWallTime;

    end


    if isfinite(Rrun.trainCPUTime) && ...
       isfinite(Rrun.testCPUTime)

        Rrun.totalCPUTime = ...
            Rrun.trainCPUTime + ...
            Rrun.testCPUTime;

    elseif isfinite(Rrun.trainCPUTime)

        Rrun.totalCPUTime = ...
            Rrun.trainCPUTime;

    elseif isfinite(Rrun.testCPUTime)

        Rrun.totalCPUTime = ...
            Rrun.testCPUTime;

    end


    % =========================================================
    % 22. Total RBF solve count
    % =========================================================

    if isfinite(Rrun.trainRBFSolves) && ...
       isfinite(Rrun.testRBFSolves)

        Rrun.totalRBFSolves = ...
            Rrun.trainRBFSolves + ...
            Rrun.testRBFSolves;

    elseif isfinite(Rrun.trainRBFSolves)

        Rrun.totalRBFSolves = ...
            Rrun.trainRBFSolves;

    elseif isfinite(Rrun.testRBFSolves)

        Rrun.totalRBFSolves = ...
            Rrun.testRBFSolves;

    end


    % =========================================================
    % 23. Total rejection count
    % =========================================================

    if isfinite(Rrun.trainRejected) && ...
       isfinite(Rrun.testRejected)

        Rrun.totalRejected = ...
            Rrun.trainRejected + ...
            Rrun.testRejected;

    elseif isfinite(Rrun.trainRejected)

        Rrun.totalRejected = ...
            Rrun.trainRejected;

    elseif isfinite(Rrun.testRejected)

        Rrun.totalRejected = ...
            Rrun.testRejected;

    end


    % =========================================================
    % 24. Complete worker result
    % =========================================================

    Rrun.RunInfo = ...
        RunInfo;


    RunCell{irun} = ...
        Rrun;


    % =========================================================
    % 25. CHECKPOINT
    %
    % Save immediately after this GGM x Kernel combination
    % finishes, just as in the TuneReg batch.
    % =========================================================

    if ~isempty(checkpointDir)

        checkpointFile = ...
            fullfile( ...
                checkpointDir, ...
                sprintf( ...
                    '%s_%s.mat', ...
                    modelName, ...
                    kernel));


        checkpointData = ...
            struct('Rrun',Rrun);


        try

            save( ...
                checkpointFile, ...
                '-fromstruct', ...
                checkpointData, ...
                '-v7.3');


        catch MEcheckpoint

            warning( ...
                'Bootstrap checkpoint failed for %s: %s', ...
                ModelAndFunction, ...
                MEcheckpoint.message);

        end

    end

end


% =============================================================
% 26. Actual elapsed parallel-section wall time
%
% This is the real elapsed time of the parallel Model x Kernel
% section and is NOT the sum of individual run wall times.
% =============================================================

Results.parallelWallTime = ...
    toc(tParallel);


% =============================================================
% 27. Merge parallel results
% =============================================================

for irun = 1:nRun

    Rrun = ...
        RunCell{irun};


    i = ...
        Rrun.i;


    j = ...
        Rrun.j;


    ModelAndFunction = ...
        Rrun.ModelAndFunction;


    % ---------------------------------------------------------
    % Detailed results.
    %
    % cvp_fixed is not duplicated here; it is stored once at
    % Results.CVInfo.cvp_fixed.
    % ---------------------------------------------------------

    Results.Run.(ModelAndFunction) = ...
        Rrun.RunInfo;


    % ---------------------------------------------------------
    % Timing.
    % ---------------------------------------------------------

    Results.trainWallTime(i,j) = ...
        Rrun.trainWallTime;


    Results.trainCPUTime(i,j) = ...
        Rrun.trainCPUTime;


    Results.testWallTime(i,j) = ...
        Rrun.testWallTime;


    Results.testCPUTime(i,j) = ...
        Rrun.testCPUTime;


    Results.totalWallTime(i,j) = ...
        Rrun.totalWallTime;


    Results.totalCPUTime(i,j) = ...
        Rrun.totalCPUTime;


    % ---------------------------------------------------------
    % Solve counts.
    % ---------------------------------------------------------

    Results.trainRBFSolves(i,j) = ...
        Rrun.trainRBFSolves;


    Results.testRBFSolves(i,j) = ...
        Rrun.testRBFSolves;


    Results.totalRBFSolves(i,j) = ...
        Rrun.totalRBFSolves;


    % ---------------------------------------------------------
    % Rejection counts.
    % ---------------------------------------------------------

    Results.trainRejected(i,j) = ...
        Rrun.trainRejected;


    Results.testRejected(i,j) = ...
        Rrun.testRejected;


    Results.totalRejected(i,j) = ...
        Rrun.totalRejected;


    % ---------------------------------------------------------
    % Status.
    % ---------------------------------------------------------

    Results.trainSuccess(i,j) = ...
        Rrun.trainSuccess;


    Results.testSuccess(i,j) = ...
        Rrun.testSuccess;


    Results.trainMessage{i,j} = ...
        Rrun.trainMessage;


    Results.testMessage{i,j} = ...
        Rrun.testMessage;


    % ---------------------------------------------------------
    % TRAINING variance summaries.
    % ---------------------------------------------------------

    if Rrun.trainSuccess && ...
       ~isempty(Rrun.RunInfo.Train)

        Results.trainMeanBootstrapVariance(i,j) = ...
            mean( ...
                Rrun.RunInfo.Train.BootstrapVariance);


        Results.trainMedianBootstrapVariance(i,j) = ...
            median( ...
                Rrun.RunInfo.Train.BootstrapVariance);


        Results.trainMeanVariance(i,j) = ...
            mean( ...
                Rrun.RunInfo.Train.Variance);


        Results.trainMedianVariance(i,j) = ...
            median( ...
                Rrun.RunInfo.Train.Variance);


        if isfield(Rrun.RunInfo.Train,'DominanceControl') && ...
           isstruct(Rrun.RunInfo.Train.DominanceControl)

            DC = ...
                Rrun.RunInfo.Train.DominanceControl;


            if isfield(DC,'nDominantPoints') && ...
               isnumeric(DC.nDominantPoints) && ...
               isscalar(DC.nDominantPoints) && ...
               isfinite(DC.nDominantPoints)

                Results.trainDominantPointsCount(i,j) = ...
                    DC.nDominantPoints;

            end


            if isfield(DC,'totalDominant') && ...
               isnumeric(DC.totalDominant) && ...
               isscalar(DC.totalDominant) && ...
               isfinite(DC.totalDominant)

                Results.trainDominantCount(i,j) = ...
                    DC.totalDominant;

            end


            if isfield(DC,'PointIndex')

                Results.trainDominantPoints{i,j} = ...
                    DC.PointIndex(:);

            end

        end

    end


    % ---------------------------------------------------------
    % TEST variance summaries.
    % ---------------------------------------------------------

    if Rrun.testSuccess && ...
       ~isempty(Rrun.RunInfo.Test)

        Results.testMeanBootstrapVariance(i,j) = ...
            mean( ...
                Rrun.RunInfo.Test.BootstrapVariance);


        Results.testMedianBootstrapVariance(i,j) = ...
            median( ...
                Rrun.RunInfo.Test.BootstrapVariance);


        Results.testMeanVariance(i,j) = ...
            mean( ...
                Rrun.RunInfo.Test.Variance);


        Results.testMedianVariance(i,j) = ...
            median( ...
                Rrun.RunInfo.Test.Variance);


        if isfield(Rrun.RunInfo.Test,'DominanceControl') && ...
           isstruct(Rrun.RunInfo.Test.DominanceControl)

            DC = ...
                Rrun.RunInfo.Test.DominanceControl;


            if isfield(DC,'nDominantPoints') && ...
               isnumeric(DC.nDominantPoints) && ...
               isscalar(DC.nDominantPoints) && ...
               isfinite(DC.nDominantPoints)

                Results.testDominantPointsCount(i,j) = ...
                    DC.nDominantPoints;

            end


            if isfield(DC,'totalDominant') && ...
               isnumeric(DC.totalDominant) && ...
               isscalar(DC.totalDominant) && ...
               isfinite(DC.totalDominant)

                Results.testDominantCount(i,j) = ...
                    DC.totalDominant;

            end


            if isfield(DC,'PointIndex')

                Results.testDominantPoints{i,j} = ...
                    DC.PointIndex(:);

            end

        end

    end

end


% =============================================================
% 28. Overall status
% =============================================================

Results.success = ...
    Results.trainSuccess & ...
    Results.testSuccess;


Results.nSuccessfulTrain = ...
    nnz(Results.trainSuccess);


Results.nSuccessfulTest = ...
    nnz(Results.testSuccess);


Results.nSuccessfulBoth = ...
    nnz(Results.success);


Results.totalRuns = ...
    nRun;


% =============================================================
% 29. Overall solve/rejection totals
% =============================================================

Results.totalTrainRBFSolves = ...
    sum(Results.trainRBFSolves(isfinite(Results.trainRBFSolves)));


Results.totalTestRBFSolves = ...
    sum(Results.testRBFSolves(isfinite(Results.testRBFSolves)));


Results.totalAllRBFSolves = ...
    sum(Results.totalRBFSolves(isfinite(Results.totalRBFSolves)));


Results.totalTrainRejected = ...
    sum(Results.trainRejected(isfinite(Results.trainRejected)));


Results.totalTestRejected = ...
    sum(Results.testRejected(isfinite(Results.testRejected)));


Results.totalAllRejected = ...
    sum(Results.totalRejected(isfinite(Results.totalRejected)));


% -------------------------------------------------------------
% Dominance-control totals.  NaN entries correspond to runs for
% which the relevant bootstrap phase did not complete successfully.
% -------------------------------------------------------------

Results.totalTrainDominantPoints = ...
    sum(Results.trainDominantPointsCount( ...
        isfinite(Results.trainDominantPointsCount)));


Results.totalTestDominantPoints = ...
    sum(Results.testDominantPointsCount( ...
        isfinite(Results.testDominantPointsCount)));


Results.totalTrainDominantCount = ...
    sum(Results.trainDominantCount( ...
        isfinite(Results.trainDominantCount)));


Results.totalTestDominantCount = ...
    sum(Results.testDominantCount( ...
        isfinite(Results.testDominantCount)));


% =============================================================
% 30. Final save
% =============================================================

if ~isempty(OutputFile)

    save( ...
        OutputFile, ...
        'Results', ...
        '-v7.3');

end

end
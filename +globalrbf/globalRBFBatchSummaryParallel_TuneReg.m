function Results = globalRBFBatchSummaryParallel_TuneReg( ...
    InputData, CaseName, ListOfModel, NameFunction_RBF, ...
    method, CMin, CMax, kfolds, RegMin, RegMax, ...
    verbose, RNG_SEED, options, OutputFile)

%GLOBALRBFBATCHSUMMARYPARALLEL_TUNEREG
%
% Batch execution and summary of global RBF training/CV and testing.
%
% RBF calculations are performed by:
%
%   globalRBFTrainingCV_TuneReg
%   globalRBFPredictTest
%
% Expected input structure:
%
%   InputData.Train.XYZ
%   InputData.Test.XYZ
%   InputData.Train.DeltaZeta.<model>
%   InputData.Test.DeltaZeta.<model>
%
% Results include prediction accuracy, search information,
% numerical diagnostics, computational cost, and run status.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 5 || isempty(method)
    method = 'ga';
end

if nargin < 6 || isempty(CMin)

    CMin.gaussian = 0.01;
    CMin.multiquadric = 0.001;

    CMin.thinplate = 1;
    CMin.cubic = 1;
    CMin.linear = 1;

end

if nargin < 7 || isempty(CMax)

    CMax.gaussian = 2;
    CMax.multiquadric = 2;

    CMax.thinplate = 1;
    CMax.cubic = 1;
    CMax.linear = 1;

end


if nargin < 8 || isempty(kfolds)
    kfolds = 10;
end

if ~isnumeric(kfolds) || ...
   ~isscalar(kfolds) || ...
   ~isfinite(kfolds) || ...
   kfolds <= 0

    error('kfolds must be a finite positive numeric scalar.');

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

if nargin < 11 || isempty(verbose)
    verbose = true;
end

if nargin < 12
    RNG_SEED = 1;
end

if isempty(RNG_SEED)
    error(['RNG_SEED must be nonempty for parallel execution ' ...
           'so all runs use the same fixed CV partition.']);
end

if ~isnumeric(RNG_SEED) || ...
   ~isscalar(RNG_SEED) || ...
   ~isfinite(RNG_SEED) || ...
   RNG_SEED < 0 || ...
   RNG_SEED ~= floor(RNG_SEED)

    error('RNG_SEED must be a finite nonnegative integer scalar.');

end

if nargin < 13 || isempty(options)
    options = struct();
end
% Solver threshold
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

% Do not calculate relative residual during GA/fallback search.

if ~isfield(options.solver,'compute_rel_residual') || ...
   isempty(options.solver.compute_rel_residual)

    options.solver.compute_rel_residual = ...
        false;

end


if ~(isscalar(options.solver.compute_rel_residual) && ...
     (islogical(options.solver.compute_rel_residual) || ...
      isnumeric(options.solver.compute_rel_residual)))

    error( ...
        ['options.solver.compute_rel_residual must be ' ...
         'a scalar logical or numeric value.']);

end


options.solver.compute_rel_residual = ...
    logical(options.solver.compute_rel_residual);


if nargin < 14
    OutputFile = '';
end

CaseName = char(CaseName);


% =============================================================
% 2. Input data
% =============================================================

X = InputData.Train.XYZ;
X_test = InputData.Test.XYZ;

if any(~isfinite(X(:))) || any(~isfinite(X_test(:)))
    error('Training and testing coordinates must contain only finite values.');
end

nModel = numel(ListOfModel);
nKernel = numel(NameFunction_RBF);


if size(X,2) ~= size(X_test,2)
    error('Training and testing coordinates must have the same dimensions.');
end

if ~isempty(OutputFile)
    outputDir = fileparts(OutputFile);
    if ~isempty(outputDir) && ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
end


% =============================================================
% 3. Initialize results
% =============================================================

Results = struct();
Results.Run = struct();

Results.CaseName = CaseName;
Results.ListOfModel = ListOfModel;
Results.NameFunction_RBF = NameFunction_RBF;


% -------------------------------------------------------------
% Experiment settings
% -------------------------------------------------------------

Results.Settings.method = method;
Results.Settings.kfolds = kfolds;
Results.Settings.RNG_SEED = RNG_SEED;
Results.Settings.verbose = verbose;
Results.Settings.CMin = CMin;
Results.Settings.CMax = CMax;

Results.Settings.RegMin = RegMin;
Results.Settings.RegMax = RegMax;
Results.Settings.options = options;
Results.Settings.GA = struct();


% -------------------------------------------------------------
% Problem size for computational/scalability reporting
% -------------------------------------------------------------

Results.DataInfo.nTrain = size(X,1);
Results.DataInfo.nTest = size(X_test,1);
Results.DataInfo.dimension = size(X,2);
Results.DataInfo.nModels = nModel;
Results.DataInfo.nKernels = nKernel;


% -------------------------------------------------------------
% Matrix interpretation
%
% Matrix interpretation for the final Results structure.
% -------------------------------------------------------------

Results.MatrixInfo.rows = ListOfModel;
Results.MatrixInfo.columns = NameFunction_RBF;
Results.MatrixInfo.description = 'Rows = input models; columns = RBF kernels';


% -------------------------------------------------------------
% Fixed CV information
%
% Store the MATLAB partition once and also store numeric fold IDs
% for easier reproducibility/reporting.
% -------------------------------------------------------------

Results.CVInfo.cvp_fixed = [];
Results.CVInfo.foldID = [];
Results.CVInfo.k_requested = kfolds;
Results.CVInfo.k_used = [];


% -------------------------------------------------------------
% Compact result matrices
% -------------------------------------------------------------

Results.reg_param = nan(nModel,nKernel);

Results.C_global = nan(nModel,nKernel);
Results.epsilon_global = nan(nModel,nKernel);

Results.cv_rmse = nan(nModel,nKernel);
Results.test_rmse = nan(nModel,nKernel);

Results.k_used = nan(nModel,nKernel);


% -------------------------------------------------------------
% Training and testing times
% -------------------------------------------------------------

Results.trainWallTime = nan(nModel,nKernel);
Results.trainCPUTime = nan(nModel,nKernel);

Results.testWallTime = nan(nModel,nKernel);
Results.testCPUTime = nan(nModel,nKernel);

Results.totalWallTime = nan(nModel,nKernel);
Results.totalCPUTime = nan(nModel,nKernel);


% -------------------------------------------------------------
% Search evaluation counts
% -------------------------------------------------------------

Results.nEvaluations = nan(nModel,nKernel);
Results.nGAEvaluations = nan(nModel,nKernel);
Results.nFallbackEvaluations = nan(nModel,nKernel);
Results.nValidEvaluations = nan(nModel,nKernel);
Results.nFailedEvaluations = nan(nModel,nKernel);


% -------------------------------------------------------------
% RBF solve counts
%
% trainRBFSolves will be populated when globalRBFTrainingCV_TuneReg
% supplies trainingInfo.nRBFSolves.
%
% A successful globalRBFPredictTest performs one final
% full-training RBF solve.
% -------------------------------------------------------------

Results.trainRBFSolves = nan(nModel,nKernel);
Results.testRBFSolves = nan(nModel,nKernel);
Results.totalRBFSolves = nan(nModel,nKernel);


% -------------------------------------------------------------
% Run-status matrices
% -------------------------------------------------------------

Results.trainSuccess = false(nModel,nKernel);
Results.testSuccess = false(nModel,nKernel);

Results.trainMessage = cell(nModel,nKernel);
Results.testMessage = cell(nModel,nKernel);


% =============================================================
% 4. Prepare parallel Model x Kernel execution
% =============================================================

% Validate model data before starting workers
for i = 1:nModel

    modelName = char(ListOfModel{i});

    if ~isfield(InputData.Train.DeltaZeta,modelName)
        error('Training data for model "%s" were not found.',modelName);
    end

    if ~isfield(InputData.Test.DeltaZeta,modelName)
        error('Testing data for model "%s" were not found.',modelName);
    end

     y = InputData.Train.DeltaZeta.(modelName);
y_test = InputData.Test.DeltaZeta.(modelName);

if size(X,1) ~= numel(y)
    error('Training coordinate/data size mismatch for model "%s".',modelName);
end

if size(X_test,1) ~= numel(y_test)
    error('Testing coordinate/data size mismatch for model "%s".',modelName);
end

if any(~isfinite(y(:))) || any(~isfinite(y_test(:)))
    error('Data for model "%s" contain NaN or Inf.',modelName);
end


end


for j = 1:nKernel

    kernel = char(NameFunction_RBF{j});
    kernelKey = lower(kernel);

    useShape = common.kernelHasShapeParameter(kernel);


    if ~isfield(CMin,kernelKey)
        error('CMin.%s was not provided.',kernelKey);
    end

    if ~isfield(CMax,kernelKey)
        error('CMax.%s was not provided.',kernelKey);
    end


    if ~isfield(RegMin,kernelKey)
        error('RegMin.%s was not provided.',kernelKey);
    end

    if ~isfield(RegMax,kernelKey)
        error('RegMax.%s was not provided.',kernelKey);
    end

    if useShape

        cmin = CMin.(kernelKey);
        cmax = CMax.(kernelKey);

        if ~isscalar(cmin) || ~isscalar(cmax) || ...
                ~isfinite(cmin) || ~isfinite(cmax)

            error('C bounds for "%s" must be finite scalars.',kernel);

        end

        if max(cmax,1e-12) < max(cmin,1e-12)
            error('CMax.%s must be >= CMin.%s.',kernelKey,kernelKey);
        end

    end

    % RegParam is required for ALL kernels.
    rmin = RegMin.(kernelKey);
    rmax = RegMax.(kernelKey);

    if ~isscalar(rmin) || ~isfinite(rmin) || rmin <= 0 || ...
            ~isscalar(rmax) || ~isfinite(rmax) || rmax < rmin

        error('Invalid RegParam bounds for kernel "%s".',kernel);

    end
end


if ~strcmpi(char(method),'ga')
    error('Only method="ga" is currently supported.');
end

% Start parallel pool
pool = gcp('nocreate');

if isempty(pool)
    pool = parpool;
end

Results.Settings.Parallel = true;
Results.Settings.NumWorkers = pool.NumWorkers;


% One independent run for each Model x Kernel combination
nRun = nModel * nKernel;
RunCell = cell(nRun,1);


% =============================================================
% 5. Parallel Model x Kernel loop
% =============================================================

checkpointDir = '';

if ~isempty(OutputFile)

    checkpointDir = fullfile( ...
        fileparts(OutputFile), ...
        [CaseName '_checkpoints']);

    if ~exist(checkpointDir,'dir')
        mkdir(checkpointDir);
    end

end

parfor irun = 1:nRun

    % Preserve original serial ordering:
    %
    % model 1: kernels 1...nKernel
    % model 2: kernels 1...nKernel
    % ...

    i = floor((irun - 1) / nKernel) + 1;
    j = mod(irun - 1,nKernel) + 1;


    modelName = char(ListOfModel{i});
    kernel = char(NameFunction_RBF{j});
    kernelKey = lower(kernel);

    useShape = common.kernelHasShapeParameter(kernel);

    ModelAndFunction = [modelName '_' kernel];


    y = InputData.Train.DeltaZeta.(modelName);
    y_test = InputData.Test.DeltaZeta.(modelName);


    y = y(:);
    y_test = y_test(:);


    % ---------------------------------------------------------
    % Local result for this Model x Kernel combination
    % ---------------------------------------------------------

    Rrun = struct();

    Rrun.i = i;
    Rrun.j = j;
    Rrun.ModelAndFunction = ModelAndFunction;

    Rrun.C_global = NaN;
    Rrun.epsilon_global = NaN;
    Rrun.cv_rmse = NaN;
    Rrun.test_rmse = NaN;
    Rrun.k_used = NaN;

    Rrun.trainWallTime = NaN;
    Rrun.trainCPUTime = NaN;
    Rrun.testWallTime = NaN;
    Rrun.testCPUTime = NaN;
    Rrun.totalWallTime = NaN;
    Rrun.totalCPUTime = NaN;

    Rrun.nEvaluations = NaN;
    Rrun.nGAEvaluations = NaN;
    Rrun.nFallbackEvaluations = NaN;
    Rrun.nValidEvaluations = NaN;
    Rrun.nFailedEvaluations = NaN;

    Rrun.trainRBFSolves = NaN;
    Rrun.testRBFSolves = NaN;
    Rrun.totalRBFSolves = NaN;

    Rrun.trainSuccess = false;
    Rrun.testSuccess = false;

    Rrun.trainMessage = '';
    Rrun.testMessage = '';

    Rrun.cvp_fixed = [];
    Rrun.foldID = [];
    Rrun.GA = [];


    % ---------------------------------------------------------
    % Initialize detailed run information
    % ---------------------------------------------------------

    RunInfo = struct();

    RunInfo.model = modelName;
    RunInfo.kernel = kernel;

    RunInfo.Train.success = false;
    RunInfo.Train.message = 'Not run yet.';
    RunInfo.Train.wallTime = NaN;
    RunInfo.Train.cpuTime = NaN;
    RunInfo.Train.nRBFSolves = [];

    RunInfo.Test.success = false;
    RunInfo.Test.message = 'Not run yet.';
    RunInfo.Test.wallTime = NaN;
    RunInfo.Test.cpuTime = NaN;
    RunInfo.Test.nRBFSolves = [];


    if verbose

        fprintf('\n====================================================\n');
        fprintf('Case: %s | Model: %s | Kernel: %s\n', ...
            CaseName,modelName,kernel);
        fprintf('====================================================\n');

    end



   % =========================================================
% 6. Kernel-specific search bounds
% =========================================================

C_min = CMin.(kernelKey);
C_max = CMax.(kernelKey);

Reg_min = RegMin.(kernelKey);
Reg_max = RegMax.(kernelKey);


  % =========================================================
% 7. Search bounds
% =========================================================

Rrun.reg_param = NaN;

if useShape

    RunInfo.C_min = C_min;
    RunInfo.C_max = C_max;

else

    RunInfo.C_min = NaN;
    RunInfo.C_max = NaN;

end
RunInfo.Reg_min = Reg_min;
RunInfo.Reg_max = Reg_max;

RunInfo.reg_param = NaN;
    % =========================================================
    % 8. Training / cross-validation
    % =========================================================

    trainingInfo = [];

    tTrain = tic;
    cpuTrain0 = cputime;

    try
       [C_global, y_cv_pred, y_diff_cv, cv_rmse, fold_rmse, ...
 C_ga, cv_rmse_ga, trainingInfo] = ...
    globalrbf.globalRBFTrainingCV_TuneReg( ...
        X, ...
        y, ...
        kernel, ...
        method, ...
        C_min, ...
        C_max, ...
        Reg_min, ...
        Reg_max, ...
        kfolds, ...
        verbose, ...
        RNG_SEED, ...
        options);

        trainWallTime = toc(tTrain);
        trainCPUTime = cputime - cpuTrain0;

    % -----------------------------------------------------
    % Selected regularization parameter
    % -----------------------------------------------------

    reg_param = trainingInfo.reg_param;

    Rrun.reg_param = reg_param;
    RunInfo.reg_param = reg_param;
        % -----------------------------------------------------
        % Fixed CV information
        % -----------------------------------------------------

        if isfield(trainingInfo,'cvp_fixed')

            Rrun.cvp_fixed = trainingInfo.cvp_fixed;

            foldID = zeros(size(X,1),1);

            for f = 1:trainingInfo.k_used
                foldID(test(trainingInfo.cvp_fixed,f)) = f;
            end

            Rrun.foldID = foldID;

        end


        % -----------------------------------------------------
        % GA settings
        % -----------------------------------------------------

        if isfield(trainingInfo,'GA')
            Rrun.GA = trainingInfo.GA;
        end


        % -----------------------------------------------------
        % Remove repeated cvpartition from per-run storage
        % -----------------------------------------------------
        trainingInfoStore = trainingInfo;

        removeFields = ...
            {'cvp_fixed','C_ga','Reg_ga','cv_rmse_ga'};

        removeFields = ...
            removeFields(isfield(trainingInfoStore,removeFields));

        if ~isempty(removeFields)
            trainingInfoStore = ...
                rmfield(trainingInfoStore,removeFields);
        end

        % -----------------------------------------------------
        % Detailed training information
        % -----------------------------------------------------

        RunInfo.Train.C_global = C_global;
        RunInfo.Train.y_cv_pred = y_cv_pred;
        RunInfo.Train.y_diff_cv = y_diff_cv;
        RunInfo.Train.errors_cv = abs(y_diff_cv);
        RunInfo.Train.cv_rmse = cv_rmse;
        RunInfo.Train.fold_rmse = fold_rmse;
        RunInfo.Train.C_ga = C_ga;
        RunInfo.Train.Reg_ga = trainingInfo.Reg_ga;
        RunInfo.Train.cv_rmse_ga = cv_rmse_ga;
        RunInfo.Train.trainingInfo = trainingInfoStore;

        RunInfo.Train.wallTime = trainWallTime;
        RunInfo.Train.cpuTime = trainCPUTime;
        RunInfo.Train.success = true;
        RunInfo.Train.message = '';


        % -----------------------------------------------------
        % Compact training information
        % -----------------------------------------------------

        Rrun.C_global = C_global;
        Rrun.cv_rmse = cv_rmse;
        Rrun.k_used = trainingInfo.k_used;

        Rrun.trainWallTime = trainWallTime;
        Rrun.trainCPUTime = trainCPUTime;

        Rrun.trainSuccess = true;
        Rrun.trainMessage = '';


        % -----------------------------------------------------
        % GA/search evaluation counts
        % -----------------------------------------------------
    Rrun.nEvaluations = trainingInfo.nEvaluations;
    Rrun.nGAEvaluations = trainingInfo.nGAEvaluations;
    Rrun.nFallbackEvaluations = trainingInfo.nFallbackEvaluations;
    Rrun.nValidEvaluations = trainingInfo.nValidEvaluations;
    Rrun.nFailedEvaluations = trainingInfo.nFailedEvaluations;

        % -----------------------------------------------------
        % Training RBF solve count
        % -----------------------------------------------------

        if isfield(trainingInfo,'nRBFSolves') && ...
                ~isempty(trainingInfo.nRBFSolves)

            Rrun.trainRBFSolves = trainingInfo.nRBFSolves;
            RunInfo.Train.nRBFSolves = trainingInfo.nRBFSolves;

        end


        if verbose

            fprintf('CV RMSE        = %.6g\n',cv_rmse);
            fprintf('Training wall  = %.3f s\n',trainWallTime);
            fprintf('Training CPU   = %.3f s\n',trainCPUTime);

            if trainingInfo.useShape
                fprintf('C_global       = %.12g\n',C_global);
            end

            fprintf('RegParam       = %.12g\n',reg_param);

        end


    catch ME

        trainWallTime = toc(tTrain);
        trainCPUTime = cputime - cpuTrain0;

        Rrun.trainWallTime = trainWallTime;
        Rrun.trainCPUTime = trainCPUTime;

        Rrun.totalWallTime = trainWallTime;
        Rrun.totalCPUTime = trainCPUTime;

        Rrun.trainSuccess = false;
        Rrun.testSuccess = false;

        Rrun.trainMessage = ME.message;
        Rrun.testMessage = 'Not run because training failed.';

        RunInfo.Train.wallTime = trainWallTime;
        RunInfo.Train.cpuTime = trainCPUTime;
        RunInfo.Train.success = false;
        RunInfo.Train.message = ME.message;

        RunInfo.Test.success = false;
        RunInfo.Test.message = 'Not run because training failed.';


        if verbose

            fprintf('[TRAIN FAILED] %s | %s | %s\n', ...
                CaseName,modelName,kernel);

            fprintf('Reason: %s\n',ME.message);

        end

    end


    % =========================================================
    % 9. Final full-training fit and testing
    % =========================================================

    if Rrun.trainSuccess

        tTest = tic;
        cpuTest0 = cputime;

        try

            [y_test_pred, y_test_diff, test_rmse, ...
                epsilon_global, modelInfo] = ...
                globalrbf.globalRBFPredictTest( ...
                X, y, X_test, y_test, trainingInfo);

            if useShape
                epsilon_report = epsilon_global;
            else
                epsilon_report = NaN;
            end


            testWallTime = toc(tTest);
            testCPUTime = cputime - cpuTest0;


            % -------------------------------------------------
            % Remove coefficients from batch storage
            % -------------------------------------------------

            modelInfoStore = modelInfo;

            if isfield(modelInfoStore,'coeffs')
                modelInfoStore = rmfield(modelInfoStore,'coeffs');
            end


            % -------------------------------------------------
            % Detailed testing information
            % -------------------------------------------------

          RunInfo.Test.epsilon_global = epsilon_report;
            RunInfo.Test.y_test_pred = y_test_pred;
            RunInfo.Test.y_test_diff = y_test_diff;
            RunInfo.Test.errors_test = abs(y_test_diff);
            RunInfo.Test.test_rmse = test_rmse;
            RunInfo.Test.modelInfo = modelInfoStore;

            RunInfo.Test.wallTime = testWallTime;
            RunInfo.Test.cpuTime = testCPUTime;
            RunInfo.Test.nRBFSolves = modelInfo.nRBFSolves;
            RunInfo.Test.success = true;
            RunInfo.Test.message = '';


            % -------------------------------------------------
            % Compact testing information
            % -------------------------------------------------

         Rrun.epsilon_global = epsilon_report;
            Rrun.test_rmse = test_rmse;

            Rrun.testWallTime = testWallTime;
            Rrun.testCPUTime = testCPUTime;

            Rrun.totalWallTime = ...
                Rrun.trainWallTime + testWallTime;

            Rrun.totalCPUTime = ...
                Rrun.trainCPUTime + testCPUTime;

            Rrun.testRBFSolves = modelInfo.nRBFSolves;

            if isfinite(Rrun.trainRBFSolves) && ...
                    isfinite(Rrun.testRBFSolves)

                Rrun.totalRBFSolves = ...
                    Rrun.trainRBFSolves + Rrun.testRBFSolves;

            end

            Rrun.testSuccess = true;
            Rrun.testMessage = '';


            if verbose

                fprintf('Test RMSE       = %.6g\n',test_rmse);
                fprintf('Testing wall    = %.3f s\n',testWallTime);
                fprintf('Testing CPU     = %.3f s\n',testCPUTime);

                if trainingInfo.useShape
                    fprintf('epsilon_global  = %.12g\n',epsilon_global);
                end

            end


        catch ME

            testWallTime = toc(tTest);
            testCPUTime = cputime - cpuTest0;

            Rrun.testWallTime = testWallTime;
            Rrun.testCPUTime = testCPUTime;

            Rrun.totalWallTime = ...
                Rrun.trainWallTime + testWallTime;

            Rrun.totalCPUTime = ...
                Rrun.trainCPUTime + testCPUTime;

            Rrun.testSuccess = false;
            Rrun.testMessage = ME.message;

            RunInfo.Test.wallTime = testWallTime;
            RunInfo.Test.cpuTime = testCPUTime;
            RunInfo.Test.success = false;
            RunInfo.Test.message = ME.message;


            if verbose

                fprintf('[TEST FAILED] %s | %s | %s\n', ...
                    CaseName,modelName,kernel);

                fprintf('Reason: %s\n',ME.message);

            end

        end

    end


    % ---------------------------------------------------------
    % Return this independent run to the client
    % ---------------------------------------------------------

    Rrun.RunInfo = RunInfo;

    RunCell{irun} = Rrun;

if ~isempty(checkpointDir)

    checkpointFile = fullfile(checkpointDir, ...
        sprintf('%s_%s.mat',modelName,kernel));

    checkpointData = struct('Rrun',Rrun);

    try
        save(checkpointFile, ...
            '-fromstruct',checkpointData, ...
            '-v7.3');
    catch MEcheckpoint
        warning('Checkpoint failed for %s: %s', ...
            ModelAndFunction,MEcheckpoint.message);
    end

end

end


% =============================================================
% 9B. Merge parallel results into Results
% =============================================================

for irun = 1:nRun

    Rrun = RunCell{irun};

    i = Rrun.i;
    j = Rrun.j;

    ModelAndFunction = Rrun.ModelAndFunction;


    % ---------------------------------------------------------
    % Detailed run information
    % ---------------------------------------------------------

    Results.Run.(ModelAndFunction) = Rrun.RunInfo;


    % ---------------------------------------------------------
    % Compact result matrices
    % ---------------------------------------------------------

    Results.reg_param(i,j) = Rrun.reg_param;

    Results.C_global(i,j) = Rrun.C_global;
    Results.epsilon_global(i,j) = Rrun.epsilon_global;

    Results.cv_rmse(i,j) = Rrun.cv_rmse;
    Results.test_rmse(i,j) = Rrun.test_rmse;

    Results.k_used(i,j) = Rrun.k_used;


    % ---------------------------------------------------------
    % Timing
    % ---------------------------------------------------------

    Results.trainWallTime(i,j) = Rrun.trainWallTime;
    Results.trainCPUTime(i,j) = Rrun.trainCPUTime;

    Results.testWallTime(i,j) = Rrun.testWallTime;
    Results.testCPUTime(i,j) = Rrun.testCPUTime;

    Results.totalWallTime(i,j) = Rrun.totalWallTime;
    Results.totalCPUTime(i,j) = Rrun.totalCPUTime;


    % ---------------------------------------------------------
    % Search evaluation counts
    % ---------------------------------------------------------

    Results.nEvaluations(i,j) = Rrun.nEvaluations;
    Results.nGAEvaluations(i,j) = Rrun.nGAEvaluations;
    Results.nFallbackEvaluations(i,j) = Rrun.nFallbackEvaluations;
    Results.nValidEvaluations(i,j) = Rrun.nValidEvaluations;
    Results.nFailedEvaluations(i,j) = Rrun.nFailedEvaluations;


    % ---------------------------------------------------------
    % RBF solve counts
    % ---------------------------------------------------------

    Results.trainRBFSolves(i,j) = Rrun.trainRBFSolves;
    Results.testRBFSolves(i,j) = Rrun.testRBFSolves;
    Results.totalRBFSolves(i,j) = Rrun.totalRBFSolves;


    % ---------------------------------------------------------
    % Status
    % ---------------------------------------------------------

    Results.trainSuccess(i,j) = Rrun.trainSuccess;
    Results.testSuccess(i,j) = Rrun.testSuccess;

    Results.trainMessage{i,j} = Rrun.trainMessage;
    Results.testMessage{i,j} = Rrun.testMessage;


    % ---------------------------------------------------------
    % Store fixed CV partition once
    % ---------------------------------------------------------

    if isempty(Results.CVInfo.cvp_fixed) && ...
            ~isempty(Rrun.cvp_fixed)

        Results.CVInfo.cvp_fixed = Rrun.cvp_fixed;
        Results.CVInfo.foldID = Rrun.foldID;
        Results.CVInfo.k_used = Rrun.k_used;

    end


    % ---------------------------------------------------------
    % Store GA settings once per kernel
    % ---------------------------------------------------------

    kernelKey = lower(char(NameFunction_RBF{j}));

    if ~isfield(Results.Settings.GA,kernelKey) && ...
            ~isempty(Rrun.GA)

        Results.Settings.GA.(kernelKey) = Rrun.GA;

    end

    % % ---------------------------------------------------------
    % % Checkpoint after each completed Model x Kernel result
    % % ---------------------------------------------------------
    % 
    % if ~isempty(OutputFile)
    %     save(OutputFile,'Results','-v7.3');
    % end

end

% =============================================================
% 10. Final save
% =============================================================

if ~isempty(OutputFile)
    save(OutputFile,'Results','-v7.3');
end

end
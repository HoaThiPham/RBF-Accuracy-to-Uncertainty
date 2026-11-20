function Results = globalRBFVarianceCalibrationBatch( ...
    FitResults, ...
    BootstrapResults, ...
    opts, ...
    OutputFile)

%GLOBALRBFVARIANCECALIBRATIONBATCH
%
% Calibrate bootstrap-based predictive variance for all
% Model x Kernel global-RBF combinations.
%
% alpha and beta are estimated ONLY from training residuals and
% training bootstrap variances.
%
% The same alpha and beta are subsequently applied to testing.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 3 || isempty(opts)

    opts = ...
        struct();

end


if nargin < 4

    OutputFile = ...
        '';

end


% =============================================================
% 2. Unwrap loaded MAT structures if needed
% =============================================================

if isfield(FitResults,'Results') && ...
   isstruct(FitResults.Results) && ...
   isscalar(FitResults.Results)

    FitResults = ...
        FitResults.Results;

end


if isfield(BootstrapResults,'Results') && ...
   isstruct(BootstrapResults.Results) && ...
   isscalar(BootstrapResults.Results)

    BootstrapResults = ...
        BootstrapResults.Results;

end


% =============================================================
% 3. Lists
% =============================================================

ListOfModel = ...
    FitResults.ListOfModel;


NameFunction_RBF = ...
    FitResults.NameFunction_RBF;


if ~isequal( ...
        cellstr(string(ListOfModel)), ...
        cellstr(string(BootstrapResults.ListOfModel)))

    error( ...
        'FitResults and BootstrapResults model lists differ.');

end


if ~isequal( ...
        cellstr(string(NameFunction_RBF)), ...
        cellstr(string(BootstrapResults.NameFunction_RBF)))

    error( ...
        'FitResults and BootstrapResults kernel lists differ.');

end


nModel = ...
    numel(ListOfModel);


nKernel = ...
    numel(NameFunction_RBF);


nRun = ...
    nModel * nKernel;


% =============================================================
% 4. Output initialization
% =============================================================

Results = ...
    struct();


Results.CaseName = ...
    FitResults.CaseName;


Results.ListOfModel = ...
    ListOfModel;


Results.NameFunction_RBF = ...
    NameFunction_RBF;


Results.Settings = ...
    opts;


Results.MatrixInfo.rows = ...
    ListOfModel;


Results.MatrixInfo.columns = ...
    NameFunction_RBF;


Results.MatrixInfo.description = ...
    'Rows = input models; columns = RBF kernels';


Results.alpha = ...
    nan(nModel,nKernel);


Results.beta = ...
    nan(nModel,nKernel);


Results.Loss = ...
    nan(nModel,nKernel);


Results.Train_ASE_RMSE_Pre = ...
    nan(nModel,nKernel);


Results.Train_RMSSE_Pre = ...
    nan(nModel,nKernel);


Results.Train_ASE_RMSE = ...
    nan(nModel,nKernel);


Results.Train_RMSSE = ...
    nan(nModel,nKernel);


Results.Test_ASE_RMSE_Pre = ...
    nan(nModel,nKernel);


Results.Test_RMSSE_Pre = ...
    nan(nModel,nKernel);


Results.Test_ASE_RMSE = ...
    nan(nModel,nKernel);


Results.Test_RMSSE = ...
    nan(nModel,nKernel);


Results.wallTime = ...
    nan(nModel,nKernel);


Results.success = ...
    false(nModel,nKernel);


Results.message = ...
    cell(nModel,nKernel);


Results.Run = ...
    struct();


% =============================================================
% 5. Parallel pool
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


RunCell = ...
    cell(nRun,1);


% =============================================================
% 6. Parallel calibration
% =============================================================

tParallel = ...
    tic;


parfor irun = 1:nRun

    i = ...
        floor((irun-1)/nKernel) + 1;


    j = ...
        mod(irun-1,nKernel) + 1;


    modelName = ...
        char(ListOfModel{i});


    kernel = ...
        char(NameFunction_RBF{j});


    runName = ...
        [modelName '_' kernel];


    Rrun = ...
        struct();


    Rrun.i = ...
        i;


    Rrun.j = ...
        j;


    Rrun.runName = ...
        runName;


    Rrun.success = ...
        false;


    Rrun.message = ...
        '';


    Rrun.Calibration = ...
        [];


    tRun = ...
        tic;


    try

        % -----------------------------------------------------
        % Training target-excluded residual
        % -----------------------------------------------------

        eTrain = ...
            FitResults.Run.(runName).Train.y_diff_cv(:);


        % -----------------------------------------------------
        % Testing residual
        % -----------------------------------------------------

        eTest = ...
            FitResults.Run.(runName).Test.y_test_diff(:);


        % -----------------------------------------------------
        % Bootstrap variances
        % -----------------------------------------------------

        vTrain = ...
            BootstrapResults.Run.(runName).Train.Variance(:);


        vTest = ...
            BootstrapResults.Run.(runName).Test.Variance(:);


        % -----------------------------------------------------
        % Consistency
        % -----------------------------------------------------

        if numel(eTrain) ~= numel(vTrain)

            error( ...
                'Training residual/variance size mismatch.');

        end


        if numel(eTest) ~= numel(vTest)

            error( ...
                'Testing residual/variance size mismatch.');

        end


        % -----------------------------------------------------
        % Train calibration + test transfer
        % -----------------------------------------------------

        Rrun.Calibration = ...
            common.calibrateRBFVarianceTrainTest( ...
                eTrain, ...
                vTrain, ...
                eTest, ...
                vTest, ...
                opts);


        Rrun.success = ...
            Rrun.Calibration.success;


    catch ME

        Rrun.success = ...
            false;


        Rrun.message = ...
            ME.message;

    end


    Rrun.wallTime = ...
        toc(tRun);


    RunCell{irun} = ...
        Rrun;

end


Results.parallelWallTime = ...
    toc(tParallel);


% =============================================================
% 7. Merge
% =============================================================

for irun = 1:nRun

    Rrun = ...
        RunCell{irun};


    i = ...
        Rrun.i;


    j = ...
        Rrun.j;


    runName = ...
        Rrun.runName;


    Results.success(i,j) = ...
        Rrun.success;


    Results.message{i,j} = ...
        Rrun.message;


    Results.wallTime(i,j) = ...
        Rrun.wallTime;


    if ~Rrun.success

        continue

    end


    C = ...
        Rrun.Calibration;


    Results.alpha(i,j) = ...
        C.alpha;


    Results.beta(i,j) = ...
        C.beta;


    Results.Loss(i,j) = ...
        C.Loss;


    Results.Train_ASE_RMSE_Pre(i,j) = ...
        C.Train.ASE_RMSE_Pre;


    Results.Train_RMSSE_Pre(i,j) = ...
        C.Train.RMSSE_Pre;


    Results.Train_ASE_RMSE(i,j) = ...
        C.Train.ASE_RMSE;


    Results.Train_RMSSE(i,j) = ...
        C.Train.RMSSE;


    Results.Test_ASE_RMSE_Pre(i,j) = ...
        C.Test.ASE_RMSE_Pre;


    Results.Test_RMSSE_Pre(i,j) = ...
        C.Test.RMSSE_Pre;


    Results.Test_ASE_RMSE(i,j) = ...
        C.Test.ASE_RMSE;


    Results.Test_RMSSE(i,j) = ...
        C.Test.RMSSE;


    Results.Run.(runName) = ...
        C;

end


% =============================================================
% 8. Final save
% =============================================================

if ~isempty(OutputFile)

    outputDir = ...
        fileparts(OutputFile);


    if ~isempty(outputDir) && ...
       ~exist(outputDir,'dir')

        mkdir(outputDir);

    end


    save( ...
        OutputFile, ...
        'Results', ...
        '-v7.3');

end


end
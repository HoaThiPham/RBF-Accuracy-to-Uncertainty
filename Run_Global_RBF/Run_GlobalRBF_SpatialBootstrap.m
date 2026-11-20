OutputDir='OutPut_RBF_2026'


nBootstrap = 200;
verbose = true;
RNG_SEED = 1;

bootstrapOptions = struct();

% Keep inner fold messages off for faster parallel execution.
bootstrapOptions.Verbose = false;

bootstrapOptions.MaxAttemptsFactor = 10;

bootstrapOptions.NumSpatialSegments = 5;   % exact current bootstrap
bootstrapOptions.MinPointsPerAnnulus = 15;


% Seeds are saved, so storing every bootstrap index is normally
% unnecessary and can substantially increase MAT-file size.
bootstrapOptions.StoreIndices = false;


%% Case 1: 
load('OutPutSplitTrainTest\InputData_Case1.mat','InputData_Case1');
ResultsCase1=load('OutPut_RBF_2026/GlobalRBF_Case1_Full_TuneReg_4')
ResultsCase1=ResultsCase1.Results;




BootstrapCase1 = ...
    globalrbf.globalRBFBootstrapBatchSummaryParallel( ...
        InputData_Case1, ...
        ResultsCase1, ...
        nBootstrap, ...
        verbose, ...
        RNG_SEED, ...
        bootstrapOptions, ...
        fullfile(OutputDir,'Case1_SpatialBootstrap.mat'));




%% Case 2: 

load('OutPutSplitTrainTest\InputData_Case2.mat','InputData_Case2');
 ResultsCase2=load('OutPut_RBF_2026/GlobalRBF_Case2_Full_TuneReg_4')
ResultsCase2=ResultsCase2.Results;

BootstrapCase2 = ...
    globalrbf.globalRBFBootstrapBatchSummaryParallel( ...
        InputData_Case2, ...
        ResultsCase2, ...
        nBootstrap, ...
        verbose, ...
        RNG_SEED, ...
        bootstrapOptions, ...
        fullfile(OutputDir,'Case2_SpatialBootstrap.mat'));
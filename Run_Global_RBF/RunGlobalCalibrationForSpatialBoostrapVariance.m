
OutputDir='OutPut_RBF_2026'
optsCalibration =   struct();
optsCalibration.gamma1 =  0.5;
optsCalibration.gamma2 =     0.5;
optsCalibration.max_gen =   500;
optsCalibration.pop_size =     500;

optsCalibration.verbose =  false;




optsCalibration.boundary_tol =  0.02;
%%  Case 1

optsCalibration.alpha_bounds = [0 0.01];

optsCalibration.beta_bounds  = [1 3.5];


ResultsCase1=load('OutPut_RBF_2026/GlobalRBF_Case1_Full_TuneReg_4')
ResultsCase1=ResultsCase1.Results;

BootstrapCase1=load('OutPut_RBF_2026/Case1_SpatialBootstrap')
BootstrapCase1=BootstrapCase1.Results;

CalibrationCase1 = ...
    globalrbf.globalRBFVarianceCalibrationBatch( ...
        ResultsCase1, ...
        BootstrapCase1, ...
        optsCalibration, ...
        fullfile( ...
            OutputDir, ...
            'Case1_SpatialVarianceCalibration.mat'));

SummaryCase1 = ...
    common.summarizeCalibrationAlphaBeta( ...
        CalibrationCase1);

%%  Case 2

optsCalibration.alpha_bounds = [0 0.01];

optsCalibration.beta_bounds  = [1 3.5];

ResultsCase2=load('OutPut_RBF_2026/GlobalRBF_Case2_Full_TuneReg_4')
ResultsCase2=ResultsCase2.Results;

BootstrapCase2=load('OutPut_RBF_2026/Case2_SpatialBootstrap')
BootstrapCase2=BootstrapCase2.Results;

CalibrationCase2 = ...
    globalrbf.globalRBFVarianceCalibrationBatch( ...
        ResultsCase2, ...
        BootstrapCase2, ...
        optsCalibration, ...
        fullfile( ...
            OutputDir, ...
            'Case2_SpatialVarianceCalibration.mat'));

SummaryCase2 = ...
    common.summarizeCalibrationAlphaBeta( ...
        CalibrationCase2);

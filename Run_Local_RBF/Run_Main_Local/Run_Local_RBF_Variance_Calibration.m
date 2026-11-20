%% ========================================================================
% LOCAL RBF VARIANCE CALIBRATION
% Mirrors the Global-RBF calibration workflow.
% Training calibration uses the residual provenance stored in the local
% bootstrap result (including pseudo-unseen mode when applicable).
% ========================================================================

clearvars
clc

%% ========================================================================
% USER INPUTS
% ========================================================================

MainResultDir = ...
    'OutPut_RBF_2026_2';

BootstrapDir = ...
    fullfile( ...
        'OutPut_RBF_2026_2', ...
        'LocalBootstrap_pseudo_unseen_B200_S0');

OutputDir = ...
    fullfile( ...
        'OutPut_RBF_2026_2', ...
        'LocalVarianceCalibration_pseudo_unseen_B200_S0');

% Set these two files to the final Local-RBF fit results used to generate
% the corresponding bootstrap results.
ResultsFileCase1 = ...
    fullfile(MainResultDir,'LocalRBF_Case1_Full_TuneReg.mat');

ResultsFileCase2 = ...
    fullfile(MainResultDir,'LocalRBF_Case2_Full_TuneReg.mat');

BootstrapFileCase1 = ...
    fullfile(BootstrapDir,'Case1_Local_Bootstrap.mat');

BootstrapFileCase2 = ...
    fullfile(BootstrapDir,'Case2_Local_Bootstrap.mat');

if ~isfolder(OutputDir)
    mkdir(OutputDir);
end

%% ========================================================================
% CALIBRATION OPTIONS
% ========================================================================

optsCalibration = struct();

optsCalibration.gamma1 = 0.5;
optsCalibration.gamma2 = 0.5;
optsCalibration.max_gen = 500;
optsCalibration.pop_size = 500;
optsCalibration.verbose = false;
optsCalibration.boundary_tol = 0.02;

optsCalibration.alpha_bounds = [0 0.01];
optsCalibration.beta_bounds  = [1 3.5];

%% ========================================================================
% CASE 1
% ========================================================================

S = load(ResultsFileCase1,'Results');
ResultsCase1 = S.Results;
clear S

S = load(BootstrapFileCase1,'Results');
BootstrapCase1 = S.Results;
clear S

CalibrationCase1 = ...
    localrbf.localRBFVarianceCalibrationBatch( ...
        ResultsCase1, ...
        BootstrapCase1, ...
        optsCalibration, ...
        fullfile( ...
            OutputDir, ...
            'Case1_Local_VarianceCalibration.mat'));

SummaryCase1 = ...
    common.summarizeCalibrationAlphaBeta( ...
        CalibrationCase1);

% Release large inputs before Case 2.
clear ResultsCase1 BootstrapCase1

%% ========================================================================
% CASE 2
% ========================================================================
optsCalibration.alpha_bounds = [0 0.01];
optsCalibration.beta_bounds  = [1 3.5];

S = load(ResultsFileCase2,'Results');
ResultsCase2 = S.Results;
clear S

S = load(BootstrapFileCase2,'Results');
BootstrapCase2 = S.Results;
clear S

CalibrationCase2 = ...
    localrbf.localRBFVarianceCalibrationBatch( ...
        ResultsCase2, ...
        BootstrapCase2, ...
        optsCalibration, ...
        fullfile( ...
            OutputDir, ...
            'Case2_Local_VarianceCalibration.mat'));

SummaryCase2 = ...
    common.summarizeCalibrationAlphaBeta( ...
        CalibrationCase2);

clear ResultsCase2 BootstrapCase2

%% ========================================================================
% FINISHED
% ========================================================================

fprintf('\n============================================================\n');
fprintf(' LOCAL RBF VARIANCE CALIBRATION FINISHED\n');
fprintf(' OutputDir = %s\n',OutputDir);
fprintf('============================================================\n');

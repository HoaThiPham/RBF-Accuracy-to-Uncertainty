%% ============================================================
% GLOBAL RBF FULL PRODUCTION BATCH
%% ============================================================

clearvars
clc

% Restart pool once so workers use the final validated functions.
delete(gcp('nocreate'));
clear functions
rehash


% =============================================================
% 1. Load COMPLETE train/test dataset
% =============================================================

load( ...
    'OutPutSplitTrainTest\InputData_Case2.mat', ...
    'InputData_Case2');

InputData = ...
    InputData_Case2;

CaseName = ...
    'Case2';


% =============================================================
% 2. Models
% =============================================================

ListOfModel = {
    'EGM2008'
    'ggmplus'
    'eigen6c4'
    'geco'
    'sgg2'
    'xgm19'
    };


% =============================================================
% 3. RBF kernels
% =============================================================

NameFunction_RBF = {
    'gaussian'
    'multiquadric'
    'thinplate'
    'cubic'
    'linear'
    };

method = ...
    'ga';


% =============================================================
% 4. Shape-parameter search bounds
%
% Shape kernels:
%   Gaussian
%   Multiquadric
%
% Shape-free kernels use C = 1 internally.
% =============================================================

CMin = struct();
CMax = struct();


CMin.gaussian = 1e-3;
CMax.gaussian = 1.0;

CMin.multiquadric = 1e-3;
CMax.multiquadric = 1.0;



% Shape-free kernels
CMin.linear = 1;
CMax.linear = 1;

CMin.cubic = 1;
CMax.cubic = 1;

CMin.thinplate = 1;
CMax.thinplate = 1;


% =============================================================
% 5. Regularization search bounds
% =============================================================

RegMin = struct();
RegMax = struct();



% ============================================================
% Regularization search range
% ============================================================

REG_MIN_COMMON = 1e-6;
REG_MAX_COMMON = 1;

RegMin.gaussian     = REG_MIN_COMMON;
RegMax.gaussian     = REG_MAX_COMMON;

RegMin.multiquadric = REG_MIN_COMMON;
RegMax.multiquadric = REG_MAX_COMMON;

RegMin.linear       = REG_MIN_COMMON;
RegMax.linear       = REG_MAX_COMMON;

RegMin.cubic        = REG_MIN_COMMON;
RegMax.cubic        = REG_MAX_COMMON;

RegMin.thinplate    = REG_MIN_COMMON;
RegMax.thinplate    = REG_MAX_COMMON;

% =============================================================
% 6. CV and reproducibility
% =============================================================

% Use your intended production number of folds.
kfolds = ...
    10;          % change only if your study uses another value

verbose = ...
    true;

RNG_SEED = ...
    11;


% =============================================================
% 7. Solver settings
% =============================================================

options = ...
    struct();

options.solver = ...
    struct();


% Production threshold -- keep validated value.
options.solver.cond_lsq = ...
    1e12;


% OFF during GA/fallback.
% Final selected CV diagnostics are enabled internally.
options.solver.compute_rel_residual = ...
    false;


% =============================================================
% 8. Output file
% =============================================================

OutputFile = ...
    fullfile( ...
        'OutPut_RBF_2026', ...
        'GlobalRBF_Case2_Full_TuneReg_4.mat');


% Optional: remove old output before a genuinely fresh production run.
if isfile(OutputFile)

    delete(OutputFile);

end


% =============================================================
% 9. Run FULL parallel batch
% =============================================================

fprintf('\n=============================================\n');
fprintf(' GLOBAL RBF FULL PRODUCTION BATCH\n');
fprintf('=============================================\n');

Results = ...
    globalrbf.globalRBFBatchSummaryParallel_TuneReg( ...
        InputData, ...
        CaseName, ...
        ListOfModel, ...
        NameFunction_RBF, ...
        method, ...
        CMin, ...
        CMax, ...
        kfolds, ...
        RegMin, ...
        RegMax, ...
        verbose, ...
        RNG_SEED, ...
        options, ...
        OutputFile);


fprintf('\n=============================================\n');
fprintf(' GLOBAL RBF FULL PRODUCTION BATCH FINISHED\n');
fprintf('=============================================\n');
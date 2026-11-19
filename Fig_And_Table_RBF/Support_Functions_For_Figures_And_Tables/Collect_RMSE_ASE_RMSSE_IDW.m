%% ========================================================================
% GATHER TQ METRICS - IDW
%
% Gather final calibrated LOCAL-RBF:
%   RMSE
%   ASE
%   RMSSE
%
% for:
%   Case 1 / Case 2
%   Train / Test
%
% Transfer rule: IDW
%
% GLOBAL is intentionally removed.
% ========================================================================

clear
clc


%% ========================================================================
% 1. USER INPUT
% ========================================================================

RootDir = '.';


% ------------------------------------------------------------------------
% LOCAL CALIBRATION - IDW
% ------------------------------------------------------------------------

LocalDir = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'LocalVarianceCalibration_pseudo_unseen_B200_S0_IDW');

LocalFileCase1 = ...
    fullfile( ...
        LocalDir, ...
        'Case1_Local_VarianceCalibration_IDW.mat');

LocalFileCase2 = ...
    fullfile( ...
        LocalDir, ...
        'Case2_Local_VarianceCalibration_IDW.mat');


% ------------------------------------------------------------------------
% OUTPUT - IDW
% ------------------------------------------------------------------------

OutputDir = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'TQ_IDW');

OutputFile = ...
    fullfile( ...
        OutputDir, ...
        'RMSE_ASE_RMSSE_Metrix_IDW.mat');


%% ========================================================================
% 2. MODELS AND KERNELS
%
% SGG1 is not included.
% ========================================================================

Models = { ...
    'EGM2008', ...
    'ggmplus', ...
    'eigen6c4', ...
    'geco', ...
    'sgg2', ...
    'xgm19'};

ModelShort = { ...
    'E08', ...
    'GGP', ...
    'EC4', ...
    'GCO', ...
    'SG2', ...
    'X19'};

Kernels = { ...
    'gaussian', ...
    'multiquadric', ...
    'thinplate', ...
    'cubic', ...
    'linear'};

KernelShort = { ...
    'G', ...
    'M', ...
    'T', ...
    'C', ...
    'L'};


nModel = ...
    numel(Models);

nKernel = ...
    numel(Kernels);


%% ========================================================================
% 3. LOAD FINAL LOCAL CALIBRATION RESULTS - IDW
% ========================================================================

S = ...
    load( ...
        LocalFileCase1, ...
        'Results');

LocalCase1 = ...
    S.Results;

clear S


S = ...
    load( ...
        LocalFileCase2, ...
        'Results');

LocalCase2 = ...
    S.Results;

clear S


%% ========================================================================
% 4. INITIALISE MATRICES
%
% rows    = EGM models
% columns = RBF kernels
% ========================================================================

RMSE_ASE_RMSSE_Metrix = ...
    struct();


Phases = { ...
    'Train', ...
    'Test'};

Cases = { ...
    'Case1', ...
    'Case2'};


for ip = 1:numel(Phases)

    for ic = 1:numel(Cases)

        RMSE_ASE_RMSSE_Metrix.(Phases{ip}). ...
            (Cases{ic}).Local.RMSE = ...
            nan(nModel,nKernel);

        RMSE_ASE_RMSSE_Metrix.(Phases{ip}). ...
            (Cases{ic}).Local.ASE = ...
            nan(nModel,nKernel);

        RMSE_ASE_RMSSE_Metrix.(Phases{ip}). ...
            (Cases{ic}).Local.RMSSE = ...
            nan(nModel,nKernel);

    end

end


%% ========================================================================
% 5. GATHER LOCAL ONLY - IDW
% ========================================================================

Source = { ...
    LocalCase1, 'Case1'; ...
    LocalCase2, 'Case2'};


for isource = 1:size(Source,1)

    Results = ...
        Source{isource,1};

    CaseName = ...
        Source{isource,2};


    for i = 1:nModel

        for j = 1:nKernel

            runName = ...
                [Models{i} '_' Kernels{j}];

            assert( ...
                isfield(Results.Run,runName), ...
                'Missing run: %s', ...
                runName);

            R = ...
                Results.Run.(runName);


            % -------------------------------------------------------------
            % TRAIN
            % -------------------------------------------------------------

            RMSE_ASE_RMSSE_Metrix.Train.(CaseName).Local.RMSE(i,j) = ...
                R.Train.RMSE;

            RMSE_ASE_RMSSE_Metrix.Train.(CaseName).Local.ASE(i,j) = ...
                R.Train.ASE;

            RMSE_ASE_RMSSE_Metrix.Train.(CaseName).Local.RMSSE(i,j) = ...
                R.Train.RMSSE;


            % -------------------------------------------------------------
            % TEST
            % -------------------------------------------------------------

            RMSE_ASE_RMSSE_Metrix.Test.(CaseName).Local.RMSE(i,j) = ...
                R.Test.RMSE;

            RMSE_ASE_RMSSE_Metrix.Test.(CaseName).Local.ASE(i,j) = ...
                R.Test.ASE;

            RMSE_ASE_RMSSE_Metrix.Test.(CaseName).Local.RMSSE(i,j) = ...
                R.Test.RMSSE;

        end

    end

end


%% ========================================================================
% 6. LABELS
% ========================================================================

RMSE_ASE_RMSSE_Metrix.Models = ...
    Models;

RMSE_ASE_RMSSE_Metrix.ModelShort = ...
    ModelShort;

RMSE_ASE_RMSSE_Metrix.Kernels = ...
    Kernels;

RMSE_ASE_RMSSE_Metrix.KernelShort = ...
    KernelShort;

RMSE_ASE_RMSSE_Metrix.TransferRule = ...
    'IDW';


%% ========================================================================
% 7. KRIGING
%
% Kept from the original script. GLOBAL RBF has been removed; this section
% only appends the same Kriging metrics if Add_Krigging_RMSE_ASE_RMSSSE_To_RBF
% is available in the MATLAB path.
% ========================================================================

Add_Krigging_RMSE_ASE_RMSSSE_To_RBF


%% ========================================================================
% 8. SAVE
% ========================================================================

if ~isfolder(OutputDir)

    mkdir(OutputDir);

end


save( ...
    OutputFile, ...
    'RMSE_ASE_RMSSE_Metrix');


%% ========================================================================
% 9. CHECK
% ========================================================================

fprintf('\n=============================================================\n');
fprintf(' TQ METRICS GATHERED - IDW\n');
fprintf('=============================================================\n');

fprintf('Models  = %d\n',nModel);
fprintf('Kernels = %d\n',nKernel);

fprintf('\nCase 1 Local Test RMSE:\n');
disp(RMSE_ASE_RMSSE_Metrix.Test.Case1.Local.RMSE);

fprintf('\nCase 1 Local Test ASE:\n');
disp(RMSE_ASE_RMSSE_Metrix.Test.Case1.Local.ASE);

fprintf('\nCase 1 Local Test RMSSE:\n');
disp(RMSE_ASE_RMSSE_Metrix.Test.Case1.Local.RMSSE);

fprintf('\nOutput = %s\n',OutputFile);

fprintf('=============================================================\n');

function Summary = summarizeCalibrationAlphaBeta(C)
%% ============================================================
% SUMMARIZE CALIBRATION ALPHA AND BETA
%
% Common function for all calibration cases.
%
% Calibration equation:
%
%     sigma_i^2 = beta * sigma_boot,i^2 + alpha
%
% Input:
%     C
%       Calibration result structure.
%
% Output:
%     Summary
%       Summary tables, raw values and boundary diagnostics.
%
% Example:
%
%     SummaryCase1 = ...
%         common.summarizeCalibrationAlphaBeta( ...
%             CalibrationCase1);
%
%     SummaryCase2 = ...
%         common.summarizeCalibrationAlphaBeta( ...
%             CalibrationCase2);
%
% =============================================================


% =============================================================
% 0. Unwrap loaded MAT-file structure if necessary
% =============================================================

if isfield(C,'Results') && ...
        isstruct(C.Results) && ...
        isscalar(C.Results)

    C = ...
        C.Results;

end


% =============================================================
% 1. Validate calibration structure
% =============================================================

assert( ...
    isstruct(C) && ...
    isscalar(C), ...
    'C must be a scalar calibration structure.');


assert( ...
    isfield(C,'Run'), ...
    'Calibration structure does not contain Run.');


assert( ...
    isfield(C,'ListOfModel'), ...
    'Calibration structure does not contain ListOfModel.');


assert( ...
    isfield(C,'NameFunction_RBF'), ...
    'Calibration structure does not contain NameFunction_RBF.');


assert( ...
    isfield(C,'success'), ...
    'Calibration structure does not contain success.');


assert( ...
    all(C.success(:)), ...
    'One or more calibration runs failed.');


% =============================================================
% 2. Basic information
% =============================================================

ListOfModel = ...
    C.ListOfModel;


NameFunction_RBF = ...
    C.NameFunction_RBF;


nModel = ...
    numel(ListOfModel);


nKernel = ...
    numel(NameFunction_RBF);


nRun = ...
    nModel * nKernel;


if isfield(C,'CaseName')

    caseName = ...
        char(C.CaseName);

else

    caseName = ...
        'Calibration';

end


assert( ...
    numel(C.success) == nRun, ...
    'Size of success does not match Model x Kernel combinations.');


% =============================================================
% 3. Preallocate
% =============================================================

Model = ...
    strings(nRun,1);


Kernel = ...
    strings(nRun,1);


Alpha = ...
    nan(nRun,1);


Beta = ...
    nan(nRun,1);


Loss = ...
    nan(nRun,1);


NearAlphaLower = ...
    false(nRun,1);


NearAlphaUpper = ...
    false(nRun,1);


NearBetaLower = ...
    false(nRun,1);


NearBetaUpper = ...
    false(nRun,1);


% =============================================================
% 4. Extract results
% =============================================================

k = ...
    0;


for i = 1:nModel

    modelName = ...
        char(ListOfModel{i});


    for j = 1:nKernel

        kernel = ...
            char(NameFunction_RBF{j});


        runName = ...
            [modelName '_' kernel];


        assert( ...
            isfield(C.Run,runName), ...
            'Calibration run "%s" is missing.', ...
            runName);


        R = ...
            C.Run.(runName);


        assert( ...
            isfield(R,'success') && ...
            R.success, ...
            'Calibration run "%s" reports failure.', ...
            runName);


        assert( ...
            isfield(R,'alpha') && ...
            isfield(R,'beta') && ...
            isfield(R,'Loss'), ...
            'Calibration parameters are missing for "%s".', ...
            runName);


        assert( ...
            isfield(R,'Optimization'), ...
            'Optimization diagnostics are missing for "%s".', ...
            runName);


        k = ...
            k + 1;


        Model(k) = ...
            string(modelName);


        Kernel(k) = ...
            string(kernel);


        Alpha(k) = ...
            R.alpha;


        Beta(k) = ...
            R.beta;


        Loss(k) = ...
            R.Loss;


        NearAlphaLower(k) = ...
            R.Optimization.nearAlphaLower;


        NearAlphaUpper(k) = ...
            R.Optimization.nearAlphaUpper;


        NearBetaLower(k) = ...
            R.Optimization.nearBetaLower;


        NearBetaUpper(k) = ...
            R.Optimization.nearBetaUpper;

    end

end


% =============================================================
% 5. Numerical validity
% =============================================================

assert( ...
    all(isfinite(Alpha)) && ...
    all(Alpha >= 0), ...
    'Invalid alpha value found.');


assert( ...
    all(isfinite(Beta)) && ...
    all(Beta > 0), ...
    'Invalid beta value found.');


assert( ...
    all(isfinite(Loss)) && ...
    all(Loss >= 0), ...
    'Invalid calibration loss found.');


% =============================================================
% 6. TABLE: ALPHA AND BETA
%
% Alpha and Beta:
%     displayed to 3 decimal places
%
% Loss:
%     scientific notation to avoid displaying small values as 0
% =============================================================

Loss_Display = ...
    compose('%.3e',Loss);


T_AlphaBeta = ...
    table( ...
        Model, ...
        Kernel, ...
        round(Alpha,4), ...
        round(Beta,4), ...
        Loss_Display, ...
        'VariableNames', ...
        { ...
        'Model', ...
        'Kernel', ...
        'Alpha', ...
        'Beta', ...
        'Loss' ...
        });


fprintf('\n============================================================\n');

fprintf( ...
    '%s: ALPHA AND BETA FOR ALL MODEL x KERNEL COMBINATIONS\n', ...
    upper(caseName));

fprintf('============================================================\n');


disp(T_AlphaBeta);


% =============================================================
% 7. OVERALL SUMMARY
% =============================================================

fprintf('\n============================================================\n');

fprintf( ...
    '%s: OVERALL SUMMARY\n', ...
    upper(caseName));

fprintf('============================================================\n');


fprintf('\nALPHA\n');


fprintf( ...
    'Range  : %.4f - %.4f\n', ...
    min(Alpha), ...
    max(Alpha));


fprintf( ...
    'Mean   : %.4f\n', ...
    mean(Alpha));


fprintf( ...
    'Median : %.4f\n', ...
    median(Alpha));


fprintf('\nBETA\n');


fprintf( ...
    'Range  : %.4f - %.4f\n', ...
    min(Beta), ...
    max(Beta));


fprintf( ...
    'Mean   : %.4f\n', ...
    mean(Beta));


fprintf( ...
    'Median : %.4f\n', ...
    median(Beta));


fprintf('\nLOSS\n');


fprintf( ...
    'Range  : %.3e - %.3e\n', ...
    min(Loss), ...
    max(Loss));


fprintf( ...
    'Mean   : %.3e\n', ...
    mean(Loss));


fprintf( ...
    'Median : %.3e\n', ...
    median(Loss));


% =============================================================
% 8. SUMMARY BY KERNEL
% =============================================================

fprintf('\n============================================================\n');

fprintf( ...
    '%s: SUMMARY BY KERNEL\n', ...
    upper(caseName));

fprintf('============================================================\n');


for j = 1:nKernel

    kernel = ...
        string(NameFunction_RBF{j});


    idx = ...
        Kernel == kernel;


    fprintf( ...
        '\n%s\n', ...
        upper(kernel));


    fprintf( ...
        'Alpha range  : %.4f - %.4f\n', ...
        min(Alpha(idx)), ...
        max(Alpha(idx)));


    fprintf( ...
        'Alpha mean   : %.4f\n', ...
        mean(Alpha(idx)));


    fprintf( ...
        'Beta range   : %.4f - %.4f\n', ...
        min(Beta(idx)), ...
        max(Beta(idx)));


    fprintf( ...
        'Beta mean    : %.4f\n', ...
        mean(Beta(idx)));


    fprintf( ...
        'Loss range   : %.3e - %.3e\n', ...
        min(Loss(idx)), ...
        max(Loss(idx)));


    fprintf( ...
        'Loss mean    : %.3e\n', ...
        mean(Loss(idx)));

end


% =============================================================
% 9. COMPACT KERNEL SUMMARY TABLE
% =============================================================

KernelSummary = ...
    strings(nKernel,1);


Alpha_Min = ...
    nan(nKernel,1);


Alpha_Max = ...
    nan(nKernel,1);


Alpha_Mean = ...
    nan(nKernel,1);


Beta_Min = ...
    nan(nKernel,1);


Beta_Max = ...
    nan(nKernel,1);


Beta_Mean = ...
    nan(nKernel,1);


Loss_Min = ...
    nan(nKernel,1);


Loss_Max = ...
    nan(nKernel,1);


Loss_Mean = ...
    nan(nKernel,1);


for j = 1:nKernel

    kernel = ...
        string(NameFunction_RBF{j});


    idx = ...
        Kernel == kernel;


    KernelSummary(j) = ...
        kernel;


    Alpha_Min(j) = ...
        min(Alpha(idx));


    Alpha_Max(j) = ...
        max(Alpha(idx));


    Alpha_Mean(j) = ...
        mean(Alpha(idx));


    Beta_Min(j) = ...
        min(Beta(idx));


    Beta_Max(j) = ...
        max(Beta(idx));


    Beta_Mean(j) = ...
        mean(Beta(idx));


    Loss_Min(j) = ...
        min(Loss(idx));


    Loss_Max(j) = ...
        max(Loss(idx));


    Loss_Mean(j) = ...
        mean(Loss(idx));

end


T_KernelSummary = ...
    table( ...
        KernelSummary, ...
        round(Alpha_Min,4), ...
        round(Alpha_Max,4), ...
        round(Alpha_Mean,4), ...
        round(Beta_Min,4), ...
        round(Beta_Max,4), ...
        round(Beta_Mean,4), ...
        compose('%.3e',Loss_Min), ...
        compose('%.3e',Loss_Max), ...
        compose('%.3e',Loss_Mean), ...
        'VariableNames', ...
        { ...
        'Kernel', ...
        'Alpha_Min', ...
        'Alpha_Max', ...
        'Alpha_Mean', ...
        'Beta_Min', ...
        'Beta_Max', ...
        'Beta_Mean', ...
        'Loss_Min', ...
        'Loss_Max', ...
        'Loss_Mean' ...
        });


fprintf('\n============================================================\n');

fprintf( ...
    '%s: KERNEL SUMMARY TABLE\n', ...
    upper(caseName));

fprintf('============================================================\n');


disp(T_KernelSummary);


% =============================================================
% 10. BOUNDARY SUMMARY
% =============================================================

fprintf('\n============================================================\n');

fprintf( ...
    '%s: BOUNDARY SUMMARY\n', ...
    upper(caseName));

fprintf('============================================================\n');


fprintf( ...
    'Alpha near lower bound : %d/%d\n', ...
    nnz(NearAlphaLower), ...
    nRun);


fprintf( ...
    'Alpha near upper bound : %d/%d\n', ...
    nnz(NearAlphaUpper), ...
    nRun);


fprintf( ...
    'Beta near lower bound  : %d/%d\n', ...
    nnz(NearBetaLower), ...
    nRun);


fprintf( ...
    'Beta near upper bound  : %d/%d\n', ...
    nnz(NearBetaUpper), ...
    nRun);


% =============================================================
% 11. Store output
% =============================================================

Summary = ...
    struct();


Summary.CaseName = ...
    caseName;


Summary.nModel = ...
    nModel;


Summary.nKernel = ...
    nKernel;


Summary.nRun = ...
    nRun;


Summary.Model = ...
    Model;


Summary.Kernel = ...
    Kernel;


Summary.Alpha = ...
    Alpha;


Summary.Beta = ...
    Beta;


Summary.Loss = ...
    Loss;


Summary.Table = ...
    T_AlphaBeta;


Summary.KernelSummaryTable = ...
    T_KernelSummary;


% -------------------------------------------------------------
% Overall statistics
% -------------------------------------------------------------

Summary.AlphaRange = ...
    [min(Alpha),max(Alpha)];


Summary.AlphaMean = ...
    mean(Alpha);


Summary.AlphaMedian = ...
    median(Alpha);


Summary.BetaRange = ...
    [min(Beta),max(Beta)];


Summary.BetaMean = ...
    mean(Beta);


Summary.BetaMedian = ...
    median(Beta);


Summary.LossRange = ...
    [min(Loss),max(Loss)];


Summary.LossMean = ...
    mean(Loss);


Summary.LossMedian = ...
    median(Loss);


% -------------------------------------------------------------
% Kernel statistics
% -------------------------------------------------------------

Summary.Kernel = ...
    Kernel;


Summary.KernelSummary = ...
    struct();


Summary.KernelSummary.Name = ...
    KernelSummary;


Summary.KernelSummary.Alpha_Min = ...
    Alpha_Min;


Summary.KernelSummary.Alpha_Max = ...
    Alpha_Max;


Summary.KernelSummary.Alpha_Mean = ...
    Alpha_Mean;


Summary.KernelSummary.Beta_Min = ...
    Beta_Min;


Summary.KernelSummary.Beta_Max = ...
    Beta_Max;


Summary.KernelSummary.Beta_Mean = ...
    Beta_Mean;


Summary.KernelSummary.Loss_Min = ...
    Loss_Min;


Summary.KernelSummary.Loss_Max = ...
    Loss_Max;


Summary.KernelSummary.Loss_Mean = ...
    Loss_Mean;


% -------------------------------------------------------------
% Boundary diagnostics
% -------------------------------------------------------------

Summary.NearAlphaLower = ...
    NearAlphaLower;


Summary.NearAlphaUpper = ...
    NearAlphaUpper;


Summary.NearBetaLower = ...
    NearBetaLower;


Summary.NearBetaUpper = ...
    NearBetaUpper;


Summary.nNearAlphaLower = ...
    nnz(NearAlphaLower);


Summary.nNearAlphaUpper = ...
    nnz(NearAlphaUpper);


Summary.nNearBetaLower = ...
    nnz(NearBetaLower);


Summary.nNearBetaUpper = ...
    nnz(NearBetaUpper);

end
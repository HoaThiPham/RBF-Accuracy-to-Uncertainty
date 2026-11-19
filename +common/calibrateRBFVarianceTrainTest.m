function Result = calibrateRBFVarianceTrainTest( ...
    residualTrain, ...
    varianceTrain, ...
    residualTest, ...
    varianceTest, ...
    opts)

%CALIBRATERBFVARIANCETRAINTEST
%
% Estimate alpha and beta using TRAINING data only:
%
%       sigma_i^2 = beta * (v_i + alpha)
%
% The SAME training-derived alpha and beta are then applied to
% testing.
%
% This function is common to global and local RBF methods.


% =============================================================
% 1. Standardize inputs
% =============================================================

residualTrain = ...
    residualTrain(:);


varianceTrain = ...
    varianceTrain(:);


residualTest = ...
    residualTest(:);


varianceTest = ...
    varianceTest(:);


% =============================================================
% 2. Basic dimensional consistency
%
% Detailed numerical validation is performed inside the common
% calibration/evaluation functions, so it is not duplicated here.
% =============================================================

if numel(residualTrain) ~= numel(varianceTrain)

    error( ...
        'Training residual and bootstrap variance lengths differ.');

end


if numel(residualTest) ~= numel(varianceTest)

    error( ...
        'Testing residual and bootstrap variance lengths differ.');

end


if isempty(residualTrain)

    error( ...
        'Training calibration data cannot be empty.');

end


if isempty(residualTest)

    error( ...
        'Testing evaluation data cannot be empty.');

end


% =============================================================
% 3. Estimate alpha and beta from TRAINING ONLY
% =============================================================

Calibration = ...
    common.calibrateBootstrapVariance( ...
        residualTrain, ...
        varianceTrain, ...
        opts);


if ~Calibration.success

    error( ...
        'Training variance calibration failed.');

end


% =============================================================
% 4. Evaluate TEST using SAME training-derived alpha and beta
%
% No test quantity is used to estimate alpha or beta.
% =============================================================

Test = ...
    common.evaluateBootstrapVarianceCalibration( ...
        residualTest, ...
        varianceTest, ...
        Calibration.alpha, ...
        Calibration.beta);


% =============================================================
% 5. Output
% =============================================================

Result = ...
    struct();


% Calibration parameters

Result.alpha = ...
    Calibration.alpha;


Result.beta = ...
    Calibration.beta;


% Eq. (18) objective

Result.Loss = ...
    Calibration.Loss;


Result.LossCheck = ...
    Calibration.LossCheck;


Result.gamma1 = ...
    Calibration.gamma1;


Result.gamma2 = ...
    Calibration.gamma2;


% Training and testing diagnostics

Result.Train = ...
    Calibration.Train;


Result.Test = ...
    Test;


% =============================================================
% 6. Optimization diagnostics
% =============================================================

Result.Optimization = ...
    struct();


Result.Optimization.alpha_bounds = ...
    Calibration.alpha_bounds;


Result.Optimization.beta_bounds = ...
    Calibration.beta_bounds;


Result.Optimization.alphaScale = ...
    Calibration.alphaScale;


Result.Optimization.nearAlphaLower = ...
    Calibration.nearAlphaLower;


Result.Optimization.nearAlphaUpper = ...
    Calibration.nearAlphaUpper;


Result.Optimization.nearBetaLower = ...
    Calibration.nearBetaLower;


Result.Optimization.nearBetaUpper = ...
    Calibration.nearBetaUpper;


Result.Optimization.exitflag = ...
    Calibration.exitflag;


Result.Optimization.generations = ...
    Calibration.generations;


Result.Optimization.funccount = ...
    Calibration.funccount;


Result.Optimization.wallTime = ...
    Calibration.wallTime;


% =============================================================
% 7. Success
% =============================================================

Result.success = ...
    Calibration.success && ...
    all(isfinite(Test.VarianceCalibrated)) && ...
    all(Test.VarianceCalibrated > 0);


end
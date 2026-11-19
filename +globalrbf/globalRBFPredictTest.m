function [y_test_pred, y_test_diff, test_rmse, ...
          epsilon_global, modelInfo] = ...
    globalRBFPredictTest( ...
        X, y, X_test, y_test, trainingInfo)

%GLOBALRBFPREDICTTEST
%
% Fit the selected global RBF model using the COMPLETE training dataset,
% predict the testing points, and calculate testing residuals and RMSE.
%
% All model settings required for prediction are transferred directly
% from globalRBFTrainingCV_TuneReg through trainingInfo.
%
% Testing observations y_test are used ONLY after prediction to evaluate
% testing accuracy. They do not influence model fitting.
%
%
% INPUTS
% ------
% X
%       [n x d] raw training coordinates.
%
% y
%       [n x 1] training observations.
%
% X_test
%       [nt x d] raw testing coordinates.
%
% y_test
%       [nt x 1] observed testing values.
%       Used only for final testing assessment.
% trainingInfo
% Structure returned by globalRBFTrainingCV_TuneReg.
%
%       Common fields used here:
%
%           kernel
%           useShape
%           S_global
%           reg_param
%           options
%
%       For kernels with a shape parameter:
%
%           C_global
%
%       For shape-free kernels:
%
% C_global =    NaN;
%
%
% OUTPUTS
% -------
% y_test_pred
%       Predictions at testing locations.
%
% y_test_diff
%       Signed testing residual:
%
%           y_test_diff = y_test - y_test_pred
%
%       This uses the same observed-minus-predicted convention
% as y_diff_cv in globalRBFTrainingCV_TuneReg.
%
% test_rmse
%       Testing RMSE.
%
% epsilon_global
%       Shape parameter used for the final full-training model.
%
%       For shape-parameter kernels:
%
%           epsilon_global = C_global * d_full
%
%       For shape-free kernels:
%
%           epsilon_global = 1
%
% modelInfo
%       Information and numerical diagnostics from the final
%       full-training RBF model.


% =============================================================
% 1. Input preparation
% =============================================================

y = ...
    y(:);

y_test = ...
    y_test(:);


if size(X,1) ~= numel(y)

    error(['Number of rows in X must equal the number ' ...
           'of training observations in y.']);

end


if size(X_test,1) ~= numel(y_test)

    error(['Number of rows in X_test must equal the number ' ...
           'of testing observations in y_test.']);

end


if size(X_test,2) ~= size(X,2)

    error(['X and X_test must have the same number ' ...
           'of coordinate dimensions.']);

end


% =============================================================
% 2. Validate and recover fixed training information
%
% trainingInfo must correspond to a successful
% globalRBFTrainingCV_TuneReg run.
% =============================================================

if ~isstruct(trainingInfo) || ...
   ~isfield(trainingInfo, 'success') || ...
   ~trainingInfo.success

error(['trainingInfo must come from a successful ' ...
       'globalRBFTrainingCV_TuneReg run.']);
end

kernel = trainingInfo.kernel;
useShape = trainingInfo.useShape;
S_global = trainingInfo.S_global;
reg_param = trainingInfo.reg_param;
options = trainingInfo.options;

if ~isscalar(reg_param) || ...
   ~isfinite(reg_param) || ...
   reg_param <= 0

    error('trainingInfo.reg_param must be a finite positive scalar.');

end


if useShape ~= common.kernelHasShapeParameter(kernel)
    error('trainingInfo.useShape is inconsistent with trainingInfo.kernel.');
end

if ~isscalar(S_global) || ...
   ~isfinite(S_global) || ...
   S_global <= 0

    error('trainingInfo.S_global must be a finite positive scalar.');

end


% =============================================================
% 3. Apply SAME global coordinate scaling used during training
%
% IMPORTANT:
%
%   - do NOT calculate a new S_global
%   - do NOT calculate a scale from X_test
%
%       X_scaled      = X      / S_global
%       X_test_scaled = X_test / S_global
% =============================================================

X_scaled = ...
    X ./ S_global;


X_test_scaled = ...
    X_test ./ S_global;


% =============================================================
% 4. Final epsilon on COMPLETE training data
%
% Shape-parameter kernel:
%
%       d_full = MeanDistant(X_scaled)
%
%       epsilon_global = C_global * d_full
%
% Shape-free kernel:
%
%       C_global       = NaN
%       d_full         = NaN
%       epsilon_global = 1
% =============================================================

if useShape

    C_global = ...
        trainingInfo.C_global;


    if ~isscalar(C_global) || ...
       ~isfinite(C_global) || ...
       C_global <= 0

        error(['trainingInfo.C_global must be a finite positive ' ...
               'scalar for kernel "%s".'], ...
               kernel);

    end


    [epsilon_global, d_full] = ...
        common.buildEpsilon( ...
            X_scaled, ...
            C_global);

else

    C_global = ...
        NaN;

    d_full = ...
        NaN;

    epsilon_global = ...
        1;

end


% =============================================================
% 5. Check polynomial admissibility
% =============================================================

[isOK, polynomialInfo] = ...
    common.checkPolynomialAdmissibility( ...
        X_scaled);


if ~isOK

    error(['Complete training dataset does not satisfy ' ...
           'the polynomial-tail admissibility requirement.']);

end


% =============================================================
% 6. Build final RBF system
%
% The COMPLETE training dataset is used.
%
% C_global has already been selected during CV and is NOT
% changed during final model fitting.
% =============================================================

S = ...
    common.buildRBFSystem( ...
        X_scaled, ...
        y, ...
        kernel, ...
        epsilon_global, ...
        reg_param);

% =============================================================
% 7. Solve final full-training RBF system
%
% solveInfo provides lightweight numerical information for the
% FINAL full-training model.
%
% Numerical diagnostics do NOT participate in model-parameter
% selection.
% =============================================================

[coeffs, solveInfo] = ...
    common.solveLinearSystem( ...
        S, ...
        options);


if ~solveInfo.success

    error( ...
        'Final global RBF model could not be solved.');

end


% =============================================================
% 8. Predict testing points
%
% B_test is retained only for the selected final model so that
% the test-site evaluation condition number can be calculated.
% =============================================================

[y_test_pred, B_test] = ...
    common.predictRBF( ...
        X_test_scaled, ...
        X_scaled, ...
        coeffs, ...
        kernel, ...
        epsilon_global);


y_test_pred = ...
    y_test_pred(:);


if ~isreal(y_test_pred) || ...
   any(~isfinite(y_test_pred))

error( ...
    'Invalid values occurred in global RBF test prediction.');
end


% =============================================================
% 9. Final selected-system numerical diagnostics
%
% Includes:
%   - raw condition number
%   - regularized condition number
%   - effective condition number
%   - training-site evaluation condition number
%   - test-site evaluation condition number
%   - relative residual
%   - backward error
% =============================================================

conditionDiagnostics = ...
    common.calculateConditionDiagnostics( ...
        S, ...
        coeffs, ...
        solveInfo, ...
        B_test);


% Reuse the detailed residual value without recalculating it.
solveInfo.rel_residual = ...
    conditionDiagnostics.rel_residual;



% =============================================================
% 10. Testing residuals and RMSE
%
% Use the SAME residual convention as CV:
%
%       observed - predicted
% =============================================================

y_test_diff = ...
    y_test - y_test_pred;


test_rmse = ...
    sqrt(mean(y_test_diff.^2));


% =============================================================
% 11. Final-model information
%
% trainingInfo.selectedFoldInfo:
%       lightweight solver information for CV folds at the
%       selected parameter configuration
%
% modelInfo.solveInfo:
%       lightweight solver information for the final
%       full-training model
%
% modelInfo.conditionDiagnostics:
%       detailed numerical diagnostics for the final
%       full-training model
% =============================================================

modelInfo = struct();


modelInfo.conditionDiagnostics = ...
    conditionDiagnostics;

modelInfo.kernel = ...
    kernel;

modelInfo.useShape = ...
    useShape;

modelInfo.C_global = ...
    C_global;

modelInfo.S_global = ...
    S_global;

modelInfo.d_full = ...
    d_full;

modelInfo.epsilon_global = ...
    epsilon_global;

modelInfo.reg_param = ...
    reg_param;

modelInfo.coeffs = ...
    coeffs;

modelInfo.solveInfo = ...
    solveInfo;

modelInfo.polynomialInfo = ...
    polynomialInfo;

modelInfo.nRBFSolves = 1;

modelInfo.success = ...
    true;


end
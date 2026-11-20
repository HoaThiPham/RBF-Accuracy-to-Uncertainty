function result = evaluateGlobalCV( ...
    X,y,kernel,C,cvp,reg_param,options,cvCache)

%EVALUATEGLOBALCV
%
% Evaluate one global RBF configuration using a fixed CV partition.
%
% IMPORTANT:
%   - X is already globally scaled before entering this function.
%   - cvp is fixed and reused for every candidate (C, RegParam) pair.
%   - no rng(...) call is made here.
%   - numerical conditioning does not select C or RegParam.
%   - result.nRBFSolves counts actual calls to solveLinearSystem.


% =============================================================
% 1. Input preparation
% =============================================================

y = y(:);

n = size(X,1);

if numel(y) ~= n
    error('X and y must contain the same number of observations.');
end

kfolds = cvp.NumTestSets;

useShape = common.kernelHasShapeParameter(kernel);


% =============================================================
% 2. Initialize outputs
% =============================================================

yhat = nan(n,1);
fold_rmse = nan(kfolds,1);

foldInfo = repmat(struct( ...
    'epsilon',       NaN, ...
    'cond_reg',      NaN, ...
    'reg_param',       NaN, ...
    'rel_residual',  NaN, ...
    'solver_used',   '', ...
    'success',       false), ...
    kfolds,1);

failureMessage = '';

nRBFSolves = 0;
success = true;

useCache = ...
    nargin >= 8 && ~isempty(cvCache);

if useCache && numel(cvCache) ~= kfolds
    error('cvCache must contain one entry for each CV fold.');
end


% =============================================================
% 3. Fixed CV evaluation
% =============================================================

for f = 1:kfolds

    % ---------------------------------------------------------
    % Fixed fold data
    % ---------------------------------------------------------

    if useCache

        idxVal = cvCache(f).idxVal;

        Xfit = cvCache(f).Xfit;
        yfit = cvCache(f).yfit(:);

        Xval = cvCache(f).Xval;
        yval = cvCache(f).yval(:);

    else

        idxTrain = training(cvp,f);
        idxVal = test(cvp,f);

        Xfit = X(idxTrain,:);
        yfit = y(idxTrain);

        Xval = X(idxVal,:);
        yval = y(idxVal);

    end


    % ---------------------------------------------------------
    % Fold-specific epsilon
    % ---------------------------------------------------------

    if useShape

        if useCache

            epsilon = ...
                max(C * cvCache(f).d,1e-12);

        else

            [epsilon, ~] = ...
                common.buildEpsilon(Xfit,C);

        end

    else

        epsilon = 1;

    end

    foldInfo(f).epsilon = epsilon;


    % ---------------------------------------------------------
    % Polynomial admissibility
    % ---------------------------------------------------------

    if useCache

        isOK = cvCache(f).isOK;

    else

        [isOK, ~] = ...
            common.checkPolynomialAdmissibility(Xfit);

    end


    if ~isOK

        failureMessage = ...
            sprintf( ...
            'Polynomial admissibility failed in CV fold %d.',f);

        success = false;
        break

    end


    % -----------------------------------------------------
    % Build regularized RBF system
    % -----------------------------------------------------

    if useCache

        S = common.buildRBFSystem( ...
            Xfit, ...
            yfit, ...
            kernel, ...
            epsilon, ...
            reg_param, ...
            cvCache(f));

    else

        S = common.buildRBFSystem( ...
            Xfit, ...
            yfit, ...
            kernel, ...
            epsilon, ...
            reg_param);

    end


    % -----------------------------------------------------
    % Solve
    % -----------------------------------------------------

    nRBFSolves = nRBFSolves + 1;

    [coeffs, solveInfo] = ...
        common.solveLinearSystem(S,options);


    foldInfo(f).cond_reg = ...
        solveInfo.cond_reg;

    foldInfo(f).reg_param = ...
        solveInfo.reg_param;

    foldInfo(f).rel_residual = ...
        solveInfo.rel_residual;

    foldInfo(f).solver_used = ...
        solveInfo.solver_used;

    foldInfo(f).success = ...
        solveInfo.success;


    if ~solveInfo.success

        failureMessage = ...
            sprintf( ...
            'Linear-system solve failed in CV fold %d.',f);

        success = false;
        break

    end


    % -----------------------------------------------------
    % Validation prediction
    % -----------------------------------------------------

    if useCache

        ypred = common.predictRBF( ...
            Xval, ...
            Xfit, ...
            coeffs, ...
            kernel, ...
            epsilon, ...
            cvCache(f));

    else

        ypred = common.predictRBF( ...
            Xval, ...
            Xfit, ...
            coeffs, ...
            kernel, ...
            epsilon);

    end


    % Force consistent prediction orientation.
    ypred = ...
        ypred(:);


    if numel(ypred) ~= numel(yval) || ...
            ~isreal(ypred) || ...
            any(~isfinite(ypred))

        failureMessage = ...
            sprintf( ...
            'Invalid prediction in CV fold %d.', ...
            f);

        success = false;

        foldInfo(f).success = ...
            false;

        break

    end


    yhat(idxVal) = ...
        ypred;


    fold_rmse_f = ...
        norm(yval - ypred) / ...
        sqrt(numel(yval));


    if ~isfinite(fold_rmse_f)

        failureMessage = ...
            sprintf( ...
            'Nonfinite RMSE in CV fold %d.', ...
            f);

        success = false;

        foldInfo(f).success = ...
            false;

        break

    end


    fold_rmse(f) = ...
        fold_rmse_f;

end


% =============================================================
% 4. Overall CV result
% =============================================================

validYhat = ...
    isreal(yhat) && ...
    all(isfinite(yhat));

validFoldRMSE = ...
    all(isfinite(fold_rmse));


if success && ...
        validYhat && ...
        validFoldRMSE

    RMSE = ...
        norm(y - yhat) / ...
        sqrt(n);

    if ~isfinite(RMSE)

        RMSE = Inf;
        success = false;

        failureMessage = ...
            'Pooled CV RMSE is nonfinite.';

    end

else

    RMSE = Inf;
    success = false;

    if isempty(failureMessage)

        if ~validYhat

            failureMessage = ...
                'CV evaluation produced invalid or incomplete predictions.';

        elseif ~validFoldRMSE

            failureMessage = ...
                'CV evaluation produced invalid or incomplete fold RMSE values.';

        else

            failureMessage = ...
                'CV evaluation failed.';

        end

    end

end


% =============================================================
% 5. Output
% =============================================================

result = struct();

result.yhat = yhat;
result.RMSE = RMSE;
result.fold_rmse = fold_rmse;
result.foldInfo = foldInfo;

result.nRBFSolves = nRBFSolves;

result.success = success;

result.failureMessage = failureMessage;
end
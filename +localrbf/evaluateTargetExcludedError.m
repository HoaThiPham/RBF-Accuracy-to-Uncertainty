function result = evaluateTargetExcludedError( ...
    cache, kernel, useShape, C, reg_param, options)
%EVALUATETARGETEXCLUDEDERROR
%
% Evaluate one parameter candidate for one fixed target-excluded
% point-radius local RBF problem.
%
% IMPORTANT:
%   - neighbourhood and geometry are fixed before entering;
%   - polynomial admissibility is checked before entering;
%   - no parameter optimization is performed here;
%   - no RNG call is made here;
%   - conditioning does not select C or RegParam;
%   - only cond_reg required by the solver is calculated;
%   - full CN/ECN/EVN diagnostics are calculated later;
%   - one call corresponds to one RBF-system solve.


% =============================================================
% 1. Candidate epsilon
% =============================================================

if useShape

    epsilon = ...
        max(C * cache.d,1e-12);

else

    C = ...
        NaN;

    epsilon = ...
        1;

end


% =============================================================
% 2. Build candidate RBF system
%
% cache.R and cache.P are reused.
% =============================================================

S = ...
    common.buildRBFSystem( ...
        cache.Xfit, ...
        cache.yfit, ...
        kernel, ...
        epsilon, ...
        reg_param, ...
        cache);


% =============================================================
% 3. Solve candidate system
%
% cond_reg is calculated inside solveLinearSystem and is used
% only for numerical solver selection.
% =============================================================

[coeffs,solveInfo] = ...
    common.solveLinearSystem( ...
        S, ...
        options);


% =============================================================
% 4. Initialize result
% =============================================================

result = struct();

result.C = ...
    C;

result.reg_param = ...
    reg_param;

result.epsilon = ...
    epsilon;

result.cond_reg = ...
    solveInfo.cond_reg;

result.rel_residual = ...
    solveInfo.rel_residual;

result.solver_used = ...
    solveInfo.solver_used;

result.nRBFSolves = ...
    1;

result.yhat = ...
    NaN;

result.y_diff = ...
    NaN;

result.AE = ...
    NaN;

result.success = ...
    false;

result.failureMessage = ...
    '';


% =============================================================
% 5. Solve validity
% =============================================================

if ~solveInfo.success

    result.failureMessage = ...
        'Linear-system solve failed.';

    return

end


% =============================================================
% 6. Predict excluded target
%
% cache.R_eval and cache.P_eval are reused.
% =============================================================

yhat = ...
    common.predictRBF( ...
        cache.Xtarget, ...
        cache.Xfit, ...
        coeffs, ...
        kernel, ...
        epsilon, ...
        cache);


yhat = ...
    yhat(:);


if numel(yhat) ~= 1 || ...
   ~isreal(yhat) || ...
   any(~isfinite(yhat))

    result.failureMessage = ...
        'Invalid target-excluded prediction.';

    return

end


% =============================================================
% 7. Target-excluded error
%
% Residual convention:
% observed - predicted
% =============================================================

y_diff = ...
    cache.ytarget - yhat;


AE = ...
    abs(y_diff);


if ~isfinite(AE)

    result.failureMessage = ...
        'Nonfinite target-excluded absolute error.';

    return

end


% =============================================================
% 8. Successful evaluation
% =============================================================

result.yhat = ...
    yhat;

result.y_diff = ...
    y_diff;

result.AE = ...
    AE;

result.success = ...
    true;

end
function diagInfo = calculateConditionDiagnostics( ...
    S, coeffs, solveInfo, B_test)
%CALCULATECONDITIONDIAGNOSTICS
% Compute detailed numerical diagnostics for a selected RBF system.
%
% Intended for selected RBF systems only, including final
% global or local models. Expensive diagnostics are not
% evaluated during parameter optimization.
%
% Diagnostics include:
%   - standard condition numbers before and after regularization
%   - effective condition number
%   - training-site evaluation condition number
%   - test-site evaluation condition number
%   - change in conditioning due to regularization
%   - effective numerical rank
%   - relative residual
%   - normwise backward error
%
% The system actually solved is S.A_reg.
%


coeffs = coeffs(:);

A = S.A_reg;


% =========================================================
% 1. Standard condition numbers
% =========================================================

% Reconstruct the raw augmented system only for diagnostics.
q = size(S.P,2);

A_raw = [S.Phi, S.P;
    S.P',   zeros(q)];

diagInfo.cond_raw = ...
    cond(A_raw);

clear A_raw

% Reuse the value already calculated by solveLinearSystem.
diagInfo.cond_reg = ...
    solveInfo.cond_reg;


% =========================================================
% 2. Change in conditioning due to regularization
% =========================================================

if isfinite(diagInfo.cond_raw) && ...
        isfinite(diagInfo.cond_reg) && ...
        diagInfo.cond_raw > 0 && ...
        diagInfo.cond_reg > 0

    diagInfo.cond_ratio = ...
        diagInfo.cond_reg / diagInfo.cond_raw;

    diagInfo.delta_log10_cond = ...
        log10(diagInfo.cond_reg) - ...
        log10(diagInfo.cond_raw);

else

    diagInfo.cond_ratio = NaN;
    diagInfo.delta_log10_cond = NaN;

end


% =========================================================
% 3. Singular values and effective numerical rank
% =========================================================

s = svd(A);

normA = s(1);

rank_tol = ...
    max(size(A)) * eps(normA);

rank_eff = ...
    sum(s > rank_tol);

diagInfo.rank_eff = ...
    rank_eff;

diagInfo.rank_tol = ...
    rank_tol;


% =========================================================
% Quantities reused below
% =========================================================

normRhs = ...
    norm(S.rhs,2);

normCoeffs = ...
    norm(coeffs,2);


% =========================================================
% 4. Effective condition number
% =========================================================

if rank_eff > 0

    sigma_k = ...
        s(rank_eff);

    diagInfo.sigma_min_retained = ...
        sigma_k;

    denom = ...
        sigma_k * normCoeffs;

    if denom > 0

        diagInfo.cond_effective = ...
            normRhs / denom;

    else

        diagInfo.cond_effective = Inf;

    end

else

    diagInfo.sigma_min_retained = 0;
    diagInfo.cond_effective = Inf;

end


% =========================================================
% 5. Evaluation condition numbers
%
% Training:
%
%   B_train = [Phi, P]
%
% Test:
%
%   B_test = [Phi_test, P_test]
%
% Evaluation condition number:
%
%   kappa_eval =
%       ||B||_2 ||coeffs||_2 / ||B*coeffs||_2
%
% These diagnostics are calculated only for the selected
% final model.
% =========================================================


% ---------------------------------------------------------
% Training-site evaluation condition number
% ---------------------------------------------------------

B_train = ...
    [S.Phi, S.P];


y_eval_train = ...
    B_train * coeffs;


normB_train = ...
    norm(B_train,2);


normEvalTrain = ...
    norm(y_eval_train,2);


if normEvalTrain > 0

    diagInfo.cond_evaluation_train = ...
        normB_train * ...
        normCoeffs / ...
        normEvalTrain;

else

    diagInfo.cond_evaluation_train = ...
        Inf;

end


clear B_train y_eval_train

% ---------------------------------------------------------
% Test-site evaluation condition number
% ---------------------------------------------------------
y_eval_test = ...
    B_test * coeffs;

normB_test = ...
    norm(B_test,2);

normEvalTest = ...
    norm(y_eval_test,2);

if normEvalTest > 0

    diagInfo.cond_evaluation_test = ...
        normB_test * ...
        normCoeffs / ...
        normEvalTest;

else

    diagInfo.cond_evaluation_test = ...
        Inf;

end


% =========================================================
% 6. Residual diagnostics
% =========================================================

residual = ...
    A*coeffs - S.rhs;

normResidual = ...
    norm(residual,2);


% Relative residual
diagInfo.rel_residual = ...
    normResidual / max(normRhs,eps);

% Normwise backward error
diagInfo.backward_error = ...
    normResidual / ...
    max(normA*normCoeffs + normRhs,realmin);



% =========================================================
% 7. Regularization and solver information
% =========================================================

diagInfo.reg_param = ...
    solveInfo.reg_param;

diagInfo.solver_used = ...
    solveInfo.solver_used;

diagInfo.success = ...
    solveInfo.success;



end
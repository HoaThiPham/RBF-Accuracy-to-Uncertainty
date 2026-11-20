function [coeffs, solveInfo] = solveLinearSystem(S, options)
%SOLVELINEARSYSTEM Solve one fixed regularized augmented RBF system.
%
% The candidate regularization is fixed before entering this
% function. Conditioning is used only for solver selection.
%
% Relative residual is calculated only when explicitly requested.

% =============================================================
% 1. Solver options
% =============================================================

cond_lsq = ...
    options.solver.cond_lsq;


if ~isnumeric(cond_lsq) || ...
   ~isscalar(cond_lsq) || ...
   ~isreal(cond_lsq) || ...
   ~isfinite(cond_lsq) || ...
   cond_lsq < 1

    error( ...
        'options.solver.cond_lsq must be a finite real scalar >= 1.');

end


% -------------------------------------------------------------
% Relative-residual option
% -------------------------------------------------------------

computeResidual = ...
    false;


if isfield(options.solver,'compute_rel_residual') && ...
   ~isempty(options.solver.compute_rel_residual)

    computeResidual = ...
        options.solver.compute_rel_residual;

end


if ~isscalar(computeResidual) || ...
   ~(islogical(computeResidual) || isnumeric(computeResidual))

    error( ...
        'options.solver.compute_rel_residual must be a logical scalar.');

end


computeResidual = ...
    logical(computeResidual);


% =============================================================
% 2. Condition number of fixed candidate system
% =============================================================

cond_reg = ...
    cond(S.A_reg);


% =============================================================
% 3. Solver selection
% =============================================================

if ~isfinite(cond_reg) || ...
   cond_reg > cond_lsq

    coeffs = ...
        lsqminnorm(S.A_reg,S.rhs);

    solver_used = ...
        'lsqminnorm';

else

    coeffs = ...
        S.A_reg \ S.rhs;

    solver_used = ...
        'backslash';

end


% =============================================================
% 4. Relative residual
%
% Calculated only when requested, normally for final selected CV.
% =============================================================

rel_residual = ...
    NaN;


if computeResidual

    residual = ...
        S.A_reg * coeffs - ...
        S.rhs;


    rel_residual = ...
        norm(residual) / ...
        max(norm(S.rhs),eps);

end


% =============================================================
% 5. Diagnostics
% =============================================================

solveInfo.cond_reg = ...
    cond_reg;

solveInfo.reg_param = ...
    S.reg_param;

solveInfo.solver_used = ...
    solver_used;

solveInfo.rel_residual = ...
    rel_residual;


solveInfo.success = ...
    all(isfinite(coeffs));


if computeResidual

    solveInfo.success = ...
        solveInfo.success && ...
        isfinite(rel_residual);

end

end
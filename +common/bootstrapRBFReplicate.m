function [yhat,info] = bootstrapRBFReplicate( ...
    Xfit,yfit,Xeval,idxBoot, ...
    kernel,C_global,reg_param,options,geometry)

%BOOTSTRAPRBFITPREDICT Fit and predict one bootstrap RBF realization.
%
% Bootstrap sampling:
%   - sampling WITH replacement;
%   - bootstrap sample size equals the original fitting-set size;
%   - duplicated observations are retained.
%
% Fixed quantities:
%   kernel
%   C_global
%   reg_param
%   solver settings
%   global coordinate scaling (already applied by caller)
%
% Shaped kernels:
%       epsilon_boot = C_global*d_boot
%
% Shape-free kernels:
%       internal epsilon = 1
%
% Numerical solution uses common.solveLinearSystem so the same
% cond(A_reg) and solver-selection rule is used as in the main
% training/testing workflow.


% =============================================================
% 1. Prepare and validate inputs
% =============================================================

yfit = yfit(:);
idxBoot = idxBoot(:);

m = size(Xfit,1);

if numel(yfit) ~= m
    error('Xfit and yfit must contain the same number of observations.');
end

if numel(idxBoot) ~= m
    error('Bootstrap sample size must equal the fitting-set size.');
end

if any(idxBoot < 1) || ...
   any(idxBoot > m) || ...
   any(idxBoot ~= round(idxBoot))

    error('Invalid bootstrap indices.');
end

if size(Xeval,2) ~= size(Xfit,2)
    error('Xfit and Xeval must have the same coordinate dimension.');
end


% =============================================================
% 2. Original geometry if no cache is supplied
% =============================================================

if nargin < 9 || isempty(geometry)

    geometry = struct();

    geometry.R = ...
        pdist2(Xfit,Xfit);

    geometry.P = ...
        [ones(m,1),Xfit];

    geometry.R_eval = ...
        pdist2(Xeval,Xfit);

    geometry.P_eval = ...
        [ones(size(Xeval,1),1),Xeval];

end


% =============================================================
% 3. Bootstrap sample
%
% IMPORTANT:
% Do NOT use unique(idxBoot).
% Repeated observations are part of ordinary bootstrap sampling.
% =============================================================

Xboot = ...
    Xfit(idxBoot,:);

yboot = ...
    yfit(idxBoot);


% =============================================================
% 4. Initialize information
% =============================================================

info = struct();

info.success = false;
info.failureStage = '';

info.admissible = false;

info.nUnique = ...
    numel(unique(idxBoot));

info.d_boot = NaN;
info.epsilon = NaN;

info.cond_reg = NaN;
info.reg_phi = NaN;
info.solver_used = '';

info.nRBFSolves = 0;


% =============================================================
% 5. Polynomial admissibility
% =============================================================

[isOK,~] = ...
    common.checkPolynomialAdmissibility(Xboot);

info.admissible = ...
    isOK;

if ~isOK

    yhat = ...
        nan(size(Xeval,1),1);

    info.failureStage = ...
        'admissibility';

    return
end


% =============================================================
% 6. Bootstrap epsilon
% =============================================================

useShape = ...
    common.kernelHasShapeParameter(kernel);

if useShape

    if ~isscalar(C_global) || ...
       ~isfinite(C_global) || ...
       C_global <= 0

        error('C_global must be finite and positive.');
    end

    [epsilon_boot,d_boot] = ...
        common.buildEpsilon( ...
            Xboot, ...
            C_global);

    info.d_boot = ...
        d_boot;

    info.epsilon = ...
        epsilon_boot;

else

    % Internal dummy value only.
    epsilon_boot = 1;

end


% =============================================================
% 7. Extract bootstrap geometry from cached original geometry
%
% Repeated idxBoot values deliberately generate repeated
% rows/columns.
% =============================================================

fitGeometry = struct();

fitGeometry.R = ...
    geometry.R(idxBoot,idxBoot);

fitGeometry.P = ...
    geometry.P(idxBoot,:);


evalGeometry = struct();

evalGeometry.R_eval = ...
    geometry.R_eval(:,idxBoot);

evalGeometry.P_eval = ...
    geometry.P_eval;


% =============================================================
% 8. Build regularized RBF system
% =============================================================

S = ...
    common.buildRBFSystem( ...
        Xboot, ...
        yboot, ...
        kernel, ...
        epsilon_boot, ...
        reg_param, ...
        fitGeometry);


% =============================================================
% 9. Solve
%
% common.solveLinearSystem already calculates cond(A_reg).
% Do NOT calculate cond(A_reg) again here.
% =============================================================

% =============================================================
% 9. Solve
%
% Regularization is applied directly as reg_param*I.
% common.solveLinearSystem calculates cond(A_reg).
% =============================================================

info.nRBFSolves = 1;

[coeffs,solveInfo] = ...
    common.solveLinearSystem( ...
        S, ...
        options);

info.cond_reg = ...
    solveInfo.cond_reg;

info.solver_used = ...
    solveInfo.solver_used;

if ~solveInfo.success

    yhat = ...
        nan(size(Xeval,1),1);

    info.failureStage = ...
        'solve';

    return
end

% =============================================================
% 10. Predict
% =============================================================

yhat = ...
    common.predictRBF( ...
        Xeval, ...
        Xboot, ...
        coeffs, ...
        kernel, ...
        epsilon_boot, ...
        evalGeometry);

yhat = ...
    yhat(:);


if any(~isfinite(yhat))

    yhat(:) = NaN;

    info.failureStage = ...
        'prediction';

    return
end


info.success = true;

end
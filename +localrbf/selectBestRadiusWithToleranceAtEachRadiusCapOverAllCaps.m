function CapResult = ...
    selectBestRadiusWithToleranceAtEachRadiusCapOverAllCaps( ...
        PointResult, tau_loc, retainDetail)
%SELECTBESTRADIUSWITHTOLERANCEATEACHRADIUSCAPOVERALLCAPS
%
% For ONE training point, apply the local radius-selection rule at EVERY
% candidate radius cap.
%
% For each cap k, among successfully optimized radii r(1),...,r(k), the
% function determines minAE, applies
%
%       AE(r) <= minAE + tau_loc
%
% and selects the SMALLEST eligible radius. The already optimized settings,
% prediction, residual, numerical information, and neighbourhood information
% associated with that selected radius are reused; nothing is re-optimized.
%
% Optional output control:
%   retainDetail = true  (default) preserves the historical complete output,
%                       including CapResult.Cap and selection-set histories.
%   retainDetail = false returns only the compact arrays required by the
%                       Stage-2 training path. Detailed per-cap structures
%                       and selection-index histories are not constructed.
%
% Residual convention:
%
%       residual = observation - prediction
%
% IMPORTANT:
%   - no parameter optimization is performed here;
%   - no RBF system is solved here;
%   - no neighbourhood is reconstructed here;
%   - no conditioning diagnostic is recalculated here;
%   - no RMSE is calculated here;
%   - retainDetail changes retained output only, never radius selection;
%   - complete optimizer histories remain authoritative in PointResult and
%     are not duplicated by this function.

% =============================================================
% 0. Optional detail control
% =============================================================

if nargin < 3 || isempty(retainDetail)
    retainDetail = true;
end

if ~(isscalar(retainDetail) && ...
     (islogical(retainDetail) || ...
      (isnumeric(retainDetail) && ...
       isreal(retainDetail) && ...
       isfinite(retainDetail) && ...
       any(retainDetail == [0 1]))))

    error('retainDetail must be true/false (or numeric 1/0).');
end

retainDetail = logical(retainDetail);

% =============================================================
% 1. Input checks
% =============================================================

if ~isstruct(PointResult)
    error('PointResult must be a structure.');
end

requiredFields = { ...
    'radii', ...
    'nRadii', ...
    'radius_success', ...
    'AE_best_all', ...
    'Radius'};

for q = 1:numel(requiredFields)
    if ~isfield(PointResult,requiredFields{q})
        error('PointResult is missing required field "%s".', ...
              requiredFields{q});
    end
end

if ~isnumeric(tau_loc) || ~isscalar(tau_loc) || ~isreal(tau_loc) || ...
   ~isfinite(tau_loc) || tau_loc < 0
    error('tau_loc must be a finite nonnegative real scalar.');
end

radii = PointResult.radii(:);
M = numel(radii);

if PointResult.nRadii ~= M
    error('PointResult.nRadii is inconsistent with PointResult.radii.');
end

AE_best = PointResult.AE_best_all(:);
radius_success = PointResult.radius_success(:);

if ~(islogical(radius_success) || isnumeric(radius_success)) || ...
   numel(radius_success) ~= M
    error(['PointResult.radius_success must be a logical or ' ...
           'numeric vector matching PointResult.radii.']);
end

if isnumeric(radius_success)
    if ~isreal(radius_success) || any(~isfinite(radius_success)) || ...
       any(radius_success ~= 0 & radius_success ~= 1)
        error(['Numeric PointResult.radius_success must contain ' ...
               'only finite values 0 or 1.']);
    end
end

radius_success = logical(radius_success);

if numel(AE_best) ~= M || numel(PointResult.Radius) ~= M
    error(['PointResult radius-level arrays and structures must ' ...
           'have the same length.']);
end

% =============================================================
% 2. Fixed point information
% =============================================================

iTarget = getFieldOrDefault(PointResult,'iTarget',NaN);
xTarget = getFieldOrDefault(PointResult,'xTarget',[]);
yTarget = getFieldOrDefault(PointResult,'yTarget',NaN);
kernel = getFieldOrDefault(PointResult,'kernel','');
useShape = getFieldOrDefault(PointResult,'useShape',false);

% =============================================================
% 3. Initialize common output
% =============================================================

CapResult = struct();
CapResult.success = false;
CapResult.iTarget = iTarget;
CapResult.xTarget = xTarget;
CapResult.yTarget = yTarget;
CapResult.kernel = kernel;
CapResult.useShape = useShape;
CapResult.tau_loc = tau_loc;
CapResult.radii = radii;
CapResult.nCaps = M;
CapResult.nSuccessfulCaps = 0;
CapResult.nFailedCaps = 0;
CapResult.message = '';

% =============================================================
% 4. Cap-level output allocation
%
% Full mode preserves the historical field set and field-creation order.
% Compact mode creates only the arrays required by Stage-2 compulsory
% training. DETAIL-only structures and variable-length neighbourhood
% vectors are never allocated.
% =============================================================

if retainDetail

    % ---------------------------------------------------------
    % Historical detailed result for every cap
    % ---------------------------------------------------------
    CapResult.Cap = repmat(emptyCapResult(),M,1);

    % ---------------------------------------------------------
    % Historical complete compact arrays
    % ---------------------------------------------------------
    CapResult.capIndex_all = (1:M).';
    CapResult.capRadius_all = radii;
    CapResult.cap_success = false(M,1);

    CapResult.selectedRadiusIndex_all = nan(M,1);
    CapResult.selectedRadius_all = nan(M,1);
    CapResult.selectedAE_all = nan(M,1);
    CapResult.minAE_all = nan(M,1);
    CapResult.errorThreshold_all = nan(M,1);
    CapResult.selectedIsExactMinimum_all = false(M,1);

    CapResult.C_selected_all = nan(M,1);
    CapResult.Reg_selected_all = nan(M,1);
    CapResult.epsilon_selected_all = nan(M,1);
    CapResult.d_local_selected_all = nan(M,1);

    CapResult.yhat_selected_all = nan(M,1);
    CapResult.residual_selected_all = nan(M,1);
    CapResult.y_diff_selected_all = nan(M,1);
    CapResult.squaredResidual_selected_all = nan(M,1);

    CapResult.cond_reg_selected_all = nan(M,1);
    CapResult.rel_residual_selected_all = nan(M,1);
    CapResult.solver_used_selected_all = cell(M,1);

    CapResult.neighbor_idx_selected_all = cell(M,1);
    CapResult.neighbor_distances_selected_all = cell(M,1);
    CapResult.neighbor_count_selected_all = nan(M,1);
    CapResult.requested_radius_selected_all = nan(M,1);
    CapResult.effective_radius_selected_all = nan(M,1);
    CapResult.selected_max_distance_all = nan(M,1);
    CapResult.n_within_radius_selected_all = nan(M,1);
    CapResult.n_added_for_minimum_selected_all = nan(M,1);
    CapResult.usedMinNeighborFallback_selected_all = false(M,1);

    CapResult.nValidRadiiWithinCap_all = zeros(M,1);
    CapResult.nExactMinimumRadii_all = zeros(M,1);
    CapResult.nEligibleRadii_all = zeros(M,1);
    CapResult.validRadiusIdx_all = cell(M,1);
    CapResult.exactMinimumRadiusIdx_all = cell(M,1);
    CapResult.eligibleRadiusIdx_all = cell(M,1);

else

    % ---------------------------------------------------------
    % Stage-2 compulsory compact arrays only
    % ---------------------------------------------------------
    CapResult.cap_success = false(M,1);

    CapResult.selectedRadiusIndex_all = nan(M,1);
    CapResult.selectedRadius_all = nan(M,1);
    CapResult.selectedAE_all = nan(M,1);

    CapResult.C_selected_all = nan(M,1);
    CapResult.Reg_selected_all = nan(M,1);
    CapResult.epsilon_selected_all = nan(M,1);
    CapResult.d_local_selected_all = nan(M,1);

    CapResult.yhat_selected_all = nan(M,1);
    CapResult.residual_selected_all = nan(M,1);
    CapResult.y_diff_selected_all = nan(M,1);
    CapResult.squaredResidual_selected_all = nan(M,1);

    CapResult.cond_reg_selected_all = nan(M,1);
    CapResult.rel_residual_selected_all = nan(M,1);
    CapResult.solver_used_selected_all = cell(M,1);

    % Exact neighbour count is retained because it is a compact selected-
    % model quantity used downstream. Variable-length neighbour indices,
    % distances, and secondary neighbourhood diagnostics are DETAIL-only.
    CapResult.neighbor_count_selected_all = nan(M,1);

end


% =============================================================
% 6. Apply selection rule to every candidate radius cap
% =============================================================

for kCap = 1:M

    % =========================================================
    % 6.1 Select best radius under this cap
    % =========================================================

    Sel = localrbf.selectBestRadiusWithToleranceAtEachRadiusCap( ...
        radii, ...
        AE_best, ...
        radius_success, ...
        kCap, ...
        tau_loc, ...
        retainDetail);

    if retainDetail
        Ck = emptyCapResult();
        Ck.capIndex = kCap;
        Ck.capRadius = radii(kCap);
        Ck.observation = yTarget;
        Ck.selection = Sel;
        Ck.success = Sel.success;
        Ck.minAE = Sel.minAE;
        Ck.errorThreshold = Sel.errorThreshold;
        Ck.validRadiusIdx = Sel.valid_idx;
        Ck.exactMinimumRadiusIdx = Sel.exactMinimum_idx;
        Ck.eligibleRadiusIdx = Sel.eligible_idx;
        Ck.nValidRadii = Sel.nValid;
        Ck.nExactMinimumRadii = Sel.nExactMinimum;
        Ck.nEligibleRadii = Sel.nEligible;

        CapResult.minAE_all(kCap) = Ck.minAE;
        CapResult.errorThreshold_all(kCap) = Ck.errorThreshold;
        CapResult.nValidRadiiWithinCap_all(kCap) = Ck.nValidRadii;
        CapResult.nExactMinimumRadii_all(kCap) = Ck.nExactMinimumRadii;
        CapResult.nEligibleRadii_all(kCap) = Ck.nEligibleRadii;
        CapResult.validRadiusIdx_all{kCap} = Ck.validRadiusIdx;
        CapResult.exactMinimumRadiusIdx_all{kCap} = Ck.exactMinimumRadiusIdx;
        CapResult.eligibleRadiusIdx_all{kCap} = Ck.eligibleRadiusIdx;
    end

    % =========================================================
    % 6.2 Failed cap
    % =========================================================

    if ~Sel.success
        if retainDetail
            Ck.message = Sel.message;
            CapResult.Cap(kCap) = Ck;
        end
        continue
    end

    % =========================================================
    % 6.3 Selected-radius validation
    % =========================================================

    jStar = Sel.radiusIndex;

    validSelectedIndex = ...
        isnumeric(jStar) && ...
        isscalar(jStar) && ...
        isreal(jStar) && ...
        isfinite(jStar) && ...
        jStar == round(jStar) && ...
        jStar >= 1 && ...
        jStar <= kCap;

    if ~validSelectedIndex
        if retainDetail
            Ck.success = false;
            Ck.message = 'Selected radius index is invalid.';
            CapResult.Cap(kCap) = Ck;
        end
        continue
    end

    RStar = PointResult.Radius(jStar);

    if ~RStar.success
        if retainDetail
            Ck.success = false;
            Ck.message = ['Selected radius is inconsistent with the ' ...
                          'stored radius-success status.'];
            CapResult.Cap(kCap) = Ck;
        end
        continue
    end

    % =========================================================
    % 6.4 Read selected radius result once
    % =========================================================

    selectedRadius = RStar.radius;
    selectedAE = RStar.AE_best;

    selectedC = RStar.C_best;
    selectedReg = RStar.reg_param_best;
    selectedEpsilon = RStar.epsilon_best;
    selectedDLocal = RStar.d_local;

    selectedYhat = RStar.yhat_best;
    selectedResidual = RStar.y_diff_best;
    selectedSquaredResidual = RStar.y_diff_best.^2;

    selectedCondReg = RStar.cond_reg_best;
    selectedRelResidual = RStar.rel_residual_best;
    selectedSolver = RStar.solver_used_best;

    % =========================================================
    % 6.5 Populate historical selected-radius detail BEFORE the
    %     final numerical-validity check when requested.
    %
    % This preserves the original full-mode failure record exactly:
    % even when yhat/residual is invalid, the selected radius, parameters,
    % numerical information, and neighbourhood remain available for
    % diagnostics. Compact mode does not construct this optional payload.
    % =========================================================

    if retainDetail
        Ck.selectedRadiusIndex = jStar;
        Ck.sourceRadiusIndex = jStar;
        Ck.selectedRadius = selectedRadius;
        Ck.selectedAE = selectedAE;
        Ck.selectedIsExactMinimum = Sel.selectedIsExactMinimum;

        Ck.C = selectedC;
        Ck.reg_param = selectedReg;
        Ck.epsilon = selectedEpsilon;
        Ck.d_local = selectedDLocal;

        Ck.yhat = selectedYhat;
        Ck.residual = selectedResidual;
        Ck.y_diff = selectedResidual;
        Ck.squaredResidual = selectedSquaredResidual;

        Ck.cond_reg = selectedCondReg;
        Ck.rel_residual = selectedRelResidual;
        Ck.solver_used = selectedSolver;

        Ck.neighbor_idx = RStar.neighbor_idx;
        Ck.neighbor_distances = RStar.neighbor_distances;
        Ck.neighbor_count = RStar.neighbor_count;
        Ck.requestedRadius = RStar.requestedRadius;
        Ck.effectiveRadius = RStar.effectiveRadius;
        Ck.selectedMaxDistance = RStar.selectedMaxDistance;
        Ck.nWithinRadius = RStar.nWithinRadius;
        Ck.nAddedForMinimum = RStar.nAddedForMinimum;
        Ck.usedMinNeighborFallback = RStar.usedMinNeighborFallback;
    end

    % =========================================================
    % 6.6 Final selected-cap validity
    % =========================================================

    selectedNumericallyValid = ...
        isscalar(selectedYhat) && ...
        isreal(selectedYhat) && ...
        isfinite(selectedYhat) && ...
        isscalar(selectedResidual) && ...
        isreal(selectedResidual) && ...
        isfinite(selectedResidual) && ...
        isscalar(selectedSquaredResidual) && ...
        isreal(selectedSquaredResidual) && ...
        isfinite(selectedSquaredResidual);

    if ~selectedNumericallyValid
        if retainDetail
            Ck.success = false;
            Ck.message = ['Selected radius produced invalid prediction ' ...
                          'or residual information.'];
            CapResult.Cap(kCap) = Ck;
        end
        continue
    end

    % =========================================================
    % 6.7 Populate compulsory compact arrays directly
    % =========================================================

    CapResult.cap_success(kCap) = true;

    CapResult.selectedRadiusIndex_all(kCap) = jStar;
    CapResult.selectedRadius_all(kCap) = selectedRadius;
    CapResult.selectedAE_all(kCap) = selectedAE;

    CapResult.C_selected_all(kCap) = selectedC;
    CapResult.Reg_selected_all(kCap) = selectedReg;
    CapResult.epsilon_selected_all(kCap) = selectedEpsilon;
    CapResult.d_local_selected_all(kCap) = selectedDLocal;

    CapResult.yhat_selected_all(kCap) = selectedYhat;
    CapResult.residual_selected_all(kCap) = selectedResidual;
    CapResult.y_diff_selected_all(kCap) = selectedResidual;
    CapResult.squaredResidual_selected_all(kCap) = selectedSquaredResidual;

    CapResult.cond_reg_selected_all(kCap) = selectedCondReg;
    CapResult.rel_residual_selected_all(kCap) = selectedRelResidual;
    CapResult.solver_used_selected_all{kCap} = selectedSolver;

    CapResult.neighbor_count_selected_all(kCap) = RStar.neighbor_count;

    % =========================================================
    % 6.8 Finalize historical detailed output only when requested
    % =========================================================

    if retainDetail
        Ck.success = true;
        Ck.message = '';
        CapResult.Cap(kCap) = Ck;

        CapResult.neighbor_idx_selected_all{kCap} = Ck.neighbor_idx;
        CapResult.neighbor_distances_selected_all{kCap} = ...
            Ck.neighbor_distances;
        CapResult.effective_radius_selected_all(kCap) = ...
            Ck.effectiveRadius;
        CapResult.selected_max_distance_all(kCap) = ...
            Ck.selectedMaxDistance;
        CapResult.n_within_radius_selected_all(kCap) = Ck.nWithinRadius;
        CapResult.n_added_for_minimum_selected_all(kCap) = ...
            Ck.nAddedForMinimum;
        CapResult.usedMinNeighborFallback_selected_all(kCap) = ...
            Ck.usedMinNeighborFallback;

        CapResult.selectedIsExactMinimum_all(kCap) = ...
            Ck.selectedIsExactMinimum;
        CapResult.requested_radius_selected_all(kCap) = ...
            Ck.requestedRadius;
    end
end

% =============================================================
% 7. Final point-level cap-selection status
% =============================================================

CapResult.nSuccessfulCaps = sum(CapResult.cap_success);
CapResult.nFailedCaps = M - CapResult.nSuccessfulCaps;
CapResult.success = CapResult.nSuccessfulCaps > 0;

if CapResult.success
    CapResult.message = '';
else
    CapResult.message = ...
        'No candidate radius cap produced a valid radius selection.';
end

end

% =============================================================
% Local helper
% =============================================================

function C = emptyCapResult()
%EMPTYCAPRESULT Uniform result structure for one candidate radius cap.

C = struct();

C.capIndex = NaN;
C.capRadius = NaN;
C.success = false;

C.selection = [];
C.minAE = NaN;
C.errorThreshold = NaN;
C.validRadiusIdx = [];
C.exactMinimumRadiusIdx = [];
C.eligibleRadiusIdx = [];
C.nValidRadii = 0;
C.nExactMinimumRadii = 0;
C.nEligibleRadii = 0;

C.selectedRadiusIndex = NaN;
C.sourceRadiusIndex = NaN;
C.selectedRadius = NaN;
C.selectedAE = NaN;
C.selectedIsExactMinimum = false;

C.C = NaN;
C.reg_param = NaN;
C.epsilon = NaN;
C.d_local = NaN;

C.observation = NaN;
C.yhat = NaN;
C.residual = NaN;
C.y_diff = NaN;
C.squaredResidual = NaN;

C.cond_reg = NaN;
C.rel_residual = NaN;
C.solver_used = '';

C.neighbor_idx = [];
C.neighbor_distances = [];
C.neighbor_count = 0;
C.requestedRadius = NaN;
C.effectiveRadius = NaN;
C.selectedMaxDistance = NaN;
C.nWithinRadius = 0;
C.nAddedForMinimum = 0;
C.usedMinNeighborFallback = false;

C.message = '';
end

% =============================================================
% Local helper
% =============================================================

function value = getFieldOrDefault(S,fieldName,defaultValue)
if isfield(S,fieldName)
    value = S.(fieldName);
else
    value = defaultValue;
end
end

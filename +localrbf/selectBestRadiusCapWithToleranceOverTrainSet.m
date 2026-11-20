function Sel = selectBestRadiusCapWithToleranceOverTrainSet( ...
    capRadii, RMSE_all, cap_success, tau_cap)
%SELECTBESTRADIUSCAPWITHTOLERANCEOVERTRAINSET
%
% Select the final radius cap using training-set RMSE.
%
% Among all valid candidate radius caps:
%
%       capRadii(1), ..., capRadii(K)
%
% first determine the exact minimum training-set RMSE:
%
%       minRMSE = min RMSE(k)
%
% Then select the SMALLEST radius cap satisfying:
%
%       RMSE(k) <= minRMSE + tau_cap
%
% Therefore:
%
%       tau_cap = 0
%
% selects the smallest radius cap attaining the exact
% minimum RMSE.
%
% IMPORTANT:
%   - RMSE_all must already have been calculated using residuals
%     from the complete training set;
%   - cap_success should indicate caps for which all required
%     training-point results are valid;
%   - no local parameter optimization is performed here;
%   - no RBF system is solved here;
%   - no neighbourhood is reconstructed here;
%   - no conditioning quantity affects cap selection;
%   - candidate radius caps must be strictly increasing;
%   - returned indices refer to the original capRadii vector.


% =============================================================
% 1. Input checks
% =============================================================

if ~isnumeric(capRadii) || ...
   isempty(capRadii) || ...
   ~isvector(capRadii) || ...
   ~isreal(capRadii) || ...
   any(~isfinite(capRadii(:))) || ...
   any(capRadii(:) <= 0)

    error( ...
        'capRadii must be a nonempty finite positive real vector.');

end


capRadii = ...
    capRadii(:);


K = ...
    numel(capRadii);


if any(diff(capRadii) <= 0)

    error( ...
        'Candidate radius caps must be strictly increasing.');

end


if ~isnumeric(RMSE_all) || ...
   ~isvector(RMSE_all) || ...
   ~isreal(RMSE_all) || ...
   numel(RMSE_all) ~= K

    error( ...
        'RMSE_all must be a real numeric vector matching capRadii.');

end


RMSE_all = ...
    RMSE_all(:);


if ~(islogical(cap_success) || isnumeric(cap_success)) || ...
   ~isvector(cap_success) || ...
   numel(cap_success) ~= K

    error( ...
        ['cap_success must be a logical or numeric vector ' ...
         'matching capRadii.']);

end


if isnumeric(cap_success)

    if ~isreal(cap_success) || ...
       any(~isfinite(cap_success(:))) || ...
       any( ...
           cap_success(:) ~= 0 & ...
           cap_success(:) ~= 1)

        error( ...
            ['Numeric cap_success must contain only finite ' ...
             'values 0 or 1.']);

    end

end


cap_success = ...
    logical(cap_success(:));


if ~isnumeric(tau_cap) || ...
   ~isscalar(tau_cap) || ...
   ~isreal(tau_cap) || ...
   ~isfinite(tau_cap) || ...
   tau_cap < 0

    error( ...
        'tau_cap must be a finite nonnegative real scalar.');

end


% =============================================================
% 2. Initialize result
% =============================================================

Sel = ...
    struct();


Sel.success = ...
    false;


Sel.tau_cap = ...
    tau_cap;


Sel.nCaps = ...
    K;


Sel.capRadii = ...
    capRadii;


Sel.RMSE_all = ...
    RMSE_all;


Sel.cap_success = ...
    cap_success;


Sel.capIndex = ...
    NaN;


Sel.capRadius = ...
    NaN;


Sel.RMSE = ...
    NaN;


Sel.minRMSE = ...
    NaN;


Sel.RMSEThreshold = ...
    NaN;


Sel.valid_idx = ...
    [];


Sel.exactMinimum_idx = ...
    [];


Sel.eligible_idx = ...
    [];


Sel.nValid = ...
    0;


Sel.nExactMinimum = ...
    0;


Sel.nEligible = ...
    0;


Sel.selectedIsExactMinimum = ...
    false;


Sel.message = ...
    '';


% =============================================================
% 3. Valid training-set cap results
%
% A cap must:
%
%   1. be marked successful;
%   2. have finite nonnegative RMSE.
% =============================================================

validMask = ...
    cap_success & ...
    isfinite(RMSE_all) & ...
    RMSE_all >= 0;


valid_idx = ...
    find(validMask);


Sel.valid_idx = ...
    valid_idx;


Sel.nValid = ...
    numel(valid_idx);


if isempty(valid_idx)

    Sel.message = ...
        'No valid radius cap has a finite training-set RMSE.';

    return

end


% =============================================================
% 4. Exact minimum training-set RMSE
% =============================================================

minRMSE = ...
    min(RMSE_all(valid_idx));


exactMinimum_idx = ...
    valid_idx( ...
        RMSE_all(valid_idx) == minRMSE);


Sel.minRMSE = ...
    minRMSE;


Sel.exactMinimum_idx = ...
    exactMinimum_idx;


Sel.nExactMinimum = ...
    numel(exactMinimum_idx);


% =============================================================
% 5. Cap-tolerance rule
%
% Eligible caps satisfy:
%
%       RMSE(k) <= minRMSE + tau_cap
% =============================================================

RMSEThreshold = ...
    minRMSE + tau_cap;


eligible_idx = ...
    valid_idx( ...
        RMSE_all(valid_idx) <= RMSEThreshold);


Sel.RMSEThreshold = ...
    RMSEThreshold;


Sel.eligible_idx = ...
    eligible_idx;


Sel.nEligible = ...
    numel(eligible_idx);


if isempty(eligible_idx)

    Sel.message = ...
        'No radius cap satisfies the cap-tolerance rule.';

    return

end


% =============================================================
% 6. Select smallest eligible radius cap
%
% capRadii are strictly increasing.
% Therefore eligible_idx(1) is the smallest eligible cap.
% No sorting is required.
% =============================================================

kStar = ...
    eligible_idx(1);


Sel.capIndex = ...
    kStar;


Sel.capRadius = ...
    capRadii(kStar);


Sel.RMSE = ...
    RMSE_all(kStar);


Sel.selectedIsExactMinimum = ...
    (RMSE_all(kStar) == minRMSE);


% =============================================================
% 7. Success
% =============================================================

Sel.success = ...
    true;


Sel.message = ...
    '';

end
function Sel = selectBestRadiusWithToleranceAtEachRadiusCap( ...
    radii, AE_best, radius_success, kCap, tau_loc, retainDetail)
%SELECTBESTRADIUSWITHTOLERANCEATEACHRADIUSCAP
%
% Select the point-specific radius under one candidate neighbourhood cap.
%
% Among valid optimized radii r(1),...,r(kCap), determine the exact
% minimum optimized absolute error and select the SMALLEST radius satisfying
%
%       AE(r) <= minAE + tau_loc
%
% Optional output control:
%   retainDetail = true  (default) preserves the historical complete
%                       selection-set output.
%   retainDetail = false returns only the scalar information required by
%                       the over-all-caps compact path. Detailed index sets
%                       are not constructed or retained.
%
% IMPORTANT:
%   - parameter optimization has already been completed;
%   - no C or RegParam optimization is performed here;
%   - no RBF solve is performed here;
%   - no conditioning quantity affects selection;
%   - only successful finite radius results are considered;
%   - candidate radii must be strictly increasing;
%   - returned indices refer to the original radii vector;
%   - retainDetail changes retained output only, never the selection rule.

% =============================================================
% 0. Optional detail control
% =============================================================

if nargin < 6 || isempty(retainDetail)
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

if ~isnumeric(radii) || isempty(radii) || ~isvector(radii) || ...
   ~isreal(radii) || any(~isfinite(radii(:))) || any(radii(:) <= 0)
    error('radii must be a nonempty finite positive real vector.');
end

radii = radii(:);
M = numel(radii);

if any(diff(radii) <= 0)
    error('Candidate radii must be strictly increasing.');
end

if ~isnumeric(AE_best) || ~isvector(AE_best) || ~isreal(AE_best) || ...
   numel(AE_best) ~= M
    error('AE_best must be a real numeric vector matching radii.');
end

AE_best = AE_best(:);

if ~(islogical(radius_success) || isnumeric(radius_success)) || ...
   ~isvector(radius_success) || numel(radius_success) ~= M
    error(['radius_success must be a logical or numeric vector ' ...
           'matching radii.']);
end

if isnumeric(radius_success)
    if ~isreal(radius_success) || any(~isfinite(radius_success(:))) || ...
       any(radius_success(:) ~= 0 & radius_success(:) ~= 1)
        error(['Numeric radius_success must contain only ' ...
               'finite values 0 or 1.']);
    end
end

radius_success = logical(radius_success(:));

if ~isnumeric(kCap) || ~isscalar(kCap) || ~isreal(kCap) || ...
   ~isfinite(kCap) || kCap ~= round(kCap) || kCap < 1 || kCap > M
    error('kCap must be an integer between 1 and numel(radii).');
end

if ~isnumeric(tau_loc) || ~isscalar(tau_loc) || ~isreal(tau_loc) || ...
   ~isfinite(tau_loc) || tau_loc < 0
    error('tau_loc must be a finite nonnegative real scalar.');
end

% =============================================================
% 2. Initialize common result
% =============================================================

Sel = struct();
Sel.success = false;
Sel.kCap = kCap;
Sel.capRadius = radii(kCap);
Sel.tau_loc = tau_loc;
Sel.radiusIndex = NaN;
Sel.radius = NaN;
Sel.AE = NaN;
Sel.minAE = NaN;
Sel.errorThreshold = NaN;

% Historical detail fields are created only when requested. In full mode,
% field creation order is preserved for backward compatibility.
if retainDetail
    Sel.cap_idx = (1:kCap).';
    Sel.valid_idx = [];
    Sel.exactMinimum_idx = [];
    Sel.eligible_idx = [];
    Sel.nCandidates = kCap;
    Sel.nValid = 0;
    Sel.nExactMinimum = 0;
    Sel.nEligible = 0;
end

Sel.selectedIsExactMinimum = false;
Sel.message = '';

% =============================================================
% 3. Valid optimized radii within this cap
% =============================================================

% idxCap is numerical working state required by the selection itself.
% In compact mode it is never copied into Sel.
idxCap = (1:kCap).';

validMask = ...
    radius_success(idxCap) & ...
    isfinite(AE_best(idxCap)) & ...
    AE_best(idxCap) >= 0;

valid_idx = idxCap(validMask);

if retainDetail
    Sel.valid_idx = valid_idx;
    Sel.nValid = numel(valid_idx);
end

if isempty(valid_idx)
    Sel.message = 'No valid optimized radius exists within this cap.';
    return
end

% =============================================================
% 4. Exact minimum optimized absolute error
% =============================================================

validAE = AE_best(valid_idx);
minAE = min(validAE);
Sel.minAE = minAE;

if retainDetail
    exactMinimum_idx = valid_idx(validAE == minAE);
    Sel.exactMinimum_idx = exactMinimum_idx;
    Sel.nExactMinimum = numel(exactMinimum_idx);
end

% =============================================================
% 5. Local tolerance rule
% =============================================================

errorThreshold = minAE + tau_loc;
Sel.errorThreshold = errorThreshold;

eligibleMask = validAE <= errorThreshold;

if retainDetail
    eligible_idx = valid_idx(eligibleMask);
    Sel.eligible_idx = eligible_idx;
    Sel.nEligible = numel(eligible_idx);

    if isempty(eligible_idx)
        Sel.message = 'No radius satisfies the local tolerance rule.';
        return
    end

    jStar = eligible_idx(1);
else
    % Avoid constructing the optional eligible-index vector.
    firstEligible = find(eligibleMask,1,'first');

    if isempty(firstEligible)
        Sel.message = 'No radius satisfies the local tolerance rule.';
        return
    end

    jStar = valid_idx(firstEligible);
end

% =============================================================
% 6. Select smallest eligible radius
% =============================================================

Sel.radiusIndex = jStar;
Sel.radius = radii(jStar);
Sel.AE = AE_best(jStar);
Sel.selectedIsExactMinimum = (AE_best(jStar) == minAE);

% =============================================================
% 7. Success
% =============================================================

Sel.success = true;
Sel.message = '';

end

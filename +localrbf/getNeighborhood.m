function Nbh = getNeighborhood( ...
    X, xCenter, radius, min_neighbors, excludeIdx, distAll)
%GETNEIGHBORHOOD
%
% Construct one local radius-based neighbourhood.
%
% If fewer than min_neighbors admissible observations lie inside
% the requested radius, the min_neighbors nearest admissible
% observations are used.
%
% For target-excluded training:
%
%       excludeIdx = iTarget
%
% For ordinary prediction:
%
%       excludeIdx = []
%
% distAll may be supplied when distances from xCenter to all X
% have already been calculated. This avoids recomputing distances
% for every candidate radius of the same target point.
%
% Default:
%
%       min_neighbors = 15
%
% IMPORTANT:
%   - the minimum-neighbour requirement is applied AFTER exclusion;
%   - exclusion is by observation index;
%   - no polynomial-admissibility check is performed here;
%   - no RBF parameter calculation or optimization is performed;
%   - X and radius must already use the same coordinate scale.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 4 || isempty(min_neighbors)

    min_neighbors = ...
        15;

end


if nargin < 5 || isempty(excludeIdx)

    excludeIdx = ...
        [];

end


if nargin < 6

    distAll = ...
        [];

end


% =============================================================
% 2. Basic dimensions
% =============================================================

n = ...
    size(X,1);


% =============================================================
% 3. Initialize output
% =============================================================

Nbh = ...
    struct();


Nbh.success = ...
    false;


Nbh.idx = ...
    [];


Nbh.distances = ...
    [];


Nbh.count = ...
    0;


Nbh.requestedRadius = ...
    radius;


Nbh.effectiveRadius = ...
    NaN;


Nbh.selectedMaxDistance = ...
    NaN;


Nbh.minNeighbors = ...
    min_neighbors;


Nbh.nAvailable = ...
    0;


Nbh.nWithinRadius = ...
    0;


Nbh.nAddedForMinimum = ...
    0;


Nbh.usedMinNeighborFallback = ...
    false;


Nbh.excludeIdx = ...
    [];


Nbh.message = ...
    '';


% =============================================================
% 4. Input checks
% =============================================================

if ~isnumeric(radius) || ...
   ~isscalar(radius) || ...
   ~isreal(radius) || ...
   ~isfinite(radius) || ...
   radius <= 0

    Nbh.message = ...
        'radius must be a finite positive real scalar.';

    return

end


if ~isnumeric(min_neighbors) || ...
   ~isscalar(min_neighbors) || ...
   ~isreal(min_neighbors) || ...
   ~isfinite(min_neighbors) || ...
   min_neighbors < 1 || ...
   min_neighbors ~= round(min_neighbors)

    Nbh.message = ...
        'min_neighbors must be a positive integer.';

    return

end


if ~isempty(excludeIdx)

    if ~isnumeric(excludeIdx) || ...
       ~isvector(excludeIdx) || ...
       any(~isfinite(excludeIdx)) || ...
       any(excludeIdx < 1) || ...
       any(excludeIdx > n) || ...
       any(excludeIdx ~= round(excludeIdx))

        Nbh.message = ...
            'excludeIdx contains invalid observation indices.';

        return

    end


    excludeIdx = ...
        unique(excludeIdx(:));

end


Nbh.excludeIdx = ...
    excludeIdx;


% =============================================================
% 5. Distances
%
% For one target point and multiple candidate radii, distAll
% should normally be calculated once by optimizePointCandidate
% and supplied here.
% =============================================================

if isempty(distAll)

    if ~isnumeric(xCenter) || ...
       size(xCenter,1) ~= 1 || ...
       size(xCenter,2) ~= size(X,2) || ...
       ~isreal(xCenter) || ...
       any(~isfinite(xCenter))

        Nbh.message = ...
            'xCenter must be a finite 1-by-size(X,2) vector.';

        return

    end


    distAll = ...
        pdist2(X,xCenter);


    distAll = ...
        distAll(:);

else

    distAll = ...
        distAll(:);


    if numel(distAll) ~= n || ...
       ~isreal(distAll) || ...
       any(~isfinite(distAll)) || ...
       any(distAll < 0)

        Nbh.message = ...
            ['distAll must contain one finite nonnegative ' ...
             'distance for every row of X.'];

        return

    end

end


% =============================================================
% 6. Admissible observations
%
% Exclusion occurs BEFORE the minimum-neighbour rule.
% =============================================================

available = ...
    true(n,1);


if ~isempty(excludeIdx)

    available(excludeIdx) = ...
        false;

end


nAvailable = ...
    sum(available);


Nbh.nAvailable = ...
    nAvailable;


if nAvailable < min_neighbors

    Nbh.message = ...
        sprintf( ...
            ['Only %d observations remain after exclusion, ' ...
             'but at least %d are required.'], ...
            nAvailable, ...
            min_neighbors);

    return

end


% =============================================================
% 7. Radius-based neighbourhood
% =============================================================

inside = ...
    available & ...
    distAll <= radius;


idxRadius = ...
    find(inside);


nWithinRadius = ...
    numel(idxRadius);


Nbh.nWithinRadius = ...
    nWithinRadius;


% =============================================================
% 8. Enforce minimum number of fitting observations
% =============================================================

if nWithinRadius >= min_neighbors

    idx = ...
        idxRadius;


    Nbh.effectiveRadius = ...
        radius;


else

    % ---------------------------------------------------------
    % Select the min_neighbors nearest admissible observations.
    %
    % Sorting uses:
    %   1. distance
    %   2. original observation index
    %
    % for deterministic treatment of equal distances.
    % ---------------------------------------------------------

    availableIdx = ...
        find(available);


    distanceIndex = ...
        [distAll(availableIdx), ...
         availableIdx];


    distanceIndex = ...
        sortrows( ...
            distanceIndex, ...
            [1 2]);


    idx = ...
        distanceIndex( ...
            1:min_neighbors, ...
            2);


    % Restore original-data order for RBF construction.
    idx = ...
        sort(idx);


    Nbh.usedMinNeighborFallback = ...
        true;


    Nbh.nAddedForMinimum = ...
        min_neighbors - ...
        nWithinRadius;

end


% =============================================================
% 9. Selected-neighbour information
% =============================================================

selectedDistances = ...
    distAll(idx);


selectedMaxDistance = ...
    max(selectedDistances);


Nbh.idx = ...
    idx;


Nbh.distances = ...
    selectedDistances;


Nbh.count = ...
    numel(idx);


Nbh.selectedMaxDistance = ...
    selectedMaxDistance;


if Nbh.usedMinNeighborFallback

    Nbh.effectiveRadius = ...
        selectedMaxDistance;

end


% =============================================================
% 10. Final validity
% =============================================================

if Nbh.count < min_neighbors

    Nbh.message = ...
        'Minimum-neighbour requirement was not satisfied.';

    return

end


if ~isempty(excludeIdx) && ...
   any(ismember(Nbh.idx,excludeIdx))

    Nbh.message = ...
        'Excluded observation remained in the neighbourhood.';

    return

end


Nbh.success = ...
    true;


Nbh.message = ...
    '';

end
function [segments,info] = buildAdaptiveRadialSegments( ...
    distanceToFit,numSegmentsRequested,minPointsPerAnnulus)

%BUILDADAPTIVERADIALSEGMENTS
%
% Build fitting-set-centered radial annuli for spatial bootstrap resampling.
%
% Initial annuli use equal radial widths from zero to the farthest
% fitting observation from the supplied fitting-set spatial centre. Sparse
% adjacent annuli are then merged deterministically until every effective annulus contains at
% least minPointsPerAnnulus observations, or only one annulus remains.
%
% No observation is discarded or duplicated during segmentation.
%
% Inputs
%   distanceToFit        : distances from the fitting-set spatial centre
%                          to all fitting points
%   numSegmentsRequested : requested number of initial radial annuli >= 1
%   minPointsPerAnnulus  : minimum fitting points required per annulus >= 1
%
% Outputs
%   segments : cell array of fitting-set indices for effective annuli
%   info     : requested/effective boundaries and counts

% =============================================================
% 1. Validate inputs
% =============================================================

if ~isnumeric(distanceToFit) || ...
   isempty(distanceToFit) || ...
   ~isvector(distanceToFit) || ...
   ~isreal(distanceToFit) || ...
   any(~isfinite(distanceToFit(:))) || ...
   any(distanceToFit(:) < 0)

    error(['distanceToFit must be a nonempty finite ' ...
           'nonnegative real numeric vector.']);
end


if ~isnumeric(numSegmentsRequested) || ...
   ~isscalar(numSegmentsRequested) || ...
   ~isfinite(numSegmentsRequested) || ...
   numSegmentsRequested < 1 || ...
   numSegmentsRequested ~= round(numSegmentsRequested)

    error('numSegmentsRequested must be an integer >= 1.');
end


if ~isnumeric(minPointsPerAnnulus) || ...
   ~isscalar(minPointsPerAnnulus) || ...
   ~isfinite(minPointsPerAnnulus) || ...
   minPointsPerAnnulus < 1 || ...
   minPointsPerAnnulus ~= round(minPointsPerAnnulus)

    error('minPointsPerAnnulus must be an integer >= 1.');
end


d = ...
    distanceToFit(:);


n = ...
    numel(d);


Krequested = ...
    double(numSegmentsRequested);


Nmin = ...
    double(minPointsPerAnnulus);


minPointsFeasible = ...
    n >= Nmin;


% =============================================================
% 2. Construct requested equal-width radial annuli
% =============================================================

dMax = ...
    max(d);


requestedEdges = ...
    linspace(0,dMax,Krequested + 1);


requestedSegments = ...
    cell(1,Krequested);


if dMax == 0

    % All fitting observations coincide in distance with the spatial centre.
    % Preserve the requested metadata but place all observations in the
    % final requested annulus before deterministic adaptation.
    for k = 1:(Krequested - 1)
        requestedSegments{k} = zeros(0,1);
    end

    requestedSegments{Krequested} = ...
        (1:n).';

else

    for k = 1:Krequested

        if k < Krequested

            idx = ...
                find( ...
                    d >= requestedEdges(k) & ...
                    d <  requestedEdges(k+1));

        else

            idx = ...
                find( ...
                    d >= requestedEdges(k) & ...
                    d <= requestedEdges(k+1));

        end

        requestedSegments{k} = ...
            idx(:);

    end

end


requestedCounts = ...
    cellfun(@numel,requestedSegments);


% -------------------------------------------------------------
% Segmentation integrity: exact partition of 1:n.
% -------------------------------------------------------------

allRequested = ...
    vertcat(requestedSegments{:});


if numel(allRequested) ~= n || ...
   ~isequal(sort(allRequested(:)),(1:n).')

    error(['Internal radial segmentation error: requested annuli ' ...
           'do not form an exact partition of the fitting set.']);
end


% =============================================================
% 3. Deterministically merge sparse adjacent annuli
% =============================================================

segments = ...
    requestedSegments;


effectiveEdges = ...
    requestedEdges;


while numel(segments) > 1

    counts = ...
        cellfun(@numel,segments);


    sparse = ...
        find(counts < Nmin);


    if isempty(sparse)
        break
    end


    % Select the sparsest deficient annulus. MATLAB min returns the first
    % occurrence in a tie, giving deterministic inner-to-outer behavior.
    [~,ii] = ...
        min(counts(sparse));


    kSparse = ...
        sparse(ii);


    Kcurrent = ...
        numel(segments);


    if kSparse == 1

        kNeighbor = 2;

    elseif kSparse == Kcurrent

        kNeighbor = Kcurrent - 1;

    elseif counts(kSparse-1) <= counts(kSparse+1)

        kNeighbor = kSparse - 1;

    else

        kNeighbor = kSparse + 1;

    end


    kLeft = ...
        min(kSparse,kNeighbor);


    kRight = ...
        kLeft + 1;


    % Merge only adjacent radial annuli. Their observation sets remain
    % disjoint, and concatenation cannot duplicate original indices.
    segments{kLeft} = ...
        [segments{kLeft}(:); segments{kRight}(:)];


    segments(kRight) = ...
        [];


    % Remove the common internal radial boundary.
    effectiveEdges(kLeft + 1) = ...
        [];

end


% =============================================================
% 4. Final integrity checks and metadata
% =============================================================

effectiveCounts = ...
    cellfun(@numel,segments);


allEffective = ...
    vertcat(segments{:});


if numel(allEffective) ~= n || ...
   ~isequal(sort(allEffective(:)),(1:n).')

    error(['Internal radial segmentation error: effective annuli ' ...
           'do not form an exact partition of the fitting set.']);
end


if numel(effectiveEdges) ~= numel(segments) + 1

    error('Internal radial segmentation edge/count inconsistency.');
end


if minPointsFeasible && ...
        any(effectiveCounts < Nmin)

    error(['Internal radial segmentation error: minimum-point ' ...
           'requirement was not satisfied after adaptation.']);
end


if ~minPointsFeasible && ...
        (numel(segments) ~= 1 || effectiveCounts(1) ~= n)

    error(['Internal radial segmentation error: infeasible minimum-' ...
           'point case did not reduce to one complete annulus.']);
end


info = struct();


info.NumSegmentsRequested = ...
    Krequested;


info.NumSegmentsEffective = ...
    numel(segments);


info.MinPointsPerAnnulus = ...
    Nmin;


info.RequestedCounts = ...
    requestedCounts(:).';


info.EffectiveCounts = ...
    effectiveCounts(:).';


info.RequestedRadiusEdges = ...
    requestedEdges(:).';


info.EffectiveRadiusEdges = ...
    effectiveEdges(:).';


info.MaxRadius = ...
    dMax;


info.WasAdapted = ...
    info.NumSegmentsEffective < info.NumSegmentsRequested;


info.MinPointsFeasible = ...
    minPointsFeasible;


info.MinPointsSatisfied = ...
    all(effectiveCounts >= Nmin);


info.nObservations = ...
    n;

end

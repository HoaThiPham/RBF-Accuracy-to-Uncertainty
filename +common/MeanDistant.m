function MD = MeanDistant(X)
%MEANDISTANT Compute a geometry-based characteristic spacing.
%
%   X  : n-by-p coordinate matrix
%   MD : geometry-based spacing scale
%
%   MD = (prod(ell) / n)^(1/p)
%
%   where ell is the coordinate range in each dimension.

% =============================================================
% 1. Dimensions
% =============================================================

[n,p] = ...
    size(X);


if n < 1 || p < 1

    error( ...
        'X must be a non-empty coordinate matrix.');

end


% =============================================================
% 2. Coordinate extent
% =============================================================

ell = ...
    max(X,[],1) - min(X,[],1);


if any(~isfinite(ell))

    error( ...
        'Coordinate ranges must be finite.');

end


if any(ell <= 0)

    error( ...
        ['Each coordinate dimension must have positive extent ' ...
         'for the geometry-based spacing calculation.']);

end


% =============================================================
% 3. Geometry-based spacing
% =============================================================

MD = ...
    (prod(ell) / n)^(1/p);


if ~isfinite(MD) || ...
   MD <= 0

    error( ...
        'Invalid geometry-based spacing scale.');

end

end
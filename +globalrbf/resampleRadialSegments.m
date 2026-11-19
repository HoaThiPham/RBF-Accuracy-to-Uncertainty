function idxBoot = resampleRadialSegments(segments,stream,nSample)

%RESAMPLERADIALSEGMENTS
%
% Apply the CURRENT nonparametric bootstrap operation independently to
% each supplied annulus and combine the resulting samples.
%
% For annulus A_k with n_k observations:
%
%   localIdx = randi(stream,n_k,[n_k,1]);
%   S_k      = A_k(localIdx);
%
% The final bootstrap index vector is S=[S_1;...;S_K]. Therefore each
% annulus preserves its original sample size and the complete bootstrap
% sample preserves nSample exactly.

% =============================================================
% 1. Validate lightweight invariants
% =============================================================

if ~iscell(segments) || isempty(segments)
    error('segments must be a nonempty cell array.');
end


if nargin < 2 || isempty(stream) || ...
        ~isa(stream,'RandStream')

    error('stream must be a valid RandStream object.');
end


counts = ...
    cellfun(@numel,segments);


totalCount = ...
    sum(counts);


if nargin < 3 || isempty(nSample)

    nSample = ...
        totalCount;

end


if ~isnumeric(nSample) || ...
   ~isscalar(nSample) || ...
   ~isfinite(nSample) || ...
   nSample < 1 || ...
   nSample ~= round(nSample)

    error('nSample must be a positive integer scalar.');
end


if totalCount ~= nSample

    error(['Radial segments contain %d observations but nSample=%d. ' ...
           'No padding or truncation is permitted.'], ...
          totalCount,nSample);
end


% =============================================================
% 2. Resample each annulus with the exact current randi operation
% =============================================================

idxBoot = ...
    zeros(nSample,1);


pos = ...
    1;


for k = 1:numel(segments)

    idxSegment = ...
        segments{k}(:);


    nk = ...
        numel(idxSegment);


    if nk == 0
        continue
    end


    % EXACT current bootstrap operation, restricted to A_k.
    localIdx = ...
        randi( ...
            stream, ...
            nk, ...
            [nk,1]);


    idxBoot(pos:(pos+nk-1)) = ...
        idxSegment(localIdx);


    pos = ...
        pos + nk;

end


if pos ~= nSample + 1

    error('Internal radial bootstrap sample-size inconsistency.');
end

end

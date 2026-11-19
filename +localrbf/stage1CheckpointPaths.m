function Paths = stage1CheckpointPaths(checkpointFile)
%STAGE1CHECKPOINTPATHS Build deterministic Stage-1 split-file paths.
%
% Input:
%   checkpointFile
%       Existing deterministic Stage-1 checkpoint filename, e.g.
%
%       ...\M001_K002_EGM2008_multiquadric_ga_G20_20.mat
%
% Output:
%   Paths.legacy
%   Paths.core
%   Paths.diag
%   Paths.traceSearch
%   Paths.traceOpt
%   Paths.traceNumeric
%
% This helper changes ONLY Stage-1 storage naming. It does not alter
% optimization, training geometry, RNG, radius selection, or Stage-2
% mathematics.

if ~(ischar(checkpointFile) || ...
     (isstring(checkpointFile) && isscalar(checkpointFile)))

    error( ...
        'checkpointFile must be a character vector or scalar string.');

end

checkpointFile = ...
    char(checkpointFile);

if isempty(strtrim(checkpointFile))

    error( ...
        'checkpointFile must be nonempty.');

end

[folderPart,namePart,extPart] = ...
    fileparts(checkpointFile);

if isempty(namePart)

    error( ...
        'checkpointFile must contain a valid filename.');

end

if isempty(extPart)

    extPart = ...
        '.mat';

elseif ~strcmpi(extPart,'.mat')

    error( ...
        'Stage-1 checkpoint extension must be .mat.');

end

% Preserve the current deterministic filename as a legacy reference.
Paths = ...
    struct();

Paths.legacy = ...
    fullfile( ...
        folderPart, ...
        [namePart extPart]);

% Authoritative compact Stage-1 data required by Stage 2.
Paths.core = ...
    fullfile( ...
        folderPart, ...
        [namePart '_CORE.mat']);

% Compact diagnostics / summaries.
Paths.diag = ...
    fullfile( ...
        folderPart, ...
        [namePart '_DIAG.mat']);

% Search/radius-decision trace.
Paths.traceSearch = ...
    fullfile( ...
        folderPart, ...
        [namePart '_TRACE_SEARCH.mat']);

% Raw optimizer evaluation trace (GA/grid/refinement).
Paths.traceOpt = ...
    fullfile( ...
        folderPart, ...
        [namePart '_TRACE_OPT.mat']);

% Detailed numerical / solver trace.
Paths.traceNumeric = ...
    fullfile( ...
        folderPart, ...
        [namePart '_TRACE_NUMERIC.mat']);

end

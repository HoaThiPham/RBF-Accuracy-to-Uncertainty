function Info = saveStage1PartAtomic(fileName,variableName,value)
%SAVESTAGE1PARTATOMIC Atomically save one Stage-1 storage component.
%
% Examples:
%
%   localrbf.saveStage1PartAtomic( ...
%       Paths.core, ...
%       'Stage1Core', ...
%       Stage1Core);
%
%   localrbf.saveStage1PartAtomic( ...
%       Paths.diag, ...
%       'Stage1Diag', ...
%       Stage1Diag);
%
% The current split-v3 Stage-1 workflow uses this helper for CORE and DIAG.
% Legacy TRACE variable names remain accepted for compatibility, although
% progressive TRACE files are finalized by finalizeStage1ProgressivePart.
%
% The temporary MAT-file is created in the SAME directory as the final
% file. The completed temporary file is moved to the deterministic final
% name only after SAVE succeeds.
%
% This function changes storage only. It does not alter numerical values,
% RNG state, optimization logic, or Stage-2 selection logic.
%
% Memory rule:
%   only one lightweight wrapper reference is created for SAVE -struct;
%   it and the local VALUE reference are released immediately after SAVE.

% =============================================================
% 1. Validate file name
% =============================================================

if ~(ischar(fileName) || ...
     (isstring(fileName) && isscalar(fileName)))

    error( ...
        'fileName must be a character vector or scalar string.');

end


fileName = ...
    char(fileName);


if isempty(strtrim(fileName))

    error( ...
        'fileName must be nonempty.');

end


[checkpointDir,~,extPart] = ...
    fileparts(fileName);


if ~strcmpi(extPart,'.mat')

    error( ...
        'Stage-1 component filename must use the .mat extension.');

end


if isempty(checkpointDir)

    checkpointDir = ...
        pwd;

end


if ~isfolder(checkpointDir)

    error( ...
        'Stage-1 checkpoint directory does not exist: %s', ...
        checkpointDir);

end


% =============================================================
% 2. Validate top-level variable name
% =============================================================

if ~(ischar(variableName) || ...
     (isstring(variableName) && isscalar(variableName)))

    error( ...
        'variableName must be a character vector or scalar string.');

end


variableName = ...
    char(variableName);


if ~isvarname(variableName)

    error( ...
        'Invalid MATLAB variable name for Stage-1 component: %s', ...
        variableName);

end


allowedVariableNames = ...
    { ...
    'Stage1Core', ...
    'Stage1Diag', ...
    'Stage1TraceSearch', ...
    'Stage1TraceOpt', ...
    'Stage1TraceNumeric' ...
    };


if ~any(strcmp(variableName,allowedVariableNames))

    error( ...
        ['Unsupported Stage-1 component variable "%s". ' ...
         'Expected one of: %s.'], ...
        variableName, ...
        strjoin(allowedVariableNames,', '));

end


% =============================================================
% 3. Validate value
%
% Stage-1 split components use scalar structures at the top level.
% Keeping this invariant makes loading and integrity checks deterministic.
% =============================================================

if ~isstruct(value) || ...
   ~isscalar(value)

    error( ...
        '%s must be a scalar structure.', ...
        variableName);

end


% =============================================================
% 4. Create temporary file in the SAME directory
% =============================================================

temporaryFile = ...
    [tempname(checkpointDir) '.mat'];


cleanupObj = ...
    onCleanup( ...
        @() deleteIfExists(temporaryFile)); %#ok<NASGU>


% =============================================================
% 5. Save exactly one top-level variable
%
% SAVE -struct gives the MAT-file the authoritative variable name
% Stage1Core / Stage1Diag / ... rather than a generic payload name.
% The wrapper uses MATLAB copy-on-write semantics; no intentional deep
% copy of VALUE is made.
% =============================================================

saveContainer = ...
    struct();


saveContainer.(variableName) = ...
    value;


save( ...
    temporaryFile, ...
    '-struct', ...
    'saveContainer', ...
    variableName, ...
    '-v7.3');


% The serialized data are now on disk. Release local references before
% filesystem validation/finalization so large CORE/DIAG values are not
% held here longer than necessary.
clear saveContainer value


% =============================================================
% 6. Minimal integrity check before publication
%
% SAVE returning successfully verifies serialization. Check only existence
% and nonzero size here; avoid rereading MAT metadata/payload because this
% helper may also be used with large legacy components and save time is a
% priority.
% =============================================================

fileInfo = ...
    dir(temporaryFile);


if isempty(fileInfo) || ...
   ~isscalar(fileInfo) || ...
   fileInfo.bytes <= 0

    error( ...
        'Temporary Stage-1 component file is missing or empty: %s', ...
        temporaryFile);

end


% =============================================================
% 7. Publish completed file
%
% Temporary and final files are in the same directory, so MOVEFILE does
% not intentionally cross filesystems. The old final file, if present, is
% replaced only after the new temporary file has been fully written.
% =============================================================

[ok,msg] = ...
    movefile( ...
        temporaryFile, ...
        fileName, ...
        'f');


if ~ok

    error( ...
        'Could not finalize Stage-1 component "%s": %s', ...
        variableName, ...
        msg);

end


% =============================================================
% 8. Verify published file and return compact save information
% =============================================================

finalInfo = ...
    dir(fileName);


if isempty(finalInfo) || ...
   ~isscalar(finalInfo) || ...
   finalInfo.bytes <= 0

    error( ...
        ['Stage-1 component "%s" was not published as a nonempty ' ...
         'file: %s'], ...
        variableName, ...
        fileName);

end


Info = ...
    struct();


Info.success = ...
    true;


Info.file = ...
    fileName;


Info.variableName = ...
    variableName;


Info.bytes = ...
    finalInfo.bytes;

end


% =============================================================
% Local helper
% =============================================================

function deleteIfExists(fileName)
%DELETEIFEXISTS Best-effort cleanup for an interrupted atomic save.

if isfile(fileName)

    try

        delete(fileName);

    catch

        % Best-effort cleanup only. Never mask the original exception.

    end

end


end

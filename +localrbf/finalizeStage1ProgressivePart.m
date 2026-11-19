function finalizeStage1ProgressivePart(State,metadata)
%FINALIZESTAGE1PROGRESSIVEPART Publish one progressive Stage-1 TRACE part.
%
% The temporary v7.3 file already contains the incrementally written
% top-level Point cell array. This function updates only the compact
% metadata variable and then moves the temporary file to the deterministic
% final filename.
%
% No Point payload is loaded into RAM here.
%
% CORE is published separately and LAST by LocalRBFTraining, so CORE remains
% the commit marker for a complete split Stage-1 checkpoint record.

% =============================================================
% 1. Validate state
% =============================================================

requiredStateFields = ...
    {'tempFile','finalFile','variableName','nPoints'};


if ~isstruct(State) || ...
   ~isscalar(State)

    error( ...
        'State must be a scalar structure.');

end


for k = 1:numel(requiredStateFields)

    if ~isfield(State,requiredStateFields{k})

        error( ...
            'Progressive Stage-1 state lacks field "%s".', ...
            requiredStateFields{k});

    end

end


if ~(ischar(State.tempFile) || ...
     (isstring(State.tempFile) && isscalar(State.tempFile))) || ...
   isempty(char(State.tempFile))

    error( ...
        'State.tempFile must be a nonempty character vector or scalar string.');

end


if ~(ischar(State.finalFile) || ...
     (isstring(State.finalFile) && isscalar(State.finalFile))) || ...
   isempty(char(State.finalFile))

    error( ...
        'State.finalFile must be a nonempty character vector or scalar string.');

end


tempFile = ...
    char(State.tempFile);


finalFile = ...
    char(State.finalFile);


if strcmp(tempFile,finalFile)

    error( ...
        'Progressive Stage-1 temporary and final filenames must differ.');

end


if ~isfile(tempFile)

    error( ...
        'Progressive Stage-1 temporary file was not found: %s', ...
        tempFile);

end


[finalDir,~,finalExt] = ...
    fileparts(finalFile);


if isempty(finalDir)

    finalDir = ...
        pwd;

end


if ~isfolder(finalDir)

    error( ...
        'Stage-1 checkpoint directory does not exist: %s', ...
        finalDir);

end


if ~strcmpi(finalExt,'.mat')

    error( ...
        'Progressive Stage-1 final component must use the .mat extension.');

end


if ~(ischar(State.variableName) || ...
     (isstring(State.variableName) && isscalar(State.variableName)))

    error( ...
        'State.variableName must be a character vector or scalar string.');

end


variableName = ...
    char(State.variableName);


allowedVariableNames = ...
    { ...
    'Stage1TraceSearch', ...
    'Stage1TraceOpt', ...
    'Stage1TraceNumeric' ...
    };


if ~any(strcmp(variableName,allowedVariableNames))

    error( ...
        'Unsupported progressive Stage-1 variable: %s', ...
        variableName);

end


if ~isnumeric(State.nPoints) || ...
   ~isscalar(State.nPoints) || ...
   ~isreal(State.nPoints) || ...
   ~isfinite(State.nPoints) || ...
   State.nPoints < 1 || ...
   State.nPoints ~= round(State.nPoints)

    error( ...
        'State.nPoints must be a positive integer.');

end


nPoints = ...
    double(State.nPoints);


% =============================================================
% 2. Validate compact metadata
% =============================================================

if ~isstruct(metadata) || ...
   ~isscalar(metadata)

    error( ...
        'metadata must be a scalar structure.');

end


if ~isfield(metadata,'complete') || ...
   ~isscalar(metadata.complete) || ...
   ~(islogical(metadata.complete) || ...
     (isnumeric(metadata.complete) && ...
      isreal(metadata.complete) && ...
      isfinite(metadata.complete) && ...
      any(metadata.complete == [0 1])))

    error( ...
        ['Progressive Stage-1 metadata must contain scalar logical ' ...
         'complete status (or numeric 1/0).']);

end


if isfield(metadata,'nPoints') && ...
   ~isequaln(double(metadata.nPoints),nPoints)

    error( ...
        'Progressive Stage-1 metadata nPoints is inconsistent with State.');

end


expectedRole = ...
    expectedRoleForVariable(variableName);


if isfield(metadata,'storageRole') && ...
   ~strcmp(char(metadata.storageRole),expectedRole)

    error( ...
        ['Progressive Stage-1 metadata storageRole "%s" does not match ' ...
         'variable "%s".'], ...
        char(metadata.storageRole), ...
        variableName);

end


% =============================================================
% 3. Verify the progressive payload layout without loading it
% =============================================================

W = ...
    whos( ...
        '-file', ...
        tempFile);


names = ...
    {W.name};


idxMeta = ...
    find(strcmp(names,variableName),1);


idxPoint = ...
    find(strcmp(names,'Point'),1);


if isempty(idxMeta)

    error( ...
        'Progressive Stage-1 temporary file lacks metadata variable %s: %s', ...
        variableName, ...
        tempFile);

end


if isempty(idxPoint)

    error( ...
        'Progressive Stage-1 temporary file lacks top-level Point payload: %s', ...
        tempFile);

end


if ~strcmp(W(idxPoint).class,'cell') || ...
   ~isequal(W(idxPoint).size,[nPoints 1])

    error( ...
        ['Progressive Stage-1 Point payload has invalid type or size in %s. ' ...
         'Expected cell(%d,1).'], ...
        tempFile, ...
        nPoints);

end


% =============================================================
% 4. Update metadata only
%
% The large Point cell remains in-place. SAVE -APPEND updates only the
% compact metadata variable and never materializes the complete trace in
% MATLAB workspace memory.
% =============================================================

C = ...
    struct();


C.(variableName) = ...
    metadata;


save( ...
    tempFile, ...
    '-struct', ...
    'C', ...
    variableName, ...
    '-append');


clear C


% =============================================================
% 5. Final file sanity check and atomic publication
% =============================================================

fileInfo = ...
    dir(tempFile);


if isempty(fileInfo) || ...
   fileInfo.bytes <= 0

    error( ...
        'Progressive Stage-1 temporary file is empty: %s', ...
        tempFile);

end


[ok,msg] = ...
    movefile( ...
        tempFile, ...
        finalFile, ...
        'f');


if ~ok

    error( ...
        'Could not finalize progressive Stage-1 component: %s', ...
        msg);

end


end


% =============================================================
% Local helper
% =============================================================

function role = expectedRoleForVariable(variableName)
%EXPECTEDROLEFORVARIABLE Keep variable/role mapping authoritative here.

switch variableName

    case 'Stage1TraceSearch'

        role = ...
            'TRACE_SEARCH';

    case 'Stage1TraceOpt'

        role = ...
            'TRACE_OPT';

    case 'Stage1TraceNumeric'

        role = ...
            'TRACE_NUMERIC';

    otherwise

        error( ...
            'Unsupported progressive Stage-1 variable: %s', ...
            variableName);

end

end

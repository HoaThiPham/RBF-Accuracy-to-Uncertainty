function State = openStage1ProgressivePart( ...
    finalFile,variableName,metadata,nPoints)
%OPENSTAGE1PROGRESSIVEPART Create a temporary writable v7.3 Stage-1 part.
%
% The temporary file is created in the SAME directory as FINALFILE.
% Point payloads are subsequently written incrementally through MATFILE:
%
%   M = matfile(State.tempFile,'Writable',true);
%   M.Point(iTarget,1) = {PointTrace};
%
% The deterministic final filename is NOT published here. Publication is
% performed only by localrbf.finalizeStage1ProgressivePart after Stage 1
% has completed successfully.
%
% This helper is storage-only and does not touch MATLAB RNG.

if ~(ischar(finalFile) || ...
     (isstring(finalFile) && isscalar(finalFile)))

    error( ...
        'finalFile must be a character vector or scalar string.');

end


finalFile = ...
    char(finalFile);


[checkpointDir,~,extPart] = ...
    fileparts(finalFile);


if isempty(checkpointDir)

    checkpointDir = ...
        pwd;

end


if ~isfolder(checkpointDir)

    error( ...
        'Stage-1 checkpoint directory does not exist: %s', ...
        checkpointDir);

end


if ~strcmpi(extPart,'.mat')

    error( ...
        'Stage-1 progressive component must use the .mat extension.');

end


if ~(ischar(variableName) || ...
     (isstring(variableName) && isscalar(variableName)))

    error( ...
        'variableName must be a character vector or scalar string.');

end


variableName = ...
    char(variableName);


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


if ~isstruct(metadata) || ...
   ~isscalar(metadata)

    error( ...
        'metadata must be a scalar structure.');

end


if ~isnumeric(nPoints) || ...
   ~isscalar(nPoints) || ...
   ~isreal(nPoints) || ...
   ~isfinite(nPoints) || ...
   nPoints < 1 || ...
   nPoints ~= round(nPoints)

    error( ...
        'nPoints must be a positive integer.');

end


temporaryFile = ...
    [tempname(checkpointDir) '.mat'];


% Write the small metadata object first, then pre-size the top-level Point
% cell. If either operation fails, remove the temporary file here. The
% caller cannot clean a temporary path that was never successfully returned
% in State.
try

    C = ...
        struct();


    C.(variableName) = ...
        metadata;


    save( ...
        temporaryFile, ...
        '-struct', ...
        'C', ...
        variableName, ...
        '-v7.3');


    % Pre-size without constructing the full trace in RAM. Individual
    % Point cells will be replaced as soon as each target finishes.
    M = ...
        matfile( ...
            temporaryFile, ...
            'Writable', ...
            true);


    M.Point(nPoints,1) = ...
        {[]};


    clear M C

catch ME

    % Release any MATFILE handle before deleting the temporary file,
    % particularly important on platforms with stricter file locking.
    clear M C


    if isfile(temporaryFile)

        try

            delete(temporaryFile);

        catch

            % Preserve the original initialization exception. Cleanup is
            % best-effort only; do not mask the actual storage failure.

        end

    end


    rethrow(ME)

end


State = ...
    struct();


State.tempFile = ...
    temporaryFile;


State.finalFile = ...
    finalFile;


State.variableName = ...
    variableName;


State.nPoints = ...
    nPoints;

end

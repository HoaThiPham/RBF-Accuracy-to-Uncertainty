%% ============================================================
% RUN LOCAL RBF BOOTSTRAP
%
% Global-bootstrap-consistent runner for LOCAL RBF.
%
% IMPORTANT:
%   - ALL user-controlled input/output files are defined in Section 0.
%   - Bootstrap consumes fixed pointwise Radius / RegParam / C from the
%     completed main local-RBF output.
%   - Bootstrap does NOT redo transfer, tau_loc/tau_cap selection,
%     Stage-1 optimization, grid search, or GA.
%   - One PARFOR iteration = one Model x Kernel combination.
%   - Bootstrap replicates remain serial inside each worker.
%% ============================================================

clearvars
clc


%% ============================================================
% 0. ALL USER INPUTS
%% ============================================================

% -------------------------------------------------------------
% 0.1 CASES TO RUN
% -------------------------------------------------------------

RUN_CASE1 = ...
     true;

RUN_CASE2 = ...
       true;


% -------------------------------------------------------------
% 0.2 INPUT DATA FILES
%
% Exact MAT-file and variable names are controlled here.
% No input-data filename is hard-coded later in the runner.
% -------------------------------------------------------------

InputDataFileCase1 = ...
    fullfile('OutPutSplitTrainTest','InputData_Case1.mat');

InputDataVariableCase1 = ...
    'InputData_Case1';

InputDataFileCase2 = ...
    fullfile('OutPutSplitTrainTest','InputData_Case2.mat');

InputDataVariableCase2 = ...
    'InputData_Case2';


% -------------------------------------------------------------
% 0.3 MAIN LOCAL-RBF RESULT FILES
%
% MainResultDir is used for:
%   (a) resolving a relative ResultsFileCase#; or
%   (b) automatic search when ResultsFileCase# is empty.
%
% Preferred for maximum reproducibility:
%   set ResultsFileCase1 / ResultsFileCase2 explicitly.
%
% If a ResultsFileCase# is empty, the corresponding ResultsPatternCase#
% is used. Automatic selection is allowed ONLY when exactly one file
% matches. Multiple matches cause an error rather than a silent choice.
% -------------------------------------------------------------

MainResultDir = ...
    'OutPut_RBF_2026_2';

ResultsFileCase1 = ...
    '';

ResultsVariableCase1 = ...
    'Results';

ResultsPatternCase1 = ...
    'LocalRBF_Case1_Full_TuneReg_IDW.mat';

ResultsFileCase2 = ...
    '';

ResultsVariableCase2 = ...
    'Results';

ResultsPatternCase2 = ...
    'LocalRBF_Case2_Full_TuneReg_IDW.mat';


% -------------------------------------------------------------
% 0.4 OUTPUT FILES / DIRECTORY
% -------------------------------------------------------------

OutputDir = ...
    'OutPut_RBF_2026_2';

OutputFileNameCase1 = ...
    'Case1_Local_Bootstrap_IDW.mat';

OutputFileNameCase2 = ...
    'Case2_Local_Bootstrap_IDW.mat';

% Separate bootstrap configurations to prevent accidental overwrite.
UseConfigurationSubDir = ...
    true;


% -------------------------------------------------------------
% 0.5 PARALLEL WORKERS
%
% One PARFOR iteration = one Model x Kernel combination.
% Bootstrap replicates are serial inside each worker.
% -------------------------------------------------------------

NUM_WORKERS = ...
    3;


% -------------------------------------------------------------
% 0.6 BOOTSTRAP SETTINGS
% -------------------------------------------------------------

nBootstrap = ...
    200;

verbose = ...
    true;

RNG_SEED = ...
    1;


% -------------------------------------------------------------
% 0.7 BOOTSTRAP OPTIONS
% -------------------------------------------------------------

bootstrapOptions = ...
    struct();

% Inner point/bootstrap printing. Keep false for faster parallel runs.
bootstrapOptions.Verbose = ...
    false;

bootstrapOptions.MaxAttemptsFactor = ...
    10;

% Seeds are saved, so storing every sampled index is normally unnecessary
% and can substantially increase MAT-file size.
bootstrapOptions.StoreIndices = ...
    false;

% false = compact result in final combined MAT file.
% true  = retain replicate-level details; full detail is checkpointed.
bootstrapOptions.SaveAll = ...
    false;

% 0  = ordinary local bootstrap.
% >0 = target-relative adaptive annular bootstrap within each fixed
%      local neighbourhood, using the same resampling convention as global.
bootstrapOptions.NumSpatialSegments = ...
    0;

bootstrapOptions.MinPointsPerAnnulus = ...
    15;

% Training bootstrap source:
%   'original'
%       final original-training Radius / RegParam / C.
%
%   'pseudo-unseen'
%       pointwise pseudo-unseen Radius / RegParam / C already stored by
%       the upstream/main local workflow. Bootstrap does NOT reconstruct
%       transfer. If those upstream fields are absent, training bootstrap
%       stops with a clear error rather than silently changing definition.
bootstrapOptions.TrainingMode = ...
    'pseudo-unseen';


%% ============================================================
% 1. VALIDATE USER INPUTS / RESOLVE OUTPUT DIRECTORY
%% ============================================================

if ~RUN_CASE1 && ~RUN_CASE2
    error('At least one case must be enabled.');
end

if ~isnumeric(NUM_WORKERS) || ...
        ~isscalar(NUM_WORKERS) || ...
        ~isreal(NUM_WORKERS) || ...
        ~isfinite(NUM_WORKERS) || ...
        NUM_WORKERS < 1 || ...
        NUM_WORKERS ~= round(NUM_WORKERS)

    error('NUM_WORKERS must be a positive integer.');
end

if ~isnumeric(nBootstrap) || ...
        ~isscalar(nBootstrap) || ...
        ~isreal(nBootstrap) || ...
        ~isfinite(nBootstrap) || ...
        nBootstrap < 2 || ...
        nBootstrap ~= round(nBootstrap)

    error('nBootstrap must be an integer >= 2.');
end

if ~isnumeric(RNG_SEED) || ...
        ~isscalar(RNG_SEED) || ...
        ~isreal(RNG_SEED) || ...
        ~isfinite(RNG_SEED) || ...
        RNG_SEED < 0 || ...
        RNG_SEED > (2^32-1) || ...
        RNG_SEED ~= round(RNG_SEED)

    error(['RNG_SEED must be a finite nonnegative integer ' ...
           'not exceeding 2^32-1.']);
end

if ~(isscalar(verbose) && ...
        (islogical(verbose) || ...
        (isnumeric(verbose) && isreal(verbose) && ...
         isfinite(verbose) && any(verbose == [0 1]))))

    error('verbose must be true/false (or numeric 1/0).');
end

verbose = ...
    logical(verbose);

if ~(isscalar(UseConfigurationSubDir) && ...
        (islogical(UseConfigurationSubDir) || ...
        (isnumeric(UseConfigurationSubDir) && ...
         isreal(UseConfigurationSubDir) && ...
         isfinite(UseConfigurationSubDir) && ...
         any(UseConfigurationSubDir == [0 1]))))

    error('UseConfigurationSubDir must be true/false (or numeric 1/0).');
end

UseConfigurationSubDir = ...
    logical(UseConfigurationSubDir);

trainingMode = ...
    lower(strtrim(char(string(bootstrapOptions.TrainingMode))));

if ~any(strcmp(trainingMode,{'original','pseudo-unseen'}))
    error(['bootstrapOptions.TrainingMode must be ''original'' ' ...
           'or ''pseudo-unseen''.']);
end

bootstrapOptions.TrainingMode = ...
    trainingMode;

if ~isfolder(OutputDir)
    mkdir(OutputDir);
end

if UseConfigurationSubDir

    ModeTag = ...
        strrep(trainingMode,'-','_');

    BootstrapRunDir = ...
        fullfile( ...
            OutputDir, ...
            sprintf( ...
                'LocalBootstrap_%s_B%d_S%d_IDW', ...
                ModeTag, ...
                nBootstrap, ...
                bootstrapOptions.NumSpatialSegments));

else

    BootstrapRunDir = ...
        OutputDir;

end

if ~isfolder(BootstrapRunDir)
    mkdir(BootstrapRunDir);
end

OutputFileCase1 = ...
    fullfile(BootstrapRunDir,OutputFileNameCase1);

OutputFileCase2 = ...
    fullfile(BootstrapRunDir,OutputFileNameCase2);


%% ============================================================
% 2. VALIDATE / RESOLVE ALL INPUT FILES BEFORE STARTING PARALLEL WORK
%% ============================================================
%
% Fail early before opening workers or starting a long bootstrap run.

if RUN_CASE1

    InputDataFileCase1 = ...
        requireInputFile( ...
            InputDataFileCase1, ...
            'Case 1 InputData');

    ResultsFileCase1Resolved = ...
        resolveMainLocalFile( ...
            ResultsFileCase1, ...
            MainResultDir, ...
            ResultsPatternCase1, ...
            'Case1');

    requireVariableInMat( ...
        InputDataFileCase1, ...
        InputDataVariableCase1, ...
        'Case 1 InputData');

    requireVariableInMat( ...
        ResultsFileCase1Resolved, ...
        ResultsVariableCase1, ...
        'Case 1 local-RBF Results');

else

    ResultsFileCase1Resolved = ...
        '';

end

if RUN_CASE2

    InputDataFileCase2 = ...
        requireInputFile( ...
            InputDataFileCase2, ...
            'Case 2 InputData');

    ResultsFileCase2Resolved = ...
        resolveMainLocalFile( ...
            ResultsFileCase2, ...
            MainResultDir, ...
            ResultsPatternCase2, ...
            'Case2');

    requireVariableInMat( ...
        InputDataFileCase2, ...
        InputDataVariableCase2, ...
        'Case 2 InputData');

    requireVariableInMat( ...
        ResultsFileCase2Resolved, ...
        ResultsVariableCase2, ...
        'Case 2 local-RBF Results');

else

    ResultsFileCase2Resolved = ...
        '';

end


%% ============================================================
% 3. PARALLEL CONFIGURATION
%% ============================================================
%
% Pass the requested worker count into the batch.  The batch performs all
% client-side compatibility/preflight checks BEFORE opening/replacing the
% pool, then creates/reuses the requested Processes pool only when needed.

bootstrapOptions.NumWorkers = NUM_WORKERS;


%% ============================================================
% 4. CASE 1
%% ============================================================

if RUN_CASE1

    InputData_Case1 = ...
        loadNamedVariable( ...
            InputDataFileCase1, ...
            InputDataVariableCase1);

    ResultsCase1 = ...
        loadNamedVariable( ...
            ResultsFileCase1Resolved, ...
            ResultsVariableCase1);

    BootstrapCase1 = ...
        localrbf.localRBFBootstrapBatchSummaryParallel( ...
            InputData_Case1, ...
            ResultsCase1, ...
            nBootstrap, ...
            verbose, ...
            RNG_SEED, ...
            bootstrapOptions, ...
            OutputFileCase1);

    % Release large source data before Case 2.
    clear InputData_Case1 ResultsCase1

end


%% ============================================================
% 5. CASE 2
%% ============================================================

if RUN_CASE2

    InputData_Case2 = ...
        loadNamedVariable( ...
            InputDataFileCase2, ...
            InputDataVariableCase2);

    ResultsCase2 = ...
        loadNamedVariable( ...
            ResultsFileCase2Resolved, ...
            ResultsVariableCase2);

    BootstrapCase2 = ...
        localrbf.localRBFBootstrapBatchSummaryParallel( ...
            InputData_Case2, ...
            ResultsCase2, ...
            nBootstrap, ...
            verbose, ...
            RNG_SEED, ...
            bootstrapOptions, ...
            OutputFileCase2);

    clear InputData_Case2 ResultsCase2

end


%% ============================================================
% 6. FINISHED
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' LOCAL RBF BOOTSTRAP FINISHED\n');
fprintf(' Training mode : %s\n',bootstrapOptions.TrainingMode);
fprintf(' nBootstrap    : %d\n',nBootstrap);
fprintf(' NumWorkers    : %d\n',NUM_WORKERS);
fprintf(' OutputDir     : %s\n',BootstrapRunDir);

if RUN_CASE1
    fprintf(' Case 1 input  : %s\n',InputDataFileCase1);
    fprintf(' Case 1 main   : %s\n',ResultsFileCase1Resolved);
    fprintf(' Case 1 output : %s\n',OutputFileCase1);
end

if RUN_CASE2
    fprintf(' Case 2 input  : %s\n',InputDataFileCase2);
    fprintf(' Case 2 main   : %s\n',ResultsFileCase2Resolved);
    fprintf(' Case 2 output : %s\n',OutputFileCase2);
end

fprintf('============================================================\n');


%% ============================================================
% LOCAL HELPERS
%% ============================================================

function fileName = requireInputFile(fileName,label)

fileName = ...
    strtrim(char(string(fileName)));

if isempty(fileName)
    error('%s file is empty.',label);
end

if ~isfile(fileName)
    error('%s file was not found:\n%s',label,fileName);
end

end


function fileName = resolveMainLocalFile( ...
    fileName, ...
    resultDir, ...
    pattern, ...
    caseName)

fileName = ...
    strtrim(char(string(fileName)));

resultDir = ...
    strtrim(char(string(resultDir)));

pattern = ...
    strtrim(char(string(pattern)));

if ~isempty(fileName)

    % First respect an exact/full path supplied by the user.
    if isfile(fileName)
        return
    end

    % For convenience, also resolve a relative filename inside MainResultDir.
    candidate = ...
        fullfile(resultDir,fileName);

    if isfile(candidate)
        fileName = ...
            candidate;
        return
    end

    error([ ...
        'Main local-RBF result file for %s was not found.\n' ...
        'Checked:\n  %s\n  %s'], ...
        caseName, ...
        fileName, ...
        candidate);

end

if isempty(resultDir)
    error('MainResultDir must be nonempty when ResultsFile%s is empty.',caseName);
end

if isempty(pattern)
    error('ResultsPattern%s must be nonempty when ResultsFile%s is empty.',caseName,caseName);
end

if ~isfolder(resultDir)
    error('MainResultDir was not found:\n%s',resultDir);
end

D = ...
    dir(fullfile(resultDir,pattern));

D = ...
    D(~[D.isdir]);

if isempty(D)

    error([ ...
        'No main local-RBF file matched for %s:\n%s\n' ...
        'Set ResultsFile%s explicitly in Section 0.'], ...
        caseName, ...
        fullfile(resultDir,pattern), ...
        caseName);

end

if numel(D) > 1

    names = ...
        string({D.name});

    error([ ...
        'Multiple main local-RBF files matched for %s.\n' ...
        'For accuracy, no file was selected automatically.\n' ...
        'Candidates:\n  %s\n' ...
        'Set ResultsFile%s explicitly in Section 0.'], ...
        caseName, ...
        strjoin(cellstr(names),'\n  '), ...
        caseName);

end

fileName = ...
    fullfile(D(1).folder,D(1).name);

end


function requireVariableInMat(fileName,variableName,label)

variableName = ...
    strtrim(char(string(variableName)));

if isempty(variableName)
    error('%s variable name is empty.',label);
end

info = ...
    whos('-file',fileName);

names = ...
    {info.name};

if ~any(strcmp(names,variableName))

    error([ ...
        '%s variable "%s" was not found in:\n%s\n' ...
        'Available variables:\n  %s'], ...
        label, ...
        variableName, ...
        fileName, ...
        strjoin(names,'\n  '));

end

end


function value = loadNamedVariable(fileName,variableName)

S = ...
    load(fileName,variableName);

if ~isfield(S,variableName)
    error('Variable "%s" was not loaded from:\n%s',variableName,fileName);
end

value = ...
    S.(variableName);

end

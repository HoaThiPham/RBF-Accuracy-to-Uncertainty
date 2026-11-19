function Results = Run_Ablation_MP_Case(CaseNumber,RootDir,NUM_WORKERS,FIXED_RADIUS_RAW,Options)
%RUN_ABLATION_MP_CASE
%
% Manuscript configuration: M_P (adaptive parameters only).
%
% M_P -- USER-FIXED LOCAL RADIUS + POINT-SPECIFIC PARAMETER TUNING
%           + AUTHORITATIVE MODEL-1 TRAIN-TO-TEST TRANSFER
%
% REVISED ABLATION IMPLEMENTATION
% ------------------------------
% M_P independently re-optimizes the point-specific kernel parameters at
% ONE pre-specified fixed radius R0. The tuning bounds are supplied by the
% master and are exactly the same C/RegParam ranges used to generate the
% random fixed-parameter references for M0/MR.
%
% Scientific definition
% ---------------------
% Radius:
%       R = R0 for EVERY training point and EVERY independent test point.
%       R0 is fixed before evaluating predictive accuracy; M_P performs
%       no radius selection or radius optimization.
%
% Training:
%   Gaussian / Multiquadric:
%       each training point independently tunes (C_i,RegParam_i) at R0.
%
%   Linear / Cubic / Thinplate:
%       each training point independently tunes RegParam_i at R0.
%
%   The target observation is excluded from its local fitting neighbourhood,
%   but its prediction error is the parameter-search objective at that point.
%
% Test:
%       the point-specific training parameters are transferred with the
%       authoritative Model-1 transfer rule. Because all donor radii equal
%       R0, the test fitting radius must remain exactly R0. For shaped
%       kernels epsilon is reconstructed from the actual R0 query
%       neighbourhood:
%
%           epsilon_q = C_q * d_q(R0).
%
% IMPORTANT
% ---------
%   * ONE fixed radius R0 for BOTH training and testing.
%   * R0 is identical to the fixed radius used by M0 for the same reference.
%   * Parameter tuning uses the SAME bounds as random fixed-reference
%     generation: C=[0.001,1.5] for shaped kernels and
%     RegParam=[1e-6,1] for all kernels (passed by the master).
%   * minNeighbors = 15 fallback.
%   * NO adaptive radius / cap selection / pseudo-unseen / tau_loc/tau_cap.
%   * NO test-data tuning or test-radius selection.
%   * Model-1 Stage-1 parameter optima are NOT reused.
%   * The authoritative Model-1 parameter-transfer rule is retained.
%   * GGM x kernel jobs are parallelized on one shared process pool.
%
if nargin < 1 || isempty(CaseNumber), CaseNumber = 1; end
if nargin < 2 || isempty(RootDir), RootDir = pwd; end
if nargin < 3 || isempty(NUM_WORKERS), NUM_WORKERS = 4; end
if nargin < 5 || isempty(Options), Options = struct(); end
Options = normalizeOptionsMP(Options);
assert(nargin >= 4 && ~isempty(FIXED_RADIUS_RAW), ...
    ['M_P requires a USER-SPECIFIED fixed radius. ' ...
     'Use the master random-robustness runner or pass R0 explicitly.']);
assert(ismember(CaseNumber,[1 2]),'CaseNumber must be 1 or 2.');
assert(isscalar(NUM_WORKERS) && NUM_WORKERS>=1 && NUM_WORKERS==round(NUM_WORKERS), ...
    'NUM_WORKERS must be a positive integer.');

%% 0. M_P SETTINGS
FIXED_RADIUS_RAW = double(FIXED_RADIUS_RAW);
assert(isscalar(FIXED_RADIUS_RAW) && isfinite(FIXED_RADIUS_RAW) && FIXED_RADIUS_RAW>0, ...
    'FIXED_RADIUS_RAW must be one positive finite scalar in metres.');

REQUIRED_RADIUS_GRID_RAW = (50000:10000:200000).';
tolR = 1e-10*max(1,abs(FIXED_RADIUS_RAW));
assert(any(abs(REQUIRED_RADIUS_GRID_RAW-FIXED_RADIUS_RAW)<=tolR), ...
    'M_P fixed radius must be one exact grid value 50:10:200 km.');


auditDependenciesMPAblation();

MIN_NEIGHBORS = 15;
USE_PARALLEL = Options.UseParallel;
ALLOW_SERIAL_FALLBACK = Options.AllowSerialFallback;
RESTART_PARALLEL_POOL = false;

caseToken = sprintf('Case%d',CaseNumber);
MAIN_RESULTS_FILE = resolveMainResultsFile(RootDir,caseToken);
INPUT_DATA_FILE = fullfile(RootDir,'OutPutSplitTrainTest', ...
    sprintf('InputData_%s.mat',caseToken));
INPUT_VARIABLE = sprintf('InputData_%s',caseToken);
AB_ROOT = fullfile(RootDir,'OutPut_RBF_2026_2', ...
    'Sensitivity_Ablation_RandomRobustness');

% Final master workflow: keep all three ablations together under each
% random-reference seed. Standalone MP calls without ReferenceSeed retain
% the radius-based compatibility path.
if ~isempty(Options.ReferenceSeed)
    OUTPUT_DIR = fullfile(AB_ROOT, ...
        sprintf('Seed_%d',Options.ReferenceSeed),'MP');
    OutputFile = fullfile(OUTPUT_DIR, ...
        sprintf('LocalRBF_%s_MP_Seed%d_FINAL.mat', ...
        caseToken,Options.ReferenceSeed));
else
    OUTPUT_DIR = fullfile(AB_ROOT,'MP');
    OutputFile = fullfile(OUTPUT_DIR, ...
        sprintf('LocalRBF_%s_MP_R%.0fkm_FINAL.mat', ...
        caseToken,FIXED_RADIUS_RAW/1000));
end
if ~isfolder(OUTPUT_DIR), mkdir(OUTPUT_DIR); end

if Options.ReuseExistingResults && isfile(OutputFile)
    E = load(OutputFile,'Results');
    if isfield(E,'Results') && isCompatibleExistingMP( ...
            E.Results,CaseNumber,FIXED_RADIUS_RAW,Options)
        Results = E.Results;
        fprintf('\nM_P %s R=%.0f km: reusing existing result.\n%s\n', ...
            caseToken,FIXED_RADIUS_RAW/1000,OutputFile);
        return
    end
end

%% 1. LOAD AUTHORITATIVE MODE-1 SETTINGS + DATA
A = load(MAIN_RESULTS_FILE,'Results');
assert(isfield(A,'Results') && isstruct(A.Results), ...
    'MAIN_RESULTS_FILE must contain Results.');
MainResults = A.Results;
S = MainResults.Settings;

B = load(INPUT_DATA_FILE,INPUT_VARIABLE);
assert(isfield(B,INPUT_VARIABLE), ...
    'INPUT_VARIABLE "%s" was not found.',INPUT_VARIABLE);
InputData = B.(INPUT_VARIABLE);

CaseName = char(string(MainResults.CaseName));
ListOfModel = cellstr(string(MainResults.ListOfModel(:)));
NameFunction_RBF = cellstr(string(MainResults.NameFunction_RBF(:)));
nModel = numel(ListOfModel);
nKernel = numel(NameFunction_RBF);
nRuns = nModel*nKernel;

Xraw = InputData.Train.XYZ;
XtestRaw = InputData.Test.XYZ;

assert(isnumeric(Xraw) && isreal(Xraw) && ismatrix(Xraw) && ...
       ~isempty(Xraw) && all(isfinite(Xraw(:))), ...
       'Training XYZ must be a nonempty finite real matrix.');
assert(isnumeric(XtestRaw) && isreal(XtestRaw) && ismatrix(XtestRaw) && ...
       ~isempty(XtestRaw) && all(isfinite(XtestRaw(:))), ...
       'Testing XYZ must be a nonempty finite real matrix.');
assert(size(Xraw,2)==size(XtestRaw,2), ...
       'Training and testing XYZ must have the same coordinate dimension.');
assert(size(Xraw,1)-1>=MIN_NEIGHBORS, ...
       'At least %d target-excluded training observations are required.',MIN_NEIGHBORS);

% Pre-extract each model response ONCE. Do not duplicate the same response
% vector inside every kernel job.
YTrainByModel = cell(nModel,1);
YTestByModel = cell(nModel,1);
for i = 1:nModel
    model = ListOfModel{i};
    assert(isfield(InputData.Train.DeltaZeta,model) && ...
           isfield(InputData.Test.DeltaZeta,model), ...
        'InputData is missing model %s.',model);
    yt = InputData.Train.DeltaZeta.(model);
    yq = InputData.Test.DeltaZeta.(model);
    assert(isnumeric(yt) && isreal(yt) && isvector(yt) && ...
           numel(yt)==size(Xraw,1) && all(isfinite(yt(:))), ...
           'Invalid training response vector for model %s.',model);
    assert(isnumeric(yq) && isreal(yq) && isvector(yq) && ...
           numel(yq)==size(XtestRaw,1) && all(isfinite(yq(:))), ...
           'Invalid testing response vector for model %s.',model);
    YTrainByModel{i} = yt(:);
    YTestByModel{i} = yq(:);
end

%% 2. PARAMETER SEARCH SETTINGS
% Search method/grid and optimization RNG remain aligned with Model 1, while
% the parameter BOUNDS come from the same domains used for random fixed-
% reference generation in the master.
CMin = Options.TuningCMin;
CMax = Options.TuningCMax;
RegMin = Options.TuningRegMin;
RegMax = Options.TuningRegMax;
Method = getSetting(S,{'Method','method'});
nGrid = getSetting(S,{'nGrid'});

numericalOptions = struct();
if isfield(S,'options') && isstruct(S.options) && ...
   isfield(S.options,'solver') && isstruct(S.options.solver)
    numericalOptions.solver = S.options.solver;
else
    numericalOptions.solver = struct('cond_lsq',1e12);
end
numericalOptions.solver.compute_rel_residual = false;

% Authoritative Model-1 stores the parameter-search seed at
% Results.Settings.RNG_SEED.  Older compatibility files may additionally
% carry options.RNG_SEED, so use that only as a fallback.
if isfield(S,'RNG_SEED') && ~isempty(S.RNG_SEED)
    RNG_SEED = S.RNG_SEED;
elseif isfield(S,'options') && isstruct(S.options) && ...
       isfield(S.options,'RNG_SEED') && ~isempty(S.options.RNG_SEED)
    RNG_SEED = S.options.RNG_SEED;
else
    RNG_SEED = [];
end

if ~isempty(RNG_SEED)
    assert(isnumeric(RNG_SEED) && isscalar(RNG_SEED) && isfinite(RNG_SEED) && ...
        RNG_SEED>=0 && RNG_SEED==round(RNG_SEED), ...
        'Results.Settings.RNG_SEED must be empty or a nonnegative integer.');
end

%% 3. INHERIT AUTHORITATIVE MODE-1 TRANSFER SETTINGS
nearThreshold = expandNumericSetting(getSetting(S, ...
    {'nearThresholdByModelKernel','near_threshold','NearThresholdRawByModelKernel'}), ...
    nModel,nKernel,'near_threshold');

nIDWSources = expandNumericSetting(getSetting(S, ...
    {'NIDWSourcesByModelKernel','NIDWSources'}), ...
    nModel,nKernel,'NIDWSources');

transferMethod = expandTextSetting(getSetting(S, ...
    {'TransferMethodByModelKernel','TransferMethod'}), ...
    nModel,nKernel,'TransferMethod');

parameterMode = expandTextSetting(getSetting(S, ...
    {'ParameterTransferModeByModelKernel','ParameterTransferMode'}), ...
    nModel,nKernel,'ParameterTransferMode');

validateAuthoritativeRadiusContainsFixed( ...
    getSetting(S,{'Radii'}),NameFunction_RBF,FIXED_RADIUS_RAW);

assert(all(nearThreshold(:)>0), ...
    'All near-threshold values must be positive.');
assert(all(nIDWSources(:)>=1 & nIDWSources(:)==round(nIDWSources(:))), ...
    'All NIDWSources values must be positive integers.');
validateTransferMethodsAndModes(transferMethod,parameterMode);

%% 4. COORDINATE SCALING + SHARED GEOMETRY
Sglobal = common.MeanDistant(Xraw);
assert(isfinite(Sglobal) && isscalar(Sglobal) && Sglobal>0, ...
    'common.MeanDistant(Xraw) returned an invalid scale.');

X = Xraw./Sglobal;
Xtest = XtestRaw./Sglobal;
fixedRadius = FIXED_RADIUS_RAW/Sglobal;

% Only matrices required by the independent test are retained.
Dtrain = pdist2(X,X);
Ptrain = [ones(size(X,1),1),X];
DtestTrain = pdist2(Xtest,X);       % nTest x nTrain, ONE copy only
Ptest = [ones(size(Xtest,1),1),Xtest];

%% 5. MODEL x KERNEL JOB DEFINITIONS
Jobs = repmat(struct(),nRuns,1);
q = 0;
for i = 1:nModel
    for j = 1:nKernel
        q = q+1;
        model = ListOfModel{i};
        kernel = NameFunction_RBF{j};
        key = lower(char(string(kernel)));

        tm = lower(strtrim(char(string(transferMethod{i,j}))));
        assert(any(strcmp(tm,{'nearest','idw','hybrid','ensemble'})), ...
            'Unsupported transfer method "%s" for %s | %s.',tm,model,kernel);

        Jobs(q).i = i;
        Jobs(q).j = j;
        Jobs(q).model = model;
        Jobs(q).kernel = kernel;
        Jobs(q).key = key;
        Jobs(q).useShape = common.kernelHasShapeParameter(kernel);
        Jobs(q).radius = fixedRadius;
        Jobs(q).radiusRaw = FIXED_RADIUS_RAW;
        Jobs(q).regMin = RegMin;
        Jobs(q).regMax = RegMax;
        Jobs(q).method = kernelTextSetting(Method,key,'Method');
        Jobs(q).nGrid = kernelGridSetting(nGrid,key,'nGrid');
        if Jobs(q).useShape
            Jobs(q).cMin = CMin;
            Jobs(q).cMax = CMax;
        else
            Jobs(q).cMin = NaN;
            Jobs(q).cMax = NaN;
        end
        Jobs(q).rngSeed = RNG_SEED;
        Jobs(q).transferMethod = tm;
        Jobs(q).parameterTransferMode = parameterMode{i,j};
        Jobs(q).nearThresholdRaw = nearThreshold(i,j);
        Jobs(q).nearThresholdScaled = nearThreshold(i,j)/Sglobal;
        Jobs(q).nIDWSources = nIDWSources(i,j);
    end
end

%% 6. PARALLEL SETUP
parallelUsed = false;
if USE_PARALLEL
    try
        pool = gcp('nocreate');
        if RESTART_PARALLEL_POOL && ~isempty(pool)
            delete(pool);
            pool = [];
        end
        if isempty(pool) || pool.NumWorkers~=NUM_WORKERS || ...
           ~isa(pool,'parallel.ProcessPool')
            if ~isempty(pool), delete(pool); end
            pool = parpool('Processes',NUM_WORKERS);
        end
        parallelUsed = true;
    catch ME
        if ~ALLOW_SERIAL_FALLBACK
            rethrow(ME);
        end
        warning('M_P falling back to serial: %s',ME.message);
    end
end

%% 7. POINT-SPECIFIC PARAMETER TUNING + INDEPENDENT TEST
% MP re-optimizes the point-specific parameters at the fixed radius because
% its tuning bounds must match the random fixed-reference parameter domains.
RunCell = cell(nRuns,1);
SummaryCell = cell(nRuns,1);
tBatch = tic;

fprintf('\n=============================================================\n');
fprintf(' M_P: FIXED RADIUS + POINT-SPECIFIC RE-OPTIMIZATION\n');
fprintf(' R0                    : %.0f km\n',FIXED_RADIUS_RAW/1000);
fprintf(' C tuning range        : [%.12g, %.12g]\n',CMin,CMax);
fprintf(' RegParam tuning range : [%.12g, %.12g]\n',RegMin,RegMax);
fprintf('=============================================================\n');

if parallelUsed
    parfor q = 1:nRuns
        [RunCell{q},SummaryCell{q}] = runOneMPRetuned( ...
            Jobs(q),YTrainByModel{Jobs(q).i},YTestByModel{Jobs(q).i}, ...
            X,Xtest,Dtrain,Ptrain,DtestTrain,Ptest, ...
            Sglobal,MIN_NEIGHBORS,numericalOptions);
    end
else
    for q = 1:nRuns
        [RunCell{q},SummaryCell{q}] = runOneMPRetuned( ...
            Jobs(q),YTrainByModel{Jobs(q).i},YTestByModel{Jobs(q).i}, ...
            X,Xtest,Dtrain,Ptrain,DtestTrain,Ptest, ...
            Sglobal,MIN_NEIGHBORS,numericalOptions);
    end
end
batchWallTime = toc(tBatch);

%% 9. RESULTS
Results = struct();
Results.Schema = 'LocalRBF.Ablation.MP.v6.FixedRadius.ReferenceDomainRetuned';
Results.Mode = 'MP';
Results.LegacySchema = 'LocalRBF.Ablation.Mode8.v6.FixedRadius.ReferenceDomainRetuned';
Results.LegacyMode = 'mode8';
Results.LegacyLabel = 'Mode 8';
Results.CaseName = CaseName;
Results.ListOfModel = ListOfModel;
Results.NameFunction_RBF = NameFunction_RBF;

Results.Run = struct();
for q = 1:nRuns
    f = matlab.lang.makeValidName(sprintf( ...
        'M%03d_K%03d_%s_%s_MP', ...
        Jobs(q).i,Jobs(q).j,Jobs(q).model,Jobs(q).kernel));
    Results.Run.(f) = RunCell{q};
end
Results.Summary = vertcat(SummaryCell{:});

Results.Settings.FixedRadiusRaw = FIXED_RADIUS_RAW;
Results.Settings.FixedRadiusKm = FIXED_RADIUS_RAW/1000;
Results.Settings.FixedRadiusScaled = fixedRadius;
Results.Settings.RadiusTuned = false;
Results.Settings.RadiusSelectionPerformed = false;
Results.Settings.UserSpecifiedFixedRadius = true;
Results.Settings.ParameterScope = 'point-specific parameters re-optimized at the user-fixed radius';
Results.Settings.ParameterSource = 'M_P pointwise re-optimization at fixed R0';
Results.Settings.TrainingObjective = 'minimum target-excluded absolute residual at the user-fixed radius';
Results.Settings.CMin = CMin;
Results.Settings.CMax = CMax;
Results.Settings.RegMin = RegMin;
Results.Settings.RegMax = RegMax;
Results.Settings.Method = Method;
Results.Settings.nGrid = nGrid;
Results.Settings.RNG_SEED = RNG_SEED;
Results.Settings.MinNeighbors = MIN_NEIGHBORS;
Results.Settings.UsesMinNeighborFallback = true;
Results.Settings.UsesPseudoUnseen = false;
Results.Settings.UsesTauLoc = false;
Results.Settings.UsesTauCap = false;
Results.Settings.PointSpecificParameterTuning = true;
Results.Settings.ReusesModel1Stage1 = false;
Results.Settings.ParameterBoundsMatchRandomReference = true;
Results.Settings.TrainingEpsilonRule = ...
    'epsilon_i=C_i*d_i from the target-excluded fixed-radius neighbourhood';
Results.Settings.TestEpsilonRule = ...
    'epsilon_q=C_q*d_q from actual fixed-radius query neighbourhood';
Results.Settings.TransferMethod = transferMethod;
Results.Settings.ParameterTransferMode = parameterMode;
Results.Settings.NIDWSources = nIDWSources;
Results.Settings.NearThresholdRaw = nearThreshold;
Results.Settings.S_global = Sglobal;

Results.Runtime.TrainingAndTestBatchWallTime = batchWallTime;
Results.Runtime.TotalMPWallTime = batchWallTime;
Results.Runtime.TotalMode8WallTime = batchWallTime; % legacy alias retained
Results.Runtime.ParallelUsed = parallelUsed;
Results.Runtime.NumWorkers = double(parallelUsed)*NUM_WORKERS;

Results.Provenance.MainResultsFile = MAIN_RESULTS_FILE;
Results.Provenance.InputDataFile = INPUT_DATA_FILE;
Results.Provenance.ReferenceSeed = Options.ReferenceSeed;
Results.Provenance.ParameterSource = ...
    'M_P independent pointwise parameter optimization at fixed R0 using random-reference bounds';
Results.Provenance.ReusedModel1Stage1 = false;
Results.Provenance.NoPseudoUnseen = true;
Results.Provenance.Mode1TransferFunction = 'localrbf.transferSettingsFromTrainToTest';
Results.Provenance.LegacyRunFieldPattern = 'M%03d_K%03d_<model>_<kernel>_MODE8';
Results.Provenance.ManuscriptRunFieldPattern = 'M%03d_K%03d_<model>_<kernel>_MP';
Results.created = datetime('now');

saveAtomicVariables(OutputFile,struct('Results',Results));
disp(Results.Summary);
fprintf('\nSaved M_P Case %d:\n%s\n',CaseNumber,OutputFile);

end

%% ========================================================================
function [R,Srow] = runOneMPRetuned(Job,y,ytest,X,Xtest,Dtrain,Ptrain,DtestTrain,Ptest, ...
    Sglobal,minNeighbors,numericalOptions)

tRun = tic;
y = y(:);
ytest = ytest(:);
n = numel(y);

% Preserve the established M_P/Model-1 parameter-search RNG seed.
if ~isempty(Job.rngSeed)
    rng(Job.rngSeed,'twister');
end

% A. Build one target-excluded fixed-radius cache per training point.
[Caches,neighborCount,dLocal,cacheSuccess,cacheMessage] = ...
    buildTrainingCachesMP(X,y,Dtrain,Ptrain,Job.radius,Job.kernel,minNeighbors);
if ~cacheSuccess
    R = failedRun(Job,cacheMessage,toc(tRun));
    Srow = makeSummaryRow(Job,R);
    return
end

% B. Independently tune parameters for every training point.
PointC = nan(n,1);
PointReg = nan(n,1);
PointResidual = nan(n,1);
PointAE = nan(n,1);
PointEpsilon = nan(n,1);
nRBFSolvesSearch = 0;

for i = 1:n
    optMessage = '';
    try
        Opt = localrbf.optimizeParameters( ...
            Caches{i},Job.kernel,Job.cMin,Job.cMax, ...
            Job.regMin,Job.regMax,numericalOptions,Job.method,Job.nGrid);
    catch ME
        Opt = [];
        optMessage = ME.message;
    end

    if isempty(Opt) || ~isstruct(Opt) || ~isfield(Opt,'success') || ~Opt.success
        if isempty(optMessage)
            optMessage = 'localrbf.optimizeParameters returned an unsuccessful result.';
        end
        R = failedRun(Job,sprintf( ...
            'Point-specific tuning failed at training point %d: %s',i,optMessage),toc(tRun));
        R.PointFailureIndex = i;
        Srow = makeSummaryRow(Job,R);
        return
    end

    if isfield(Opt,'searchInfo') && isstruct(Opt.searchInfo)
        if isfield(Opt.searchInfo,'nRBFSolves') && isfinite(Opt.searchInfo.nRBFSolves)
            nRBFSolvesSearch = nRBFSolvesSearch + double(Opt.searchInfo.nRBFSolves);
        elseif isfield(Opt.searchInfo,'nEvaluations') && isfinite(Opt.searchInfo.nEvaluations)
            nRBFSolvesSearch = nRBFSolvesSearch + double(Opt.searchInfo.nEvaluations);
        end
    end

    PointC(i) = Opt.C;
    PointReg(i) = Opt.reg_param;
    PointResidual(i) = Opt.y_diff;
    PointAE(i) = Opt.AE;
    PointEpsilon(i) = Opt.epsilon;

    if ~isfinite(PointReg(i)) || PointReg(i)<=0 || ...
       ~isfinite(PointResidual(i)) || ~isfinite(PointAE(i)) || ...
       ~isfinite(PointEpsilon(i)) || PointEpsilon(i)<=0 || ...
       (Job.useShape && (~isfinite(PointC(i)) || PointC(i)<=0))
        R = failedRun(Job,sprintf( ...
            'Invalid optimized parameters at training point %d.',i),toc(tRun));
        R.PointFailureIndex = i;
        Srow = makeSummaryRow(Job,R);
        return
    end
end

TrainRMSE = sqrt(mean(PointResidual.^2));
TrainMAE = mean(abs(PointResidual));
TrainMaxAE = max(abs(PointResidual));

% C. Transfer final point-specific parameters with the Model-1 rule.
FinalSettings = struct();
FinalSettings.success = true;
FinalSettings.radius = repmat(Job.radius,n,1);
FinalSettings.reg_param = PointReg;
FinalSettings.radiusIndex = ones(n,1);
if Job.useShape, FinalSettings.C = PointC; end

idwRadiusCompatibility = max(2*Job.nearThresholdScaled,max(Job.radius,eps));
try
    Transfer = localrbf.transferSettingsFromTrainToTest( ...
        X,Xtest,FinalSettings,Job.kernel, ...
        Job.nearThresholdScaled,idwRadiusCompatibility,DtestTrain,false, ...
        Job.transferMethod,Job.nIDWSources,Job.parameterTransferMode);
catch ME
    R = failedRun(Job,sprintf('Mode-1 transfer failed: %s',ME.message),toc(tRun));
    Srow = makeSummaryRow(Job,R);
    return
end

% D. Independent test; epsilon is rebuilt from each query neighbourhood.
TestEval = evaluateTestMP( ...
    X,Xtest,y,ytest,Dtrain,Ptrain,DtestTrain,Ptest, ...
    Transfer,Job.kernel,minNeighbors,numericalOptions);

finiteR = TestEval.Radius(isfinite(TestEval.Radius));
if ~isempty(finiteR)
    tolFixed = 1e-10*max(1,abs(Job.radius));
    assert(all(abs(finiteR-Job.radius)<=tolFixed), ...
        ['M_P radius invariant violated: transferred test radius changed ' ...
         'from the user-fixed R0.']);
end

R = struct();
R.success = TestEval.success;
R.model = Job.model;
R.kernel = Job.kernel;
R.mode = 'MP';
R.message = '';
R.FixedRadiusRaw = Job.radiusRaw;
R.FixedRadiusScaled = Job.radius;
R.ParameterScope = 'point-specific-per-training-point';
R.PointSpecificParameterTuning = true;
R.ParameterSource = 'M_P re-optimization at fixed R0 using random-reference parameter bounds';
R.UsesPseudoUnseen = false;
R.UsesTauLoc = false;
R.UsesTauCap = false;

R.Training.RMSE = TrainRMSE;
R.Training.MAE = TrainMAE;
R.Training.MaxAE = TrainMaxAE;
R.Training.Residual = PointResidual;
R.Training.AE = PointAE;
R.Training.C = PointC;
R.Training.RegParam = PointReg;
R.Training.dLocal = dLocal;
R.Training.Epsilon = PointEpsilon;
R.Training.NeighborCount = neighborCount;
R.Training.RadiusRaw = repmat(Job.radiusRaw,n,1);
R.Training.RadiusScaled = repmat(Job.radius,n,1);

R.Transfer.Method = Job.transferMethod;
R.Transfer.ParameterTransferMode = Job.parameterTransferMode;
R.Transfer.NIDWSources = Job.nIDWSources;
R.Transfer.NearThresholdRaw = Job.nearThresholdRaw;
R.Transfer.NearThresholdScaled = Job.nearThresholdScaled;
R.Transfer.Success = isfield(Transfer,'success') && logical(Transfer.success);

R.Test.RMSE = TestEval.RMSE;
R.Test.MAE = TestEval.MAE;
R.Test.MaxAE = TestEval.MaxAE;
R.Test.Residual = TestEval.Residual;
R.Test.Prediction = TestEval.Prediction;
R.Test.NeighborCount = TestEval.NeighborCount;
R.Test.TransferredRadiusScaled = TestEval.Radius;
R.Test.TransferredRadiusRaw = TestEval.Radius*Sglobal;
R.Test.RequestedFixedRadiusRaw = repmat(Job.radiusRaw,numel(ytest),1);
R.Test.RequestedFixedRadiusScaled = repmat(Job.radius,numel(ytest),1);
R.Test.TransferredRegParam = TestEval.RegParam;
R.Test.TransferredC = TestEval.C;
R.Test.dLocal = TestEval.dLocal;
R.Test.Epsilon = TestEval.Epsilon;
R.Test.TransferMethodByPoint = TestEval.TransferMethodByPoint;
R.Test.TransferSourceCount = TestEval.TransferSourceCount;

R.Runtime.TrainingSearchRBFSolves = nRBFSolvesSearch;
R.Runtime.TestRBFSolves = TestEval.nRBFSolves;
R.Runtime.TotalRBFSolves = nRBFSolvesSearch + TestEval.nRBFSolves;
R.Runtime.ActualNewRBFSolves = nRBFSolvesSearch + TestEval.nRBFSolves;
R.Runtime.ReusedModel1Stage1 = false;
R.Runtime.WallTime = toc(tRun);

Srow = makeSummaryRow(Job,R);
end

%% ========================================================================
function [Caches,neighborCount,dLocal,ok,message] = buildTrainingCachesMP( ...
    X,y,Dtrain,Ptrain,radius,kernel,minNeighbors)

n = numel(y);
useShape = common.kernelHasShapeParameter(kernel);
Caches = cell(n,1);
neighborCount = nan(n,1);
dLocal = nan(n,1);
ok = true;
message = '';

for i = 1:n
    try
        Nbh = localrbf.getNeighborhood(X,X(i,:),radius,minNeighbors,i,Dtrain(:,i));
    catch ME
        ok = false;
        message = sprintf('Neighborhood construction failed at training point %d: %s',i,ME.message);
        return
    end
    if ~isstruct(Nbh) || ~isfield(Nbh,'success') || ~Nbh.success
        ok = false;
        message = sprintf('Neighborhood failed at training point %d.',i);
        return
    end

    idx = Nbh.idx(:);
    Xfit = X(idx,:);
    yfit = y(idx);
    try
        [isOK,~] = common.checkPolynomialAdmissibility(Xfit);
    catch
        isOK = false;
    end
    if ~isOK
        ok = false;
        message = sprintf('Polynomial admissibility failed at training point %d.',i);
        return
    end

    if useShape
        try
            [~,di] = common.buildEpsilon(Xfit,1);
        catch ME
            ok = false;
            message = sprintf('Local geometry scale failed at training point %d: %s',i,ME.message);
            return
        end
        if ~isfinite(di) || di<=0
            ok = false;
            message = sprintf('Invalid local geometry scale at training point %d.',i);
            return
        end
    else
        di = NaN;
    end

    C = struct();
    C.Xfit = Xfit;
    C.yfit = yfit;
    C.Xtarget = X(i,:);
    C.ytarget = y(i);
    C.R = Dtrain(idx,idx);
    C.P = Ptrain(idx,:);
    C.R_eval = reshape(Dtrain(i,idx),1,[]);
    C.P_eval = Ptrain(i,:);
    C.d = di;
    Caches{i} = C;
    neighborCount(i) = numel(idx);
    dLocal(i) = di;
end
end

%% ========================================================================
function T = evaluateTestMP( ...
    X,Xtest,y,ytest,Dtrain,Ptrain,DtestTrain,Ptest, ...
    Transfer,kernel,minNeighbors,numericalOptions)

nTest = numel(ytest);
useShape = common.kernelHasShapeParameter(kernel);

prediction = nan(nTest,1);
residual = nan(nTest,1);
neighborCount = nan(nTest,1);
radiusUsed = nan(nTest,1);
regUsed = nan(nTest,1);
cUsed = nan(nTest,1);
dLocalUsed = nan(nTest,1);
epsilonUsed = nan(nTest,1);
sourceCount = nan(nTest,1);
transferMethodByPoint = strings(nTest,1);

nSolves = 0;

for it = 1:nTest

    if ~isfield(Transfer,'querySuccess') || ...
       numel(Transfer.querySuccess)<it || ...
       ~Transfer.querySuccess(it)
        continue
    end

    method_i = lower(char(string(Transfer.transferMethod{it})));
    transferMethodByPoint(it) = string(method_i);

    if isfield(Transfer,'nTransferSources') && numel(Transfer.nTransferSources)>=it
        sourceCount(it) = Transfer.nTransferSources(it);
    elseif strcmp(method_i,'nearest')
        sourceCount(it) = 1;
    elseif strcmp(method_i,'ensemble') && ...
           isfield(Transfer,'ensembleNUniqueTuples') && ...
           numel(Transfer.ensembleNUniqueTuples)>=it
        sourceCount(it) = Transfer.ensembleNUniqueTuples(it);
    else
        sourceCount(it) = NaN;
    end

    if strcmp(method_i,'ensemble')

        if ~isfield(Transfer,'ensembleUniqueRadius') || ...
           numel(Transfer.ensembleUniqueRadius)<it
            continue
        end

        radii = Transfer.ensembleUniqueRadius{it}(:);
        regs = Transfer.ensembleUniqueRegParam{it}(:);
        weights = Transfer.ensembleUniqueWeight{it}(:);

        if useShape
            Cs = Transfer.ensembleUniqueC{it}(:);
        else
            Cs = nan(size(radii));
        end

        if isempty(radii) || numel(regs)~=numel(radii) || ...
           numel(weights)~=numel(radii) || ...
           (useShape && numel(Cs)~=numel(radii))
            continue
        end

        if any(~isfinite(radii)) || any(radii<=0) || ...
           any(~isfinite(regs)) || any(regs<=0) || ...
           any(~isfinite(weights)) || any(weights<0) || sum(weights)<=0 || ...
           (useShape && (any(~isfinite(Cs)) || any(Cs<=0)))
            continue
        end

        weights = weights./sum(weights);

    else

        radii = Transfer.radius(it);
        regs = Transfer.reg_param(it);
        weights = 1;

        if useShape
            Cs = Transfer.C(it);
        else
            Cs = NaN;
        end
    end

    yhTuple = nan(numel(radii),1);
    ncTuple = nan(numel(radii),1);
    dTuple = nan(numel(radii),1);
    epsTuple = nan(numel(radii),1);

    allOK = true;

    for u = 1:numel(radii)

        try
            Nbh = localrbf.getNeighborhood( ...
                X,Xtest(it,:),radii(u),minNeighbors,[],DtestTrain(it,:).');
        catch
            allOK = false;
            break
        end

        if ~isstruct(Nbh) || ~isfield(Nbh,'success') || ~Nbh.success
            allOK = false;
            break
        end

        idx = Nbh.idx(:);
        Xfit = X(idx,:);
        yfit = y(idx);

        try
            [isOK,~] = common.checkPolynomialAdmissibility(Xfit);
        catch
            isOK = false;
        end

        if ~isOK
            allOK = false;
            break
        end

        if useShape

            try
                [~,dq] = common.buildEpsilon(Xfit,1);
                epsilon = max(Cs(u)*dq,1e-12);
            catch
                epsilon = NaN;
                dq = NaN;
            end

            if ~isfinite(epsilon) || epsilon<=0 || ~isfinite(dq) || dq<=0
                allOK = false;
                break
            end

        else

            dq = NaN;
            epsilon = 1;
        end

        geom = struct();
        geom.R = Dtrain(idx,idx);
        geom.P = Ptrain(idx,:);
        geom.R_eval = reshape(DtestTrain(it,idx),1,[]);
        geom.P_eval = Ptest(it,:);

        try
            Sys = common.buildRBFSystem( ...
                Xfit,yfit,kernel,epsilon,regs(u),geom);

            nSolves = nSolves+1;

            [coeffs,solveInfo] = ...
                common.solveLinearSystem(Sys,numericalOptions);

            if ~solveInfo.success || isempty(coeffs) || ...
               any(~isfinite(coeffs(:)))
                allOK = false;
                break
            end

            yhat = common.predictRBF( ...
                Xtest(it,:),Xfit,coeffs,kernel,epsilon,geom);

            if ~isscalar(yhat) || ~isfinite(yhat)
                allOK = false;
                break
            end

            yhTuple(u) = yhat;
            ncTuple(u) = numel(idx);
            dTuple(u) = dq;
            epsTuple(u) = epsilon;

        catch

            allOK = false;
            break
        end
    end

    if ~allOK || any(~isfinite(yhTuple))
        continue
    end

    yhatFinal = sum(weights.*yhTuple);

    prediction(it) = yhatFinal;
    residual(it) = ytest(it)-yhatFinal;

    radiusUsed(it) = sum(weights.*radii);
    regUsed(it) = weightedPositiveMean(regs,weights,'log');

    if useShape
        cUsed(it) = weightedPositiveMean(Cs,weights,'log');
        dLocalUsed(it) = sum(weights.*dTuple);
        epsilonUsed(it) = sum(weights.*epsTuple);
    else
        cUsed(it) = NaN;
        dLocalUsed(it) = NaN;
        epsilonUsed(it) = 1;
    end

    neighborCount(it) = sum(weights.*ncTuple);
end

T = struct();
T.success = all(isfinite(residual));
T.Prediction = prediction;
T.Residual = residual;
T.NeighborCount = neighborCount;
T.Radius = radiusUsed;
T.RegParam = regUsed;
T.C = cUsed;
T.dLocal = dLocalUsed;
T.Epsilon = epsilonUsed;
T.TransferMethodByPoint = transferMethodByPoint;
T.TransferSourceCount = sourceCount;
T.nRBFSolves = nSolves;

if T.success
    T.RMSE = sqrt(mean(residual.^2));
    T.MAE = mean(abs(residual));
    T.MaxAE = max(abs(residual));
else
    T.RMSE = NaN;
    T.MAE = NaN;
    T.MaxAE = NaN;
end

end

%% ========================================================================
function v = weightedPositiveMean(x,w,mode)

x = x(:);
w = w(:);
w = w./sum(w);

if strcmpi(mode,'log')
    v = 10^(sum(w.*log10(x)));
else
    v = sum(w.*x);
end

end

%% ========================================================================
function R = failedRun(Job,msg,wall)

R = struct();
R.success = false;
R.model = Job.model;
R.kernel = Job.kernel;
R.mode = 'MP';
R.message = msg;
R.FixedRadiusRaw = Job.radiusRaw;

R.Training = struct( ...
    'RMSE',NaN, ...
    'MAE',NaN, ...
    'MaxAE',NaN);

R.Test = struct( ...
    'RMSE',NaN, ...
    'MAE',NaN, ...
    'MaxAE',NaN);

R.Runtime = struct( ...
    'TrainingSearchRBFSolves',NaN, ...
    'TestRBFSolves',NaN, ...
    'TotalRBFSolves',NaN, ...
    'WallTime',wall);

end

%% ========================================================================
function Srow = makeSummaryRow(Job,R)

if isfield(R,'Training') && isfield(R.Training,'RMSE')
    tr = R.Training.RMSE;
else
    tr = NaN;
end

if isfield(R,'Test') && isfield(R.Test,'RMSE')
    te = R.Test.RMSE;
else
    te = NaN;
end

if isfield(R,'Training') && isfield(R.Training,'MaxAE')
    tm = R.Training.MaxAE;
else
    tm = NaN;
end

if isfield(R,'Test') && isfield(R.Test,'MaxAE')
    xm = R.Test.MaxAE;
else
    xm = NaN;
end

if isfield(R,'Runtime') && isfield(R.Runtime,'TotalRBFSolves')
    ns = R.Runtime.TotalRBFSolves;
else
    ns = NaN;
end

if isfield(R,'Runtime') && isfield(R.Runtime,'WallTime')
    wt = R.Runtime.WallTime;
else
    wt = NaN;
end

if isfield(R,'Training') && isfield(R.Training,'RegParam') && ...
   all(isfinite(R.Training.RegParam))
    meanReg = mean(R.Training.RegParam);
    minReg = min(R.Training.RegParam);
    maxReg = max(R.Training.RegParam);
else
    meanReg = NaN;
    minReg = NaN;
    maxReg = NaN;
end

if Job.useShape && isfield(R,'Training') && isfield(R.Training,'C') && ...
   all(isfinite(R.Training.C))
    meanC = mean(R.Training.C);
    minC = min(R.Training.C);
    maxC = max(R.Training.C);
else
    meanC = NaN;
    minC = NaN;
    maxC = NaN;
end

Srow = table( ...
    string(Job.model), ...
    string(Job.kernel), ...
    "MP", ...
    Job.radiusRaw, ...
    string(Job.method), ...
    string(Job.transferMethod), ...
    string(Job.parameterTransferMode), ...
    meanC,minC,maxC, ...
    meanReg,minReg,maxReg, ...
    tr,te,tm,xm,ns,wt,logical(R.success), ...
    'VariableNames',{ ...
    'Model','Kernel','Mode','FixedRadiusRaw', ...
    'TuningMethod','TestTransferMethod','ParameterTransferMode', ...
    'MeanTrainC','MinTrainC','MaxTrainC', ...
    'MeanTrainRegParam','MinTrainRegParam','MaxTrainRegParam', ...
    'TrainRMSE','TestRMSE','TrainMaxAE','TestMaxAE', ...
    'TotalRBFSolves','WallTime','Success'});

end

%% ========================================================================
function value = getSetting(S,names)

for k = 1:numel(names)
    if isfield(S,names{k}) && ~isempty(S.(names{k}))
        value = S.(names{k});
        return
    end
end

error('Required setting not found: %s',strjoin(names,', '));

end

%% ========================================================================
function A = expandNumericSetting(value,nM,nK,name)

assert(isnumeric(value) && isreal(value), ...
    '%s must be numeric.',name);

if isscalar(value)
    A = repmat(double(value),nM,nK);
else
    A = double(value);
end

assert(isequal(size(A),[nM nK]), ...
    '%s must be scalar or nModel x nKernel.',name);

assert(all(isfinite(A(:))), ...
    '%s contains nonfinite values.',name);

end

%% ========================================================================
function C = expandTextSetting(value,nM,nK,name)

if ischar(value) || (isstring(value) && isscalar(value))

    C = repmat({char(string(value))},nM,nK);

elseif isstring(value)

    C = cellstr(value);

elseif iscell(value)

    C = cellfun( ...
        @(x) char(string(x)), ...
        value, ...
        'UniformOutput',false);

else

    error('%s must be text or a text matrix.',name);
end

assert(isequal(size(C),[nM nK]), ...
    '%s must be scalar or nModel x nKernel.',name);

end

%% ========================================================================
function t = kernelTextSetting(A,key,name)

if ischar(A) || (isstring(A) && isscalar(A))

    t = lower(strtrim(char(string(A))));

elseif isstruct(A) && isfield(A,key)

    t = lower(strtrim(char(string(A.(key)))));

else

    error('%s must be scalar text or contain field %s.',name,key);
end

assert(any(strcmp(t,{'ga','grid'})), ...
    '%s.%s must be ga or grid.',name,key);

end

%% ========================================================================
function g = kernelGridSetting(A,key,name)

if isnumeric(A)

    g = double(A);

elseif isstruct(A) && isfield(A,key)

    g = double(A.(key));

else

    error('%s must be numeric or contain field %s.',name,key);
end

g = g(:).';

assert(any(numel(g)==[1 2]) && ...
       all(isfinite(g)) && ...
       all(g>=1) && ...
       all(g==round(g)), ...
    '%s for %s must be integer scalar or [nC nReg].',name,key);

end

%% ========================================================================
function auditDependenciesMPAblation()
required = { ...
    'localrbf.optimizeParameters', ...
    'localrbf.getNeighborhood', ...
    'localrbf.transferSettingsFromTrainToTest', ...
    'common.MeanDistant', ...
    'common.buildEpsilon', ...
    'common.checkPolynomialAdmissibility', ...
    'common.buildRBFSystem', ...
    'common.solveLinearSystem', ...
    'common.predictRBF'};
for k=1:numel(required)
    assert(~isempty(which(required{k})),'Required function not on MATLAB path: %s',required{k});
end
narg=nargin('localrbf.transferSettingsFromTrainToTest');
assert(narg<0 || narg>=11, ...
    'Obsolete localrbf.transferSettingsFromTrainToTest detected; 11-input API required.');
end

%% ========================================================================
function Options = normalizeOptionsMP(Options)
if ~isfield(Options,'ReuseExistingResults'), Options.ReuseExistingResults = true; end
if ~isfield(Options,'UseParallel'), Options.UseParallel = true; end
if ~isfield(Options,'AllowSerialFallback'), Options.AllowSerialFallback = false; end
if ~isfield(Options,'ReferenceSeed'), Options.ReferenceSeed = []; end
assert(isfield(Options,'TuningCMin') && isfield(Options,'TuningCMax') && ...
       isfield(Options,'TuningRegMin') && isfield(Options,'TuningRegMax'), ...
    ['M_P requires TuningCMin/TuningCMax/TuningRegMin/TuningRegMax ' ...
     'from the master random-reference design.']);
Options.ReuseExistingResults = logical(Options.ReuseExistingResults);
Options.UseParallel = logical(Options.UseParallel);
Options.AllowSerialFallback = logical(Options.AllowSerialFallback);
if ~isempty(Options.ReferenceSeed)
    Options.ReferenceSeed = double(Options.ReferenceSeed);
    assert(isscalar(Options.ReferenceSeed) && isfinite(Options.ReferenceSeed) && ...
           Options.ReferenceSeed>=0 && Options.ReferenceSeed==round(Options.ReferenceSeed), ...
        'ReferenceSeed must be empty or one nonnegative integer.');
end
Options.TuningCMin = double(Options.TuningCMin);
Options.TuningCMax = double(Options.TuningCMax);
Options.TuningRegMin = double(Options.TuningRegMin);
Options.TuningRegMax = double(Options.TuningRegMax);
assert(isscalar(Options.TuningCMin) && isfinite(Options.TuningCMin) && Options.TuningCMin>0 && ...
       isscalar(Options.TuningCMax) && isfinite(Options.TuningCMax) && Options.TuningCMax>=Options.TuningCMin, ...
    'Invalid M_P C tuning range.');
assert(isscalar(Options.TuningRegMin) && isfinite(Options.TuningRegMin) && Options.TuningRegMin>0 && ...
       isscalar(Options.TuningRegMax) && isfinite(Options.TuningRegMax) && Options.TuningRegMax>=Options.TuningRegMin, ...
    'Invalid M_P RegParam tuning range.');
end

%% ========================================================================
function tf = isCompatibleExistingMP(Results,CaseNumber,fixedRadiusRaw,Options)
tf = false;
EXPECTED_SCHEMA = 'LocalRBF.Ablation.MP.v6.FixedRadius.ReferenceDomainRetuned';
if ~isstruct(Results) || ~isfield(Results,'Schema') || ...
   ~strcmp(char(string(Results.Schema)),EXPECTED_SCHEMA) || ...
   ~isfield(Results,'Mode') || ~strcmpi(char(string(Results.Mode)),'MP') || ...
   ~isfield(Results,'Settings') || ~isstruct(Results.Settings)
    return
end
if ~isfield(Results.Settings,'FixedRadiusRaw'), return, end
r = double(Results.Settings.FixedRadiusRaw);
tol = 1e-10*max([1,abs(r),abs(fixedRadiusRaw)]);
if ~isscalar(r) || ~isfinite(r) || abs(r-fixedRadiusRaw)>tol, return, end
required = {'CMin','CMax','RegMin','RegMax','ReusesModel1Stage1','ParameterBoundsMatchRandomReference'};
for k=1:numel(required)
    if ~isfield(Results.Settings,required{k}), return, end
end
valsSaved = double([Results.Settings.CMin Results.Settings.CMax Results.Settings.RegMin Results.Settings.RegMax]);
valsNeed = double([Options.TuningCMin Options.TuningCMax Options.TuningRegMin Options.TuningRegMax]);
if max(abs(valsSaved-valsNeed)./max(1,abs(valsNeed)))>1e-12, return, end
if logical(Results.Settings.ReusesModel1Stage1), return, end
if ~logical(Results.Settings.ParameterBoundsMatchRandomReference), return, end
if isfield(Results,'CaseName')
    expected = sprintf('Case%d',CaseNumber);
    if ~strcmpi(char(string(Results.CaseName)),expected), return, end
end
if ~isfield(Results,'Summary') || ~istable(Results.Summary) || height(Results.Summary)~=30
    return
end
if ~isempty(Options.ReferenceSeed)
    if ~isfield(Results,'Provenance') || ~isstruct(Results.Provenance) || ...
       ~isfield(Results.Provenance,'ReferenceSeed') || ...
       isempty(Results.Provenance.ReferenceSeed) || ...
       double(Results.Provenance.ReferenceSeed)~=double(Options.ReferenceSeed)
        return
    end
end
tf = true;
end

%% ========================================================================
function saveAtomicVariables(fileName,V)
folder = fileparts(fileName);
if ~isfolder(folder), mkdir(folder); end
tmp = [tempname(folder) '.mat'];
cleanup = onCleanup(@() cleanupTempFileMP(tmp));
names = fieldnames(V);
for k = 1:numel(names)
    eval([names{k} ' = V.(names{k});']); %#ok<EVLDIR>
end
try
    save(tmp,names{:},'-v7');
catch
    save(tmp,names{:},'-v7.3');
end
[ok,msg] = movefile(tmp,fileName,'f');
assert(ok,'Could not finalize MAT file %s: %s',fileName,msg);
clear cleanup
end

function cleanupTempFileMP(f)
if isfile(f)
    try, delete(f); catch, end
end
end

%% ========================================================================
function validateTransferMethodsAndModes(methodCell,modeCell)
allowedMethods = {'nearest','idw','hybrid','ensemble'};
allowedModes = {'arithmetic_arithmetic','arith_arith','m21', ...
                'arithmetic_log','arith_log','m23', ...
                'log_arithmetic','log_arith','m22', ...
                'log_log','m24'};

for k = 1:numel(methodCell)
    m = lower(strtrim(char(string(methodCell{k}))));
    assert(any(strcmp(m,allowedMethods)), ...
        'Unsupported transfer method: %s',m);
end
for k = 1:numel(modeCell)
    m = lower(strtrim(char(string(modeCell{k}))));
    assert(any(strcmp(m,allowedModes)), ...
        'Unsupported parameter-transfer mode: %s',m);
end
end

%% ========================================================================
function validateAuthoritativeRadiusContainsFixed(Radii,kernels,fixedRadiusRaw)
for j = 1:numel(kernels)
    key = lower(char(string(kernels{j})));
    if isstruct(Radii)
        assert(isfield(Radii,key),'Mode-1 Radii.%s is missing.',key);
        r = double(Radii.(key)(:));
    else
        r = double(Radii(:));
    end
    tol = 1e-10*max(1,abs(fixedRadiusRaw));
    assert(any(abs(r-fixedRadiusRaw)<=tol), ...
        'Authoritative Model-1 radius grid for %s does not contain %.0f km.', ...
        key,fixedRadiusRaw/1000);
end
end

%% ========================================================================
function fileName = resolveMainResultsFile(RootDir,caseToken)
baseDirs = {
    fullfile(RootDir,'OutPut_RBF_2026_2')
    fullfile(RootDir,'OutPut_RBF_2026_1')
    };

exactHits = {};
for b = 1:numel(baseDirs)
    exact = fullfile(baseDirs{b}, ...
        sprintf('LocalRBF_%s_Full_TuneReg.mat',caseToken));
    if isfile(exact)
        exactHits{end+1,1} = exact; %#ok<AGROW>
    end
end

if numel(exactHits)==1
    fileName = exactHits{1};
    return
elseif numel(exactHits)>1
    error('More than one exact authoritative main result was found for %s.',caseToken);
end

hits = {};
for b = 1:numel(baseDirs)
    if ~isfolder(baseDirs{b}), continue, end
    F = dir(fullfile(baseDirs{b}, ...
        sprintf('LocalRBF_%s*TuneReg*.mat',caseToken)));
    for k = 1:numel(F)
        if F(k).isdir, continue, end
        nm = lower(F(k).name);
        if contains(nm,'stage1') || contains(nm,'sensitivity')
            continue
        end
        hits{end+1,1} = fullfile(F(k).folder,F(k).name); %#ok<AGROW>
    end
end

assert(numel(hits)==1, ...
    'Could not uniquely resolve authoritative main result for %s.',caseToken);
fileName = hits{1};
end

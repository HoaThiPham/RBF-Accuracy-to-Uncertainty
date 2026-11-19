function Results = Run_Ablation_MR_Case(CaseNumber,RootDir,NUM_WORKERS,Reference,Options)
%RUN_ABLATION_MR_CASE
%
% Manuscript configuration: M_R (adaptive neighbourhood only).
%
% M_R -- ADAPTIVE NEIGHBOURHOOD / POINT-SPECIFIC RADIUS WITHIN EACH CAP
%             + FIX ALL RBF PARAMETERS
%
% Scientific definition
% ---------------------
% Candidate radii/caps:
%       50:10:200 km, exactly the required Mode-1 radius grid.
%
% At each candidate cap R_cap(k), each TRAINING point i may choose its own
% best radius from the candidates satisfying R_j <= R_cap(k):
%
%       R_i*(R_cap) = argmin_{R_j <= R_cap} |e_i(R_j)|
%
% where e_i(R_j) is the ordinary target-excluded TRAINING residual at the
% exact requested radius R_j.  Exact ties choose the smaller radius.
%
% The cap-level TRAINING RMSE is formed from those point-specific best
% residuals. Each cap is then evaluated by STRICT PSEUDO-UNSEEN prediction,
% and the FINAL cap is the smallest cap attaining the exact minimum strict
% pseudo-unseen RMSE.
%
% Fixed parameters
% ----------------
%       RegParam and C are supplied by ONE pre-specified random reference
%       realization and are identical to the values used by M_0.
%
%       Gaussian / Multiquadric:
%           epsilon = C_fixed,k * dLocal(actual current support)
%       using common.buildEpsilon(Xfit,1), matching Model-1's C-to-epsilon
%       parameterization while keeping C itself fixed.
%
%       Linear/Cubic/Thinplate: shape-free.
%
%       No C or RegParam tuning is performed inside M_R.
%
% Test handling
% -------------
% Test data NEVER select radius/cap/shape/RegParam.  After the final cap is
% selected from strict PSEUDO-UNSEEN training evaluation only, the selected
% point-specific TRAINING radii
% are transferred to each test location using the authoritative Mode-1
% transfer method for that GGM x kernel (nearest / IDW / hybrid / ensemble).
% Only the radius is effectively variable; RegParam and shape remain the
% fixed M_R values above.  Ensemble combines predictions at the
% prediction level, matching the production meaning of ensemble transfer.
%
% IMPORTANT
% ---------
%   * NO C tuning.
%   * NO RegParam tuning.
%   * STRICT pseudo-unseen is used to select the final cap.
%   * NO tau_loc.
%   * NO tau_cap.
%   * NO test-data selection.
%   * minNeighbors = 15 fallback is ACTIVE. If fewer than 15 target-excluded
%     training observations lie inside the requested/transferred radius, the
%     nearest available observations are added until 15 neighbours are used.
%   * The same fixed C/RegParam reference is used by M_0 and M_R.
%
% Parallelism: one PARFOR job = one GGM x Kernel combination.

if nargin < 1 || isempty(CaseNumber), CaseNumber = 1; end
if nargin < 2 || isempty(RootDir), RootDir = pwd; end
if nargin < 3 || isempty(NUM_WORKERS), NUM_WORKERS = 4; end
if nargin < 5 || isempty(Options), Options = struct(); end
Options = normalizeOptionsMR(Options);
Reference = validateReferenceMR(Reference);
auditDependenciesMR();
assert(ismember(CaseNumber,[1 2]),'CaseNumber must be 1 or 2.');
assert(isscalar(NUM_WORKERS) && NUM_WORKERS>=1 && NUM_WORKERS==round(NUM_WORKERS), ...
    'NUM_WORKERS must be a positive integer.');

%% 0. MODEL-12 SETTINGS
MR_RADII_RAW = (50000:10000:200000).';
FIXED_REGPARAM_BY_KERNEL = double(Reference.FixedRegParamByKernel(:).');
FIXED_C_BY_KERNEL = double(Reference.FixedCByKernel(:).');
MIN_NEIGHBORS = 15;
KERNEL_ORDER_REG = {'gaussian','multiquadric','linear','cubic','thinplate'};
KERNEL_ORDER_C   = {'gaussian','multiquadric'};

USE_PARALLEL = Options.UseParallel;
ALLOW_SERIAL_FALLBACK = Options.AllowSerialFallback;
RESTART_PARALLEL_POOL = false;
STORE_POINTWISE = Options.StorePointwise;

caseToken = sprintf('Case%d',CaseNumber);
MAIN_RESULTS_FILE = resolveMainResultsFile(RootDir,caseToken);
INPUT_DATA_FILE = fullfile(RootDir,'OutPutSplitTrainTest', ...
    sprintf('InputData_%s.mat',caseToken));
INPUT_VARIABLE = sprintf('InputData_%s',caseToken);
AB_ROOT = fullfile(RootDir,'OutPut_RBF_2026_2', ...
    'Sensitivity_Ablation_RandomRobustness');
OUTPUT_DIR = fullfile(AB_ROOT,sprintf('Seed_%d',Reference.Seed),'MR');
if ~isfolder(OUTPUT_DIR), mkdir(OUTPUT_DIR); end
OutputFile = fullfile(OUTPUT_DIR, ...
    sprintf('LocalRBF_%s_MR_Seed%d_FINAL.mat',caseToken,Reference.Seed));

if Options.ReuseExistingResults && isfile(OutputFile)
    E0 = load(OutputFile,'Results');
    if isfield(E0,'Results') && isCompatibleExistingMR(E0.Results,CaseNumber,Reference)
        Results = E0.Results;
        fprintf('\nM_R %s seed %d: reusing existing result.\n%s\n', ...
            caseToken,Reference.Seed,OutputFile);
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
assert(numel(FIXED_REGPARAM_BY_KERNEL)==5, ...
    'FIXED_REGPARAM_BY_KERNEL must contain exactly 5 values.');
assert(numel(FIXED_C_BY_KERNEL)==2, ...
    'FIXED_C_BY_KERNEL must contain exactly 2 values: Gaussian and Multiquadric.');
nRuns = nModel*nKernel;

Xraw = double(InputData.Train.XYZ);
XtestRaw = double(InputData.Test.XYZ);
assert(isreal(Xraw) && ~isempty(Xraw) && all(isfinite(Xraw(:))), ...
    'Training XYZ is invalid.');
assert(isreal(XtestRaw) && ~isempty(XtestRaw) && all(isfinite(XtestRaw(:))), ...
    'Test XYZ is invalid.');
assert(size(Xraw,2)==size(XtestRaw,2),'Train/test coordinate dimension mismatch.');
assert(size(Xraw,1)-1>=MIN_NEIGHBORS, ...
    'Insufficient target-excluded training observations for minNeighbors=%d.',MIN_NEIGHBORS);

SensitivityRanges = validateFixedReferenceAgainstModel1MR( ...
    S,NameFunction_RBF,FIXED_REGPARAM_BY_KERNEL,FIXED_C_BY_KERNEL,Reference);

%% 2. VERIFY THE SAME 50:10:200 km RADIUS GRID AS MODE 10 / MODE 1
RadiiMain = getSetting(S,{'Radii'});
validateRequiredMRRadii(RadiiMain,NameFunction_RBF,MR_RADII_RAW);

%% 3. AUTHORITATIVE MODE-1 RADIUS-TRANSFER SETTINGS
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

%% 4. COMMON COORDINATE SCALING + SHARED GEOMETRY
% Coordinate scaling follows Model 1.  Epsilon is NOT global: for shaped
% kernels it is reconstructed from each actual local fitting support as
% epsilon = C_fixed * dLocal.
Sglobal = common.MeanDistant(Xraw);
assert(isfinite(Sglobal) && isscalar(Sglobal) && Sglobal>0, ...
    'common.MeanDistant(Xraw) returned an invalid coordinate scale.');
X = Xraw./Sglobal;
Xtest = XtestRaw./Sglobal;

Dtrain = pdist2(X,X);
Ptrain = [ones(size(X,1),1),X];
Dtest = pdist2(X,Xtest);            % nTrain x nTest
Ptest = [ones(size(Xtest,1),1),Xtest];
radiiScaled = MR_RADII_RAW./Sglobal;

% Geometry-only exact-radius supports are reused by every GGM/kernel job.
% This removes repeated radius searches and admissibility checks.
TrainSupport = precomputeTrainingSupportMR( ...
    X,Dtrain,radiiScaled,MIN_NEIGHBORS);

numericalOptions = struct();
if isfield(S,'options') && isstruct(S.options) && ...
   isfield(S.options,'solver') && isstruct(S.options.solver)
    numericalOptions.solver = S.options.solver;
else
    numericalOptions.solver = struct('cond_lsq',1e12);
end
numericalOptions.solver.compute_rel_residual = false;


%% 5. MODEL x KERNEL JOBS
Jobs = repmat(struct(),nRuns,1);
q = 0;
for i = 1:nModel
    model = ListOfModel{i};
    assert(isfield(InputData.Train.DeltaZeta,model) && ...
           isfield(InputData.Test.DeltaZeta,model), ...
        'InputData is missing model %s.',model);
    ytrain0 = InputData.Train.DeltaZeta.(model);
    ytest0 = InputData.Test.DeltaZeta.(model);

    for j = 1:nKernel
        q = q+1;
        kernel = NameFunction_RBF{j};
        key = lower(char(string(kernel)));
        useShape = common.kernelHasShapeParameter(kernel);

        regIdx = find(strcmp(KERNEL_ORDER_REG,key),1);
        assert(~isempty(regIdx), ...
            'Kernel %s is not present in KERNEL_ORDER_REG.',kernel);

        if strcmp(key,'gaussian')
            fixedC = FIXED_C_BY_KERNEL(1);
            shapeRule = 'epsilon = C_gaussian,fixed * dLocal(actual support)';
        elseif strcmp(key,'multiquadric')
            fixedC = FIXED_C_BY_KERNEL(2);
            shapeRule = 'epsilon = C_multiquadric,fixed * dLocal(actual support)';
        elseif useShape
            error('M_R has no prescribed fixed C for shaped kernel %s.',kernel);
        else
            fixedC = NaN;
            shapeRule = 'shape-free';
        end

        tm = lower(strtrim(char(string(transferMethod{i,j}))));
        assert(any(strcmp(tm,{'nearest','idw','hybrid','ensemble'})), ...
            'Unsupported transfer method "%s" for %s | %s.',tm,model,kernel);

        Jobs(q).i = i;
        Jobs(q).j = j;
        Jobs(q).model = model;
        Jobs(q).kernel = kernel;
        Jobs(q).key = key;
        Jobs(q).useShape = useShape;
        Jobs(q).ytrain = ytrain0(:);
        Jobs(q).ytest = ytest0(:);
        Jobs(q).radiiRaw = MR_RADII_RAW;
        Jobs(q).radiiScaled = radiiScaled;
        Jobs(q).fixedRegParam = FIXED_REGPARAM_BY_KERNEL(regIdx);
        Jobs(q).minNeighbors = MIN_NEIGHBORS;
        Jobs(q).fixedC = fixedC;
        Jobs(q).shapeRule = shapeRule;
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
            delete(pool); pool = [];
        end
        if isempty(pool) || pool.NumWorkers~=NUM_WORKERS || ...
           ~isa(pool,'parallel.ProcessPool')
            if ~isempty(pool), delete(pool); end
            pool = parpool('Processes',NUM_WORKERS);
        end
        parallelUsed = true;
    catch ME
        if ~ALLOW_SERIAL_FALLBACK, rethrow(ME); end
        warning('M_R falling back to serial: %s',ME.message);
    end
end

%% 7. RUN MODEL x KERNEL JOBS
RunCell = cell(nRuns,1);
SummaryCell = cell(nRuns,1);
tBatch = tic;
if parallelUsed
    parfor q = 1:nRuns
        [RunCell{q},SummaryCell{q}] = runOneMR( ...
            Jobs(q),X,Xtest,Dtrain,Ptrain,Dtest,Ptest,TrainSupport, ...
            numericalOptions,STORE_POINTWISE);
    end
else
    for q = 1:nRuns
        [RunCell{q},SummaryCell{q}] = runOneMR( ...
            Jobs(q),X,Xtest,Dtrain,Ptrain,Dtest,Ptest,TrainSupport, ...
            numericalOptions,STORE_POINTWISE);
    end
end
batchWallTime = toc(tBatch);

%% 8. RESULTS
Results = struct();
Results.Schema = 'LocalRBF.Ablation.MR.v7.GlobalRegRange.LocalEpsilon';
Results.Mode = 'MR';
Results.LegacySchema = 'LocalRBF.Ablation.Mode12.v7.GlobalRegRange.LocalEpsilon';
Results.LegacyMode = 'mode12';
Results.LegacyLabel = 'Model 12';
Results.CaseName = CaseName;
Results.ListOfModel = ListOfModel;
Results.NameFunction_RBF = NameFunction_RBF;
Results.Reference = Reference;
Results.Run = struct();
for q = 1:nRuns
    f = matlab.lang.makeValidName(sprintf( ...
        'M%03d_K%03d_%s_%s_MR',Jobs(q).i,Jobs(q).j, ...
        Jobs(q).model,Jobs(q).kernel));
    Results.Run.(f) = RunCell{q};
end
Results.Summary = vertcat(SummaryCell{:});
Results.Settings.RadiiRaw = MR_RADII_RAW;
Results.Settings.RadiiKm = MR_RADII_RAW./1000;
Results.Settings.RadiusRule = ...
    'at each cap, each training point chooses exact-min-AE radius <= cap';
Results.Settings.PointSpecificRadiusSelection = true;
Results.Settings.FinalCapRule = ...
    'smallest cap attaining exact minimum strict pseudo-unseen RMSE';
Results.Settings.FixedRegParamByKernel = FIXED_REGPARAM_BY_KERNEL;
Results.Settings.FixedRegParamKernelOrder = KERNEL_ORDER_REG;
Results.Settings.RegParamTuned = false;
Results.Settings.FixedCByKernel = FIXED_C_BY_KERNEL;
Results.Settings.FixedCKernelOrder = KERNEL_ORDER_C;
Results.Settings.CommonSensitivityRegMin = SensitivityRanges.CommonRegMin;
Results.Settings.CommonSensitivityRegMax = SensitivityRanges.CommonRegMax;
Results.Settings.CommonSensitivityCMin = SensitivityRanges.CommonCMin;
Results.Settings.CommonSensitivityCMax = SensitivityRanges.CommonCMax;
Results.Settings.Model1InvestigatedRegMinByKernel = SensitivityRanges.Model1RegMinByKernel;
Results.Settings.Model1InvestigatedRegMaxByKernel = SensitivityRanges.Model1RegMaxByKernel;
Results.Settings.Model1InvestigatedCMinByKernel = SensitivityRanges.Model1CMinByKernel;
Results.Settings.Model1InvestigatedCMaxByKernel = SensitivityRanges.Model1CMaxByKernel;
Results.Settings.ReferenceSamplingRangeRule = ...
    'RegParam from GLOBAL-RBF input range; C from common local Gaussian/MQ envelope';
Results.Settings.FixedReferenceRegMin = SensitivityRanges.CommonRegMin;
Results.Settings.FixedReferenceRegMax = SensitivityRanges.CommonRegMax;
Results.Settings.RegRangeSource = 'GLOBAL-RBF input range';
Results.Settings.StorePointwiseIntermediates = STORE_POINTWISE;
Results.Settings.PointwiseStorageRule = ...
    'When false, redundant all-radius pointwise matrices are omitted; selected-radius arrays, cap curves, pseudo-unseen curves, and final test outputs are retained';
Results.Settings.EpsilonDefinition = ...
    'epsilon = C_fixed,k * dLocal(actual local support), using common.buildEpsilon(Xfit,1)';
Results.Settings.GaussianShapeRule = 'Fixed C_gaussian; local epsilon reconstructed from actual support';
Results.Settings.MultiquadricShapeRule = 'Fixed C_multiquadric; local epsilon reconstructed from actual support';
Results.Settings.GaussianUsesFixedC = true;
Results.Settings.MultiquadricUsesFixedC = true;
Results.Settings.ShapeTuned = false;
Results.Settings.UsesPseudoUnseen = true;
Results.Settings.UsesTauLoc = false;
Results.Settings.UsesTauCap = false;
Results.Settings.StrictRadiusNoFallback = false;
Results.Settings.MinNeighbors = MIN_NEIGHBORS;
Results.Settings.UsesMinNeighborsFallback = true;
Results.Settings.NeighborhoodRule = ...
    'requested radius; if fewer than 15 eligible observations, add nearest observations until 15';
Results.Settings.PseudoRadiusTransferMethod = transferMethod;
Results.Settings.ParameterTransferMode = parameterMode;
Results.Settings.PseudoDefinition = ...
    'target excluded from transfer donors and fitting set; no nested re-optimization';
Results.Settings.TestRadiusTransferMethod = transferMethod;
Results.Settings.NIDWSources = nIDWSources;
Results.Settings.NearThresholdRaw = nearThreshold;
Results.Settings.S_global = Sglobal;
Results.Settings.ReferenceSeed = Reference.Seed;
Results.Settings.ReferenceID = char(string(Reference.ID));
Results.Runtime.BatchWallTime = batchWallTime;
Results.Runtime.ParallelUsed = parallelUsed;
Results.Runtime.NumWorkers = double(parallelUsed)*NUM_WORKERS;
Results.Provenance.MainResultsFile = MAIN_RESULTS_FILE;
Results.Provenance.InputDataFile = INPUT_DATA_FILE;
Results.Provenance.NoPseudoUnseen = false;
Results.created = datetime('now');

saveAtomicVariablesMR(OutputFile,struct('Results',Results));
disp(Results.Summary);
fprintf('\nSaved M_R %s seed %d:\n%s\n',caseToken,Reference.Seed,OutputFile);
end

%% ========================================================================
function [R,Srow] = runOneMR(Job,X,Xtest,Dtrain,Ptrain,Dtest,Ptest,TrainSupport, ...
    numericalOptions,storePointwise)

tRun = tic;
y = Job.ytrain(:);
ytest = Job.ytest(:);
K = numel(Job.radiiScaled);
n = numel(y);

% ------------------------------------------------------------
% A. Evaluate each exact requested radius ONCE for every training point.
%    minNeighbors=15 fallback may supplement sparse neighbourhoods.
% ------------------------------------------------------------
ResidualByRadius = nan(n,K);
NeighborCountByRadius = nan(n,K);
dLocalByRadius = nan(n,K);
EpsilonByRadius = nan(n,K);
ExactRadiusRMSE = nan(K,1);
nRBFSolvesTrain = 0;

for k = 1:K
    E = evaluateTrainingAtCommonRadius( ...
        X,y,Dtrain,Ptrain,TrainSupport,k,Job.kernel, ...
        Job.fixedC,Job.fixedRegParam,numericalOptions);
    nRBFSolvesTrain = nRBFSolvesTrain + E.nRBFSolves;
    ResidualByRadius(:,k) = E.Residual;
    NeighborCountByRadius(:,k) = E.NeighborCount;
    dLocalByRadius(:,k) = E.dLocal;
    EpsilonByRadius(:,k) = E.Epsilon;
    if all(isfinite(E.Residual))
        ExactRadiusRMSE(k) = sqrt(mean(E.Residual.^2));
    end
end

% ------------------------------------------------------------
% B. For each cap, choose each point's best radius <= cap.
%    No extra RBF solve is needed here.
% ------------------------------------------------------------
CapRMSE = nan(K,1);
CapMAE = nan(K,1);
CapMaxAE = nan(K,1);
CapValid = false(K,1);
BestRadiusIndexByCap = nan(n,K);
SelectedResidualByCap = nan(n,K);
SelectedNeighborCountByCap = nan(n,K);

for kCap = 1:K
    selectedResidual = nan(n,1);
    selectedN = nan(n,1);
    selectedIndex = nan(n,1);

    for i = 1:n
        valid = find(isfinite(ResidualByRadius(i,1:kCap)));
        if isempty(valid), continue; end
        [~,jj] = min(abs(ResidualByRadius(i,valid)));
        kBest = valid(jj);  % ascending radii => smaller radius wins exact tie
        selectedIndex(i) = kBest;
        selectedResidual(i) = ResidualByRadius(i,kBest);
        selectedN(i) = NeighborCountByRadius(i,kBest);
    end

    if all(isfinite(selectedResidual)) && all(isfinite(selectedIndex))
        CapValid(kCap) = true;
        CapRMSE(kCap) = sqrt(mean(selectedResidual.^2));
        CapMAE(kCap) = mean(abs(selectedResidual));
        CapMaxAE(kCap) = max(abs(selectedResidual));
        BestRadiusIndexByCap(:,kCap) = selectedIndex;
        SelectedResidualByCap(:,kCap) = selectedResidual;
        SelectedNeighborCountByCap(:,kCap) = selectedN;
    end
end

% Internal consistency: because each cap selects the minimum absolute
% residual point-by-point from radii <= cap, CapRMSE cannot exceed the
% RMSE obtained at the exact cap radius itself (when that exact-radius
% evaluation is complete).
for kCheck = 1:K
    if CapValid(kCheck) && isfinite(ExactRadiusRMSE(kCheck))
        tolCheck = 1e-12*max([1,abs(CapRMSE(kCheck)),abs(ExactRadiusRMSE(kCheck))]);
        assert(CapRMSE(kCheck) <= ExactRadiusRMSE(kCheck)+tolCheck, ...
            'Mode 12 internal inconsistency at cap %.0f km: CapRMSE exceeds ExactRadiusRMSE.', ...
            Job.radiiRaw(kCheck)/1000);
    end
end

validCaps = find(CapValid & isfinite(CapRMSE));
if isempty(validCaps)
    R = failedRun(Job,'No cap supplied a valid best-radius choice for every training point.',toc(tRun));
    R.Selection.CapRMSE = CapRMSE;
    Srow = makeSummaryRow(Job,R);
    return
end

% ------------------------------------------------------------
% C. STRICT PSEUDO-UNSEEN EVALUATION FOR CAP SELECTION.
%    For each candidate cap, target i is excluded from BOTH:
%      (1) transfer donors, and (2) local fitting observations.
%    The cap-specific pointwise radii selected above are transferred from
%    the remaining training points using the authoritative Mode-1 transfer
%    method. Fixed RegParam and fixed shape are NEVER re-optimized.
%    Official pseudo RMSE is available only when EVERY training target has
%    a valid pseudo-unseen prediction. Identical selected-radius patterns
%    across caps are evaluated only once and reused.
% ------------------------------------------------------------
Pseudo = evaluatePseudoUnseenCapsMR( ...
    X,y,Dtrain,Ptrain,BestRadiusIndexByCap,CapValid,Job,numericalOptions);

validPseudoCaps = find(Pseudo.CapSuccess & isfinite(Pseudo.RMSE));
if isempty(validPseudoCaps)
    R = failedRun(Job,'No cap supplied a complete strict pseudo-unseen prediction for every training point.',toc(tRun));
    R.Selection.CapRMSE = CapRMSE;
    R.Pseudo = Pseudo;
    Srow = makeSummaryRow(Job,R);
    return
end

[~,jj] = min(Pseudo.RMSE(validPseudoCaps));
kStar = validPseudoCaps(jj);  % ascending caps => smaller cap wins exact tie
finalCapRaw = Job.radiiRaw(kStar);
finalCapScaled = Job.radiiScaled(kStar);
finalBestIndex = BestRadiusIndexByCap(:,kStar);
finalRadiusScaledTrain = Job.radiiScaled(finalBestIndex);
finalRadiusRawTrain = Job.radiiRaw(finalBestIndex);
finalResidualTrain = SelectedResidualByCap(:,kStar);
finalNeighborCountTrain = SelectedNeighborCountByCap(:,kStar);
finalLinearIndex = sub2ind(size(dLocalByRadius),(1:n).',finalBestIndex);
finaldLocalTrain = dLocalByRadius(finalLinearIndex);
finalEpsilonTrain = EpsilonByRadius(finalLinearIndex);

% ------------------------------------------------------------
% D. Transfer the selected training radii to independent Test with the
%    AUTHORITATIVE Model-1 transfer function.  Fixed C/RegParam are carried
%    through the same API only as an audit: because every donor has the
%    same fixed parameter, valid transfer must preserve those constants.
% ------------------------------------------------------------
FinalSettings = struct();
FinalSettings.success = true;
FinalSettings.radius = finalRadiusScaledTrain(:);
FinalSettings.reg_param = repmat(Job.fixedRegParam,n,1);
FinalSettings.radiusIndex = finalBestIndex(:);
if common.kernelHasShapeParameter(Job.kernel)
    FinalSettings.C = repmat(Job.fixedC,n,1);
end

idwRadiusCompatibility = max(2*Job.nearThresholdScaled,max(Job.radiiScaled));
try
    TransferTest = localrbf.transferSettingsFromTrainToTest( ...
        X,Xtest,FinalSettings,Job.kernel, ...
        Job.nearThresholdScaled,idwRadiusCompatibility,Dtest.',false, ...
        Job.transferMethod,Job.nIDWSources,Job.parameterTransferMode);
catch ME
    R = failedRun(Job,sprintf('Independent-test transfer failed: %s',ME.message),toc(tRun));
    R.Selection.CapRMSE = CapRMSE;
    R.Pseudo = Pseudo;
    Srow = makeSummaryRow(Job,R);
    return
end

TestEval = evaluateTestTransferMR( ...
    X,Xtest,y,ytest,Dtrain,Ptrain,Dtest,Ptest,TransferTest, ...
    Job.kernel,Job.fixedC,Job.fixedRegParam,Job.minNeighbors,numericalOptions);

R = struct();
R.success = CapValid(kStar) && Pseudo.CapSuccess(kStar) && TestEval.success;
R.model = Job.model;
R.kernel = Job.kernel;
R.mode = 'MR';
R.message = '';
R.FixedRegParam = Job.fixedRegParam;
R.FixedC = Job.fixedC;
R.FixedEpsilon = NaN;  % epsilon is support-specific, not one fixed scalar
R.ShapeRule = Job.shapeRule;
R.UsesPseudoUnseen = true;
R.UsesTauLoc = false;
R.UsesTauCap = false;
R.PointSpecificRadiusSelection = true;
R.Selection.FinalCapIndex = kStar;
R.Selection.FinalCapRaw = finalCapRaw;
R.Selection.FinalCapScaled = finalCapScaled;
R.Selection.FinalRule = 'exact minimum strict pseudo-unseen RMSE after pointwise min-AE radius selection within cap';
R.Selection.RadiusRule = 'each training point chooses exact-min-AE radius <= current cap';
R.Selection.CapRMSE = CapRMSE;
R.Selection.CapMAE = CapMAE;
R.Selection.CapMaxAE = CapMaxAE;
R.Selection.CapValid = CapValid;
R.Selection.ExactRadiusRMSE = ExactRadiusRMSE;
k60 = find(Job.radiiRaw==60000,1);
if ~isempty(k60)
    R.Selection.ExactRadiusRMSE60 = ExactRadiusRMSE(k60);
    R.Selection.CapRMSE60 = CapRMSE(k60);
else
    R.Selection.ExactRadiusRMSE60 = NaN;
    R.Selection.CapRMSE60 = NaN;
end
R.Selection.PseudoRMSE = Pseudo.RMSE;
R.Selection.PseudoMaxAE = Pseudo.MaxAE;
R.Selection.PseudoCapSuccess = Pseudo.CapSuccess;
R.Training.RMSE = CapRMSE(kStar);
R.Training.MAE = CapMAE(kStar);
R.Training.MaxAE = CapMaxAE(kStar);
R.Training.Residual = finalResidualTrain;
R.Training.SelectedRadiusIndex = finalBestIndex;
R.Training.SelectedRadiusRaw = finalRadiusRawTrain;
R.Training.SelectedRadiusScaled = finalRadiusScaledTrain;
R.Training.NeighborCount = finalNeighborCountTrain;
R.Training.dLocal = finaldLocalTrain;
R.Training.Epsilon = finalEpsilonTrain;
R.Pseudo.RMSE = Pseudo.RMSE(kStar);
R.Pseudo.MAE = Pseudo.MAE(kStar);
R.Pseudo.MaxAE = Pseudo.MaxAE(kStar);
R.Pseudo.Residual = Pseudo.Residual(:,kStar);
R.Pseudo.Success = Pseudo.CapSuccess(kStar);
R.Pseudo.NValid = Pseudo.NValid(kStar);
R.Pseudo.NFailed = Pseudo.NFailed(kStar);
R.Pseudo.RMSEAllCaps = Pseudo.RMSE;
R.Pseudo.MaxAEAllCaps = Pseudo.MaxAE;
R.Pseudo.CapSuccess = Pseudo.CapSuccess;
R.Pseudo.ReusedFromCap = Pseudo.ReusedFromCap;
R.Test.RMSE = TestEval.RMSE;
R.Test.MAE = TestEval.MAE;
R.Test.MaxAE = TestEval.MaxAE;
R.Test.Residual = TestEval.Residual;
R.Test.TransferredRadiusScaled = TestEval.EffectiveRadiusScaled;
R.Test.TransferredRadiusRaw = TestEval.EffectiveRadiusScaled .* (finalCapRaw/finalCapScaled);
R.Test.NeighborCount = TestEval.NeighborCount;
R.Test.Prediction = TestEval.Prediction;
R.Test.dLocal = TestEval.dLocal;
R.Test.Epsilon = TestEval.Epsilon;
R.Test.TransferMethod = Job.transferMethod;
R.Test.TransferMethodByPoint = TestEval.TransferMethodByPoint;
R.Test.TransferSourceCount = TestEval.SourceCount;
R.Runtime.TrainingRBFSolves = nRBFSolvesTrain;
R.Runtime.PseudoRBFSolves = Pseudo.nRBFSolves;
R.Runtime.TestRBFSolves = TestEval.nRBFSolves;
R.Runtime.TotalRBFSolves = nRBFSolvesTrain + Pseudo.nRBFSolves + TestEval.nRBFSolves;
R.Runtime.WallTime = toc(tRun);

if storePointwise
    R.Pointwise.TrainingResidualByRadius = ResidualByRadius;
    R.Pointwise.TrainingNeighborCountByRadius = NeighborCountByRadius;
    R.Pointwise.TrainingdLocalByRadius = dLocalByRadius;
    R.Pointwise.TrainingEpsilonByRadius = EpsilonByRadius;
    R.Pointwise.BestRadiusIndexByCap = BestRadiusIndexByCap;
    R.Pointwise.SelectedResidualByCap = SelectedResidualByCap;
    R.Pointwise.TestTransfer = TransferTest;
end

Srow = makeSummaryRow(Job,R);
end

%% ========================================================================
function E = evaluateTrainingAtCommonRadius(X,y,Dtrain,Ptrain,TrainSupport,kRadius, ...
    kernel,fixedC,regParam,numericalOptions)

n = numel(y);
useShape = common.kernelHasShapeParameter(kernel);
residual = nan(n,1);
neighborCount = nan(n,1);
dLocalUsed = nan(n,1);
epsilonUsed = nan(n,1);
nSolves = 0;

for i = 1:n
    if ~TrainSupport.Success(i,kRadius), continue, end
    idx = TrainSupport.Idx{i,kRadius};
    if isempty(idx), continue, end
    Xfit = X(idx,:);
    yfit = y(idx);

    if useShape
        [okEpsilon,dLocal,epsilon] = localEpsilonMR(Xfit,fixedC);
        if ~okEpsilon, continue, end
    else
        dLocal = NaN;
        epsilon = 1;
    end

    geom = struct();
    geom.R = Dtrain(idx,idx);
    geom.P = Ptrain(idx,:);
    geom.R_eval = reshape(Dtrain(i,idx),1,[]);
    geom.P_eval = Ptrain(i,:);

    try
        Sys = common.buildRBFSystem(Xfit,yfit,kernel,epsilon,regParam,geom);
        nSolves = nSolves+1;
        [coeffs,solveInfo] = common.solveLinearSystem(Sys,numericalOptions);
        if ~solveInfo.success || isempty(coeffs) || any(~isfinite(coeffs(:))), continue, end
        yhat = common.predictRBF(X(i,:),Xfit,coeffs,kernel,epsilon,geom);
        if ~isscalar(yhat) || ~isfinite(yhat), continue, end
        residual(i) = y(i)-yhat;
        neighborCount(i) = numel(idx);
        dLocalUsed(i) = dLocal;
        epsilonUsed(i) = epsilon;
    catch
    end
end

E = struct();
E.Residual = residual;
E.NeighborCount = neighborCount;
E.dLocal = dLocalUsed;
E.Epsilon = epsilonUsed;
E.nRBFSolves = nSolves;
E.success = all(isfinite(residual));
if E.success
    E.RMSE = sqrt(mean(residual.^2));
    E.MAE = mean(abs(residual));
    E.MaxAE = max(abs(residual));
else
    E.RMSE = NaN; E.MAE = NaN; E.MaxAE = NaN;
end
end

%% ========================================================================
function P = evaluatePseudoUnseenCapsMR( ...
    X,y,Dtrain,Ptrain,BestRadiusIndexByCap,CapValid,Job,numericalOptions)
%EVALUATEPSEUDOUNSEENCAPSMR
% Strict pseudo-unseen evaluation used ONLY to select the final cap.
%
% For pseudo target i and cap k:
%   1) target i is removed from transfer donors;
%   2) target i is removed from the fitting set;
%   3) cap-specific pointwise radii from the remaining training points are
%      transferred to i using the authoritative Mode-1 transfer method;
%   4) fixed M_R RegParam and fixed C are retained; shaped-kernel
%      epsilon is reconstructed from the actual pseudo-unseen support;
%   5) minNeighbors=15 fallback is applied using donor observations only;
%   6) no parameter/radius re-optimization is performed.
%
% Official pseudo RMSE is finite only when ALL training targets succeed.
% Identical selected-radius patterns across caps are reused without new
% pseudo RBF solves.

n = numel(y);
K = size(BestRadiusIndexByCap,2);
useShape = common.kernelHasShapeParameter(Job.kernel);

P = struct();
P.RMSE = nan(K,1);
P.MAE = nan(K,1);
P.MaxAE = nan(K,1);
P.CapSuccess = false(K,1);
P.NValid = zeros(K,1);
P.NFailed = n*ones(K,1);
P.Residual = nan(n,K);
P.ReusedFromCap = nan(K,1);
P.nRBFSolves = 0;
P.nTransferFailures = 0;
P.nPredictionFailures = 0;

nearThreshold = Job.nearThresholdScaled;
if ~isfinite(nearThreshold) || nearThreshold<=0
    nearThreshold = 1; % compatibility only for transfer modes that ignore it
end
idwRadiusCompatibility = max(2*nearThreshold,max(Job.radiiScaled));

for kCap = 1:K
    if ~CapValid(kCap) || any(~isfinite(BestRadiusIndexByCap(:,kCap)))
        continue
    end

    % Reuse pseudo result if this cap produces exactly the same selected
    % radius index at every training point as an earlier cap.
    reused = false;
    for kPrev = 1:kCap-1
        if CapValid(kPrev) && ...
           isequal(BestRadiusIndexByCap(:,kPrev),BestRadiusIndexByCap(:,kCap)) && ...
           isfinite(P.ReusedFromCap(kPrev))
            P.RMSE(kCap) = P.RMSE(kPrev);
            P.MAE(kCap) = P.MAE(kPrev);
            P.MaxAE(kCap) = P.MaxAE(kPrev);
            P.CapSuccess(kCap) = P.CapSuccess(kPrev);
            P.NValid(kCap) = P.NValid(kPrev);
            P.NFailed(kCap) = P.NFailed(kPrev);
            P.Residual(:,kCap) = P.Residual(:,kPrev);
            P.ReusedFromCap(kCap) = kPrev;
            reused = true;
            break
        end
    end
    if reused, continue; end
    P.ReusedFromCap(kCap) = kCap;

    radiusIndex = BestRadiusIndexByCap(:,kCap);
    radiusTrain = Job.radiiScaled(radiusIndex);
    residual = nan(n,1);

    for iTarget = 1:n
        donorIdx = (1:n).';
        donorIdx(iTarget) = [];

        FinalDonor = struct();
        FinalDonor.success = true;
        FinalDonor.radius = radiusTrain(donorIdx);
        FinalDonor.reg_param = repmat(Job.fixedRegParam,numel(donorIdx),1);
        if useShape
            FinalDonor.C = repmat(Job.fixedC,numel(donorIdx),1);
        end

        DqueryDonor = reshape(Dtrain(iTarget,donorIdx),1,[]);
        try
            Transfer = localrbf.transferSettingsFromTrainToTest( ...
                X(donorIdx,:),X(iTarget,:),FinalDonor,Job.kernel, ...
                nearThreshold,idwRadiusCompatibility,DqueryDonor,false, ...
                Job.transferMethod,Job.nIDWSources,Job.parameterTransferMode);
        catch
            P.nTransferFailures = P.nTransferFailures + 1;
            continue
        end

        if ~isfield(Transfer,'querySuccess') || ~Transfer.querySuccess(1)
            P.nTransferFailures = P.nTransferFailures + 1;
            continue
        end

        assertFixedParameterTransferMR(Transfer,1,Job);

        if strcmp(Transfer.transferMethod{1},'ensemble')
            radii = Transfer.ensembleUniqueRadius{1};
            weights = Transfer.ensembleUniqueWeight{1};
        else
            radii = Transfer.radius(1);
            weights = 1;
        end

        [ok,yhat,nSolves] = evaluateOnePseudoTargetMR( ...
            X,y,Dtrain,Ptrain,iTarget,donorIdx,radii,weights, ...
            Job.kernel,Job.fixedC,Job.fixedRegParam, ...
            Job.minNeighbors,numericalOptions);
        P.nRBFSolves = P.nRBFSolves + nSolves;

        if ~ok || ~isfinite(yhat)
            P.nPredictionFailures = P.nPredictionFailures + 1;
            continue
        end
        residual(iTarget) = y(iTarget)-yhat;
    end

    valid = isfinite(residual);
    P.NValid(kCap) = sum(valid);
    P.NFailed(kCap) = n-P.NValid(kCap);
    P.Residual(:,kCap) = residual;

    if P.NValid(kCap)==n
        P.CapSuccess(kCap) = true;
        P.RMSE(kCap) = sqrt(mean(residual.^2));
        P.MAE(kCap) = mean(abs(residual));
        P.MaxAE(kCap) = max(abs(residual));
    end
end
end

%% ========================================================================
function assertFixedParameterTransferMR(Transfer,it,Job)
useShape=common.kernelHasShapeParameter(Job.kernel);
method_i=lower(char(string(Transfer.transferMethod{it})));
if strcmp(method_i,'ensemble')
    regs=Transfer.ensembleUniqueRegParam{it}(:);
    if useShape, Cs=Transfer.ensembleUniqueC{it}(:); else, Cs=[]; end
else
    regs=Transfer.reg_param(it);
    if useShape, Cs=Transfer.C(it); else, Cs=[]; end
end
regTol=1e-10*max([1,abs(Job.fixedRegParam),max(abs(regs))]);
assert(all(abs(regs-Job.fixedRegParam)<=regTol), ...
    'M_R pseudo transfer changed fixed RegParam.');
if useShape
    cTol=1e-10*max([1,abs(Job.fixedC),max(abs(Cs))]);
    assert(all(abs(Cs-Job.fixedC)<=cTol), ...
        'M_R pseudo transfer changed fixed C.');
end
end

%% ========================================================================
function [success,yhat,nSolves] = evaluateOnePseudoTargetMR( ...
    X,y,Dtrain,Ptrain,iTarget,donorIdx,radii,weights,kernel, ...
    fixedC,regParam,minNeighbors,numericalOptions)
%EVALUATEONEPSEUDOTARGETMR Strict target-excluded pseudo prediction.

success = false;
yhat = NaN;
nSolves = 0;

radii = radii(:);
weights = weights(:);
if isempty(radii) || numel(radii)~=numel(weights) || ...
   any(~isfinite(radii)) || any(radii<=0) || ...
   any(~isfinite(weights)) || any(weights<0) || sum(weights)<=0
    return
end
weights = weights./sum(weights);

useShape = common.kernelHasShapeParameter(kernel);
yhatTuple = nan(numel(radii),1);

for u = 1:numel(radii)
    dDonor = reshape(Dtrain(iTarget,donorIdx),[],1);
    idxLocal = radiusWithMinNeighborsIndices(dDonor,radii(u),[],minNeighbors);
    if isempty(idxLocal), return; end
    idx = donorIdx(idxLocal);

    Xfit = X(idx,:);
    yfit = y(idx);
    try
        [isOK,~] = common.checkPolynomialAdmissibility(Xfit);
    catch
        isOK = false;
    end
    if ~isOK, return; end

    if useShape
        [okEpsilon,~,epsilon] = localEpsilonMR(Xfit,fixedC);
        if ~okEpsilon, return, end
    else
        epsilon = 1;
    end

    geom = struct();
    geom.R = Dtrain(idx,idx);
    geom.P = Ptrain(idx,:);
    geom.R_eval = reshape(Dtrain(iTarget,idx),1,[]);
    geom.P_eval = Ptrain(iTarget,:);

    try
        Sys = common.buildRBFSystem(Xfit,yfit,kernel,epsilon,regParam,geom);
        nSolves = nSolves+1;
        [coeffs,solveInfo] = common.solveLinearSystem(Sys,numericalOptions);
        if ~solveInfo.success || isempty(coeffs) || any(~isfinite(coeffs(:)))
            return
        end
        yh = common.predictRBF(X(iTarget,:),Xfit,coeffs,kernel,epsilon,geom);
        if ~isscalar(yh) || ~isfinite(yh), return; end
        yhatTuple(u) = yh;
    catch
        return
    end
end

if all(isfinite(yhatTuple))
    yhat = sum(weights.*yhatTuple);
    success = isfinite(yhat);
end
end

function T = evaluateTestTransferMR(X,Xtest,y,ytest,Dtrain,Ptrain,Dtest,Ptest, ...
    Transfer,kernel,fixedC,fixedRegParam,minNeighbors,numericalOptions)
%EVALUATETESTTRANSFERMR
% Independent-test evaluation from the authoritative Model-1 transfer
% payload.  C and RegParam are expected to remain exactly the fixed
% reference values; only radius may vary spatially.

nTest = numel(ytest);
useShape = common.kernelHasShapeParameter(kernel);

prediction = nan(nTest,1);
residual = nan(nTest,1);
neighborCount = nan(nTest,1);
effectiveRadius = nan(nTest,1);
sourceCount = nan(nTest,1);
dLocalUsed = nan(nTest,1);
epsilonUsed = nan(nTest,1);
transferMethodByPoint = strings(nTest,1);
nSolves = 0;

for it=1:nTest
    if ~isfield(Transfer,'querySuccess') || numel(Transfer.querySuccess)<it || ...
       ~Transfer.querySuccess(it)
        continue
    end

    method_i=lower(char(string(Transfer.transferMethod{it})));
    transferMethodByPoint(it)=string(method_i);

    if isfield(Transfer,'nTransferSources') && numel(Transfer.nTransferSources)>=it
        sourceCount(it)=Transfer.nTransferSources(it);
    elseif strcmp(method_i,'nearest')
        sourceCount(it)=1;
    elseif strcmp(method_i,'ensemble') && isfield(Transfer,'ensembleNUniqueTuples') && ...
           numel(Transfer.ensembleNUniqueTuples)>=it
        sourceCount(it)=Transfer.ensembleNUniqueTuples(it);
    end

    if strcmp(method_i,'ensemble')
        if ~isfield(Transfer,'ensembleUniqueRadius') || ...
           numel(Transfer.ensembleUniqueRadius)<it
            continue
        end
        radii=Transfer.ensembleUniqueRadius{it}(:);
        regs=Transfer.ensembleUniqueRegParam{it}(:);
        weights=Transfer.ensembleUniqueWeight{it}(:);
        if useShape
            Cs=Transfer.ensembleUniqueC{it}(:);
        else
            Cs=nan(size(radii));
        end
    else
        radii=Transfer.radius(it);
        regs=Transfer.reg_param(it);
        weights=1;
        if useShape, Cs=Transfer.C(it); else, Cs=NaN; end
    end

    radii=radii(:); regs=regs(:); weights=weights(:); Cs=Cs(:);
    if isempty(radii) || numel(regs)~=numel(radii) || ...
       numel(weights)~=numel(radii) || (useShape && numel(Cs)~=numel(radii))
        continue
    end
    if any(~isfinite(radii)) || any(radii<=0) || ...
       any(~isfinite(regs)) || any(regs<=0) || ...
       any(~isfinite(weights)) || any(weights<0) || sum(weights)<=0 || ...
       (useShape && (any(~isfinite(Cs)) || any(Cs<=0)))
        continue
    end
    weights=weights./sum(weights);

    % Fixed-parameter invariant: the transfer API must not alter a constant
    % donor field.
    regTol=1e-10*max([1,abs(fixedRegParam),max(abs(regs))]);
    assert(all(abs(regs-fixedRegParam)<=regTol), ...
        'M_R transfer changed fixed RegParam.');
    if useShape
        cTol=1e-10*max([1,abs(fixedC),max(abs(Cs))]);
        assert(all(abs(Cs-fixedC)<=cTol), ...
            'M_R transfer changed fixed C.');
    end

    yhTuple=nan(numel(radii),1);
    ncTuple=nan(numel(radii),1);
    dTuple=nan(numel(radii),1);
    epsTuple=nan(numel(radii),1);
    allOK=true;

    for u=1:numel(radii)
        idx=radiusWithMinNeighborsIndices(Dtest(:,it),radii(u),[],minNeighbors);
        if isempty(idx), allOK=false; break, end
        Xfit=X(idx,:); yfit=y(idx);

        try
            [isOK,~]=common.checkPolynomialAdmissibility(Xfit);
        catch
            isOK=false;
        end
        if ~isOK, allOK=false; break, end

        if useShape
            [okEpsilon,dLocal,epsilon]=localEpsilonMR(Xfit,Cs(u));
            if ~okEpsilon, allOK=false; break, end
        else
            dLocal=NaN; epsilon=1;
        end

        geom=struct();
        geom.R=Dtrain(idx,idx);
        geom.P=Ptrain(idx,:);
        geom.R_eval=reshape(Dtest(idx,it),1,[]);
        geom.P_eval=Ptest(it,:);

        try
            Sys=common.buildRBFSystem(Xfit,yfit,kernel,epsilon,regs(u),geom);
            nSolves=nSolves+1;
            [coeffs,solveInfo]=common.solveLinearSystem(Sys,numericalOptions);
            if ~solveInfo.success || isempty(coeffs) || any(~isfinite(coeffs(:)))
                allOK=false; break
            end
            yh=common.predictRBF(Xtest(it,:),Xfit,coeffs,kernel,epsilon,geom);
            if ~isscalar(yh) || ~isfinite(yh), allOK=false; break, end
            yhTuple(u)=yh;
            ncTuple(u)=numel(idx);
            dTuple(u)=dLocal;
            epsTuple(u)=epsilon;
        catch
            allOK=false; break
        end
    end

    if ~allOK || any(~isfinite(yhTuple)), continue, end

    prediction(it)=sum(weights.*yhTuple);
    residual(it)=ytest(it)-prediction(it);
    neighborCount(it)=sum(weights.*ncTuple);
    effectiveRadius(it)=sum(weights.*radii);
    if useShape
        dLocalUsed(it)=sum(weights.*dTuple);
        epsilonUsed(it)=sum(weights.*epsTuple);
    else
        dLocalUsed(it)=NaN;
        epsilonUsed(it)=1;
    end
end

T=struct();
T.success=all(isfinite(residual));
T.Prediction=prediction;
T.Residual=residual;
T.NeighborCount=neighborCount;
T.dLocal=dLocalUsed;
T.Epsilon=epsilonUsed;
T.EffectiveRadiusScaled=effectiveRadius;
T.SourceCount=sourceCount;
T.TransferMethodByPoint=transferMethodByPoint;
T.nRBFSolves=nSolves;
if T.success
    T.RMSE=sqrt(mean(residual.^2));
    T.MAE=mean(abs(residual));
    T.MaxAE=max(abs(residual));
else
    T.RMSE=NaN; T.MAE=NaN; T.MaxAE=NaN;
end
end

%% ========================================================================
function R = failedRun(Job,msg,wall)
R = struct();
R.success = false; R.model = Job.model; R.kernel = Job.kernel; R.mode = 'MR';
R.message = msg; R.FixedRegParam = Job.fixedRegParam;
R.FixedC = Job.fixedC;
R.FixedEpsilon = NaN; R.ShapeRule = Job.shapeRule;
R.Selection = struct('FinalCapIndex',NaN,'FinalCapRaw',NaN);
R.Training = struct('RMSE',NaN,'MAE',NaN,'MaxAE',NaN);
R.Pseudo = struct('RMSE',NaN,'MAE',NaN,'MaxAE',NaN,'Success',false);
R.Test = struct('RMSE',NaN,'MAE',NaN,'MaxAE',NaN);
R.Runtime = struct('TrainingRBFSolves',NaN,'PseudoRBFSolves',NaN,'TestRBFSolves',NaN, ...
    'TotalRBFSolves',NaN,'WallTime',wall);
end

function Srow = makeSummaryRow(Job,R)
%MAKESUMMARYROW Always return exactly ONE table row.
%
% Successful Mode-12 runs store scalar final-cap metrics in R.Training,
% R.Pseudo and R.Test.  Some failure branches intentionally preserve the
% complete cap-wise vectors in R.Pseudo (e.g. Pseudo.RMSE is Kx1).  Those
% diagnostic vectors must not be passed directly to table(), otherwise
% MATLAB reports "All table variables must have the same number of rows."
%
% Therefore the summary uses a metric only when it is scalar; otherwise
% the summary value is NaN while the full diagnostic vector remains in R.

fc = getScalarField(R,{'Selection','FinalCapRaw'},NaN);

tr = getScalarField(R,{'Training','RMSE'},NaN);
pr = getScalarField(R,{'Pseudo','RMSE'},NaN);
te = getScalarField(R,{'Test','RMSE'},NaN);

tm = getScalarField(R,{'Training','MaxAE'},NaN);
pm = getScalarField(R,{'Pseudo','MaxAE'},NaN);
xm = getScalarField(R,{'Test','MaxAE'},NaN);

ns = getScalarField(R,{'Runtime','TotalRBFSolves'},NaN);
wt = getScalarField(R,{'Runtime','WallTime'},NaN);

if isfield(R,'Training') && isfield(R.Training,'SelectedRadiusRaw') && ...
        ~isempty(R.Training.SelectedRadiusRaw) && ...
        all(isfinite(R.Training.SelectedRadiusRaw(:)))
    rr = R.Training.SelectedRadiusRaw(:);
    meanR = mean(rr);
    minR = min(rr);
    maxR = max(rr);
else
    meanR = NaN;
    minR = NaN;
    maxR = NaN;
end

er60 = getScalarField(R,{'Selection','ExactRadiusRMSE60'},NaN);
cr60 = getScalarField(R,{'Selection','CapRMSE60'},NaN);

successValue = false;
if isfield(R,'success') && isscalar(R.success)
    successValue = logical(R.success);
end

% Defensive scalarization of Job fields.
fixedReg = scalarOrNaN(Job.fixedRegParam);
fixedC = scalarOrNaN(Job.fixedC);
fixedEpsilon = NaN;

Srow = table( ...
    string(Job.model), ...
    string(Job.kernel), ...
    "MR", ...
    scalarOrNaN(fc), ...
    scalarOrNaN(meanR), ...
    scalarOrNaN(minR), ...
    scalarOrNaN(maxR), ...
    fixedReg, ...
    fixedC, ...
    fixedEpsilon, ...
    string(Job.shapeRule), ...
    scalarOrNaN(er60), ...
    scalarOrNaN(cr60), ...
    string(Job.transferMethod), ...
    scalarOrNaN(tr), ...
    scalarOrNaN(pr), ...
    scalarOrNaN(te), ...
    scalarOrNaN(tm), ...
    scalarOrNaN(pm), ...
    scalarOrNaN(xm), ...
    scalarOrNaN(ns), ...
    scalarOrNaN(wt), ...
    successValue, ...
    'VariableNames',{ ...
    'Model','Kernel','Mode','FinalCapRaw', ...
    'MeanTrainRadiusRaw','MinTrainRadiusRaw','MaxTrainRadiusRaw', ...
    'FixedRegParam','FixedC','FixedEpsilon','ShapeRule', ...
    'ExactRadiusRMSE60','CapRMSE60','TestRadiusTransfer', ...
    'TrainRMSE','PseudoRMSE','TestRMSE', ...
    'TrainMaxAE','PseudoMaxAE','TestMaxAE', ...
    'TotalRBFSolves','WallTime','Success'});
end

function value = getScalarField(S,path,defaultValue)
value = defaultValue;
try
    x = S;
    for k = 1:numel(path)
        if ~isstruct(x) || ~isfield(x,path{k})
            return
        end
        x = x.(path{k});
    end
    if isnumeric(x) || islogical(x)
        if isscalar(x) && isfinite(double(x))
            value = double(x);
        end
    end
catch
    value = defaultValue;
end
end

function value = scalarOrNaN(x)
if (isnumeric(x) || islogical(x)) && isscalar(x) && isfinite(double(x))
    value = double(x);
else
    value = NaN;
end
end

%% ========================================================================
function validateRequiredMRRadii(Radii,kernels,required)
for j = 1:numel(kernels)
    key = lower(char(string(kernels{j})));
    if isstruct(Radii)
        assert(isfield(Radii,key),'Mode-1 Radii.%s is missing.',key);
        r = Radii.(key);
    else
        r = Radii;
    end
    r = double(r(:));
    assert(isequal(r,double(required(:))), ...
        ['M_R requires the exact Mode-1 grid 50:10:200 km. ' ...
         'Radii.%s does not match this grid.'],key);
end
end

function idx = radiusWithMinNeighborsIndices(distanceVector,radius,excludeIndex,minNeighbors)
% Identical fixed-radius/min-neighbour rule as M_0.
d=distanceVector(:);
maskFinite=isfinite(d);
maskRadius=maskFinite & (d<=radius);
if nargin>=3 && ~isempty(excludeIndex)
    maskRadius(excludeIndex)=false;
    maskFinite(excludeIndex)=false;
end
idx=find(maskRadius);
if nargin<4 || isempty(minNeighbors) || ~isfinite(minNeighbors) || minNeighbors<=0
    return
end
if numel(idx) >= minNeighbors
    return
end
allIdx=find(maskFinite);
if isempty(allIdx)
    return
end
[~,ord]=sort(d(allIdx),'ascend');
allIdx=allIdx(ord);
keepN=min(minNeighbors,numel(allIdx));
idx=unique([idx; allIdx(1:keepN)],'stable');
end

function auditDependenciesMR()
required={'localrbf.transferSettingsFromTrainToTest','common.MeanDistant', ...
    'common.buildEpsilon','common.checkPolynomialAdmissibility', ...
    'common.buildRBFSystem','common.solveLinearSystem','common.predictRBF'};
for k=1:numel(required)
    assert(~isempty(which(required{k})),'Required function not on MATLAB path: %s',required{k});
end
narg=nargin('localrbf.transferSettingsFromTrainToTest');
assert(narg<0 || narg>=11, ...
    'Obsolete localrbf.transferSettingsFromTrainToTest detected; 11-input API required.');
end

%% ========================================================================
function Support = precomputeTrainingSupportMR(X,Dtrain,radiiScaled,minNeighbors)
n = size(Dtrain,1);
K = numel(radiiScaled);
Support.Idx = cell(n,K);
Support.Success = false(n,K);
Support.NeighborCount = nan(n,K);
for k = 1:K
    r = radiiScaled(k);
    for i = 1:n
        idx = radiusWithMinNeighborsIndices(Dtrain(:,i),r,i,minNeighbors);
        if isempty(idx), continue, end
        try
            [isOK,~] = common.checkPolynomialAdmissibility(X(idx,:));
        catch
            isOK = false;
        end
        if ~isOK, continue, end
        Support.Idx{i,k} = idx(:);
        Support.Success(i,k) = true;
        Support.NeighborCount(i,k) = numel(idx);
    end
end
end

%% ========================================================================
function [ok,dLocal,epsilon] = localEpsilonMR(Xfit,C)
ok = false; dLocal = NaN; epsilon = NaN;
if ~isscalar(C) || ~isfinite(C) || C<=0, return, end
try
    [~,dLocal] = common.buildEpsilon(Xfit,1);
catch
    return
end
if ~isscalar(dLocal) || ~isfinite(dLocal) || dLocal<=0, return, end
epsilon = max(C*dLocal,1e-12);
ok = isfinite(epsilon) && epsilon>0;
end

%% ========================================================================
function Reference = validateReferenceMR(Reference)
assert(isstruct(Reference) && isscalar(Reference),'Reference must be one struct.');
required={'Seed','ID','FixedRadiusRaw','FixedRegParamByKernel','FixedCByKernel'};
assert(all(isfield(Reference,required)),'Reference is missing required fields.');
assert(isnumeric(Reference.Seed) && isscalar(Reference.Seed) && isfinite(Reference.Seed) && ...
    Reference.Seed>=0 && Reference.Seed==round(Reference.Seed),'Invalid Reference.Seed.');
r=double(Reference.FixedRegParamByKernel(:));
c=double(Reference.FixedCByKernel(:));
assert(numel(r)==5 && all(isfinite(r)) && all(r>0),'Reference RegParam must contain 5 positive values.');
assert(numel(c)==2 && all(isfinite(c)) && all(c>0),'Reference C must contain 2 positive values.');
end

function Options = normalizeOptionsMR(Options)
if ~isfield(Options,'ReuseExistingResults'), Options.ReuseExistingResults=true; end
if ~isfield(Options,'UseParallel'), Options.UseParallel=true; end
if ~isfield(Options,'AllowSerialFallback'), Options.AllowSerialFallback=false; end
if ~isfield(Options,'StorePointwise'), Options.StorePointwise=false; end
Options.ReuseExistingResults=logical(Options.ReuseExistingResults);
Options.UseParallel=logical(Options.UseParallel);
Options.AllowSerialFallback=logical(Options.AllowSerialFallback);
Options.StorePointwise=logical(Options.StorePointwise);
end

function tf = isCompatibleExistingMR(R,CaseNumber,Reference)
tf=false;
expectedSchema='LocalRBF.Ablation.MR.v7.GlobalRegRange.LocalEpsilon';
if ~isstruct(R) || ~isfield(R,'Schema') || ...
   ~strcmp(char(string(R.Schema)),expectedSchema) || ...
   ~isfield(R,'Mode') || ~strcmpi(char(string(R.Mode)),'MR') || ...
   ~isfield(R,'Reference') || ~isstruct(R.Reference) || ...
   ~isfield(R,'Summary') || ~istable(R.Summary) || height(R.Summary)~=30
    return
end
if isfield(R,'CaseName') && ~strcmpi(char(string(R.CaseName)),sprintf('Case%d',CaseNumber)), return, end
try
    x=[double(R.Reference.FixedRadiusRaw);double(R.Reference.FixedRegParamByKernel(:));double(R.Reference.FixedCByKernel(:))];
    y=[double(Reference.FixedRadiusRaw);double(Reference.FixedRegParamByKernel(:));double(Reference.FixedCByKernel(:))];
    tol=1e-12*max(1,max(abs([x;y])));
    tf=double(R.Reference.Seed)==double(Reference.Seed) && ...
       strcmp(char(string(R.Reference.ID)),char(string(Reference.ID))) && ...
       numel(x)==numel(y) && max(abs(x-y))<=tol;
catch
    tf=false;
end
end

function Ranges=validateFixedReferenceAgainstModel1MR(S,kernels,regVals,cVals,Reference)
%VALIDATEFIXEDREFERENCEAGAINSTMODEL1MR
% Validate random fixed parameters. RegParam is checked against the
% GLOBAL-RBF input range carried by the reference; the narrower Model-1 local
% RegParam ranges are retained only as provenance. C is checked against the
% common envelope spanning the local Gaussian/MQ investigated C ranges.

RegMin=getSetting(S,{'RegMin'}); RegMax=getSetting(S,{'RegMax'});
CMin=getSetting(S,{'CMin'}); CMax=getSetting(S,{'CMax'});
regOrder={'gaussian','multiquadric','linear','cubic','thinplate'};
cOrder={'gaussian','multiquadric'};

Ranges=struct();
Ranges.Model1RegMinByKernel=nan(1,numel(regOrder));
Ranges.Model1RegMaxByKernel=nan(1,numel(regOrder));
for k=1:numel(regOrder)
    Ranges.Model1RegMinByKernel(k)=extractKernelBoundMR( ...
        RegMin,regOrder{k},kernels,numel(kernels),'RegMin');
    Ranges.Model1RegMaxByKernel(k)=extractKernelBoundMR( ...
        RegMax,regOrder{k},kernels,numel(kernels),'RegMax');
end

Ranges.Model1CMinByKernel=nan(1,numel(cOrder));
Ranges.Model1CMaxByKernel=nan(1,numel(cOrder));
for k=1:numel(cOrder)
    Ranges.Model1CMinByKernel(k)=extractKernelBoundMR( ...
        CMin,cOrder{k},kernels,numel(kernels),'CMin');
    Ranges.Model1CMaxByKernel(k)=extractKernelBoundMR( ...
        CMax,cOrder{k},kernels,numel(kernels),'CMax');
end

if nargin>=5 && isstruct(Reference) && isfield(Reference,'FixedReferenceRegMin')
    Ranges.CommonRegMin=double(Reference.FixedReferenceRegMin);
    Ranges.CommonRegMax=double(Reference.FixedReferenceRegMax);
elseif nargin>=5 && isstruct(Reference) && isfield(Reference,'CommonSensitivityRegMin')
    % Backward-compatible alias from the master reference structure.
    Ranges.CommonRegMin=double(Reference.CommonSensitivityRegMin);
    Ranges.CommonRegMax=double(Reference.CommonSensitivityRegMax);
else
    error('Reference does not contain fixed-reference RegParam range provenance.');
end
assert(isfinite(Ranges.CommonRegMin) && isfinite(Ranges.CommonRegMax) && ...
       Ranges.CommonRegMin>0 && Ranges.CommonRegMax>=Ranges.CommonRegMin, ...
    'Invalid fixed-reference RegParam range.');

Ranges.CommonCMin=min(Ranges.Model1CMinByKernel);
Ranges.CommonCMax=max(Ranges.Model1CMaxByKernel);

regVals=double(regVals(:).');
cVals=double(cVals(:).');

tolReg=1e-12*max([1,abs(Ranges.CommonRegMin),abs(Ranges.CommonRegMax),max(abs(regVals))]);
assert(all(regVals>=Ranges.CommonRegMin-tolReg & regVals<=Ranges.CommonRegMax+tolReg), ...
    'One or more fixed RegParam values are outside the GLOBAL-RBF input reference range.');

tolC=1e-12*max([1,abs(Ranges.CommonCMin),abs(Ranges.CommonCMax),max(abs(cVals))]);
assert(all(cVals>=Ranges.CommonCMin-tolC & cVals<=Ranges.CommonCMax+tolC), ...
    'One or more fixed C values are outside the common sensitivity range.');

if nargin>=5 && isstruct(Reference)
    if isfield(Reference,'CommonSensitivityRegMin')
        assert(abs(double(Reference.CommonSensitivityRegMin)-Ranges.CommonRegMin)<=tolReg && ...
               abs(double(Reference.CommonSensitivityRegMax)-Ranges.CommonRegMax)<=tolReg, ...
            'Reference/master RegParam provenance is internally inconsistent.');
    end
    if isfield(Reference,'CommonSensitivityCMin')
        assert(abs(double(Reference.CommonSensitivityCMin)-Ranges.CommonCMin)<=tolC && ...
               abs(double(Reference.CommonSensitivityCMax)-Ranges.CommonCMax)<=tolC, ...
            'Reference/master common C range does not match Model-1 settings.');
    end
end
end

function v=extractKernelBoundMR(A,key,kernels,nKernel,name)
if isnumeric(A) && isscalar(A)
    v=double(A);
elseif isstruct(A) && isfield(A,key)
    x=double(A.(key)); assert(isscalar(x),'%s.%s must be scalar.',name,key); v=x;
elseif isnumeric(A)
    A=double(A); j=find(strcmpi(kernels,key),1); assert(~isempty(j),'Kernel %s not found.',key);
    if isvector(A) && numel(A)==nKernel
        x=A(:); v=x(j);
    elseif size(A,2)==nKernel
        x=A(:,j); assert(all(isfinite(x)),'%s nonfinite for %s.',name,key);
        assert(max(x)-min(x)<=1e-12*max(1,max(abs(x))),'%s varies across GGMs for %s.',name,key);
        v=x(1);
    else
        error('%s has unsupported numeric shape.',name);
    end
else
    error('%s has unsupported representation.',name);
end
assert(isscalar(v)&&isfinite(v)&&v>0,'%s for %s must be positive finite.',name,key);
end

function saveAtomicVariablesMR(fileName,V)
folder=fileparts(fileName); if ~isfolder(folder), mkdir(folder); end
tmp=[tempname(folder) '.mat']; cleanup=onCleanup(@() cleanupTempMR(tmp));
names=fieldnames(V);
for k=1:numel(names)
    eval([names{k} ' = V.(names{k});']); %#ok<EVLDIR>
end
try
    save(tmp,names{:},'-v7');
catch
    save(tmp,names{:},'-v7.3');
end
[ok,msg]=movefile(tmp,fileName,'f');
assert(ok,'Could not finalize MAT file %s: %s',fileName,msg);
clear cleanup
end

function cleanupTempMR(f)
if isfile(f), try, delete(f); catch, end, end
end

%% ========================================================================
function value = getSetting(S,names)
for k = 1:numel(names)
    if isfield(S,names{k}) && ~isempty(S.(names{k}))
        value = S.(names{k}); return
    end
end
error('Required setting not found: %s',strjoin(names,', '));
end

function A = expandNumericSetting(value,nM,nK,name)
assert(isnumeric(value) && isreal(value),'%s must be numeric.',name);
if isscalar(value)
    A = repmat(double(value),nM,nK);
else
    A = double(value);
end
assert(isequal(size(A),[nM nK]),'%s must be scalar or nModel x nKernel.',name);
assert(all(isfinite(A(:))),'%s contains nonfinite values.',name);
end

function C = expandTextSetting(value,nM,nK,name)
if ischar(value) || (isstring(value) && isscalar(value))
    C = repmat({char(string(value))},nM,nK);
elseif isstring(value)
    C = cellstr(value);
elseif iscell(value)
    C = cellfun(@(x) char(string(x)),value,'UniformOutput',false);
else
    error('%s must be text or a text matrix.',name);
end
assert(isequal(size(C),[nM nK]),'%s must be scalar or nModel x nKernel.',name);
end

function saveAtomicResult(fileName,varName,value)
parent = fileparts(fileName);
if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
tmp = [tempname(parent) '.mat'];
S = struct(); S.(varName) = value;
save(tmp,'-struct','S','-v7.3');
[ok,msg] = movefile(tmp,fileName,'f');
if ~ok
    if isfile(tmp), delete(tmp); end
    error('Could not publish MAT file %s: %s',fileName,msg);
end
end

function fileName = resolveMainResultsFile(RootDir,caseToken)
baseDir = fullfile(RootDir,'OutPut_RBF_2026_2');
exact = fullfile(baseDir,sprintf('LocalRBF_%s_Full_TuneReg.mat',caseToken));
if isfile(exact), fileName = exact; return; end
F = dir(fullfile(baseDir,sprintf('LocalRBF_%s*TuneReg*.mat',caseToken)));
F = F(~[F.isdir]);
assert(numel(F)==1,['Could not uniquely resolve authoritative main result for %s. ' ...
    'Expected %s, or exactly one LocalRBF_%s*TuneReg*.mat file.'], ...
    caseToken,exact,caseToken);
fileName = fullfile(F(1).folder,F(1).name);
end

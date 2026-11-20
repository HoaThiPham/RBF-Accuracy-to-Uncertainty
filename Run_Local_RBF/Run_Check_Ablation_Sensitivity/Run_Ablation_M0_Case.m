function Results = Run_Ablation_M0_Case(CaseNumber,RootDir,NUM_WORKERS,Reference,Options)
%RUN_ABLATION_M0_CASE
%
% Manuscript configuration: M_0 (both neighbourhood and parameters fixed).
%
% M_0 -- FIXED NEIGHBOURHOOD + FIXED KERNEL PARAMETERS
%
% Scientific definition
% ---------------------
% For one pre-specified random reference realization:
%
%   * ONE fixed radius R0 is used for every training and test location.
%   * ONE fixed RegParam is used per kernel for all GGMs/points/cases.
%   * ONE fixed C is used for Gaussian and Multiquadric for all
%     GGMs/points/cases.
%   * No radius, C, or RegParam is optimized inside M_0.
%   * No pseudo-unseen selection and no test-data selection is performed.
%
% To keep the RBF mathematical formulation aligned with Model 1, shaped
% kernels reconstruct epsilon from the ACTUAL local support:
%
%       epsilon_i = C_fixed,k * dLocal_i(R0)
%
% where dLocal_i is obtained from common.buildEpsilon(Xfit,1).
%
% This means the ablation fixes the selectable quantities (R,C,RegParam)
% without changing Model-1's C -> epsilon parameterization.
%
% minNeighbors = 15 fallback is retained.
%
% Reference must contain:
%   Seed
%   ID
%   FixedRadiusRaw
%   FixedRegParamByKernel = [G MQ L C TPS]
%   FixedCByKernel        = [G MQ]
%
% Full pointwise final predictions/residuals/neighbour counts/dLocal/epsilon
% are saved. Redundant optimization caches do not exist because M_0
% performs no optimization.

if nargin < 1 || isempty(CaseNumber), CaseNumber = 1; end
if nargin < 2 || isempty(RootDir), RootDir = pwd; end
if nargin < 3 || isempty(NUM_WORKERS), NUM_WORKERS = 4; end
if nargin < 5 || isempty(Options), Options = struct(); end
Options = normalizeOptionsM0(Options);

assert(ismember(CaseNumber,[1 2]),'CaseNumber must be 1 or 2.');
assert(isscalar(NUM_WORKERS) && isfinite(NUM_WORKERS) && NUM_WORKERS>=1 && ...
       NUM_WORKERS==round(NUM_WORKERS),'NUM_WORKERS must be a positive integer.');
Reference = validateReferenceShape(Reference);
auditDependenciesM0();

MIN_NEIGHBORS = 15;
KERNEL_ORDER_REG = {'gaussian','multiquadric','linear','cubic','thinplate'};
KERNEL_ORDER_C   = {'gaussian','multiquadric'};

caseToken = sprintf('Case%d',CaseNumber);
MAIN_RESULTS_FILE = resolveMainResultsFileM0(RootDir,caseToken);
INPUT_DATA_FILE = fullfile(RootDir,'OutPutSplitTrainTest', ...
    sprintf('InputData_%s.mat',caseToken));
INPUT_VARIABLE = sprintf('InputData_%s',caseToken);

AB_ROOT = fullfile(RootDir,'OutPut_RBF_2026_2', ...
    'Sensitivity_Ablation_RandomRobustness');
OUTPUT_DIR = fullfile(AB_ROOT,sprintf('Seed_%d',Reference.Seed),'M0');
if ~isfolder(OUTPUT_DIR), mkdir(OUTPUT_DIR); end
OutputFile = fullfile(OUTPUT_DIR, ...
    sprintf('LocalRBF_%s_M0_Seed%d_FINAL.mat',caseToken,Reference.Seed));

if Options.ReuseExistingResults && isfile(OutputFile)
    E = load(OutputFile,'Results');
    if isfield(E,'Results') && isCompatibleExistingM0(E.Results,CaseNumber,Reference)
        Results = E.Results;
        fprintf('\nM_0 %s seed %d: reusing existing result.\n%s\n', ...
            caseToken,Reference.Seed,OutputFile);
        return
    end
end

%% 1. LOAD AUTHORITATIVE MODEL-1 SETTINGS + DATA
A = load(MAIN_RESULTS_FILE,'Results');
assert(isfield(A,'Results') && isstruct(A.Results), ...
    'MAIN_RESULTS_FILE must contain Results.');
MainResults = A.Results;
assert(isfield(MainResults,'Settings') && isstruct(MainResults.Settings), ...
    'Model-1 Results.Settings is missing.');
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

assert(nModel==6 && nKernel==5, ...
    'Expected 6 GGMs x 5 RBF kernels; found %d x %d.',nModel,nKernel);

Xraw = double(InputData.Train.XYZ);
XtestRaw = double(InputData.Test.XYZ);
assert(isreal(Xraw) && ~isempty(Xraw) && all(isfinite(Xraw(:))), ...
    'Training XYZ is invalid.');
assert(isreal(XtestRaw) && ~isempty(XtestRaw) && all(isfinite(XtestRaw(:))), ...
    'Test XYZ is invalid.');
assert(size(Xraw,2)==size(XtestRaw,2),'Train/test coordinate dimension mismatch.');
assert(size(Xraw,1)-1>=MIN_NEIGHBORS, ...
    'Insufficient target-excluded training observations for minNeighbors=%d.',MIN_NEIGHBORS);

%% 2. VALIDATE RANDOM FIXED REFERENCE
% RegParam is validated against the GLOBAL-RBF input range carried by the
% master-generated reference. The Model-1 local RegParam ranges are retained
% only as provenance. C is still validated against the common envelope
% spanning the investigated local Gaussian/MQ C ranges.
RegMin = getSettingM0(S,{'RegMin'});
RegMax = getSettingM0(S,{'RegMax'});
CMin = getSettingM0(S,{'CMin'});
CMax = getSettingM0(S,{'CMax'});

RefReg = double(Reference.FixedRegParamByKernel(:).');
RefC = double(Reference.FixedCByKernel(:).');
assert(numel(RefReg)==5 && numel(RefC)==2,'Invalid fixed-reference parameter lengths.');

Model1RegMinByKernel = nan(1,5);
Model1RegMaxByKernel = nan(1,5);
for k = 1:5
    key = KERNEL_ORDER_REG{k};
    Model1RegMinByKernel(k) = extractKernelBoundM0(RegMin,key,NameFunction_RBF,nKernel,'RegMin');
    Model1RegMaxByKernel(k) = extractKernelBoundM0(RegMax,key,NameFunction_RBF,nKernel,'RegMax');
end
% RegParam random-reference bounds come from the GLOBAL-RBF input range.
if isfield(Reference,'FixedReferenceRegMin')
    CommonRegMin = double(Reference.FixedReferenceRegMin);
    CommonRegMax = double(Reference.FixedReferenceRegMax);
elseif isfield(Reference,'CommonSensitivityRegMin')
    % Backward-compatible alias used by the master output structure.
    CommonRegMin = double(Reference.CommonSensitivityRegMin);
    CommonRegMax = double(Reference.CommonSensitivityRegMax);
else
    error('Reference does not contain fixed-reference RegParam range provenance.');
end
assert(isfinite(CommonRegMin) && isfinite(CommonRegMax) && ...
       CommonRegMin>0 && CommonRegMax>=CommonRegMin, ...
    'Invalid fixed-reference RegParam range.');

Model1CMinByKernel = nan(1,2);
Model1CMaxByKernel = nan(1,2);
for k = 1:2
    key = KERNEL_ORDER_C{k};
    Model1CMinByKernel(k) = extractKernelBoundM0(CMin,key,NameFunction_RBF,nKernel,'CMin');
    Model1CMaxByKernel(k) = extractKernelBoundM0(CMax,key,NameFunction_RBF,nKernel,'CMax');
end
CommonCMin = min(Model1CMinByKernel);
CommonCMax = max(Model1CMaxByKernel);

tolReg = 1e-12*max([1,abs(CommonRegMin),abs(CommonRegMax),max(abs(RefReg))]);
assert(all(RefReg>=CommonRegMin-tolReg & RefReg<=CommonRegMax+tolReg), ...
    'One or more fixed RegParam values are outside the GLOBAL-RBF input reference range.');

tolC = 1e-12*max([1,abs(CommonCMin),abs(CommonCMax),max(abs(RefC))]);
assert(all(RefC>=CommonCMin-tolC & RefC<=CommonCMax+tolC), ...
    'One or more fixed C values are outside the common sensitivity range.');

% Cross-check the master-generated provenance when present.
if isfield(Reference,'CommonSensitivityRegMin')
    assert(abs(double(Reference.CommonSensitivityRegMin)-CommonRegMin)<=tolReg && ...
           abs(double(Reference.CommonSensitivityRegMax)-CommonRegMax)<=tolReg, ...
        'Reference/master RegParam provenance is internally inconsistent.');
end
if isfield(Reference,'CommonSensitivityCMin')
    assert(abs(double(Reference.CommonSensitivityCMin)-CommonCMin)<=tolC && ...
           abs(double(Reference.CommonSensitivityCMax)-CommonCMax)<=tolC, ...
        'Reference/master common C range does not match Model-1 settings.');
end

validateRadiusGridM0(getSettingM0(S,{'Radii'}), ...
    NameFunction_RBF,double(Reference.FixedRadiusRaw));

%% 3. COORDINATE SCALING + SHARED GEOMETRY
Sglobal = common.MeanDistant(Xraw);
assert(isscalar(Sglobal) && isfinite(Sglobal) && Sglobal>0, ...
    'common.MeanDistant(Xraw) returned an invalid scale.');
X = Xraw./Sglobal;
Xtest = XtestRaw./Sglobal;
fixedRadiusScaled = double(Reference.FixedRadiusRaw)/Sglobal;

Dtrain = pdist2(X,X);
Ptrain = [ones(size(X,1),1),X];
Dtest = pdist2(X,Xtest);     % nTrain x nTest
Ptest = [ones(size(Xtest,1),1),Xtest];

numericalOptions = struct();
if isfield(S,'options') && isstruct(S.options) && ...
   isfield(S.options,'solver') && isstruct(S.options.solver)
    numericalOptions.solver = S.options.solver;
else
    numericalOptions.solver = struct('cond_lsq',1e12);
end
numericalOptions.solver.compute_rel_residual = false;

%% 4. BUILD MODEL x KERNEL JOBS
Jobs = repmat(struct(),nRuns,1);
q = 0;
for i = 1:nModel
    model = ListOfModel{i};
    assert(isfield(InputData.Train.DeltaZeta,model) && ...
           isfield(InputData.Test.DeltaZeta,model), ...
        'InputData is missing model %s.',model);
    y0 = InputData.Train.DeltaZeta.(model);
    ytest0 = InputData.Test.DeltaZeta.(model);
    y = double(y0(:));
    ytest = double(ytest0(:));
    assert(numel(y)==size(X,1) && all(isfinite(y)), ...
        'Invalid training response for %s.',model);
    assert(numel(ytest)==size(Xtest,1) && all(isfinite(ytest)), ...
        'Invalid test response for %s.',model);

    for j = 1:nKernel
        q = q+1;
        kernel = NameFunction_RBF{j};
        key = lower(strtrim(char(string(kernel))));
        regIdx = find(strcmp(KERNEL_ORDER_REG,key),1);
        assert(~isempty(regIdx),'Unsupported kernel %s.',kernel);

        Jobs(q).i = i;
        Jobs(q).j = j;
        Jobs(q).model = model;
        Jobs(q).kernel = kernel;
        Jobs(q).key = key;
        Jobs(q).useShape = common.kernelHasShapeParameter(kernel);
        Jobs(q).y = y;
        Jobs(q).ytest = ytest;
        Jobs(q).radiusRaw = double(Reference.FixedRadiusRaw);
        Jobs(q).radiusScaled = fixedRadiusScaled;
        Jobs(q).regParam = RefReg(regIdx);
        if strcmp(key,'gaussian')
            Jobs(q).C = RefC(1);
        elseif strcmp(key,'multiquadric')
            Jobs(q).C = RefC(2);
        elseif Jobs(q).useShape
            error('No fixed C reference is defined for shaped kernel %s.',kernel);
        else
            Jobs(q).C = NaN;
        end
    end
end

%% 5. PARALLEL SETUP
parallelUsed = setupProcessPoolM0(Options.UseParallel, ...
    Options.AllowSerialFallback,NUM_WORKERS);

%% 6. RUN 30 GGM x KERNEL JOBS
RunCell = cell(nRuns,1);
SummaryCell = cell(nRuns,1);
tBatch = tic;
if parallelUsed
    parfor q = 1:nRuns
        [RunCell{q},SummaryCell{q}] = runOneM0( ...
            Jobs(q),X,Xtest,Dtrain,Ptrain,Dtest,Ptest, ...
            numericalOptions,MIN_NEIGHBORS);
    end
else
    for q = 1:nRuns
        [RunCell{q},SummaryCell{q}] = runOneM0( ...
            Jobs(q),X,Xtest,Dtrain,Ptrain,Dtest,Ptest, ...
            numericalOptions,MIN_NEIGHBORS);
    end
end
batchWallTime = toc(tBatch);

%% 7. PACKAGE FULL RESULTS
Results = struct();
Results.Schema = 'LocalRBF.Ablation.M0.v3.GlobalRegRange.LocalEpsilon';
Results.Mode = 'M0';
Results.LegacySchema = 'LocalRBF.Ablation.Mode11.v3.GlobalRegRange.LocalEpsilon';
Results.LegacyMode = 'mode11';
Results.LegacyLabel = 'Model 11';
Results.CaseName = CaseName;
Results.ListOfModel = ListOfModel;
Results.NameFunction_RBF = NameFunction_RBF;
Results.Reference = Reference;
Results.Run = struct();

for q = 1:nRuns
    f = matlab.lang.makeValidName(sprintf( ...
        'M%03d_K%03d_%s_%s_M0', ...
        Jobs(q).i,Jobs(q).j,Jobs(q).model,Jobs(q).kernel));
    Results.Run.(f) = RunCell{q};
end
Results.Summary = vertcat(SummaryCell{:});

Results.Settings.FixedRadiusRaw = double(Reference.FixedRadiusRaw);
Results.Settings.FixedRadiusKm = double(Reference.FixedRadiusRaw)/1000;
Results.Settings.FixedRadiusScaled = fixedRadiusScaled;
Results.Settings.FixedRegParamByKernel = RefReg;
Results.Settings.FixedRegParamKernelOrder = KERNEL_ORDER_REG;
Results.Settings.FixedCByKernel = RefC;
Results.Settings.FixedCKernelOrder = KERNEL_ORDER_C;
Results.Settings.CommonSensitivityRegMin = CommonRegMin;
Results.Settings.CommonSensitivityRegMax = CommonRegMax;
Results.Settings.CommonSensitivityCMin = CommonCMin;
Results.Settings.CommonSensitivityCMax = CommonCMax;
Results.Settings.Model1InvestigatedRegMinByKernel = Model1RegMinByKernel;
Results.Settings.Model1InvestigatedRegMaxByKernel = Model1RegMaxByKernel;
Results.Settings.Model1InvestigatedCMinByKernel = Model1CMinByKernel;
Results.Settings.Model1InvestigatedCMaxByKernel = Model1CMaxByKernel;
Results.Settings.ReferenceSamplingRangeRule = ...
    'RegParam from GLOBAL-RBF input range; C from common local Gaussian/MQ envelope';
Results.Settings.FixedReferenceRegMin = CommonRegMin;
Results.Settings.FixedReferenceRegMax = CommonRegMax;
Results.Settings.RegRangeSource = 'GLOBAL-RBF input range';
Results.Settings.RadiusTuned = false;
Results.Settings.ParameterTuned = false;
Results.Settings.UsesPseudoUnseen = false;
Results.Settings.UsesTestSelection = false;
Results.Settings.MinNeighbors = MIN_NEIGHBORS;
Results.Settings.UsesMinNeighborsFallback = true;
Results.Settings.EpsilonDefinition = ...
    'epsilon = C_fixed,k * dLocal(actual fixed-radius support), using common.buildEpsilon(Xfit,1)';
Results.Settings.S_global = Sglobal;

Results.Runtime.BatchWallTime = batchWallTime;
Results.Runtime.ParallelUsed = parallelUsed;
Results.Runtime.NumWorkers = double(parallelUsed)*NUM_WORKERS;
Results.Provenance.MainResultsFile = MAIN_RESULTS_FILE;
Results.Provenance.InputDataFile = INPUT_DATA_FILE;
Results.Provenance.ReferenceID = char(string(Reference.ID));
Results.Provenance.ReferenceSeed = double(Reference.Seed);
Results.created = datetime('now');

saveAtomicVariablesM0(OutputFile,struct('Results',Results));
fprintf('\nSaved M_0 %s seed %d:\n%s\n', ...
    caseToken,Reference.Seed,OutputFile);
disp(Results.Summary);

end

%% ========================================================================
function [R,Srow] = runOneM0(Job,X,Xtest,Dtrain,Ptrain,Dtest,Ptest, ...
    numericalOptions,minNeighbors)

t0 = tic;
Train = evaluateFixedSupportM0(X,[],Job.y,Job.y,Dtrain,Dtrain,Ptrain,Ptrain, ...
    Job.radiusScaled,Job.kernel,Job.C,Job.regParam,numericalOptions, ...
    minNeighbors,true);
Test = evaluateFixedSupportM0(X,Xtest,Job.y,Job.ytest,Dtest,Dtrain,Ptrain,Ptest, ...
    Job.radiusScaled,Job.kernel,Job.C,Job.regParam,numericalOptions, ...
    minNeighbors,false);

R = struct();
R.success = Train.success && Test.success;
R.model = Job.model;
R.kernel = Job.kernel;
R.mode = 'M0';
R.message = '';
R.FixedRadiusRaw = Job.radiusRaw;
R.FixedRadiusScaled = Job.radiusScaled;
R.FixedRegParam = Job.regParam;
R.FixedC = Job.C;
R.ShapeRule = 'epsilon=C_fixed*dLocal(actual support)';
R.Training = Train;
R.Test = Test;
R.Runtime.TrainingRBFSolves = Train.nRBFSolves;
R.Runtime.TestRBFSolves = Test.nRBFSolves;
R.Runtime.TotalRBFSolves = Train.nRBFSolves + Test.nRBFSolves;
R.Runtime.WallTime = toc(t0);

Srow = table(string(Job.model),string(Job.kernel),"M0", ...
    Job.radiusRaw,Job.regParam,Job.C, ...
    Train.RMSE,Test.RMSE,Train.MAE,Test.MAE,Train.MaxAE,Test.MaxAE, ...
    meanFiniteM0(Train.NeighborCount),meanFiniteM0(Test.NeighborCount), ...
    R.Runtime.TotalRBFSolves,R.Runtime.WallTime,logical(R.success), ...
    'VariableNames',{'Model','Kernel','Mode','FixedRadiusRaw','FixedRegParam', ...
    'FixedC','TrainRMSE','TestRMSE','TrainMAE','TestMAE','TrainMaxAE','TestMaxAE', ...
    'MeanTrainNeighbors','MeanTestNeighbors','TotalRBFSolves','WallTime','Success'});
end

%% ========================================================================
function E = evaluateFixedSupportM0(Xtrain,Xquery,ytrain,ytrue,Dquery,DtrainGeom,Ptrain,Pquery, ...
    radius,kernel,C,regParam,numericalOptions,minNeighbors,isTraining)

if isTraining
    nQuery = size(Xtrain,1);
else
    nQuery = size(Xquery,1);
end
useShape = common.kernelHasShapeParameter(kernel);

prediction = nan(nQuery,1);
residual = nan(nQuery,1);
neighborCount = nan(nQuery,1);
dLocalUsed = nan(nQuery,1);
epsilonUsed = nan(nQuery,1);
nSolves = 0;

for iq = 1:nQuery
    if isTraining
        dvec = Dquery(:,iq);
        excludeIndex = iq;
        Xq = Xtrain(iq,:);
        Pq = Pquery(iq,:);
    else
        dvec = Dquery(:,iq);
        excludeIndex = [];
        Xq = Xquery(iq,:);
        Pq = Pquery(iq,:);
    end

    idx = radiusWithMinNeighborsIndicesM0(dvec,radius,excludeIndex,minNeighbors);
    if isempty(idx), continue, end
    Xfit = Xtrain(idx,:);
    yfit = ytrain(idx);

    try
        [isOK,~] = common.checkPolynomialAdmissibility(Xfit);
    catch
        isOK = false;
    end
    if ~isOK, continue, end

    if useShape
        try
            [~,dLocal] = common.buildEpsilon(Xfit,1);
        catch
            dLocal = NaN;
        end
        if ~isscalar(dLocal) || ~isfinite(dLocal) || dLocal<=0 || ...
           ~isscalar(C) || ~isfinite(C) || C<=0
            continue
        end
        epsilon = max(C*dLocal,1e-12);
    else
        dLocal = NaN;
        epsilon = 1;
    end

    geom = struct();
    if isTraining
        geom.R = DtrainGeom(idx,idx);
        geom.R_eval = reshape(Dquery(iq,idx),1,[]);
    else
        geom.R = DtrainGeom(idx,idx);
        geom.R_eval = reshape(Dquery(idx,iq),1,[]);
    end
    geom.P = Ptrain(idx,:);
    geom.P_eval = Pq;

    try
        Sys = common.buildRBFSystem(Xfit,yfit,kernel,epsilon,regParam,geom);
        nSolves = nSolves+1;
        [coeffs,solveInfo] = common.solveLinearSystem(Sys,numericalOptions);
        if ~solveInfo.success || isempty(coeffs) || any(~isfinite(coeffs(:)))
            continue
        end
        yh = common.predictRBF(Xq,Xfit,coeffs,kernel,epsilon,geom);
        if ~isscalar(yh) || ~isfinite(yh), continue, end
    catch
        continue
    end

    prediction(iq) = yh;
    residual(iq) = ytrue(iq)-yh;
    neighborCount(iq) = numel(idx);
    dLocalUsed(iq) = dLocal;
    epsilonUsed(iq) = epsilon;
end

E = struct();
E.success = all(isfinite(residual));
E.Prediction = prediction;
E.Residual = residual;
E.NeighborCount = neighborCount;
E.dLocal = dLocalUsed;
E.Epsilon = epsilonUsed;
E.RequestedRadiusScaled = repmat(radius,nQuery,1);
E.nRBFSolves = nSolves;

if E.success
    E.RMSE = sqrt(mean(residual.^2));
    E.MAE = mean(abs(residual));
    E.MaxAE = max(abs(residual));
else
    E.RMSE = NaN;
    E.MAE = NaN;
    E.MaxAE = NaN;
end
end

%% ========================================================================
function idx = radiusWithMinNeighborsIndicesM0(distanceVector,radius,excludeIndex,minNeighbors)
d = double(distanceVector(:));
maskFinite = isfinite(d);
maskRadius = maskFinite & d<=radius;
if nargin>=3 && ~isempty(excludeIndex)
    maskRadius(excludeIndex)=false;
    maskFinite(excludeIndex)=false;
end
idx = find(maskRadius);
if numel(idx)>=minNeighbors, return, end
allIdx = find(maskFinite);
if isempty(allIdx), idx=[]; return, end
[~,ord] = sort(d(allIdx),'ascend');
allIdx = allIdx(ord);
keepN = min(minNeighbors,numel(allIdx));
idx = unique([idx;allIdx(1:keepN)],'stable');
end

%% ========================================================================
function auditDependenciesM0()
required={'common.MeanDistant','common.buildEpsilon','common.checkPolynomialAdmissibility', ...
    'common.buildRBFSystem','common.solveLinearSystem','common.predictRBF'};
for k=1:numel(required)
    assert(~isempty(which(required{k})),'Required function not on MATLAB path: %s',required{k});
end
end

%% ========================================================================
function Reference = validateReferenceShape(Reference)
assert(isstruct(Reference) && isscalar(Reference),'Reference must be one struct.');
required = {'Seed','ID','FixedRadiusRaw','FixedRegParamByKernel','FixedCByKernel'};
assert(all(isfield(Reference,required)), ...
    'Reference is missing one or more required fields.');
assert(isnumeric(Reference.Seed) && isscalar(Reference.Seed) && ...
    isfinite(Reference.Seed) && Reference.Seed>=0 && Reference.Seed==round(Reference.Seed), ...
    'Reference.Seed must be a nonnegative integer.');
assert(isnumeric(Reference.FixedRadiusRaw) && isscalar(Reference.FixedRadiusRaw) && ...
    isfinite(Reference.FixedRadiusRaw) && Reference.FixedRadiusRaw>0, ...
    'Reference.FixedRadiusRaw is invalid.');
rr = double(Reference.FixedRegParamByKernel(:));
cc = double(Reference.FixedCByKernel(:));
assert(numel(rr)==5 && all(isfinite(rr)) && all(rr>0), ...
    'Reference.FixedRegParamByKernel must contain 5 positive finite values.');
assert(numel(cc)==2 && all(isfinite(cc)) && all(cc>0), ...
    'Reference.FixedCByKernel must contain 2 positive finite values.');
end

function Options = normalizeOptionsM0(Options)
if ~isfield(Options,'ReuseExistingResults'), Options.ReuseExistingResults=true; end
if ~isfield(Options,'UseParallel'), Options.UseParallel=true; end
if ~isfield(Options,'AllowSerialFallback'), Options.AllowSerialFallback=false; end
Options.ReuseExistingResults = logical(Options.ReuseExistingResults);
Options.UseParallel = logical(Options.UseParallel);
Options.AllowSerialFallback = logical(Options.AllowSerialFallback);
end

function tf = setupProcessPoolM0(useParallel,allowSerial,numWorkers)
tf = false;
if ~useParallel, return, end
try
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers~=numWorkers || ~isa(pool,'parallel.ProcessPool')
        if ~isempty(pool), delete(pool); end
        pool = parpool('Processes',numWorkers); %#ok<NASGU>
    end
    tf = true;
catch ME
    if ~allowSerial, rethrow(ME); end
    warning('M_0 falling back to serial: %s',ME.message);
end
end

function tf = isCompatibleExistingM0(R,CaseNumber,Reference)
tf = false;
expectedSchema = 'LocalRBF.Ablation.M0.v3.GlobalRegRange.LocalEpsilon';
if ~isstruct(R) || ~isfield(R,'Schema') || ...
   ~strcmp(char(string(R.Schema)),expectedSchema) || ...
   ~isfield(R,'Mode') || ~strcmpi(char(string(R.Mode)),'M0') || ...
   ~isfield(R,'Reference') || ~isstruct(R.Reference) || ...
   ~isfield(R,'Summary') || ~istable(R.Summary) || height(R.Summary)~=30
    return
end
if isfield(R,'CaseName') && ~strcmpi(char(string(R.CaseName)),sprintf('Case%d',CaseNumber))
    return
end
tf = referencesEqualM0(R.Reference,Reference);
end

function tf = referencesEqualM0(A,B)
tf = false;
try
    if double(A.Seed)~=double(B.Seed), return, end
    if ~strcmp(char(string(A.ID)),char(string(B.ID))), return, end
    x = [double(A.FixedRadiusRaw);double(A.FixedRegParamByKernel(:));double(A.FixedCByKernel(:))];
    y = [double(B.FixedRadiusRaw);double(B.FixedRegParamByKernel(:));double(B.FixedCByKernel(:))];
    tol = 1e-12*max(1,max(abs([x;y])));
    tf = numel(x)==numel(y) && max(abs(x-y))<=tol;
catch
    tf = false;
end
end

function validateRadiusGridM0(Radii,kernels,fixedRadiusRaw)
required = (50000:10000:200000).';
for j=1:numel(kernels)
    key = lower(char(string(kernels{j})));
    if isstruct(Radii)
        assert(isfield(Radii,key),'Model-1 Radii.%s is missing.',key);
        r = double(Radii.(key)(:));
    else
        r = double(Radii(:));
    end
    assert(numel(r)==numel(required) && ...
        max(abs(r-required))<=1e-10*max(1,max(abs(required))), ...
        'Model-1 radius grid for %s is not exactly 50:10:200 km.',key);
    assert(any(abs(r-fixedRadiusRaw)<=1e-10*max(1,abs(fixedRadiusRaw))), ...
        'Fixed radius is not present for %s.',key);
end
end

function v = extractKernelBoundM0(A,key,kernels,nKernel,name)
if isnumeric(A) && isscalar(A)
    v = double(A);
elseif isstruct(A) && isfield(A,key)
    x = double(A.(key));
    assert(isscalar(x),'%s.%s must be scalar.',name,key);
    v = x;
elseif isnumeric(A)
    A = double(A);
    j = find(strcmpi(kernels,key),1);
    assert(~isempty(j),'Kernel %s not found.',key);
    if isvector(A) && numel(A)==nKernel
        x=A(:); v=x(j);
    elseif size(A,2)==nKernel
        x=A(:,j);
        assert(all(isfinite(x)),'%s has nonfinite values for %s.',name,key);
        assert(max(x)-min(x)<=1e-12*max(1,max(abs(x))), ...
            '%s varies across GGMs for %s.',name,key);
        v=x(1);
    else
        error('%s has unsupported numeric shape.',name);
    end
else
    error('%s has unsupported representation.',name);
end
assert(isscalar(v) && isfinite(v) && v>0,'%s for %s must be positive finite.',name,key);
end

function value = getSettingM0(S,names)
for k=1:numel(names)
    if isfield(S,names{k}) && ~isempty(S.(names{k}))
        value=S.(names{k}); return
    end
end
error('Required setting not found: %s',strjoin(names,', '));
end

function fileName = resolveMainResultsFileM0(RootDir,caseToken)
baseDirs = {fullfile(RootDir,'OutPut_RBF_2026_2'),fullfile(RootDir,'OutPut_RBF_2026_1')};
hits={};
for b=1:numel(baseDirs)
    exact=fullfile(baseDirs{b},sprintf('LocalRBF_%s_Full_TuneReg.mat',caseToken));
    if isfile(exact), hits{end+1,1}=exact; end %#ok<AGROW>
end
if numel(hits)==1, fileName=hits{1}; return, end
assert(isempty(hits),'More than one exact Model-1 result found for %s.',caseToken);
for b=1:numel(baseDirs)
    if ~isfolder(baseDirs{b}), continue, end
    F=dir(fullfile(baseDirs{b},sprintf('LocalRBF_%s*TuneReg*.mat',caseToken)));
    for k=1:numel(F)
        if F(k).isdir, continue, end
        nm=lower(F(k).name);
        if contains(nm,'stage1') || contains(nm,'sensitivity'), continue, end
        hits{end+1,1}=fullfile(F(k).folder,F(k).name); %#ok<AGROW>
    end
end
assert(numel(hits)==1,'Could not uniquely resolve Model-1 result for %s.',caseToken);
fileName=hits{1};
end

function saveAtomicVariablesM0(fileName,V)
folder=fileparts(fileName); if ~isfolder(folder), mkdir(folder); end
tmp=[tempname(folder) '.mat'];
cleanup=onCleanup(@() cleanupTempM0(tmp));
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

function cleanupTempM0(f)
if isfile(f), try, delete(f); catch, end, end
end

function m = meanFiniteM0(x)
x=double(x(:)); x=x(isfinite(x));
if isempty(x), m=NaN; else, m=mean(x); end
end

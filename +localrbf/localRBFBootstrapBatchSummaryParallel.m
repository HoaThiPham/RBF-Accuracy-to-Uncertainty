function Results = localRBFBootstrapBatchSummaryParallel( ...
    InputData, FitResults, nBootstrap, verbose, RNG_SEED, bootstrapOptions, OutputFile)
%LOCALRBFBOOTSTRAPBATCHSUMMARYPARALLEL Parallel bootstrap for final LOCAL RBF runs.
%
% INPUT: InputData uses the same Train/Test XYZ + DeltaZeta contract as the
% global bootstrap batch. FitResults is the local-RBF fit result.
%
% Compatible with the split Stage-2 storage produced by
% localrbf.localRBFBatchSummaryParallel_TuneReg. Bootstrap consumes only
% the final fitted model; it never re-optimizes radii/C/Reg, never repeats
% test-parameter transfer, and does not require Stage-2 DETAIL.
%
% bootstrapOptions.TrainingMode = 'original' (default) or 'pseudo-unseen'.
% Both modes consume already-fixed Radius/RegParam/C from the main output;
% bootstrap itself never performs transfer. Pseudo-unseen mode therefore
% requires pointwise pseudo settings/residuals to be present upstream.
%
% IMPORTANT RADIUS STORAGE CONTRACT FOR SPLIT STAGE-2 CORE:
%   Core.Parameters.Train.Radius  : RAW coordinate units
%   Core.Parameters.Test.Radius   : RAW coordinate units
%   Core.Parameters.Pseudo.Radius : already SCALED by S_global
% The pseudo-unseen path must therefore NOT divide Pseudo.Radius by
% S_global a second time.  The raw pseudo radius is reconstructed only for
% metadata as radius_raw = radius * S_global.
%
% NumSpatialSegments / MinPointsPerAnnulus follow the GLOBAL bootstrap
% convention. A rare sequential variance-dominance safeguard is applied to
% final bootstrap statistics: the current largest variance contributor is
% treated as dominant only while its contribution exceeds 80%, with the
% dominant fraction capped at 3% of the originally generated B predictions.
% SaveAll=false (default)
% keeps compact statistics. StoreIndices=false (default) avoids
% bootstrap-index storage.

if nargin < 3 || isempty(nBootstrap), nBootstrap = 200; end
if nargin < 4 || isempty(verbose), verbose = true; end
if nargin < 5 || isempty(RNG_SEED), RNG_SEED = 1; end
if nargin < 6 || isempty(bootstrapOptions), bootstrapOptions = struct(); end
if nargin < 7, OutputFile = ''; end

% Input data contract is intentionally the same as the global bootstrap
% batch. Coordinates are shared once across all Model x Kernel runs instead
% of being duplicated inside split Stage-2 files.
if ~isstruct(InputData) || ~isscalar(InputData) || ...
        ~isfield(InputData,'Train') || ~isfield(InputData,'Test') || ...
        ~isstruct(InputData.Train) || ~isstruct(InputData.Test) || ...
        ~isfield(InputData.Train,'XYZ') || ~isfield(InputData.Test,'XYZ') || ...
        ~isfield(InputData.Train,'DeltaZeta') || ~isfield(InputData.Test,'DeltaZeta') || ...
        ~isstruct(InputData.Train.DeltaZeta) || ~isstruct(InputData.Test.DeltaZeta)
    error('InputData does not contain the required Train/Test fields.');
end
XtrainRaw=InputData.Train.XYZ; XtestRaw=InputData.Test.XYZ;
if ~isnumeric(XtrainRaw)||isempty(XtrainRaw)||~ismatrix(XtrainRaw)||any(~isfinite(XtrainRaw(:))) || ...
        ~isnumeric(XtestRaw)||isempty(XtestRaw)||~ismatrix(XtestRaw)||any(~isfinite(XtestRaw(:)))
    error('InputData Train/Test XYZ coordinates must be finite numeric matrices.');
end
if size(XtrainRaw,2)~=size(XtestRaw,2), error('Training/test coordinate dimensions differ.'); end

if isstruct(FitResults) && isscalar(FitResults) && isfield(FitResults,'Results') && ...
        isstruct(FitResults.Results) && isscalar(FitResults.Results)
    FitResults = FitResults.Results;
end
if ~isstruct(FitResults) || ~isscalar(FitResults) || ...
        ~isfield(FitResults,'Run') || ~isstruct(FitResults.Run)
    error('FitResults.Run is required.');
end
if ~isnumeric(nBootstrap) || ~isscalar(nBootstrap) || ~isfinite(nBootstrap) || ...
        nBootstrap<2 || nBootstrap~=round(nBootstrap)
    error('nBootstrap must be an integer >= 2.');
end
verbose = validateLogicalScalar(verbose,'verbose');
if ~isnumeric(RNG_SEED) || ~isscalar(RNG_SEED) || ~isfinite(RNG_SEED) || ...
        RNG_SEED<0 || RNG_SEED>(2^32-1) || RNG_SEED~=round(RNG_SEED)
    error('RNG_SEED must be a finite nonnegative integer not exceeding 2^32-1.');
end
if ~isstruct(bootstrapOptions) || ~isscalar(bootstrapOptions)
    error('bootstrapOptions must be a scalar structure.');
end
if ~isfield(bootstrapOptions,'Verbose') || isempty(bootstrapOptions.Verbose)
    bootstrapOptions.Verbose = false;
end
if ~isfield(bootstrapOptions,'MaxAttemptsFactor') || isempty(bootstrapOptions.MaxAttemptsFactor)
    bootstrapOptions.MaxAttemptsFactor = 10;
end
if ~isfield(bootstrapOptions,'SaveAll') || isempty(bootstrapOptions.SaveAll)
    bootstrapOptions.SaveAll = false;
end
if ~isfield(bootstrapOptions,'StoreIndices') || isempty(bootstrapOptions.StoreIndices)
    bootstrapOptions.StoreIndices = false;
end
if ~isfield(bootstrapOptions,'NumSpatialSegments') || isempty(bootstrapOptions.NumSpatialSegments)
    bootstrapOptions.NumSpatialSegments = 0;
end
if ~isfield(bootstrapOptions,'MinPointsPerAnnulus') || isempty(bootstrapOptions.MinPointsPerAnnulus)
    bootstrapOptions.MinPointsPerAnnulus = 5;
end
if ~isfield(bootstrapOptions,'TrainingMode') || isempty(bootstrapOptions.TrainingMode)
    bootstrapOptions.TrainingMode = 'original';
end
if ~isfield(bootstrapOptions,'NumWorkers') || isempty(bootstrapOptions.NumWorkers)
    bootstrapOptions.NumWorkers = [];
end
if ~isfield(bootstrapOptions,'DominanceLimit') || isempty(bootstrapOptions.DominanceLimit)
    bootstrapOptions.DominanceLimit = 0.80;
end
if ~isfield(bootstrapOptions,'MaxDominantFraction') || isempty(bootstrapOptions.MaxDominantFraction)
    bootstrapOptions.MaxDominantFraction = 0.03;
end
bootstrapOptions.Verbose = validateLogicalScalar(bootstrapOptions.Verbose,'bootstrapOptions.Verbose');
bootstrapOptions.SaveAll = validateLogicalScalar(bootstrapOptions.SaveAll,'bootstrapOptions.SaveAll');
bootstrapOptions.StoreIndices = validateLogicalScalar(bootstrapOptions.StoreIndices,'bootstrapOptions.StoreIndices');
if ~isnumeric(bootstrapOptions.MaxAttemptsFactor) || ~isscalar(bootstrapOptions.MaxAttemptsFactor) || ...
        ~isfinite(bootstrapOptions.MaxAttemptsFactor) || bootstrapOptions.MaxAttemptsFactor<1
    error('bootstrapOptions.MaxAttemptsFactor must be a finite scalar >= 1.');
end
if ~isnumeric(bootstrapOptions.NumSpatialSegments) || ~isscalar(bootstrapOptions.NumSpatialSegments) || ...
        ~isfinite(bootstrapOptions.NumSpatialSegments) || bootstrapOptions.NumSpatialSegments<0 || ...
        bootstrapOptions.NumSpatialSegments~=round(bootstrapOptions.NumSpatialSegments)
    error('bootstrapOptions.NumSpatialSegments must be a nonnegative integer.');
end
if ~isnumeric(bootstrapOptions.MinPointsPerAnnulus) || ~isscalar(bootstrapOptions.MinPointsPerAnnulus) || ...
        ~isfinite(bootstrapOptions.MinPointsPerAnnulus) || bootstrapOptions.MinPointsPerAnnulus<1 || ...
        bootstrapOptions.MinPointsPerAnnulus~=round(bootstrapOptions.MinPointsPerAnnulus)
    error('bootstrapOptions.MinPointsPerAnnulus must be a positive integer.');
end
if ~isempty(bootstrapOptions.NumWorkers) && ...
        (~isnumeric(bootstrapOptions.NumWorkers) || ~isscalar(bootstrapOptions.NumWorkers) || ...
         ~isfinite(bootstrapOptions.NumWorkers) || bootstrapOptions.NumWorkers<1 || ...
         bootstrapOptions.NumWorkers~=round(bootstrapOptions.NumWorkers))
    error('bootstrapOptions.NumWorkers must be empty or a positive integer.');
end
if ~isnumeric(bootstrapOptions.DominanceLimit) || ~isscalar(bootstrapOptions.DominanceLimit) || ...
        ~isfinite(bootstrapOptions.DominanceLimit) || bootstrapOptions.DominanceLimit<=0 || bootstrapOptions.DominanceLimit>=1
    error('bootstrapOptions.DominanceLimit must be a finite scalar strictly between 0 and 1.');
end
if ~isnumeric(bootstrapOptions.MaxDominantFraction) || ~isscalar(bootstrapOptions.MaxDominantFraction) || ...
        ~isfinite(bootstrapOptions.MaxDominantFraction) || bootstrapOptions.MaxDominantFraction<0 || ...
        bootstrapOptions.MaxDominantFraction>=1
    error('bootstrapOptions.MaxDominantFraction must be a finite scalar in [0,1).');
end
trainingMode = lower(strtrim(char(string(bootstrapOptions.TrainingMode))));
if ~any(strcmp(trainingMode,{'original','pseudo-unseen'}))
    error('bootstrapOptions.TrainingMode must be ''original'' or ''pseudo-unseen''.');
end
bootstrapOptions.TrainingMode = trainingMode;
saveAll = bootstrapOptions.SaveAll;
TestRNG_SEED = mod(double(RNG_SEED)+1000003,2^32);

if isfield(FitResults,'CaseName') && ~isempty(FitResults.CaseName)
    CaseName = char(string(FitResults.CaseName));
else
    CaseName = 'LocalRBF';
end
sourceRunNames = fieldnames(FitResults.Run);
if isempty(sourceRunNames), error('FitResults.Run is empty.'); end

% -------------------------------------------------------------------------
% Resolve every source run and its Stage-2 CORE on the client. No DETAIL is
% loaded for normal bootstrap operation.
% -------------------------------------------------------------------------
RunRecords = repmat(struct('runField','','run',struct(),'core',struct(), ...
    'model','','kernel','','legacy',false),numel(sourceRunNames),1);
modelTmp = strings(0,1); kernelTmp = strings(0,1);
for k=1:numel(sourceRunNames)
    runField = sourceRunNames{k};
    Z = FitResults.Run.(runField);
    if ~isstruct(Z), continue; end
    [Core,isLegacy] = readStage2CoreOrLegacy(Z,runField);
    [modelName,kernel] = resolveRunIdentity(Z,Core);
    if isempty(modelName) || isempty(kernel), continue; end
    RunRecords(k).runField = runField;
    RunRecords(k).run = Z;
    RunRecords(k).core = Core;
    RunRecords(k).model = modelName;
    RunRecords(k).kernel = kernel;
    RunRecords(k).legacy = isLegacy;
    modelTmp(end+1,1)=string(modelName); %#ok<AGROW>
    kernelTmp(end+1,1)=string(kernel); %#ok<AGROW>
end

if isfield(FitResults,'ListOfModel') && ~isempty(FitResults.ListOfModel)
    ListOfModel = cellstr(string(FitResults.ListOfModel(:)));
else
    ListOfModel = cellstr(unique(modelTmp,'stable'));
end
if isfield(FitResults,'NameFunction_RBF') && ~isempty(FitResults.NameFunction_RBF)
    NameFunction_RBF = cellstr(string(FitResults.NameFunction_RBF(:)));
else
    NameFunction_RBF = cellstr(unique(kernelTmp,'stable'));
end
nModel=numel(ListOfModel); nKernel=numel(NameFunction_RBF);
if nModel<1 || nKernel<1, error('No valid model/kernel identities were found.'); end
for j=1:nKernel, common.kernelHasShapeParameter(NameFunction_RBF{j}); end

% Extract only the response vectors actually required by workers. The full
% InputData structure is never broadcast through PARFOR.
YTrainModel=cell(nModel,1); YTestModel=cell(nModel,1);
for iModel=1:nModel
    modelName=ListOfModel{iModel};
    if ~isfield(InputData.Train.DeltaZeta,modelName) || ~isfield(InputData.Test.DeltaZeta,modelName)
        error('InputData is missing response vectors for model %s.',modelName);
    end
    yt=InputData.Train.DeltaZeta.(modelName); yq=InputData.Test.DeltaZeta.(modelName);
    yt=yt(:); yq=yq(:);
    if numel(yt)~=size(XtrainRaw,1) || numel(yq)~=size(XtestRaw,1) || ...
            any(~isfinite(yt)) || any(~isfinite(yq))
        error('InputData response sizes/values are invalid for model %s.',modelName);
    end
    YTrainModel{iModel}=yt; YTestModel{iModel}=yq;
end

RunFieldMatrix=cell(nModel,nKernel);
SourceRunCell=cell(nModel,nKernel);
CoreCell=cell(nModel,nKernel);
LegacyCell=false(nModel,nKernel);
OriginalTrainOK=false(nModel,nKernel);
OriginalTestOK=false(nModel,nKernel);
for k=1:numel(RunRecords)
    if isempty(RunRecords(k).runField), continue; end
    i=find(strcmpi(ListOfModel,RunRecords(k).model),1);
    j=find(strcmpi(NameFunction_RBF,RunRecords(k).kernel),1);
    if isempty(i)||isempty(j), error('Stored run "%s" cannot be mapped.',RunRecords(k).runField); end
    if ~isempty(RunFieldMatrix{i,j})
        error('More than one stored run exists for model "%s" and kernel "%s".', ...
            RunRecords(k).model,RunRecords(k).kernel);
    end
    RunFieldMatrix{i,j}=RunRecords(k).runField;
    SourceRunCell{i,j}=RunRecords(k).run;
    CoreCell{i,j}=RunRecords(k).core;
    LegacyCell(i,j)=RunRecords(k).legacy;
    if RunRecords(k).legacy
        Z=RunRecords(k).run;
        OriginalTrainOK(i,j)=isfield(Z,'Train')&&isstruct(Z.Train)&& ...
            isfield(Z.Train,'TrainingResult')&&isstruct(Z.Train.TrainingResult)&& ...
            isfield(Z.Train.TrainingResult,'success')&&logical(Z.Train.TrainingResult.success);
        OriginalTestOK(i,j)=isfield(Z,'Test')&&isstruct(Z.Test)&& ...
            isfield(Z.Test,'TestingResult')&&isstruct(Z.Test.TestingResult)&& ...
            isfield(Z.Test.TestingResult,'success')&&logical(Z.Test.TestingResult.success);
    else
        Core=RunRecords(k).core;
        OriginalTrainOK(i,j)=getCoreStatus(Core,'TrainSuccess');
        OriginalTestOK(i,j)=getCoreStatus(Core,'TestSuccess');
    end
end
nStoredRuns=nnz(~cellfun(@isempty,RunFieldMatrix));
if nStoredRuns<1, error('No valid completed local-RBF runs were found.'); end

% Shared coordinates come directly from InputData, exactly as in the global
% bootstrap batch. Split Stage-2 files therefore never need to duplicate
% coordinate matrices merely for downstream bootstrap.
[nTrainReference,p]=size(XtrainRaw);
[nTestReference,pTest]=size(XtestRaw);
if pTest~=p, error('Training/test coordinate dimensions differ.'); end

% All runs use the same training geometry and therefore the same S_global.
SglobalReference=NaN;
for i=1:nModel
    for j=1:nKernel
        if isempty(RunFieldMatrix{i,j}), continue; end
        s=getRunScale(SourceRunCell{i,j},CoreCell{i,j},LegacyCell(i,j));
        if ~isfinite(SglobalReference), SglobalReference=s;
        elseif abs(s-SglobalReference)>1e-12*max(1,abs(SglobalReference))
            error('S_global differs between stored local-RBF runs.');
        end
        validateRunDimensions(SourceRunCell{i,j},CoreCell{i,j},LegacyCell(i,j), ...
            nTrainReference,nTestReference);
    end
end
if ~isfinite(SglobalReference)||SglobalReference<=0, error('No valid S_global was found.'); end
SglobalInput=common.MeanDistant(XtrainRaw);
if ~isfinite(SglobalInput) || SglobalInput<=0 || ...
        abs(SglobalInput-SglobalReference)>1e-12*max(1,abs(SglobalReference))
    error('InputData training geometry is inconsistent with fitted S_global.');
end
XtrainScaled=XtrainRaw./SglobalReference;
XtestScaled=XtestRaw./SglobalReference;
geometryCache=struct();
geometryCache.D=pdist2(XtrainScaled,XtrainScaled);
geometryCache.P=[ones(nTrainReference,1),XtrainScaled];
% Query-to-training geometry is also common to every Model x Kernel run.
% Build it once here rather than once inside every testing bootstrap worker.
geometryCache.D_query_train=pdist2(XtestScaled,XtrainScaled);
geometryCache.P_query=[ones(nTestReference,1),XtestScaled];

% Numerical options shared by all compact Stage-2 inputs.
solverOptions=struct();
if isfield(FitResults,'Settings')&&isstruct(FitResults.Settings)&& ...
        isfield(FitResults.Settings,'options')&&isstruct(FitResults.Settings.options)
    solverOptions=FitResults.Settings.options;
end
if isfield(solverOptions,'execution'), solverOptions=rmfield(solverOptions,'execution'); end

Results=struct(); Results.Run=struct();
Results.CaseName=CaseName;
Results.ListOfModel=ListOfModel;
Results.NameFunction_RBF=NameFunction_RBF;
Results.RunFieldMatrix=RunFieldMatrix;
Results.Settings.nBootstrap=nBootstrap;
Results.Settings.RNG_SEED_Train=RNG_SEED;
Results.Settings.RNG_SEED_Test=TestRNG_SEED;
Results.Settings.RNGAlgorithm='mt19937ar';
Results.Settings.verbose=verbose;
Results.Settings.SaveAll=saveAll;
Results.Settings.StoreIndices=bootstrapOptions.StoreIndices;
Results.Settings.NumSpatialSegments=bootstrapOptions.NumSpatialSegments;
Results.Settings.MinPointsPerAnnulus=bootstrapOptions.MinPointsPerAnnulus;
Results.Settings.TrainingMode=trainingMode;
Results.Settings.DominanceLimit=bootstrapOptions.DominanceLimit;
Results.Settings.MaxDominantFraction=bootstrapOptions.MaxDominantFraction;
Results.Settings.DominanceControlMethod='sequential_largest_variance_contributor';
if strcmp(trainingMode,'pseudo-unseen')
    Results.Settings.TrainingRadiusStorage='scaled';
    Results.Settings.TrainingRadiusRawReconstruction='radius_raw = radius * S_global';
else
    Results.Settings.TrainingRadiusStorage='raw_in_Stage2Core_then_scaled_for_bootstrap';
    Results.Settings.TrainingRadiusRawReconstruction='stored directly in Stage2Core.Parameters.Train.Radius';
end
Results.Settings.bootstrapOptions=bootstrapOptions;
Results.Source.RunFieldMatrix=RunFieldMatrix;
if isfield(FitResults,'Settings'), Results.Source.Settings=FitResults.Settings; end
if isfield(FitResults,'train_rmse'), Results.Source.train_rmse=FitResults.train_rmse; end
if isfield(FitResults,'test_rmse'), Results.Source.test_rmse=FitResults.test_rmse; end
if isfield(FitResults,'finalCapIndex'), Results.Source.finalCapIndex=FitResults.finalCapIndex; end
if isfield(FitResults,'finalCapRadiusRaw'), Results.Source.finalCapRadiusRaw=FitResults.finalCapRadiusRaw; end
Results.DataInfo.nTrain=nTrainReference;
Results.DataInfo.nTest=nTestReference;
Results.DataInfo.dimension=p;
Results.DataInfo.nModels=nModel;
Results.DataInfo.nKernels=nKernel;
Results.DataInfo.nStoredRuns=nStoredRuns;
Results.MatrixInfo.rows=ListOfModel;
Results.MatrixInfo.columns=NameFunction_RBF;
Results.MatrixInfo.description='Rows = input models; columns = RBF kernels';

% Compact matrices.
matNames={'trainMeanBootstrapVariance','trainMedianBootstrapVariance','trainMeanVariance','trainMedianVariance', ...
    'testMeanBootstrapVariance','testMedianBootstrapVariance','testMeanVariance','testMedianVariance', ...
    'trainWallTime','trainCPUTime','testWallTime','testCPUTime','totalWallTime','totalCPUTime', ...
    'trainRBFSolves','testRBFSolves','totalRBFSolves','trainRejected','testRejected','totalRejected', ...
    'trainDominantPointsCount','testDominantPointsCount', ...
    'trainDominantCount','testDominantCount'};
for q=1:numel(matNames), Results.(matNames{q})=nan(nModel,nKernel); end
Results.originalTrainSuccess=OriginalTrainOK;
Results.originalTestSuccess=OriginalTestOK;
Results.trainSuccess=false(nModel,nKernel);
Results.testSuccess=false(nModel,nKernel);
Results.success=false(nModel,nKernel);
Results.trainMessage=cell(nModel,nKernel);
Results.testMessage=cell(nModel,nKernel);
% Direct point markers for quick audit of where the dominance safeguard was used.
Results.trainDominantPoints=cell(nModel,nKernel);
Results.testDominantPoints=cell(nModel,nKernel);

% Fail before opening the parallel pool when pseudo-unseen source data are
% incompatible. Bootstrap has no ensemble branch and requires one fixed
% scalar Radius/Reg/C tuple plus finite residual per training target.
if strcmp(trainingMode,'pseudo-unseen')
    for i=1:nModel
        for j=1:nKernel
            if isempty(RunFieldMatrix{i,j}) || ~OriginalTrainOK(i,j)
                continue
            end
            validatePseudoBootstrapSourceLocal( ...
                SourceRunCell{i,j},CoreCell{i,j},LegacyCell(i,j), ...
                nTrainReference,NameFunction_RBF{j});

            % Split Stage-2 pseudo radii are already scaled.  Verify the
            % corrected interpretation against the saved pointwise
            % neighbour counts before opening the parallel pool.
            if ~LegacyCell(i,j)
                validatePseudoNeighborhoodConsistencyLocal( ...
                    CoreCell{i,j},XtrainScaled,geometryCache.D);
            end
        end
    end
end

checkpointDir='';
if ~isempty(OutputFile)
    outputDir=fileparts(OutputFile);
    if ~isempty(outputDir)&&~exist(outputDir,'dir'), mkdir(outputDir); end
    checkpointDir=fullfile(outputDir,[CaseName '_local_bootstrap_checkpoints']);
    if ~exist(checkpointDir,'dir'), mkdir(checkpointDir); end
end

pool=gcp('nocreate');
requestedWorkers=bootstrapOptions.NumWorkers;
if isempty(pool)
    if isempty(requestedWorkers), pool=parpool;
    else, pool=parpool('Processes',requestedWorkers); end
elseif ~isempty(requestedWorkers) && pool.NumWorkers~=requestedWorkers
    delete(pool);
    pool=parpool('Processes',requestedWorkers);
end
Results.Settings.Parallel=true; Results.Settings.NumWorkers=pool.NumWorkers;

nRun=nModel*nKernel;
RunCell=cell(nRun,1);
% Explicit linear run lists give PARFOR sliced variables and avoid
% broadcasting the complete Model x Kernel source-cell matrices.
RunFieldList=cell(nRun,1); SourceRunList=cell(nRun,1); CoreRunList=cell(nRun,1);
LegacyRunList=false(nRun,1); IRun=zeros(nRun,1); JRun=zeros(nRun,1);
for irun=1:nRun
    ii=floor((irun-1)/nKernel)+1; jj=mod(irun-1,nKernel)+1;
    IRun(irun)=ii; JRun(irun)=jj;
    RunFieldList{irun}=RunFieldMatrix{ii,jj};
    LegacyRunList(irun)=LegacyCell(ii,jj);

    % Split Stage-2 bootstrap input is reconstructed entirely from CORE.
    % Do not broadcast the complete source run to every worker unless this
    % is a legacy run that genuinely requires it.
    if LegacyRunList(irun)
        SourceRunList{irun}=SourceRunCell{ii,jj};
    else
        SourceRunList{irun}=struct();
    end

    CoreRunList{irun}=CoreCell{ii,jj};
end

% The linear worker lists above are now authoritative. Release redundant
% client-side containers before PARFOR to reduce peak memory. This does not
% discard any information used by Results or checkpoints.
clear RunRecords SourceRunCell CoreCell sourceRunNames modelTmp kernelTmp

tParallel=tic;
parfor irun=1:nRun
    i=IRun(irun); j=JRun(irun);
    R=emptyWorkerResult(); R.i=i; R.j=j;
    runField=RunFieldList{irun}; R.runField=runField;
    if isempty(runField)
        R.trainMessage='Source run is missing.'; R.testMessage='Source run is missing.';
        RunCell{irun}=R; continue
    end
    R.model=ListOfModel{i}; R.kernel=NameFunction_RBF{j};
    Z=SourceRunList{irun}; Core=CoreRunList{irun}; isLegacy=LegacyRunList(irun);

    % Always create a compact nonempty run record before any early exit so
    % failed pseudo-unseen runs cannot appear as unexplained empty structs.
    RunInfo=initializeRunInfoLocal(R.model,R.kernel,runField,trainingMode,saveAll);
    R.RunInfo=RunInfo;

    if ~OriginalTrainOK(i,j)
        R.trainMessage='Original local-RBF training run was unsuccessful.';
        R.testMessage='Bootstrap skipped because original training was unsuccessful.';
        R.RunInfo=finalizeFailureRunInfoLocal(R.RunInfo,R);
        RunCell{irun}=R; continue
    end

    try
        [TR,TT] = buildBootstrapInputs(Z,Core,isLegacy,XtrainRaw,XtestRaw,solverOptions,YTrainModel{i},YTestModel{i},R.model,trainingMode);
    catch ME
        R.trainMessage=ME.message; R.testMessage=ME.message;
        R.RunInfo=finalizeFailureRunInfoLocal(R.RunInfo,R);
        RunCell{irun}=R; continue
    end

    t=tic; c0=cputime;
    try
        Btrain=localrbf.localRBFBootstrapForTraining(TR,nBootstrap,RNG_SEED,bootstrapOptions,geometryCache);
        R.trainWallTime=toc(t); R.trainCPUTime=cputime-c0;
        R.trainSuccess=isfield(Btrain,'success')&&logical(Btrain.success);
        if R.trainSuccess, R.trainMessage=''; else, R.trainMessage=getText(Btrain,'message','Training bootstrap unsuccessful.'); end
        R.trainRBFSolves=getNestedScalar(Btrain,{'Computation','totalRBFSolves'},NaN);
        R.trainRejected=getNestedScalar(Btrain,{'Computation','totalRejected'},NaN);
    catch ME
        Btrain=[]; R.trainWallTime=toc(t); R.trainCPUTime=cputime-c0;
        R.trainSuccess=false; R.trainMessage=ME.message;
    end

    if OriginalTestOK(i,j)
        t=tic; c0=cputime;
        try
            Btest=localrbf.localRBFBootstrapForTesting(TR,TT,nBootstrap,TestRNG_SEED,bootstrapOptions,geometryCache);
            R.testWallTime=toc(t); R.testCPUTime=cputime-c0;
            R.testSuccess=isfield(Btest,'success')&&logical(Btest.success);
            if R.testSuccess, R.testMessage=''; else, R.testMessage=getText(Btest,'message','Testing bootstrap unsuccessful.'); end
            R.testRBFSolves=getNestedScalar(Btest,{'Computation','totalRBFSolves'},NaN);
            R.testRejected=getNestedScalar(Btest,{'Computation','totalRejected'},NaN);
        catch ME
            Btest=[]; R.testWallTime=toc(t); R.testCPUTime=cputime-c0;
            R.testSuccess=false; R.testMessage=ME.message;
        end
    else
        Btest=[];
        if ~OriginalTestOK(i,j), R.testMessage='Original local-RBF test run was unsuccessful.'; end
    end

    R.totalWallTime=sumFinitePair(R.trainWallTime,R.testWallTime);
    R.totalCPUTime=sumFinitePair(R.trainCPUTime,R.testCPUTime);
    R.totalRBFSolves=sumFinitePair(R.trainRBFSolves,R.testRBFSolves);
    R.totalRejected=sumFinitePair(R.trainRejected,R.testRejected);

    RunInfoFull=RunInfo;
    RunInfoFull.Train=Btrain; RunInfoFull.Test=Btest;
    RunInfoFull.TrainSuccess=R.trainSuccess; RunInfoFull.TestSuccess=R.testSuccess;
    RunInfoFull.TrainMessage=R.trainMessage; RunInfoFull.TestMessage=R.testMessage;
    RunInfoFull.totalWallTime=R.totalWallTime; RunInfoFull.totalCPUTime=R.totalCPUTime;

    checkpointFile=''; checkpointSaved=false;
    if ~isempty(checkpointDir)
        checkpointFile=fullfile(checkpointDir,[matlab.lang.makeValidName(runField) '.mat']);
        RunInfoFull.Storage.CheckpointFile=checkpointFile;
        % The checkpoint contains full replicate detail exactly when SaveAll
        % is enabled. Record that fact inside the checkpoint itself.
        RunInfoFull.Storage.FullDetailSaved=saveAll;
        try
            % Keep the temporary full checkpoint object inside a helper
            % workspace.  This preserves the intended early memory release
            % without using CLEAR directly inside PARFOR (which violates
            % MATLAB workspace-transparency rules).
            saveBootstrapCheckpointAtomic(checkpointFile,R,RunInfoFull);
            checkpointSaved=true;
        catch MEcheckpoint
            warning('localRBF:BootstrapCheckpointFailed','Bootstrap checkpoint failed for %s: %s',runField,MEcheckpoint.message);
        end
    end

    if saveAll && checkpointSaved
        RunInfoCompact=RunInfoFull;
        RunInfoCompact.Train=compactBootstrapResult(Btrain);
        RunInfoCompact.Test=compactBootstrapResult(Btest);
        RunInfoCompact.Storage.FullDetailSaved=true;
        RunInfoCompact.Storage.CheckpointFile=checkpointFile;
        R.RunInfo=RunInfoCompact;
    else
        % SaveAll=false is already compact. SaveAll=true without durable
        % checkpoint returns full detail to avoid requested-information loss.
        RunInfoFull.Storage.FullDetailSaved=saveAll&&checkpointSaved;
        RunInfoFull.Storage.CheckpointFile=checkpointFile;
        R.RunInfo=RunInfoFull;
    end
    % No CLEAR here: PARFOR iterations have isolated workspaces, and these
    % temporaries are released automatically when this iteration returns.
    RunCell{irun}=R;
end
Results.parallelWallTime=toc(tParallel);

for irun=1:nRun
    R=RunCell{irun}; i=R.i; j=R.j; runField=R.runField;
    if isempty(runField)
        Results.trainMessage{i,j}=R.trainMessage; Results.testMessage{i,j}=R.testMessage; continue
    end
    Results.Run.(runField)=R.RunInfo;
    Results.trainWallTime(i,j)=R.trainWallTime; Results.trainCPUTime(i,j)=R.trainCPUTime;
    Results.testWallTime(i,j)=R.testWallTime; Results.testCPUTime(i,j)=R.testCPUTime;
    Results.totalWallTime(i,j)=R.totalWallTime; Results.totalCPUTime(i,j)=R.totalCPUTime;
    Results.trainRBFSolves(i,j)=R.trainRBFSolves; Results.testRBFSolves(i,j)=R.testRBFSolves;
    Results.totalRBFSolves(i,j)=R.totalRBFSolves;
    Results.trainRejected(i,j)=R.trainRejected; Results.testRejected(i,j)=R.testRejected;
    Results.totalRejected(i,j)=R.totalRejected;
    Results.trainSuccess(i,j)=R.trainSuccess; Results.testSuccess(i,j)=R.testSuccess;
    Results.success(i,j)=R.trainSuccess&&R.testSuccess;
    Results.trainMessage{i,j}=R.trainMessage; Results.testMessage{i,j}=R.testMessage;
    if R.trainSuccess && isfield(R.RunInfo,'Train') && ~isempty(R.RunInfo.Train)
        b=R.RunInfo.Train;
        Results.trainMeanBootstrapVariance(i,j)=mean(b.BootstrapVariance);
        Results.trainMedianBootstrapVariance(i,j)=median(b.BootstrapVariance);
        Results.trainMeanVariance(i,j)=mean(b.Variance);
        Results.trainMedianVariance(i,j)=median(b.Variance);
        Results.trainDominantPointsCount(i,j)=getNestedScalar(b,{'DominanceControl','nDominantPoints'},0);
        Results.trainDominantCount(i,j)=getNestedScalar(b,{'DominanceControl','totalDominant'},0);
        if isfield(b,'DominanceControl')&&isstruct(b.DominanceControl)&&isfield(b.DominanceControl,'PointIndex')
            Results.trainDominantPoints{i,j}=b.DominanceControl.PointIndex(:);
        end
    end
    if R.testSuccess && isfield(R.RunInfo,'Test') && ~isempty(R.RunInfo.Test)
        b=R.RunInfo.Test;
        Results.testMeanBootstrapVariance(i,j)=mean(b.BootstrapVariance);
        Results.testMedianBootstrapVariance(i,j)=median(b.BootstrapVariance);
        Results.testMeanVariance(i,j)=mean(b.Variance);
        Results.testMedianVariance(i,j)=median(b.Variance);
        Results.testDominantPointsCount(i,j)=getNestedScalar(b,{'DominanceControl','nDominantPoints'},0);
        Results.testDominantCount(i,j)=getNestedScalar(b,{'DominanceControl','totalDominant'},0);
        if isfield(b,'DominanceControl')&&isstruct(b.DominanceControl)&&isfield(b.DominanceControl,'PointIndex')
            Results.testDominantPoints{i,j}=b.DominanceControl.PointIndex(:);
        end
    end
end

Results.nSuccessfulTrain=nnz(Results.trainSuccess);
Results.nSuccessfulTest=nnz(Results.testSuccess);
Results.nSuccessfulBoth=nnz(Results.success);
Results.totalStoredRuns=nStoredRuns;
Results.totalTrainRBFSolves=sum(Results.trainRBFSolves(isfinite(Results.trainRBFSolves)));
Results.totalTestRBFSolves=sum(Results.testRBFSolves(isfinite(Results.testRBFSolves)));
Results.totalAllRBFSolves=sum(Results.totalRBFSolves(isfinite(Results.totalRBFSolves)));
Results.totalTrainRejected=sum(Results.trainRejected(isfinite(Results.trainRejected)));
Results.totalTestRejected=sum(Results.testRejected(isfinite(Results.testRejected)));
Results.totalAllRejected=sum(Results.totalRejected(isfinite(Results.totalRejected)));
Results.totalTrainDominantPoints=sum(Results.trainDominantPointsCount(isfinite(Results.trainDominantPointsCount)));
Results.totalTestDominantPoints=sum(Results.testDominantPointsCount(isfinite(Results.testDominantPointsCount)));
Results.totalTrainDominantCount=sum(Results.trainDominantCount(isfinite(Results.trainDominantCount)));
Results.totalTestDominantCount=sum(Results.testDominantCount(isfinite(Results.testDominantCount)));
Results.Summary=buildSummaryTable(Results,RunFieldMatrix,ListOfModel,NameFunction_RBF,nStoredRuns);
Results.OutputFile=OutputFile;
Results.CheckpointDir=checkpointDir;

if ~isempty(OutputFile)
    saveNamedVariableAtomic(OutputFile,'Results',Results);
end
end

% =========================================================================
% Helpers
% =========================================================================
function [Core,isLegacy] = readStage2CoreOrLegacy(Z,runField)
Core=struct(); isLegacy=false;
if isfield(Z,'Stage2Core') && isstruct(Z.Stage2Core) && ~isempty(fieldnames(Z.Stage2Core))
    Core=Z.Stage2Core; return
end
if isfield(Z,'Stage2Paths') && isstruct(Z.Stage2Paths) && isfield(Z.Stage2Paths,'core') && ...
        ~isempty(Z.Stage2Paths.core) && isfile(char(Z.Stage2Paths.core))
    S=load(char(Z.Stage2Paths.core),'Stage2Core');
    if ~isfield(S,'Stage2Core') || ~isstruct(S.Stage2Core)
        error('Invalid Stage-2 CORE file for run %s.',runField);
    end
    Core=S.Stage2Core;
    if ~isfield(Core,'Storage') || ~isstruct(Core.Storage) || ...
            ~isfield(Core.Storage,'Complete') || ~logical(Core.Storage.Complete)
        error('Stage-2 CORE is not a completed commit for run %s.',runField);
    end
    return
end
if isfield(Z,'Train') && isstruct(Z.Train) && isfield(Z.Train,'TrainingResult') && ...
        isstruct(Z.Train.TrainingResult)
    isLegacy=true; return
end
error('Run %s has neither a valid split Stage-2 CORE nor a legacy TrainingResult.',runField);
end

function [modelName,kernel] = resolveRunIdentity(Z,Core)
modelName=''; kernel='';
if isfield(Z,'model')&&~isempty(Z.model), modelName=char(string(Z.model)); end
if isfield(Z,'kernel')&&~isempty(Z.kernel), kernel=char(string(Z.kernel)); end
if isempty(modelName)&&isstruct(Core)&&isfield(Core,'Setup')&&isfield(Core.Setup,'Model')
    modelName=char(string(Core.Setup.Model));
end
if isempty(kernel)&&isstruct(Core)&&isfield(Core,'Setup')&&isfield(Core.Setup,'Kernel')
    kernel=char(string(Core.Setup.Kernel));
end
end

function tf=getCoreStatus(Core,name)
tf=false;
if isstruct(Core)&&isfield(Core,'Status')&&isstruct(Core.Status)&&isfield(Core.Status,name)
    v=Core.Status.(name); if isscalar(v), tf=logical(v); end
end
end

function s=getRunScale(Z,Core,isLegacy)
if isLegacy
    s=Z.Train.TrainingResult.S_global;
else
    s=Core.Setup.SGlobal;
end
if ~isnumeric(s)||~isscalar(s)||~isfinite(s)||s<=0, error('Invalid S_global in source run.'); end
end

function validateRunDimensions(Z,Core,isLegacy,nTrain,nTest)
if isLegacy
    TR=Z.Train.TrainingResult;
    if size(TR.X,1)~=nTrain, error('Legacy training size mismatch.'); end
    if isfield(Z,'Test')&&isfield(Z.Test,'TestingResult')&&isstruct(Z.Test.TestingResult) && ...
            size(Z.Test.TestingResult.Xtest,1)~=nTest
        error('Legacy testing size mismatch.');
    end
else
    if Core.Setup.NTrain~=nTrain||Core.Setup.NTest~=nTest
        error('Stage-2 CORE train/test size mismatch.');
    end
end
end

function [TR,TT]=buildBootstrapInputs(Z,Core,isLegacy,XtrainRaw,XtestRaw,sharedOptions,yTrainInput,yTestInput,modelName,trainingMode)
%BUILDBOOTSTRAPINPUTS Convert main output to fixed-setting bootstrap inputs.
% Bootstrap consumes settings only; it never transfers them.

if isLegacy
    TR=Z.Train.TrainingResult;
    if isfield(Z,'Test')&&isfield(Z.Test,'TestingResult'), TT=Z.Test.TestingResult; else, TT=struct(); end
    yTrainInput=yTrainInput(:); yTestInput=yTestInput(:);
    if ~isequaln(double(TR.y(:)),double(yTrainInput)), error('InputData training response does not match legacy TrainingResult for model %s.',modelName); end
    if isfield(TR,'X_raw'), Xlegacy=TR.X_raw; else, Xlegacy=TR.X.*TR.S_global; end
    if ~isequaln(double(Xlegacy),double(XtrainRaw)), error('InputData training coordinates do not match legacy TrainingResult.'); end
    if ~isempty(fieldnames(TT))
        if isfield(TT,'Xtest_raw'), Xq=TT.Xtest_raw; else, Xq=TT.Xtest.*TR.S_global; end
        if ~isequaln(double(Xq),double(XtestRaw)), error('InputData test coordinates do not match legacy TestingResult.'); end
    end
    if strcmp(trainingMode,'pseudo-unseen')
        [P,res,pred] = legacyPseudoSourceLocal(Z,numel(yTrainInput));
        TR.Final=P; TR.success=P.success; TR.ReferenceResidual=res; TR.ReferencePrediction=pred;
    else
        TR.ReferenceResidual=getLegacyResidualLocal(TR,numel(yTrainInput));
        TR.ReferencePrediction=getLegacyPredictionLocal(TR,numel(yTrainInput));
    end
    if ~isempty(fieldnames(TT))
        TT.ReferenceResidual=getLegacyTestResidualLocal(TT,numel(yTestInput));
        TT.ReferencePrediction=getLegacyTestPredictionLocal(TT,numel(yTestInput));
        TT.ReferenceObserved=yTestInput;
    end
    return
end

s=Core.Setup.SGlobal; nTrain=Core.Setup.NTrain; nTest=Core.Setup.NTest;
if size(XtrainRaw,1)~=nTrain||size(XtestRaw,1)~=nTest, error('Coordinate/Core size mismatch.'); end
yTrainInput=yTrainInput(:); yTestInput=yTestInput(:);
if numel(yTrainInput)~=nTrain||numel(yTestInput)~=nTest||any(~isfinite(yTrainInput))||any(~isfinite(yTestInput)), error('InputData response sizes/values are invalid for model %s.',modelName); end
yTrainCore=Core.Accuracy.Train.Observed(:); yTestCore=Core.Accuracy.Test.Observed(:);
if ~isequaln(double(yTrainCore),double(yTrainInput))||~isequaln(double(yTestCore),double(yTestInput)), error('InputData responses do not exactly match Stage-2 CORE for model %s.',modelName); end
kernel=char(Core.Setup.Kernel); useShape=common.kernelHasShapeParameter(kernel);
options=sharedOptions; if ~isfield(options,'solver')||~isstruct(options.solver), options.solver=struct(); end
if isfield(Core.Setup,'SolverCondThreshold')&&isfinite(Core.Setup.SolverCondThreshold), options.solver.cond_lsq=Core.Setup.SolverCondThreshold; end
TR=struct(); TR.success=getCoreStatus(Core,'TrainSuccess'); TR.kernel=kernel; TR.useShape=useShape;
TR.X=XtrainRaw./s; TR.X_raw=XtrainRaw; TR.y=yTrainCore; TR.S_global=s; TR.minNeighbors=Core.Setup.MinNeighbors; TR.options=options;

if strcmp(trainingMode,'original')
    P=Core.Parameters.Train;
    TR.Final=parameterBlockToFinalLocal(P,nTrain,s,TR.success);
    TR.ReferenceResidual=getVectorNestedLocal(Core,{'Accuracy','Train','Residual'},nTrain,'original training residual');
    TR.ReferencePrediction=getOptionalVectorNestedLocal(Core,{'Accuracy','Train','Prediction'},nTrain);
else
    % Canonical pseudo-unseen bootstrap input. This must be generated by the
    % main/upstream workflow. No transfer is reconstructed in bootstrap.
    if ~isfield(Core,'Parameters')||~isstruct(Core.Parameters)||~isfield(Core.Parameters,'Pseudo')||~isstruct(Core.Parameters.Pseudo)
        error(['TrainingMode=''pseudo-unseen'' requires Core.Parameters.Pseudo with pointwise Radius/Reg/C produced upstream. ' ...
            'The bootstrap package intentionally does not reconstruct transfer settings.']);
    end
    if ~isfield(Core,'Accuracy')||~isstruct(Core.Accuracy)||~isfield(Core.Accuracy,'Pseudo')||~isstruct(Core.Accuracy.Pseudo)
        error('TrainingMode=''pseudo-unseen'' requires Core.Accuracy.Pseudo with pointwise residuals produced upstream.');
    end
    P=Core.Parameters.Pseudo;
    pseudoSuccess=true;
    if isfield(P,'PointSuccess')&&numel(P.PointSuccess)==nTrain, pseudoSuccess=all(logical(P.PointSuccess(:))); end
    % Pseudo Radius is already in the scaled coordinate system used by
    % localrbf.getNeighborhood.  Do not divide by S_global again.
    TR.Final=pseudoParameterBlockToFinalLocal(P,nTrain,s,pseudoSuccess);
    TR.success=TR.Final.success;
    TR.ReferenceResidual=getVectorNestedLocal(Core,{'Accuracy','Pseudo','Residual'},nTrain,'pseudo-unseen residual');
    TR.ReferencePrediction=getOptionalVectorNestedLocal(Core,{'Accuracy','Pseudo','Prediction'},nTrain);
    TR.ReferenceNumerical=getPseudoNumericalLocal(Core,nTrain);
end

TT=struct(); TT.success=getCoreStatus(Core,'TestSuccess'); TT.kernel=kernel; TT.useShape=useShape;
TT.nTrain=nTrain; TT.nTest=nTest; TT.Xtest_raw=XtestRaw; TT.Xtest=XtestRaw./s; TT.S_global=s;
TT.minNeighbors=Core.Setup.MinNeighbors; TT.point_success=true(nTest,1);
TT.radius_raw=Core.Parameters.Test.Radius(:); TT.radius=TT.radius_raw./s;
TT.C=Core.Parameters.Test.C(:); TT.reg_param=Core.Parameters.Test.Reg(:);
TT.neighbor_count=Core.Parameters.Test.NeighborCount(:);
TT.ReferenceResidual=getVectorNestedLocal(Core,{'Accuracy','Test','Residual'},nTest,'test residual');
TT.ReferencePrediction=getOptionalVectorNestedLocal(Core,{'Accuracy','Test','Prediction'},nTest);
TT.ReferenceObserved=yTestCore;
end

function F=parameterBlockToFinalLocal(P,n,s,successFlag)
%PARAMETERBLOCKTOFINALLOCAL Convert RAW-radius Train storage to scaled form.
req={'Radius','C','Reg'}; for k=1:numel(req), if ~isfield(P,req{k}), error('Fixed parameter block is missing %s.',req{k}); end, end
F=struct(); F.success=logical(successFlag); F.point_success=true(n,1);
F.radius_raw=P.Radius(:); F.radius=F.radius_raw./s; F.C=P.C(:); F.reg_param=P.Reg(:);
if numel(F.radius)~=n||numel(F.C)~=n||numel(F.reg_param)~=n, error('Fixed parameter block dimensions are inconsistent with nTrain.'); end
if isfield(P,'NeighborCount')&&numel(P.NeighborCount)==n, F.neighbor_count=P.NeighborCount(:); else, F.neighbor_count=nan(n,1); end
if isfield(P,'Epsilon')&&numel(P.Epsilon)==n, F.epsilon=P.Epsilon(:); end
if isfield(P,'PointSuccess')&&numel(P.PointSuccess)==n, F.point_success=logical(P.PointSuccess(:)); F.success=F.success&&all(F.point_success); end
end

function F=pseudoParameterBlockToFinalLocal(P,n,s,successFlag)
%PSEUDOPARAMETERBLOCKTOFINALLOCAL Convert split Stage-2 pseudo settings.
%
% Storage contract:
%   P.Radius is already scaled by S_global because it is copied from the
%   strict pseudo-unseen transfer/evaluation path (Transfer.radius).
% Therefore F.radius=P.Radius exactly.  F.radius_raw is reconstructed only
% for metadata/diagnostics and must not feed back into neighbourhood search.
req={'Radius','C','Reg'};
for k=1:numel(req)
    if ~isfield(P,req{k})
        error('Fixed pseudo parameter block is missing %s.',req{k});
    end
end
F=struct();
F.success=logical(successFlag);
F.point_success=true(n,1);
F.radius=P.Radius(:);
F.radius_raw=F.radius.*s;
F.C=P.C(:);
F.reg_param=P.Reg(:);
if numel(F.radius)~=n||numel(F.C)~=n||numel(F.reg_param)~=n
    error('Fixed pseudo parameter block dimensions are inconsistent with nTrain.');
end
if any(~isfinite(F.radius))||any(F.radius<=0)
    error('Fixed pseudo Radius must contain finite positive scaled values.');
end
if any(~isfinite(F.reg_param))||any(F.reg_param<=0)
    error('Fixed pseudo Reg must contain finite positive values.');
end
if isfield(P,'NeighborCount')&&numel(P.NeighborCount)==n
    F.neighbor_count=P.NeighborCount(:);
else
    F.neighbor_count=nan(n,1);
end
if isfield(P,'Epsilon')&&numel(P.Epsilon)==n
    F.epsilon=P.Epsilon(:);
end
if isfield(P,'PointSuccess')&&numel(P.PointSuccess)==n
    F.point_success=logical(P.PointSuccess(:));
    F.success=F.success&&all(F.point_success);
end
end

function validatePseudoBootstrapSourceLocal(Z,Core,isLegacy,n,kernel)
useShape=common.kernelHasShapeParameter(kernel);
if isLegacy
    [F,res,~]=legacyPseudoSourceLocal(Z,n);
    if ~isfield(F,'radius')||~isfield(F,'reg_param')||numel(F.radius)~=n||numel(F.reg_param)~=n
        error('Legacy pseudo-unseen source lacks pointwise Radius/RegParam.');
    end
    if any(~isfinite(F.radius(:)))||any(F.radius(:)<=0)||any(~isfinite(F.reg_param(:)))||any(F.reg_param(:)<=0)
        error('Legacy pseudo-unseen Radius/RegParam contains invalid values.');
    end
    if useShape && (~isfield(F,'C')||numel(F.C)~=n||any(~isfinite(F.C(:)))||any(F.C(:)<=0))
        error('Legacy pseudo-unseen C contains invalid values for a shaped kernel.');
    end
    if numel(res)~=n||any(~isfinite(res)), error('Legacy pseudo-unseen residual is invalid.'); end
    return
end
if ~isfield(Core,'Parameters')||~isstruct(Core.Parameters)||~isfield(Core.Parameters,'Pseudo')||~isstruct(Core.Parameters.Pseudo)
    error('Pseudo-unseen bootstrap requires Stage2Core.Parameters.Pseudo. Re-run Stage 2 with the revised main local code.');
end
P=Core.Parameters.Pseudo;
if isfield(P,'TransferMethod')&&strcmpi(char(string(P.TransferMethod)),'ensemble')
    error('Pseudo-unseen bootstrap does not support ensemble transfer. Use scalar nearest/IDW/hybrid pseudo settings upstream.');
end
if isfield(P,'ScalarPointwiseParametersAvailable')&&~logical(P.ScalarPointwiseParametersAvailable)
    error('Pseudo-unseen scalar pointwise parameters are unavailable. Bootstrap does not reconstruct or approximate them.');
end
req={'Radius','Reg','C','NeighborCount','PointSuccess'};
for q=1:numel(req), if ~isfield(P,req{q}), error('Stage2Core.Parameters.Pseudo.%s is required.',req{q}); end, end
if numel(P.Radius)~=n||numel(P.Reg)~=n||numel(P.C)~=n|| ...
        numel(P.NeighborCount)~=n||numel(P.PointSuccess)~=n
    error('Stage2Core.Parameters.Pseudo dimensions are inconsistent with nTrain.');
end
if ~all(logical(P.PointSuccess(:))), error('Pseudo-unseen pointwise settings are incomplete.'); end
if any(~isfinite(P.Radius(:)))||any(P.Radius(:)<=0)||any(~isfinite(P.Reg(:)))||any(P.Reg(:)<=0)
    error('Pseudo-unseen Radius/Reg contains invalid values.');
end
if any(~isfinite(P.NeighborCount(:)))||any(P.NeighborCount(:)<1)|| ...
        any(P.NeighborCount(:)~=round(P.NeighborCount(:)))
    error('Pseudo-unseen NeighborCount contains invalid values.');
end
if useShape && (any(~isfinite(P.C(:)))||any(P.C(:)<=0)), error('Pseudo-unseen C contains invalid values for shaped kernel.'); end
res=getVectorNestedLocal(Core,{'Accuracy','Pseudo','Residual'},n,'pseudo-unseen residual'); %#ok<NASGU>
end

function validatePseudoNeighborhoodConsistencyLocal(Core,Xscaled,D)
%VALIDATEPSEUDONEIGHBORHOODCONSISTENCYLOCAL
% Confirm the bootstrap interpretation of the stored pseudo Radius before
% expensive resampling starts.  Stage-2 pseudo Radius is already scaled.
if ~isstruct(Core)||~isfield(Core,'Parameters')||~isfield(Core.Parameters,'Pseudo')
    error('Stage2Core.Parameters.Pseudo is required for pseudo neighbourhood validation.');
end
if ~isfield(Core,'Setup')||~isfield(Core.Setup,'MinNeighbors')
    error('Stage2Core.Setup.MinNeighbors is required for pseudo neighbourhood validation.');
end
P=Core.Parameters.Pseudo;
n=size(Xscaled,1);
minNeighbors=Core.Setup.MinNeighbors;
if ~isnumeric(minNeighbors)||~isscalar(minNeighbors)||~isfinite(minNeighbors)|| ...
        minNeighbors<1||minNeighbors~=round(minNeighbors)
    error('Stage2Core.Setup.MinNeighbors is invalid.');
end
if size(D,1)~=n||size(D,2)~=n
    error('Pseudo neighbourhood validation geometry has invalid dimensions.');
end
for iTarget=1:n
    if ~logical(P.PointSuccess(iTarget))
        continue
    end
    radiusScaled=P.Radius(iTarget);
    Nbh=localrbf.getNeighborhood( ...
        Xscaled,Xscaled(iTarget,:),radiusScaled,minNeighbors, ...
        iTarget,D(iTarget,:).');
    if ~isstruct(Nbh)||~isfield(Nbh,'success')||~Nbh.success||~isfield(Nbh,'idx')
        error('Failed to reconstruct pseudo-unseen neighbourhood at training point %d.',iTarget);
    end
    nReconstructed=numel(Nbh.idx);
    nSaved=P.NeighborCount(iTarget);
    if nReconstructed~=nSaved
        error([ ...
            'Pseudo-unseen neighbour count mismatch at training point %d: ' ...
            'reconstructed=%d, saved=%d, scaled Radius=%.17g. ' ...
            'Stage2Core.Parameters.Pseudo.Radius is interpreted as already scaled.'], ...
            iTarget,nReconstructed,nSaved,radiusScaled);
    end
end
end

function N=getPseudoNumericalLocal(Core,n)
N=struct();
if ~isfield(Core,'Numerical')||~isstruct(Core.Numerical)||~isfield(Core.Numerical,'Pseudo')||~isstruct(Core.Numerical.Pseudo), return; end
P=Core.Numerical.Pseudo;
fields={'Cond1','Cond2','Cond3','RelResidual','BackwardError','Success'};
for q=1:numel(fields)
    f=fields{q};
    if isfield(P,f)&&numel(P.(f))==n, N.(f)=P.(f)(:); end
end
if isfield(P,'Solver')&&iscell(P.Solver)&&numel(P.Solver)==n, N.Solver=P.Solver(:); end
end

function [F,res,pred]=legacyPseudoSourceLocal(Z,n)
F=[]; res=[]; pred=nan(n,1);
if isfield(Z,'Train')&&isstruct(Z.Train)&&isfield(Z.Train,'PseudoTrainingResult')&&isstruct(Z.Train.PseudoTrainingResult)
    Q=Z.Train.PseudoTrainingResult;
elseif isfield(Z,'PseudoTrainingResult')&&isstruct(Z.PseudoTrainingResult)
    Q=Z.PseudoTrainingResult;
else
    error(['TrainingMode=''pseudo-unseen'' requires an upstream legacy PseudoTrainingResult containing fixed pointwise settings. ' ...
        'Bootstrap does not calculate transfer.']);
end
if isfield(Q,'Final')&&isstruct(Q.Final), F=Q.Final; else, error('PseudoTrainingResult.Final is required.'); end
if ~isfield(F,'success'), F.success=true; end
if ~isfield(F,'point_success'), F.point_success=true(n,1); end
if isfield(Q,'residual'), res=Q.residual(:); elseif isfield(Q,'y_diff'), res=Q.y_diff(:); else, error('PseudoTrainingResult residual is required for calibration.'); end
if numel(res)~=n, error('PseudoTrainingResult residual size mismatch.'); end
if isfield(Q,'yhat')&&numel(Q.yhat)==n, pred=Q.yhat(:); end
end
function v=getLegacyResidualLocal(TR,n), v=firstLegacyVectorLocal(TR,{'residual','y_diff','y_diff_train'},n,'training residual'); end
function v=getLegacyPredictionLocal(TR,n), v=firstLegacyVectorOptionalLocal(TR,{'yhat','y_train_pred'},n); end
function v=getLegacyTestResidualLocal(TT,n), v=firstLegacyVectorLocal(TT,{'residual','y_diff','test_diff'},n,'test residual'); end
function v=getLegacyTestPredictionLocal(TT,n), v=firstLegacyVectorOptionalLocal(TT,{'yhat','y_pred_test'},n); end
function v=firstLegacyVectorLocal(S,names,n,label)
for k=1:numel(names), if isfield(S,names{k})&&isnumeric(S.(names{k}))&&numel(S.(names{k}))==n, v=S.(names{k})(:); return, end, end
error('Legacy %s was not found.',label);
end
function v=firstLegacyVectorOptionalLocal(S,names,n)
v=nan(n,1); for k=1:numel(names), if isfield(S,names{k})&&isnumeric(S.(names{k}))&&numel(S.(names{k}))==n, v=S.(names{k})(:); return, end, end
end
function v=getVectorNestedLocal(S,path,n,label)
T=S; for k=1:numel(path), if ~isstruct(T)||~isfield(T,path{k}), error('Stage-2 CORE %s is missing.',label); end, T=T.(path{k}); end
if ~isnumeric(T)||numel(T)~=n, error('Stage-2 CORE %s has invalid size/type.',label); end
v=T(:); if any(~isfinite(v)), error('Stage-2 CORE %s contains NaN/Inf.',label); end
end
function v=getOptionalVectorNestedLocal(S,path,n)
T=S; for k=1:numel(path), if ~isstruct(T)||~isfield(T,path{k}), v=nan(n,1); return, end, T=T.(path{k}); end
if isnumeric(T)&&numel(T)==n, v=T(:); else, v=nan(n,1); end
end

function B=compactBootstrapResult(B)
if ~isstruct(B)||isempty(fieldnames(B)), return; end
if isfield(B,'Predictions'), B=rmfield(B,'Predictions'); end
if isfield(B,'BootstrapIndices'), B=rmfield(B,'BootstrapIndices'); end
if isfield(B,'Diagnostics')&&isstruct(B.Diagnostics)
    f={'d_bootstrap','epsilon_bootstrap','cond_reg','reg_phi','solver_used','nUnique'};
    for k=1:numel(f), if isfield(B.Diagnostics,f{k}), B.Diagnostics=rmfield(B.Diagnostics,f{k}); end, end
end
if isfield(B,'Local')&&isstruct(B.Local)&&isfield(B.Local,'neighbor_idx')
    B.Local=rmfield(B.Local,'neighbor_idx');
end
end

function RunInfo=initializeRunInfoLocal(model,kernel,runField,trainingMode,saveAll)
%INITIALIZERUNINFOLOCAL Nonempty run record used for success and failure.
RunInfo=struct();
RunInfo.model=model;
RunInfo.kernel=kernel;
RunInfo.SourceRunField=runField;
RunInfo.TrainingMode=trainingMode;
RunInfo.Train=[];
RunInfo.Test=[];
RunInfo.TrainSuccess=false;
RunInfo.TestSuccess=false;
RunInfo.TrainMessage='';
RunInfo.TestMessage='';
RunInfo.totalWallTime=NaN;
RunInfo.totalCPUTime=NaN;
RunInfo.Storage=struct( ...
    'SaveAllRequested',saveAll, ...
    'FullDetailSaved',false, ...
    'CheckpointFile','');
end

function RunInfo=finalizeFailureRunInfoLocal(RunInfo,R)
%FINALIZEFAILURERUNINFOLOCAL Preserve exact failure context in Results.Run.
RunInfo.TrainSuccess=logical(R.trainSuccess);
RunInfo.TestSuccess=logical(R.testSuccess);
RunInfo.TrainMessage=R.trainMessage;
RunInfo.TestMessage=R.testMessage;
RunInfo.totalWallTime=R.totalWallTime;
RunInfo.totalCPUTime=R.totalCPUTime;
end

function R=emptyWorkerResult()
R=struct('i',NaN,'j',NaN,'model','','kernel','','runField','', ...
    'trainWallTime',NaN,'trainCPUTime',NaN,'testWallTime',NaN,'testCPUTime',NaN, ...
    'totalWallTime',NaN,'totalCPUTime',NaN,'trainRBFSolves',NaN,'testRBFSolves',NaN, ...
    'totalRBFSolves',NaN,'trainRejected',NaN,'testRejected',NaN,'totalRejected',NaN, ...
    'trainSuccess',false,'testSuccess',false,'trainMessage','','testMessage','','RunInfo',struct());
end

function T=buildSummaryTable(Results,RunFieldMatrix,ListOfModel,NameFunction_RBF,nStoredRuns)
T=table('Size',[nStoredRuns 19], ...
    'VariableTypes',{'string','string','string','logical','logical', ...
    'double','double','double','double','double','double','double','double', ...
    'double','double','double','double','double','double'}, ...
    'VariableNames',{'RunField','Model','Kernel','TrainSuccess','TestSuccess', ...
    'TrainMeanBootstrapVariance','TrainMedianBootstrapVariance','TrainMeanVariance','TrainMedianVariance', ...
    'TestMeanBootstrapVariance','TestMedianBootstrapVariance','TestMeanVariance','TestMedianVariance', ...
    'TrainDominantPoints','TestDominantPoints', ...
    'TrainDominantCount','TestDominantCount','TrainWallTime','TestWallTime'});
r=0;
for i=1:numel(ListOfModel)
    for j=1:numel(NameFunction_RBF)
        runField=RunFieldMatrix{i,j}; if isempty(runField), continue; end
        r=r+1; T.RunField(r)=string(runField); T.Model(r)=string(ListOfModel{i}); T.Kernel(r)=string(NameFunction_RBF{j});
        T.TrainSuccess(r)=Results.trainSuccess(i,j); T.TestSuccess(r)=Results.testSuccess(i,j);
        T.TrainMeanBootstrapVariance(r)=Results.trainMeanBootstrapVariance(i,j);
        T.TrainMedianBootstrapVariance(r)=Results.trainMedianBootstrapVariance(i,j);
        T.TrainMeanVariance(r)=Results.trainMeanVariance(i,j); T.TrainMedianVariance(r)=Results.trainMedianVariance(i,j);
        T.TestMeanBootstrapVariance(r)=Results.testMeanBootstrapVariance(i,j);
        T.TestMedianBootstrapVariance(r)=Results.testMedianBootstrapVariance(i,j);
        T.TestMeanVariance(r)=Results.testMeanVariance(i,j); T.TestMedianVariance(r)=Results.testMedianVariance(i,j);
        T.TrainDominantPoints(r)=Results.trainDominantPointsCount(i,j);
        T.TestDominantPoints(r)=Results.testDominantPointsCount(i,j);
        T.TrainDominantCount(r)=Results.trainDominantCount(i,j);
        T.TestDominantCount(r)=Results.testDominantCount(i,j);
        T.TrainWallTime(r)=Results.trainWallTime(i,j); T.TestWallTime(r)=Results.testWallTime(i,j);
    end
end
end

function saveBootstrapCheckpointAtomic(fileName,R,RunInfoFull)
%SAVEBootstrapCheckpointAtomic Build/save the full checkpoint in an
%isolated function workspace so its temporary copy is released on return.
Rcheckpoint=R;
Rcheckpoint.RunInfo=RunInfoFull;
saveNamedVariableAtomic(fileName,'Rrun',Rcheckpoint);
end

function saveNamedVariableAtomic(fileName,variableName,value)
if isempty(fileName), return; end
parent=fileparts(fileName); if ~isempty(parent)&&~exist(parent,'dir'), mkdir(parent); end
if ~(ischar(fileName)||(isstring(fileName)&&isscalar(fileName))) || isempty(char(fileName))
    error('fileName must be a nonempty character vector or scalar string.');
end
if ~(ischar(variableName)||(isstring(variableName)&&isscalar(variableName))) || isempty(char(variableName))
    error('variableName must be a nonempty character vector or scalar string.');
end
fileName=char(fileName); variableName=char(variableName);
parent=fileparts(fileName); if isempty(parent), parent=pwd; elseif ~exist(parent,'dir'), mkdir(parent); end
tmp=[tempname(parent) '.mat'];
cleanup=onCleanup(@()deleteIfExists(tmp)); %#ok<NASGU>
S=struct(); S.(variableName)=value;
save(tmp,'-struct','S','-v7.3');
info=dir(tmp); if isempty(info)||info.bytes<=0, error('Atomic MAT temporary save failed: %s',tmp); end
[ok,msg]=movefile(tmp,fileName,'f'); if ~ok, error('Atomic MAT publish failed: %s',msg); end
end

function deleteIfExists(f)
if isfile(f), try, delete(f); catch, end, end
end

function v=getNestedScalar(S,path,defaultValue)
v=defaultValue; T=S;
for k=1:numel(path)
    if ~isstruct(T)||~isfield(T,path{k}), return; end
    T=T.(path{k});
end
if isnumeric(T)&&isscalar(T)&&isfinite(T), v=T; end
end

function txt=getText(S,f,defaultValue)
txt=defaultValue;
if isstruct(S)&&isfield(S,f)&&(ischar(S.(f))||(isstring(S.(f))&&isscalar(S.(f))))
    txt=char(S.(f));
end
end

function s=sumFinitePair(a,b)
vals=[a b]; vals=vals(isfinite(vals)); if isempty(vals), s=NaN; else, s=sum(vals); end
end

function tf=validateLogicalScalar(value,name)
if ~(isscalar(value)&&(islogical(value)||(isnumeric(value)&&isreal(value)&&isfinite(value)&&any(value==[0 1]))))
    error('%s must be true/false (or numeric 1/0).',name);
end
tf=logical(value);
end

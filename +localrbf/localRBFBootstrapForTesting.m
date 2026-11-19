function Bootstrap = localRBFBootstrapForTesting( ...
    TrainingResult, TestingResult, nBootstrap, RNG_SEED, bootstrapOptions, geometryCache)
%LOCALRBFBOOTSTRAPFORTESTING Bootstrap uncertainty for fixed local-RBF test settings.
%
% Radius / RegParam / C are consumed exactly as already supplied by the
% main local-RBF output. Bootstrap performs NO optimization, cap selection,
% or setting transfer and does not inspect the transfer method.
%
% Variance/resampling definitions intentionally match the global bootstrap:
% B valid predictions are generated first. A rare sequential variance-
% dominance control then treats the current largest variance contributor
% as dominant only while its contribution exceeds 80%, capped at 3% of B. Eq. (25)
% is evaluated on the retained predictions, followed by Eq. (26)
% nUsed/(nUsed-1) finite-sample adjustment. NumSpatialSegments=0 gives the
% ordinary bootstrap; NumSpatialSegments>0 uses the same global adaptive
% target-relative annular resampling helpers inside each fixed local
% neighbourhood.

if nargin < 3 || isempty(nBootstrap), nBootstrap=200; end
if nargin < 4 || isempty(RNG_SEED), RNG_SEED=1; end
if nargin < 5 || isempty(bootstrapOptions), bootstrapOptions=struct(); end
if nargin < 6, geometryCache=[]; end
bootstrapOptions=normalizeBootstrapOptionsLocal(bootstrapOptions);
verbose=bootstrapOptions.Verbose; saveAll=bootstrapOptions.SaveAll;
storeIndices=bootstrapOptions.StoreIndices; maxAttemptsFactor=bootstrapOptions.MaxAttemptsFactor;
numSpatialSegments=bootstrapOptions.NumSpatialSegments; minPointsPerAnnulus=bootstrapOptions.MinPointsPerAnnulus;
dominanceLimit=bootstrapOptions.DominanceLimit;
maxDominantFraction=bootstrapOptions.MaxDominantFraction;
useSpatialBootstrap=numSpatialSegments>0;
validateBootstrapScalarInputsLocal(nBootstrap,RNG_SEED,maxAttemptsFactor,numSpatialSegments,minPointsPerAnnulus);
if useSpatialBootstrap, requireSpatialHelpersLocal(); end
maxAttempts=ceil(maxAttemptsFactor*nBootstrap);

% Training / test fixed-setting contracts.
if ~isstruct(TrainingResult)||~isscalar(TrainingResult), error('TrainingResult must be a scalar structure.'); end
reqT={'success','kernel','useShape','X','y','S_global','minNeighbors','Final'};
for k=1:numel(reqT), if ~isfield(TrainingResult,reqT{k}), error('TrainingResult is missing required field "%s".',reqT{k}); end, end
if ~logical(TrainingResult.success)||~isstruct(TrainingResult.Final)||~isfield(TrainingResult.Final,'success')||~logical(TrainingResult.Final.success)
    error('TrainingResult does not contain a successful final model.');
end
if ~isstruct(TestingResult)||~isscalar(TestingResult), error('TestingResult must be a scalar structure.'); end
reqQ={'success','kernel','useShape','nTrain','nTest','Xtest','point_success','radius','C','reg_param'};
for k=1:numel(reqQ), if ~isfield(TestingResult,reqQ{k}), error('TestingResult is missing required field "%s".',reqQ{k}); end, end
if ~logical(TestingResult.success), error('TestingResult does not represent completely successful fixed test settings.'); end

Xtrain=TrainingResult.X; ytrain=TrainingResult.y(:); Xtest=TestingResult.Xtest;
if ~isnumeric(Xtrain)||isempty(Xtrain)||~ismatrix(Xtrain)||~isreal(Xtrain)||any(~isfinite(Xtrain(:))), error('TrainingResult.X must be a nonempty finite real matrix.'); end
if ~isnumeric(ytrain)||isempty(ytrain)||~isvector(ytrain)||~isreal(ytrain)||any(~isfinite(ytrain)), error('TrainingResult.y must be a nonempty finite real vector.'); end
if ~isnumeric(Xtest)||isempty(Xtest)||~ismatrix(Xtest)||~isreal(Xtest)||any(~isfinite(Xtest(:))), error('TestingResult.Xtest must be a nonempty finite real matrix.'); end
[nTrain,p]=size(Xtrain); [nTest,pTest]=size(Xtest);
if numel(ytrain)~=nTrain||pTest~=p, error('Training/test data dimensions are inconsistent.'); end
if TestingResult.nTrain~=nTrain||TestingResult.nTest~=nTest, error('TestingResult nTrain/nTest are inconsistent.'); end
if numel(TestingResult.point_success)~=nTest||numel(TestingResult.radius)~=nTest||numel(TestingResult.C)~=nTest||numel(TestingResult.reg_param)~=nTest
    error('TestingResult pointwise arrays are inconsistent with nTest.');
end
if ~all(logical(TestingResult.point_success(:))), error('TestingResult contains unsuccessful test points.'); end

kernel=char(TrainingResult.kernel); if ~strcmpi(char(TestingResult.kernel),kernel), error('TrainingResult and TestingResult kernels are inconsistent.'); end
useShape=common.kernelHasShapeParameter(kernel);
if ~isscalar(TrainingResult.useShape)||logical(TrainingResult.useShape)~=useShape||~isscalar(TestingResult.useShape)||logical(TestingResult.useShape)~=useShape
    error('TrainingResult/TestingResult useShape is inconsistent with kernel.');
end
S_global=TrainingResult.S_global;
if ~isnumeric(S_global)||~isscalar(S_global)||~isfinite(S_global)||S_global<=0, error('TrainingResult.S_global must be finite and positive.'); end
if isfield(TestingResult,'S_global') && (~isscalar(TestingResult.S_global)||~isfinite(TestingResult.S_global)||abs(TestingResult.S_global-S_global)>1e-12*max(1,abs(S_global)))
    error('TrainingResult and TestingResult S_global are inconsistent.');
end
minNeighbors=TrainingResult.minNeighbors;
if ~isnumeric(minNeighbors)||~isscalar(minNeighbors)||~isfinite(minNeighbors)||minNeighbors<1||minNeighbors~=round(minNeighbors), error('TrainingResult.minNeighbors is invalid.'); end
if isfield(TestingResult,'minNeighbors')&&isfinite(TestingResult.minNeighbors)&&TestingResult.minNeighbors~=minNeighbors, error('TrainingResult and TestingResult minNeighbors are inconsistent.'); end

radius=TestingResult.radius(:); regParam=TestingResult.reg_param(:);
if any(~isfinite(radius))||any(radius<=0), error('TestingResult.radius must contain finite positive scaled values.'); end
if any(~isfinite(regParam))||any(regParam<=0), error('TestingResult.reg_param must contain finite positive values.'); end
if useShape
    C=TestingResult.C(:); if any(~isfinite(C))||any(C<=0), error('TestingResult.C must contain finite positive values for shaped kernels.'); end
else, C=nan(nTest,1); end

haveStoredNeighborCount=isfield(TestingResult,'neighbor_count')&&isnumeric(TestingResult.neighbor_count)&&numel(TestingResult.neighbor_count)==nTest;
if haveStoredNeighborCount, storedNeighborCount=TestingResult.neighbor_count(:); else, storedNeighborCount=nan(nTest,1); end
haveStoredNeighborIdx=isfield(TestingResult,'neighbor_idx')&&iscell(TestingResult.neighbor_idx)&&numel(TestingResult.neighbor_idx)==nTest;

referenceResidual=nan(nTest,1); referencePrediction=nan(nTest,1); referenceObserved=nan(nTest,1);
if isfield(TestingResult,'ReferenceResidual')&&isnumeric(TestingResult.ReferenceResidual)&&numel(TestingResult.ReferenceResidual)==nTest, referenceResidual=TestingResult.ReferenceResidual(:); end
if isfield(TestingResult,'ReferencePrediction')&&isnumeric(TestingResult.ReferencePrediction)&&numel(TestingResult.ReferencePrediction)==nTest, referencePrediction=TestingResult.ReferencePrediction(:); end
if isfield(TestingResult,'ReferenceObserved')&&isnumeric(TestingResult.ReferenceObserved)&&numel(TestingResult.ReferenceObserved)==nTest, referenceObserved=TestingResult.ReferenceObserved(:); end

if isfield(TrainingResult,'options')&&isstruct(TrainingResult.options), options=TrainingResult.options; else, options=struct(); end
if isfield(options,'execution'), options=rmfield(options,'execution'); end
if ~isfield(options,'solver')||~isstruct(options.solver), options.solver=struct(); end
if ~isfield(options.solver,'cond_lsq')||isempty(options.solver.cond_lsq), options.solver.cond_lsq=1e12; end
if ~isscalar(options.solver.cond_lsq)||~isfinite(options.solver.cond_lsq)||options.solver.cond_lsq<=0, error('options.solver.cond_lsq must be finite and positive.'); end

% Geometry once.
tTotal=tic; tGeometry=tic; geometryReused=~isempty(geometryCache);
if geometryReused
    validateTestGeometryLocal(geometryCache,nTrain,nTest,p);
    D_train_train=geometryCache.D; P_train=geometryCache.P;
    D_query_train=geometryCache.D_query_train; P_query=geometryCache.P_query;
else
    D_train_train=pdist2(Xtrain,Xtrain); P_train=[ones(nTrain,1),Xtrain];
    D_query_train=pdist2(Xtest,Xtrain); P_query=[ones(nTest,1),Xtest];
end
geometryWallTime=toc(tGeometry);

PredictionMean=nan(nTest,1); BootstrapVariance=nan(nTest,1); nValid=zeros(nTest,1); nUsed=nan(nTest,1);
dOriginal=nan(nTest,1); epsilonOriginal=nan(nTest,1); epsilonStored=nan(nTest,1);
epsilonDifference=nan(nTest,1); epsilonConsistent=true(nTest,1); TargetRNG_SEED=nan(nTest,1);
nAttempts=zeros(nTest,1); nRejected=zeros(nTest,1); nDominant=zeros(nTest,1); nRejectedAdmissibility=zeros(nTest,1);
nRejectedSolve=zeros(nTest,1); nRejectedPrediction=zeros(nTest,1); nRBFSolves=zeros(nTest,1);
NumSegmentsEffective=nan(nTest,1); MinPointsFeasible=false(nTest,1); MinPointsSatisfied=false(nTest,1);
SegmentCountsRequested=cell(nTest,1); SegmentCountsEffective=cell(nTest,1);
RadiusEdgesRequestedScaled=cell(nTest,1); RadiusEdgesEffectiveScaled=cell(nTest,1);
RadiusEdgesRequested=cell(nTest,1); RadiusEdgesEffective=cell(nTest,1);
if saveAll, condRegSummaryValues=[]; else, condRegSummaryValues=nan(nTest*nBootstrap,1); end
nCondSummary=0; nBackslash=0; nLsqminnorm=0;
if saveAll
    Predictions=nan(nTest,nBootstrap); dBootstrap=nan(nTest,nBootstrap); epsilonBootstrap=nan(nTest,nBootstrap);
    condReg=nan(nTest,nBootstrap); regPhi=nan(nTest,nBootstrap); solverUsed=cell(nTest,nBootstrap); nUnique=nan(nTest,nBootstrap); neighborIndices=cell(nTest,1);
else
    Predictions=[]; dBootstrap=[]; epsilonBootstrap=[]; condReg=[]; regPhi=[]; solverUsed=[]; nUnique=[]; neighborIndices=[];
end
if storeIndices, BootstrapIndices=cell(nTest,nBootstrap); else, BootstrapIndices=[]; end

for iTest=1:nTest
    if haveStoredNeighborIdx&&~isempty(TestingResult.neighbor_idx{iTest}), idxFit=TestingResult.neighbor_idx{iTest}(:);
    else
        Nbh=localrbf.getNeighborhood(Xtrain,Xtest(iTest,:),radius(iTest),minNeighbors,[],D_query_train(iTest,:).');
        if ~isstruct(Nbh)||~isfield(Nbh,'success')||~Nbh.success, error('Failed to reconstruct fixed neighbourhood at test point %d.',iTest); end
        idxFit=Nbh.idx(:);
    end
    validateTestNeighborhoodLocal(idxFit,nTrain,minNeighbors,iTest);
    if haveStoredNeighborCount&&isfinite(storedNeighborCount(iTest))&&storedNeighborCount(iTest)~=numel(idxFit), error('Reconstructed/saved neighbour count mismatch at test point %d.',iTest); end
    if saveAll, neighborIndices{iTest}=idxFit; end

    Xfit=Xtrain(idxFit,:); yfit=ytrain(idxFit); Xeval=Xtest(iTest,:); m=numel(idxFit); nUsed(iTest)=m;
    if useShape, Ci=C(iTest); [epsilonOriginal(iTest),dOriginal(iTest)]=common.buildEpsilon(Xfit,Ci);
    else, Ci=1; dOriginal(iTest)=NaN; epsilonOriginal(iTest)=1; end
    if isfield(TestingResult,'epsilon')&&isnumeric(TestingResult.epsilon)&&numel(TestingResult.epsilon)==nTest&&isfinite(TestingResult.epsilon(iTest))
        epsilonStored(iTest)=TestingResult.epsilon(iTest); epsilonDifference(iTest)=abs(epsilonOriginal(iTest)-epsilonStored(iTest));
        tolEps=1e-10*max(1,abs(epsilonStored(iTest))); if ~useShape, tolEps=1e-12; end
        epsilonConsistent(iTest)=epsilonDifference(iTest)<=tolEps;
        if ~epsilonConsistent(iTest), error('Recomputed epsilon differs from stored fixed epsilon at test point %d.',iTest); end
    end

    Regi=regParam(iTest); geometry=struct('R',D_train_train(idxFit,idxFit),'P',P_train(idxFit,:), ...
        'R_eval',reshape(D_query_train(iTest,idxFit),1,[]),'P_eval',P_query(iTest,:));
    segments=[];
    if useSpatialBootstrap
        [segments,segmentInfo]=globalrbf.buildAdaptiveRadialSegments(geometry.R_eval,numSpatialSegments,minPointsPerAnnulus);
        NumSegmentsEffective(iTest)=segmentInfo.NumSegmentsEffective; MinPointsFeasible(iTest)=segmentInfo.MinPointsFeasible; MinPointsSatisfied(iTest)=segmentInfo.MinPointsSatisfied;
        SegmentCountsRequested{iTest}=segmentInfo.RequestedCounts; SegmentCountsEffective{iTest}=segmentInfo.EffectiveCounts;
        RadiusEdgesRequestedScaled{iTest}=segmentInfo.RequestedRadiusEdges; RadiusEdgesEffectiveScaled{iTest}=segmentInfo.EffectiveRadiusEdges;
        RadiusEdgesRequested{iTest}=segmentInfo.RequestedRadiusEdges.*S_global; RadiusEdgesEffective{iTest}=segmentInfo.EffectiveRadiusEdges.*S_global;
        targetSeed=mod(double(RNG_SEED)+104729*double(iTest-1),2^32);
    else
        targetSeed=mod(double(RNG_SEED)+double(iTest-1),2^32); % preserve historical ordinary local path
    end
    TargetRNG_SEED(iTest)=targetSeed; stream=RandStream('mt19937ar','Seed',targetSeed);

    % Compact per-target prediction vector used for final statistics.
    predPoint=nan(1,nBootstrap);
    b=0; attempts=0; rejectAdmissibility=0; rejectSolve=0; rejectPrediction=0; rbfSolvesPoint=0;
    while b<nBootstrap&&attempts<maxAttempts
        attempts=attempts+1;
        if useSpatialBootstrap, idxBoot=globalrbf.resampleRadialSegments(segments,stream,m); else, idxBoot=randi(stream,m,[m,1]); end
        [yhat,bootInfo]=common.bootstrapRBFReplicate(Xfit,yfit,Xeval,idxBoot,kernel,Ci,Regi,options,geometry);
        rbfSolvesPoint=rbfSolvesPoint+bootInfo.nRBFSolves;
        if ~bootInfo.success
            switch bootInfo.failureStage
                case 'admissibility', rejectAdmissibility=rejectAdmissibility+1;
                case 'solve', rejectSolve=rejectSolve+1;
                case 'prediction', rejectPrediction=rejectPrediction+1;
                otherwise, error('Unknown bootstrap failure stage "%s".',bootInfo.failureStage);
            end
            continue
        end
        b=b+1;
        predPoint(b)=yhat(1);
        if saveAll
            Predictions(iTest,b)=yhat(1); dBootstrap(iTest,b)=bootInfo.d_boot; epsilonBootstrap(iTest,b)=bootInfo.epsilon;
            condReg(iTest,b)=bootInfo.cond_reg; if isfield(bootInfo,'reg_phi'), regPhi(iTest,b)=bootInfo.reg_phi; end
            solverUsed{iTest,b}=bootInfo.solver_used; nUnique(iTest,b)=bootInfo.nUnique;
        end
        if storeIndices, BootstrapIndices{iTest,b}=idxBoot; end
        if ~saveAll&&isfinite(bootInfo.cond_reg), nCondSummary=nCondSummary+1; condRegSummaryValues(nCondSummary)=bootInfo.cond_reg; end
        if strcmp(bootInfo.solver_used,'backslash'), nBackslash=nBackslash+1; elseif strcmp(bootInfo.solver_used,'lsqminnorm'), nLsqminnorm=nLsqminnorm+1; end
    end
    nAttempts(iTest)=attempts; nRejectedAdmissibility(iTest)=rejectAdmissibility; nRejectedSolve(iTest)=rejectSolve;
    nRejectedPrediction(iTest)=rejectPrediction; nRejected(iTest)=rejectAdmissibility+rejectSolve+rejectPrediction; nRBFSolves(iTest)=rbfSolvesPoint;
    if b<nBootstrap, error('Only %d/%d valid bootstrap realizations were obtained for test point %d after %d attempts.',b,nBootstrap,iTest,attempts); end
    if any(~isfinite(predPoint)), error('Local test bootstrap predictions are incomplete at point %d.',iTest); end

    [predPointStats,nDominant(iTest)] = applyDominanceControlLocal( ...
        predPoint,dominanceLimit,maxDominantFraction);

    PredictionMean(iTest)=mean(predPointStats);
    BootstrapVariance(iTest)=var(predPointStats,0);
    nValid(iTest)=nBootstrap;
    if verbose, fprintf('[local bootstrap testing] point %d/%d: %d accepted, %d rejected, %d RBF solves.\n',iTest,nTest,nBootstrap,nRejected(iTest),nRBFSolves(iTest)); end
end

if saveAll
    if any(~isfinite(Predictions(:))), error('Local testing bootstrap prediction matrix is incomplete or contains NaN/Inf.'); end
end
BootstrapStd=sqrt(BootstrapVariance); if any(nUsed<=1), error('Every local fitting neighbourhood must contain at least two observations.'); end
AdjustmentFactor=nUsed./(nUsed-1); AdjustedVariance=BootstrapVariance.*AdjustmentFactor; AdjustedStd=sqrt(AdjustedVariance);
if saveAll, condVals=condReg(isfinite(condReg)); else, condVals=condRegSummaryValues(1:nCondSummary); end
if isempty(condVals), condRegMin=NaN; condRegMedian=NaN; condRegMax=NaN; else, condRegMin=min(condVals); condRegMedian=median(condVals); condRegMax=max(condVals); end
clear condVals condRegSummaryValues

totalWallTime=toc(tTotal); Bootstrap=struct(); Bootstrap.kernel=kernel; Bootstrap.useShape=useShape;
Bootstrap.S_global=S_global; Bootstrap.minNeighbors=minNeighbors; Bootstrap.Local=struct(); Bootstrap.Local.radius=radius;
if isfield(TestingResult,'radius_raw')&&isnumeric(TestingResult.radius_raw)&&numel(TestingResult.radius_raw)==nTest, Bootstrap.Local.radius_raw=TestingResult.radius_raw(:); else, Bootstrap.Local.radius_raw=radius.*S_global; end
Bootstrap.Local.C=C; Bootstrap.Local.reg_param=regParam; Bootstrap.Local.nUsed=nUsed; Bootstrap.Local.d_original=dOriginal;
Bootstrap.Local.epsilon_original=epsilonOriginal; Bootstrap.Local.epsilon_stored=epsilonStored;
Bootstrap.Local.epsilon_difference=epsilonDifference; Bootstrap.Local.epsilon_consistent=epsilonConsistent;
if saveAll, Bootstrap.Local.neighbor_idx=neighborIndices; end

Bootstrap.Reference=struct(); Bootstrap.Reference.Mode='independent-test'; Bootstrap.Reference.Observed=referenceObserved;
Bootstrap.Reference.Prediction=referencePrediction; Bootstrap.Reference.Residual=referenceResidual;
if saveAll, Bootstrap.Predictions=Predictions; end
Bootstrap.Mean=PredictionMean; Bootstrap.BootstrapVariance=BootstrapVariance; Bootstrap.BootstrapStd=BootstrapStd;
Bootstrap.AdjustedVariance=AdjustedVariance; Bootstrap.AdjustedStd=AdjustedStd; Bootstrap.Variance=AdjustedVariance; Bootstrap.Std=AdjustedStd;
Bootstrap.nUsed=nUsed; Bootstrap.AdjustmentFactor=AdjustmentFactor; Bootstrap.nValid=nValid;

Bootstrap.DominanceControl=struct();
Bootstrap.DominanceControl.DominanceLimit=dominanceLimit;
Bootstrap.DominanceControl.MaxDominantFraction=maxDominantFraction;
Bootstrap.DominanceControl.MaxAllowedDominant=min(floor(maxDominantFraction*nBootstrap),max(0,nBootstrap-2));
Bootstrap.DominanceControl.PointMarked=(nDominant>0);
Bootstrap.DominanceControl.PointIndex=find(nDominant>0);
Bootstrap.DominanceControl.NDominantByPoint=nDominant;
Bootstrap.DominanceControl.nDominantPoints=nnz(nDominant>0);
Bootstrap.DominanceControl.totalDominant=sum(nDominant);
if isempty(nDominant), Bootstrap.DominanceControl.maxDominantAtPoint=0;
else, Bootstrap.DominanceControl.maxDominantAtPoint=max(nDominant); end
Bootstrap.Original=struct('nTrain',nTrain,'nTest',nTest,'nFit',nUsed,'d',dOriginal,'epsilon',epsilonOriginal);
Bootstrap.Diagnostics=struct(); Bootstrap.Diagnostics.Layout='point_by_bootstrap'; Bootstrap.Diagnostics.nBackslash=nBackslash;
Bootstrap.Diagnostics.nLsqminnorm=nLsqminnorm; Bootstrap.Diagnostics.cond_reg_min=condRegMin;
Bootstrap.Diagnostics.cond_reg_median=condRegMedian; Bootstrap.Diagnostics.cond_reg_max=condRegMax;
if saveAll
    Bootstrap.Diagnostics.d_bootstrap=dBootstrap; Bootstrap.Diagnostics.epsilon_bootstrap=epsilonBootstrap;
    Bootstrap.Diagnostics.cond_reg=condReg; Bootstrap.Diagnostics.reg_phi=regPhi;
    Bootstrap.Diagnostics.solver_used=solverUsed; Bootstrap.Diagnostics.nUnique=nUnique;
end
Bootstrap.Computation=struct(); Bootstrap.Computation.nAccepted=nValid; Bootstrap.Computation.nAttempts=nAttempts;
Bootstrap.Computation.nRejected=nRejected; Bootstrap.Computation.nRejectedAdmissibility=nRejectedAdmissibility;
Bootstrap.Computation.nRejectedSolve=nRejectedSolve; Bootstrap.Computation.nRejectedPrediction=nRejectedPrediction;
Bootstrap.Computation.nRBFSolves=nRBFSolves; Bootstrap.Computation.totalAttempts=sum(nAttempts);
Bootstrap.Computation.totalRejected=sum(nRejected); Bootstrap.Computation.totalRBFSolves=sum(nRBFSolves);
Bootstrap.Computation.geometryWallTime=geometryWallTime; Bootstrap.Computation.totalWallTime=totalWallTime; Bootstrap.Computation.geometryReused=geometryReused;
Bootstrap.Settings=struct(); Bootstrap.Settings.nBootstrap=nBootstrap; Bootstrap.Settings.RNG_SEED=RNG_SEED;
Bootstrap.Settings.RNGAlgorithm='mt19937ar'; Bootstrap.Settings.TargetRNG_SEED=TargetRNG_SEED;
Bootstrap.Settings.TargetSeedStride=ternaryLocal(useSpatialBootstrap,104729,1); Bootstrap.Settings.MaxAttemptsFactor=maxAttemptsFactor;
Bootstrap.Settings.SaveAll=saveAll; Bootstrap.Settings.StoreIndices=storeIndices; Bootstrap.Settings.NumSpatialSegments=numSpatialSegments;
Bootstrap.Settings.MinPointsPerAnnulus=minPointsPerAnnulus; Bootstrap.Settings.SpatialBootstrapEnabled=useSpatialBootstrap;
Bootstrap.Settings.DominanceLimit=dominanceLimit;
Bootstrap.Settings.MaxDominantFraction=maxDominantFraction;
Bootstrap.Settings.DominanceControlMethod='sequential_largest_variance_contributor';
if useSpatialBootstrap, Bootstrap.Settings.ResamplingLayout='local_target_specific_annular'; else, Bootstrap.Settings.ResamplingLayout='local_target_specific_ordinary'; end
Bootstrap.Settings.options=options;
if useSpatialBootstrap
    Bootstrap.Spatial=struct(); Bootstrap.Spatial.NumSegmentsRequested=numSpatialSegments; Bootstrap.Spatial.NumSegmentsEffective=NumSegmentsEffective;
    Bootstrap.Spatial.MinPointsPerAnnulus=minPointsPerAnnulus; Bootstrap.Spatial.MinPointsFeasible=MinPointsFeasible; Bootstrap.Spatial.MinPointsSatisfied=MinPointsSatisfied;
    Bootstrap.Spatial.SegmentCountsRequested=SegmentCountsRequested; Bootstrap.Spatial.SegmentCountsEffective=SegmentCountsEffective;
    Bootstrap.Spatial.RadiusEdgesRequestedScaled=RadiusEdgesRequestedScaled; Bootstrap.Spatial.RadiusEdgesEffectiveScaled=RadiusEdgesEffectiveScaled;
    Bootstrap.Spatial.RadiusEdgesRequested=RadiusEdgesRequested; Bootstrap.Spatial.RadiusEdgesEffective=RadiusEdgesEffective;
end
if storeIndices, Bootstrap.BootstrapIndices=BootstrapIndices; end
Bootstrap.success=all(nValid==nBootstrap)&&all(epsilonConsistent); Bootstrap.message='';
if ~Bootstrap.success, Bootstrap.message='Local testing bootstrap did not complete consistently.'; end
end

function [pStats,nDominant]=applyDominanceControlLocal(p,dominanceLimit,maxDominantFraction)
%APPLYDOMINANCECONTROLLOCAL Apply control only to overwhelmingly dominant bootstrap
% realizations in final statistics. Candidate selection is mathematically
% aligned with variance: largest current squared deviation from current mean.
pStats=p(:).';
Boriginal=numel(pStats);
maxDominant=min(floor(maxDominantFraction*Boriginal),max(0,Boriginal-2));
nDominant=0;
while nDominant<maxDominant
    meanP=mean(pStats);
    sqDev=(pStats-meanP).^2;
    totalSS=sum(sqDev);
    if ~isfinite(totalSS)||totalSS<=0, break; end
    [maxSqDev,idxExtreme]=max(sqDev);
    dominance=maxSqDev/totalSS;
    if ~isfinite(dominance)||dominance<=dominanceLimit, break; end
    pStats(idxExtreme)=[];
    nDominant=nDominant+1;
end
end

function O=normalizeBootstrapOptionsLocal(O)
if ~isstruct(O)||~isscalar(O), error('bootstrapOptions must be a scalar structure.'); end
if ~isfield(O,'Verbose')||isempty(O.Verbose), O.Verbose=true; end
if ~isfield(O,'MaxAttemptsFactor')||isempty(O.MaxAttemptsFactor), O.MaxAttemptsFactor=10; end
if ~isfield(O,'SaveAll')||isempty(O.SaveAll), O.SaveAll=false; end
if ~isfield(O,'StoreIndices')||isempty(O.StoreIndices), O.StoreIndices=false; end
if ~isfield(O,'NumSpatialSegments')||isempty(O.NumSpatialSegments), O.NumSpatialSegments=0; end
if ~isfield(O,'MinPointsPerAnnulus')||isempty(O.MinPointsPerAnnulus), O.MinPointsPerAnnulus=5; end
if ~isfield(O,'DominanceLimit')||isempty(O.DominanceLimit), O.DominanceLimit=0.80; end
if ~isfield(O,'MaxDominantFraction')||isempty(O.MaxDominantFraction), O.MaxDominantFraction=0.03; end
O.Verbose=validateLogicalScalar(O.Verbose,'bootstrapOptions.Verbose'); O.SaveAll=validateLogicalScalar(O.SaveAll,'bootstrapOptions.SaveAll');
O.StoreIndices=validateLogicalScalar(O.StoreIndices,'bootstrapOptions.StoreIndices');
if ~isnumeric(O.DominanceLimit)||~isscalar(O.DominanceLimit)||~isfinite(O.DominanceLimit)||O.DominanceLimit<=0||O.DominanceLimit>=1
    error('bootstrapOptions.DominanceLimit must be a finite scalar strictly between 0 and 1.');
end
if ~isnumeric(O.MaxDominantFraction)||~isscalar(O.MaxDominantFraction)||~isfinite(O.MaxDominantFraction)||O.MaxDominantFraction<0||O.MaxDominantFraction>=1
    error('bootstrapOptions.MaxDominantFraction must be a finite scalar in [0,1).');
end
end
function validateBootstrapScalarInputsLocal(B,seed,maxF,nSeg,minPts)
if ~isnumeric(B)||~isscalar(B)||~isreal(B)||~isfinite(B)||B<2||B~=round(B), error('nBootstrap must be an integer >= 2.'); end
if ~isnumeric(seed)||~isscalar(seed)||~isreal(seed)||~isfinite(seed)||seed<0||seed>(2^32-1)||seed~=round(seed), error('RNG_SEED must be a finite nonnegative integer not exceeding 2^32-1.'); end
if ~isnumeric(maxF)||~isscalar(maxF)||~isreal(maxF)||~isfinite(maxF)||maxF<1, error('MaxAttemptsFactor must be a finite scalar >= 1.'); end
if ~isnumeric(nSeg)||~isscalar(nSeg)||~isfinite(nSeg)||nSeg<0||nSeg~=round(nSeg), error('NumSpatialSegments must be a nonnegative integer.'); end
if ~isnumeric(minPts)||~isscalar(minPts)||~isfinite(minPts)||minPts<1||minPts~=round(minPts), error('MinPointsPerAnnulus must be a positive integer.'); end
end
function requireSpatialHelpersLocal()
if exist('globalrbf.buildAdaptiveRadialSegments','file')~=2||exist('globalrbf.resampleRadialSegments','file')~=2
    error(['Spatial local bootstrap uses the same helpers as global bootstrap. ' ...
        'globalrbf.buildAdaptiveRadialSegments and globalrbf.resampleRadialSegments must be on the MATLAB path.']);
end
end
function validateTestGeometryLocal(G,nTrain,nTest,p)
req={'D','P','D_query_train','P_query'}; for k=1:numel(req), if ~isfield(G,req{k}), error('geometryCache.%s is required.',req{k}); end, end
if ~isnumeric(G.D)||~isequal(size(G.D),[nTrain nTrain])||any(~isfinite(G.D(:))), error('geometryCache.D has invalid size or values.'); end
if ~isnumeric(G.P)||~isequal(size(G.P),[nTrain p+1])||any(~isfinite(G.P(:))), error('geometryCache.P has invalid size or values.'); end
if ~isnumeric(G.D_query_train)||~isequal(size(G.D_query_train),[nTest nTrain])||any(~isfinite(G.D_query_train(:))), error('geometryCache.D_query_train has invalid size or values.'); end
if ~isnumeric(G.P_query)||~isequal(size(G.P_query),[nTest p+1])||any(~isfinite(G.P_query(:))), error('geometryCache.P_query has invalid size or values.'); end
end
function validateTestNeighborhoodLocal(idx,nTrain,minN,iTest)
if isempty(idx)||any(~isfinite(idx))||any(idx~=round(idx))||any(idx<1)||any(idx>nTrain)||numel(unique(idx))~=numel(idx), error('Fixed testing neighbourhood is invalid at point %d.',iTest); end
if numel(idx)<minN, error('Fixed testing neighbourhood at point %d contains fewer than minNeighbors.',iTest); end
end
function tf=validateLogicalScalar(value,name)
if ~(isscalar(value)&&(islogical(value)||(isnumeric(value)&&isreal(value)&&isfinite(value)&&any(value==[0 1])))), error('%s must be true/false (or numeric 1/0).',name); end
tf=logical(value);
end
function v=ternaryLocal(tf,a,b), if tf, v=a; else, v=b; end, end

function Bootstrap = localRBFBootstrapForTraining( ...
    TrainingResult, nBootstrap, RNG_SEED, bootstrapOptions, geometryCache)
%LOCALRBFBOOTSTRAPFORTRAINING Bootstrap uncertainty for fixed local-RBF training settings.
%
% IMPORTANT ARCHITECTURE
% ----------------------
% This function is downstream of the main local-RBF pipeline. It NEVER
% optimizes, selects a cap/radius, or transfers settings. For every training
% target it consumes the already-fixed Radius / RegParam / C supplied in
% TrainingResult.Final and bootstraps only the corresponding target-excluded
% local fitting set.
%
% TrainingResult may represent either:
%   bootstrapOptions.TrainingMode = 'original'
%       main final per-training-point settings;
%   bootstrapOptions.TrainingMode = 'pseudo-unseen'
%       upstream pseudo-unseen per-target settings already produced by the
%       main workflow. Bootstrap still performs NO transfer calculation.
%
% Global-bootstrap consistency:
%   - selected settings stay fixed;
%   - B valid bootstrap predictions per target;
%   - rare sequential variance-dominance control before final statistics:
%       flag the current largest variance contributor as dominant only while its
%       contribution exceeds 80%, with the dominant fraction capped at 3% of B;
%   - Eq. (25): sample variance of the retained bootstrap predictions;
%   - Eq. (26): finite-sample adjustment nUsed/(nUsed-1);
%   - ordinary bootstrap when NumSpatialSegments=0;
%   - adaptive target-relative annular bootstrap when NumSpatialSegments>0;
%   - deterministic target RNG streams;
%   - serial bootstrap replicates inside the outer Model x Kernel worker.
%
% Spatial bootstrap requires the same helpers used by the global code:
%   globalrbf.buildAdaptiveRadialSegments
%   globalrbf.resampleRadialSegments
%
% SaveAll=false (default) retains compulsory statistics and compact
% diagnostics only. StoreIndices=false (default) avoids bootstrap-index
% storage. No optimization history or transfer provenance is stored here.

if nargin < 2 || isempty(nBootstrap), nBootstrap = 200; end
if nargin < 3 || isempty(RNG_SEED), RNG_SEED = 1; end
if nargin < 4 || isempty(bootstrapOptions), bootstrapOptions = struct(); end
if nargin < 5, geometryCache = []; end

bootstrapOptions = normalizeBootstrapOptionsLocal(bootstrapOptions);
verbose = bootstrapOptions.Verbose;
saveAll = bootstrapOptions.SaveAll;
storeIndices = bootstrapOptions.StoreIndices;
maxAttemptsFactor = bootstrapOptions.MaxAttemptsFactor;
numSpatialSegments = bootstrapOptions.NumSpatialSegments;
minPointsPerAnnulus = bootstrapOptions.MinPointsPerAnnulus;
trainingMode = bootstrapOptions.TrainingMode;
dominanceLimit = bootstrapOptions.DominanceLimit;
maxDominantFraction = bootstrapOptions.MaxDominantFraction;
useSpatialBootstrap = numSpatialSegments > 0;
validateBootstrapScalarInputsLocal(nBootstrap,RNG_SEED,maxAttemptsFactor, ...
    numSpatialSegments,minPointsPerAnnulus);
if useSpatialBootstrap
    requireSpatialHelpersLocal();
end
maxAttempts = ceil(maxAttemptsFactor*nBootstrap);

% -------------------------------------------------------------------------
% Fixed-setting training contract.
% -------------------------------------------------------------------------
if ~isstruct(TrainingResult) || ~isscalar(TrainingResult)
    error('TrainingResult must be a scalar structure.');
end
required = {'success','kernel','useShape','X','y','S_global','minNeighbors','Final'};
for k = 1:numel(required)
    if ~isfield(TrainingResult,required{k})
        error('TrainingResult is missing required field "%s".',required{k});
    end
end
if ~logical(TrainingResult.success)
    error('TrainingResult does not represent a successful fixed-setting source.');
end
if ~isstruct(TrainingResult.Final) || ~isscalar(TrainingResult.Final) || ...
        ~isfield(TrainingResult.Final,'success') || ~logical(TrainingResult.Final.success)
    error('TrainingResult.Final does not contain successful fixed settings.');
end
Final = TrainingResult.Final;
requiredFinal = {'point_success','radius','C','reg_param'};
for k = 1:numel(requiredFinal)
    if ~isfield(Final,requiredFinal{k})
        error('TrainingResult.Final is missing required field "%s".',requiredFinal{k});
    end
end

X = TrainingResult.X;
y = TrainingResult.y(:);
if ~isnumeric(X) || isempty(X) || ~ismatrix(X) || ~isreal(X) || any(~isfinite(X(:)))
    error('TrainingResult.X must be a nonempty finite real matrix.');
end
if ~isnumeric(y) || isempty(y) || ~isvector(y) || ~isreal(y) || any(~isfinite(y))
    error('TrainingResult.y must be a nonempty finite real vector.');
end
[n,p] = size(X);
if numel(y) ~= n, error('TrainingResult.X and TrainingResult.y are inconsistent.'); end
if numel(Final.point_success) ~= n || numel(Final.radius) ~= n || ...
        numel(Final.C) ~= n || numel(Final.reg_param) ~= n
    error('TrainingResult.Final pointwise arrays are inconsistent with nTrain.');
end
if ~all(logical(Final.point_success(:)))
    error('Fixed training settings are not successful at every training target.');
end

kernel = char(TrainingResult.kernel);
useShape = common.kernelHasShapeParameter(kernel);
if ~isscalar(TrainingResult.useShape) || logical(TrainingResult.useShape) ~= useShape
    error('TrainingResult.useShape is inconsistent with kernel.');
end
S_global = TrainingResult.S_global;
if ~isnumeric(S_global) || ~isscalar(S_global) || ~isreal(S_global) || ...
        ~isfinite(S_global) || S_global <= 0
    error('TrainingResult.S_global must be finite and positive.');
end
minNeighbors = TrainingResult.minNeighbors;
if ~isnumeric(minNeighbors) || ~isscalar(minNeighbors) || ~isfinite(minNeighbors) || ...
        minNeighbors < 1 || minNeighbors ~= round(minNeighbors)
    error('TrainingResult.minNeighbors is invalid.');
end

radius = Final.radius(:);
regParam = Final.reg_param(:);
if any(~isfinite(radius)) || any(radius <= 0)
    error('Final.radius must contain finite positive scaled values.');
end
if any(~isfinite(regParam)) || any(regParam <= 0)
    error('Final.reg_param must contain finite positive values.');
end
if useShape
    C = Final.C(:);
    if any(~isfinite(C)) || any(C <= 0)
        error('Final.C must contain finite positive values for shaped kernels.');
    end
else
    C = nan(n,1);
end

haveStoredNeighborCount = isfield(Final,'neighbor_count') && ...
    isnumeric(Final.neighbor_count) && numel(Final.neighbor_count)==n;
if haveStoredNeighborCount, storedNeighborCount = Final.neighbor_count(:); else, storedNeighborCount = nan(n,1); end
haveStoredNeighborIdx = isfield(Final,'neighbor_idx') && ...
    iscell(Final.neighbor_idx) && numel(Final.neighbor_idx)==n;

% Reference deterministic residual used later for variance calibration.
referenceResidual = nan(n,1);
if isfield(TrainingResult,'ReferenceResidual') && isnumeric(TrainingResult.ReferenceResidual) && ...
        numel(TrainingResult.ReferenceResidual)==n
    referenceResidual = TrainingResult.ReferenceResidual(:);
end
referencePrediction = nan(n,1);
if isfield(TrainingResult,'ReferencePrediction') && isnumeric(TrainingResult.ReferencePrediction) && ...
        numel(TrainingResult.ReferencePrediction)==n
    referencePrediction = TrainingResult.ReferencePrediction(:);
end
referenceObserved = y;

% Numerical solver options.
if isfield(TrainingResult,'options') && isstruct(TrainingResult.options), options = TrainingResult.options; else, options = struct(); end
if isfield(options,'execution'), options = rmfield(options,'execution'); end
if ~isfield(options,'solver') || ~isstruct(options.solver), options.solver = struct(); end
if ~isfield(options.solver,'cond_lsq') || isempty(options.solver.cond_lsq), options.solver.cond_lsq = 1e12; end
if ~isscalar(options.solver.cond_lsq) || ~isfinite(options.solver.cond_lsq) || options.solver.cond_lsq <= 0
    error('options.solver.cond_lsq must be finite and positive.');
end

% Shared geometry once.
tTotal = tic;
tGeometry = tic;
geometryReused = ~isempty(geometryCache);
if geometryReused
    validateTrainingGeometryLocal(geometryCache,n,p);
    D = geometryCache.D; P = geometryCache.P;
else
    D = pdist2(X,X); P = [ones(n,1),X];
end
geometryWallTime = toc(tGeometry);

% Compulsory outputs.
PredictionMean = nan(n,1); BootstrapVariance = nan(n,1); nValid = zeros(n,1);
nUsed = nan(n,1); dOriginal = nan(n,1); epsilonOriginal = nan(n,1);
epsilonStored = nan(n,1); epsilonDifference = nan(n,1); epsilonConsistent = true(n,1);
TargetRNG_SEED = nan(n,1); nAttempts = zeros(n,1); nRejected = zeros(n,1);
nDominant = zeros(n,1);
nRejectedAdmissibility = zeros(n,1); nRejectedSolve = zeros(n,1);
nRejectedPrediction = zeros(n,1); nRBFSolves = zeros(n,1);

% Spatial metadata follows the global-bootstrap naming.
NumSegmentsEffective = nan(n,1); MinPointsFeasible = false(n,1); MinPointsSatisfied = false(n,1);
SegmentCountsRequested = cell(n,1); SegmentCountsEffective = cell(n,1);
RadiusEdgesRequestedScaled = cell(n,1); RadiusEdgesEffectiveScaled = cell(n,1);
RadiusEdgesRequested = cell(n,1); RadiusEdgesEffective = cell(n,1);

if saveAll, condRegSummaryValues = []; else, condRegSummaryValues = nan(n*nBootstrap,1); end
nCondSummary = 0; nBackslash = 0; nLsqminnorm = 0;

if saveAll
    Predictions = nan(n,nBootstrap); dBootstrap = nan(n,nBootstrap);
    epsilonBootstrap = nan(n,nBootstrap); condReg = nan(n,nBootstrap);
    regPhi = nan(n,nBootstrap); solverUsed = cell(n,nBootstrap); nUnique = nan(n,nBootstrap);
    neighborIndices = cell(n,1);
else
    Predictions = []; dBootstrap = []; epsilonBootstrap = []; condReg = [];
    regPhi = []; solverUsed = []; nUnique = []; neighborIndices = [];
end
if storeIndices, BootstrapIndices = cell(n,nBootstrap); else, BootstrapIndices = []; end

for iTarget = 1:n
    if haveStoredNeighborIdx && ~isempty(Final.neighbor_idx{iTarget})
        idxFit = Final.neighbor_idx{iTarget}(:);
    else
        Nbh = localrbf.getNeighborhood(X,X(iTarget,:),radius(iTarget),minNeighbors,iTarget,D(iTarget,:).');
        if ~isstruct(Nbh) || ~isfield(Nbh,'success') || ~Nbh.success
            error('Failed to reconstruct fixed neighbourhood at training point %d.',iTarget);
        end
        idxFit = Nbh.idx(:);
    end
    validateTrainingNeighborhoodLocal(idxFit,n,minNeighbors,iTarget);
    if haveStoredNeighborCount && isfinite(storedNeighborCount(iTarget)) && storedNeighborCount(iTarget) ~= numel(idxFit)
        error('Reconstructed/saved neighbour count mismatch at training point %d.',iTarget);
    end
    if saveAll, neighborIndices{iTarget} = idxFit; end

    Xfit = X(idxFit,:); yfit = y(idxFit); Xeval = X(iTarget,:); m = numel(idxFit); nUsed(iTarget) = m;
    if useShape
        Ci = C(iTarget); [epsilonOriginal(iTarget),dOriginal(iTarget)] = common.buildEpsilon(Xfit,Ci);
    else
        Ci = 1; dOriginal(iTarget)=NaN; epsilonOriginal(iTarget)=1;
    end
    if isfield(Final,'epsilon') && isnumeric(Final.epsilon) && numel(Final.epsilon)==n && isfinite(Final.epsilon(iTarget))
        epsilonStored(iTarget)=Final.epsilon(iTarget);
        epsilonDifference(iTarget)=abs(epsilonOriginal(iTarget)-epsilonStored(iTarget));
        tolEps=1e-10*max(1,abs(epsilonStored(iTarget))); if ~useShape, tolEps=1e-12; end
        epsilonConsistent(iTarget)=epsilonDifference(iTarget)<=tolEps;
        if ~epsilonConsistent(iTarget), error('Recomputed epsilon differs from stored fixed epsilon at training point %d.',iTarget); end
    end

    Regi = regParam(iTarget);
    geometry = struct('R',D(idxFit,idxFit),'P',P(idxFit,:), ...
        'R_eval',reshape(D(iTarget,idxFit),1,[]),'P_eval',P(iTarget,:));

    % Same spatial segmentation convention as global bootstrap, but applied
    % inside this target's already-fixed local neighbourhood.
    segments = [];
    if useSpatialBootstrap
        [segments,segmentInfo] = globalrbf.buildAdaptiveRadialSegments( ...
            geometry.R_eval,numSpatialSegments,minPointsPerAnnulus);
        NumSegmentsEffective(iTarget)=segmentInfo.NumSegmentsEffective;
        MinPointsFeasible(iTarget)=segmentInfo.MinPointsFeasible;
        MinPointsSatisfied(iTarget)=segmentInfo.MinPointsSatisfied;
        SegmentCountsRequested{iTarget}=segmentInfo.RequestedCounts;
        SegmentCountsEffective{iTarget}=segmentInfo.EffectiveCounts;
        RadiusEdgesRequestedScaled{iTarget}=segmentInfo.RequestedRadiusEdges;
        RadiusEdgesEffectiveScaled{iTarget}=segmentInfo.EffectiveRadiusEdges;
        RadiusEdgesRequested{iTarget}=segmentInfo.RequestedRadiusEdges.*S_global;
        RadiusEdgesEffective{iTarget}=segmentInfo.EffectiveRadiusEdges.*S_global;
        targetSeed=mod(double(RNG_SEED)+104729*double(iTarget-1),2^32);
    else
        targetSeed=mod(double(RNG_SEED)+double(iTarget-1),2^32); % preserve historical ordinary path
    end
    TargetRNG_SEED(iTarget)=targetSeed;
    stream=RandStream('mt19937ar','Seed',targetSeed);

    % Keep one compact prediction vector for final statistics regardless of
    % SaveAll. SaveAll additionally stores the complete raw prediction matrix.
    predPoint=nan(1,nBootstrap);
    b=0; attempts=0; rejectAdmissibility=0; rejectSolve=0; rejectPrediction=0; rbfSolvesPoint=0;
    while b<nBootstrap && attempts<maxAttempts
        attempts=attempts+1;
        if useSpatialBootstrap
            idxBoot=globalrbf.resampleRadialSegments(segments,stream,m);
        else
            idxBoot=randi(stream,m,[m,1]);
        end
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
            Predictions(iTarget,b)=yhat(1); dBootstrap(iTarget,b)=bootInfo.d_boot;
            epsilonBootstrap(iTarget,b)=bootInfo.epsilon; condReg(iTarget,b)=bootInfo.cond_reg;
            if isfield(bootInfo,'reg_phi'), regPhi(iTarget,b)=bootInfo.reg_phi; end
            solverUsed{iTarget,b}=bootInfo.solver_used; nUnique(iTarget,b)=bootInfo.nUnique;
        end
        if storeIndices, BootstrapIndices{iTarget,b}=idxBoot; end
        if ~saveAll && isfinite(bootInfo.cond_reg)
            nCondSummary=nCondSummary+1; condRegSummaryValues(nCondSummary)=bootInfo.cond_reg;
        end
        if strcmp(bootInfo.solver_used,'backslash'), nBackslash=nBackslash+1;
        elseif strcmp(bootInfo.solver_used,'lsqminnorm'), nLsqminnorm=nLsqminnorm+1; end
    end

    nAttempts(iTarget)=attempts; nRejectedAdmissibility(iTarget)=rejectAdmissibility;
    nRejectedSolve(iTarget)=rejectSolve; nRejectedPrediction(iTarget)=rejectPrediction;
    nRejected(iTarget)=rejectAdmissibility+rejectSolve+rejectPrediction; nRBFSolves(iTarget)=rbfSolvesPoint;
    if b<nBootstrap
        error('Only %d/%d valid bootstrap realizations were obtained for training point %d after %d attempts.',b,nBootstrap,iTarget,attempts);
    end
    if any(~isfinite(predPoint)), error('Local training bootstrap predictions are incomplete at point %d.',iTarget); end

    % Sequential dominance control. This acts only on the statistics vector;
    % successfully generated bootstrap predictions remain accepted and are
    % still preserved in Predictions when SaveAll=true.
    [predPointStats,nDominant(iTarget)] = applyDominanceControlLocal( ...
        predPoint,dominanceLimit,maxDominantFraction);

    PredictionMean(iTarget)=mean(predPointStats);
    BootstrapVariance(iTarget)=var(predPointStats,0);
    nValid(iTarget)=nBootstrap;
    if verbose
        fprintf('[local bootstrap training:%s] point %d/%d: %d accepted, %d rejected, %d RBF solves.\n', ...
            trainingMode,iTarget,n,nBootstrap,nRejected(iTarget),nRBFSolves(iTarget));
    end
end

if saveAll
    if any(~isfinite(Predictions(:))), error('Local training bootstrap prediction matrix is incomplete or contains NaN/Inf.'); end
end
BootstrapStd=sqrt(BootstrapVariance);
if any(nUsed<=1), error('Every local fitting neighbourhood must contain at least two observations.'); end
AdjustmentFactor=nUsed./(nUsed-1); AdjustedVariance=BootstrapVariance.*AdjustmentFactor; AdjustedStd=sqrt(AdjustedVariance);

if saveAll, condVals=condReg(isfinite(condReg)); else, condVals=condRegSummaryValues(1:nCondSummary); end
if isempty(condVals), condRegMin=NaN; condRegMedian=NaN; condRegMax=NaN;
else, condRegMin=min(condVals); condRegMedian=median(condVals); condRegMax=max(condVals); end
clear condVals condRegSummaryValues

totalWallTime=toc(tTotal);
Bootstrap=struct(); Bootstrap.kernel=kernel; Bootstrap.useShape=useShape;
Bootstrap.S_global=S_global; Bootstrap.minNeighbors=minNeighbors;
Bootstrap.Local=struct(); Bootstrap.Local.radius=radius;
if isfield(Final,'radius_raw') && isnumeric(Final.radius_raw) && numel(Final.radius_raw)==n
    Bootstrap.Local.radius_raw=Final.radius_raw(:);
else
    Bootstrap.Local.radius_raw=radius.*S_global;
end
Bootstrap.Local.C=C; Bootstrap.Local.reg_param=regParam; Bootstrap.Local.nUsed=nUsed;
Bootstrap.Local.d_original=dOriginal; Bootstrap.Local.epsilon_original=epsilonOriginal;
Bootstrap.Local.epsilon_stored=epsilonStored; Bootstrap.Local.epsilon_difference=epsilonDifference;
Bootstrap.Local.epsilon_consistent=epsilonConsistent;
if saveAll, Bootstrap.Local.neighbor_idx=neighborIndices; end

Bootstrap.Reference=struct(); Bootstrap.Reference.Mode=trainingMode;
Bootstrap.Reference.Observed=referenceObserved; Bootstrap.Reference.Prediction=referencePrediction;
Bootstrap.Reference.Residual=referenceResidual;
if isfield(TrainingResult,'ReferenceNumerical') && isstruct(TrainingResult.ReferenceNumerical)
    Bootstrap.Reference.Numerical=TrainingResult.ReferenceNumerical;
end

if saveAll, Bootstrap.Predictions=Predictions; end
Bootstrap.Mean=PredictionMean; Bootstrap.BootstrapVariance=BootstrapVariance; Bootstrap.BootstrapStd=BootstrapStd;
Bootstrap.AdjustedVariance=AdjustedVariance; Bootstrap.AdjustedStd=AdjustedStd;
Bootstrap.Variance=AdjustedVariance; Bootstrap.Std=AdjustedStd;
Bootstrap.nUsed=nUsed; Bootstrap.AdjustmentFactor=AdjustmentFactor; Bootstrap.nValid=nValid;

% Mark exactly which target points used the dominance safeguard and how
% many successfully generated bootstrap predictions were classified as
% dominant for final mean/variance at each marked point.
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

Bootstrap.Original=struct(); Bootstrap.Original.nFit=nUsed; Bootstrap.Original.d=dOriginal; Bootstrap.Original.epsilon=epsilonOriginal;

Bootstrap.Diagnostics=struct(); Bootstrap.Diagnostics.Layout='point_by_bootstrap';
Bootstrap.Diagnostics.nBackslash=nBackslash; Bootstrap.Diagnostics.nLsqminnorm=nLsqminnorm;
Bootstrap.Diagnostics.cond_reg_min=condRegMin; Bootstrap.Diagnostics.cond_reg_median=condRegMedian; Bootstrap.Diagnostics.cond_reg_max=condRegMax;
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
Bootstrap.Computation.geometryWallTime=geometryWallTime; Bootstrap.Computation.totalWallTime=totalWallTime;
Bootstrap.Computation.geometryReused=geometryReused;

Bootstrap.Settings=struct(); Bootstrap.Settings.nBootstrap=nBootstrap; Bootstrap.Settings.RNG_SEED=RNG_SEED;
Bootstrap.Settings.RNGAlgorithm='mt19937ar'; Bootstrap.Settings.TargetRNG_SEED=TargetRNG_SEED;
Bootstrap.Settings.TargetSeedStride=ternaryLocal(useSpatialBootstrap,104729,1);
Bootstrap.Settings.MaxAttemptsFactor=maxAttemptsFactor; Bootstrap.Settings.SaveAll=saveAll;
Bootstrap.Settings.StoreIndices=storeIndices; Bootstrap.Settings.NumSpatialSegments=numSpatialSegments;
Bootstrap.Settings.MinPointsPerAnnulus=minPointsPerAnnulus; Bootstrap.Settings.SpatialBootstrapEnabled=useSpatialBootstrap;
Bootstrap.Settings.DominanceLimit=dominanceLimit;
Bootstrap.Settings.MaxDominantFraction=maxDominantFraction;
Bootstrap.Settings.DominanceControlMethod='sequential_largest_variance_contributor';
Bootstrap.Settings.TrainingMode=trainingMode;
if useSpatialBootstrap, Bootstrap.Settings.ResamplingLayout='local_target_specific_annular';
else, Bootstrap.Settings.ResamplingLayout='local_target_specific_ordinary'; end
Bootstrap.Settings.options=options;
if useSpatialBootstrap
    Bootstrap.Spatial=struct(); Bootstrap.Spatial.NumSegmentsRequested=numSpatialSegments;
    Bootstrap.Spatial.NumSegmentsEffective=NumSegmentsEffective; Bootstrap.Spatial.MinPointsPerAnnulus=minPointsPerAnnulus;
    Bootstrap.Spatial.MinPointsFeasible=MinPointsFeasible; Bootstrap.Spatial.MinPointsSatisfied=MinPointsSatisfied;
    Bootstrap.Spatial.SegmentCountsRequested=SegmentCountsRequested; Bootstrap.Spatial.SegmentCountsEffective=SegmentCountsEffective;
    Bootstrap.Spatial.RadiusEdgesRequestedScaled=RadiusEdgesRequestedScaled; Bootstrap.Spatial.RadiusEdgesEffectiveScaled=RadiusEdgesEffectiveScaled;
    Bootstrap.Spatial.RadiusEdgesRequested=RadiusEdgesRequested; Bootstrap.Spatial.RadiusEdgesEffective=RadiusEdgesEffective;
end
if storeIndices, Bootstrap.BootstrapIndices=BootstrapIndices; end
Bootstrap.success=all(nValid==nBootstrap) && all(epsilonConsistent);
Bootstrap.message=''; if ~Bootstrap.success, Bootstrap.message='Local training bootstrap did not complete consistently.'; end
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
if ~isfield(O,'TrainingMode')||isempty(O.TrainingMode), O.TrainingMode='original'; end
if ~isfield(O,'DominanceLimit')||isempty(O.DominanceLimit), O.DominanceLimit=0.80; end
if ~isfield(O,'MaxDominantFraction')||isempty(O.MaxDominantFraction), O.MaxDominantFraction=0.03; end
O.Verbose=validateLogicalScalar(O.Verbose,'bootstrapOptions.Verbose');
O.SaveAll=validateLogicalScalar(O.SaveAll,'bootstrapOptions.SaveAll');
O.StoreIndices=validateLogicalScalar(O.StoreIndices,'bootstrapOptions.StoreIndices');
if ~isnumeric(O.DominanceLimit)||~isscalar(O.DominanceLimit)||~isfinite(O.DominanceLimit)||O.DominanceLimit<=0||O.DominanceLimit>=1
    error('bootstrapOptions.DominanceLimit must be a finite scalar strictly between 0 and 1.');
end
if ~isnumeric(O.MaxDominantFraction)||~isscalar(O.MaxDominantFraction)||~isfinite(O.MaxDominantFraction)||O.MaxDominantFraction<0||O.MaxDominantFraction>=1
    error('bootstrapOptions.MaxDominantFraction must be a finite scalar in [0,1).');
end
mode=lower(strtrim(char(string(O.TrainingMode))));
if ~any(strcmp(mode,{'original','pseudo-unseen'})), error('bootstrapOptions.TrainingMode must be ''original'' or ''pseudo-unseen''.'); end
O.TrainingMode=mode;
end
function validateBootstrapScalarInputsLocal(B,seed,maxF,nSeg,minPts)
if ~isnumeric(B)||~isscalar(B)||~isreal(B)||~isfinite(B)||B<2||B~=round(B), error('nBootstrap must be an integer >= 2.'); end
if ~isnumeric(seed)||~isscalar(seed)||~isreal(seed)||~isfinite(seed)||seed<0||seed>(2^32-1)||seed~=round(seed), error('RNG_SEED must be a finite nonnegative integer not exceeding 2^32-1.'); end
if ~isnumeric(maxF)||~isscalar(maxF)||~isreal(maxF)||~isfinite(maxF)||maxF<1, error('MaxAttemptsFactor must be a finite scalar >= 1.'); end
if ~isnumeric(nSeg)||~isscalar(nSeg)||~isfinite(nSeg)||nSeg<0||nSeg~=round(nSeg), error('NumSpatialSegments must be a nonnegative integer.'); end
if ~isnumeric(minPts)||~isscalar(minPts)||~isfinite(minPts)||minPts<1||minPts~=round(minPts), error('MinPointsPerAnnulus must be a positive integer.'); end
end
function requireSpatialHelpersLocal()
if exist('globalrbf.buildAdaptiveRadialSegments','file')~=2 || exist('globalrbf.resampleRadialSegments','file')~=2
    error(['Spatial local bootstrap uses the same helpers as global bootstrap. ' ...
        'globalrbf.buildAdaptiveRadialSegments and globalrbf.resampleRadialSegments must be on the MATLAB path.']);
end
end
function validateTrainingGeometryLocal(G,n,p)
if ~isstruct(G)||~isfield(G,'D')||~isfield(G,'P'), error('geometryCache must contain D and P.'); end
if ~isnumeric(G.D)||~isequal(size(G.D),[n n])||any(~isfinite(G.D(:))), error('geometryCache.D has invalid size or values.'); end
if ~isnumeric(G.P)||~isequal(size(G.P),[n p+1])||any(~isfinite(G.P(:))), error('geometryCache.P has invalid size or values.'); end
end
function validateTrainingNeighborhoodLocal(idx,n,minN,iTarget)
if isempty(idx)||any(~isfinite(idx))||any(idx~=round(idx))||any(idx<1)||any(idx>n)||numel(unique(idx))~=numel(idx)||any(idx==iTarget)
    error('Fixed training neighbourhood is invalid at point %d.',iTarget);
end
if numel(idx)<minN, error('Fixed training neighbourhood at point %d contains fewer than minNeighbors.',iTarget); end
end
function tf=validateLogicalScalar(value,name)
if ~(isscalar(value)&&(islogical(value)||(isnumeric(value)&&isreal(value)&&isfinite(value)&&any(value==[0 1])))), error('%s must be true/false (or numeric 1/0).',name); end
tf=logical(value);
end
function v=ternaryLocal(tf,a,b), if tf, v=a; else, v=b; end, end

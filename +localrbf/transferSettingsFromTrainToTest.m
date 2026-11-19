function Transfer = transferSettingsFromTrainToTest( ...
    Xtrain, Xquery, FinalSettings, kernel, ...
    near_threshold, idw_radius, D_query_train, retainDetail, ...
    transferMode, nIDWSources, parameterTransferMode)
%TRANSFERSETTINGSFROMTRAINTOTEST
%
% Transfer trained point-specific local-RBF settings from training centres
% to query points.
%
% Transfer modes:
%   'hybrid'  : nearest training point when
%               nearestDistance <= near_threshold; otherwise use the
%               N nearest training points with normalized
%               inverse-square-distance weights.
%   'idw'     : always use the N nearest training points with normalized
%               inverse-square-distance weights.
%   'nearest' : always use the nearest training point.
%   'ensemble': nearest training point when
%               nearestDistance <= near_threshold; otherwise use the
%               N nearest training points with normalized inverse-square
%               distance weights, but DO NOT aggregate Radius/RegParam/C.
%               Each unique complete donor tuple is retained for separate
%               RBF prediction; the final prediction must be combined in
%               LocalRBFTesting using the returned ensemble weights.
%
% nIDWSources controls N for IDW-based and ensemble transfer modes.
%
% For IDW/hybrid, Radius is transferred by weighted arithmetic mean.
% For ensemble, Radius/RegParam/C are not aggregated.
%
% RegParam and C aggregation can be selected independently through
% parameterTransferMode:
%
%   'arithmetic_arithmetic' : RegParam arithmetic, C arithmetic
%   'arithmetic_log'        : RegParam arithmetic, C log10-geometric
%   'log_arithmetic'        : RegParam log10-geometric, C arithmetic
%   'log_log'               : RegParam log10-geometric, C log10-geometric
%
% Short aliases are also accepted:
%   'arith_arith', 'arith_log', 'log_arith', 'log_log'.
% Reference-method aliases from Check_Transfer_Sensitivity_LOO_All:
%   M21 -> arithmetic_arithmetic
%   M22 -> log_arithmetic
%   M23 -> arithmetic_log
%   M24 -> log_log
%
% For shape-free kernels only the RegParam component is used. Direct
% nearest-point transfer is exact and is independent of this option.
%
% IMPORTANT FOR ENSEMBLE:
%   parameterTransferMode is completely inactive. Radius, RegParam and C
%   are not averaged. Complete donor tuples are preserved and predictions
%   are combined later at the prediction level.
%
% Backward-compatible default: parameterTransferMode = 'log_log'.
%
% Epsilon is not transferred for shaped kernels and must later be
% recomputed from the query neighbourhood.
%
% Optional output-retention control:
%   retainDetail = true  (default): preserve the historical complete
%       transfer output, including source indices/distances/weights/source
%       settings and per-query consistency arrays.
%   retainDetail = false: return only the compact transfer quantities needed
%       by the Stage-2 compulsory testing path. Detail-only cell arrays and
%       consistency-history arrays are never allocated.
%
% IMPORTANT:
%   - retainDetail changes retained output only, never the transfer rule;
%   - transferMode changes only the intended source-selection strategy;
%   - IDW aggregation itself is unchanged across 'hybrid' and 'idw';
%   - ensemble uses the same N-nearest/inverse-square source weights but
%     does not aggregate Radius/RegParam/C;
%   - idw_radius is retained for backward compatibility/metadata only and
%     is not used for source selection;
%   - D_query_train is never duplicated in Transfer.
%
% Backward-compatible defaults:
%   retainDetail = true
%   transferMode = 'hybrid'
%   nIDWSources  = 7
%   parameterTransferMode = 'log_log'

% =============================================================
% 1. Optional transfer/detail controls
% =============================================================

if nargin < 8 || isempty(retainDetail)
    retainDetail = true;
end

if ~(isscalar(retainDetail) && ...
     (islogical(retainDetail) || ...
      (isnumeric(retainDetail) && isreal(retainDetail) && ...
       isfinite(retainDetail) && any(retainDetail == [0 1]))))
    error('retainDetail must be true/false (or numeric 1/0).');
end

retainDetail = logical(retainDetail);

if nargin < 9 || isempty(transferMode)
    transferMode = 'hybrid';
end

if ~(ischar(transferMode) || ...
     (isstring(transferMode) && isscalar(transferMode)))
    error(['transferMode must be ''hybrid'', ''idw'', ''nearest'', ' ...
           'or ''ensemble''.']);
end

transferMode = lower(strtrim(char(transferMode)));

if ~any(strcmp(transferMode,{'hybrid','idw','nearest','ensemble'}))
    error(['transferMode must be ''hybrid'', ''idw'', ''nearest'', ' ...
           'or ''ensemble''.']);
end

if nargin < 10 || isempty(nIDWSources)
    nIDWSources = 7;
end

if ~isnumeric(nIDWSources) || ~isscalar(nIDWSources) || ...
   ~isreal(nIDWSources) || ~isfinite(nIDWSources) || ...
   nIDWSources < 1 || nIDWSources ~= round(nIDWSources)
    error('nIDWSources must be a positive integer scalar.');
end

nIDWSourcesRequested = double(nIDWSources);


% =============================================================
% 1.1 RegParam/C aggregation mode
% =============================================================

if strcmp(transferMode,'ensemble')

    % Ensemble combines predictions, not transferred parameters.
    % Therefore ParameterTransferMode is intentionally inactive.
    parameterTransferMode = ...
        'not-applicable';

    regParamTransferScale = ...
        'not-applicable';

    CTransferScale = ...
        'not-applicable';

else

    if nargin < 11 || isempty(parameterTransferMode)

        parameterTransferMode = ...
            'log_log';

    end


    [ ...
        parameterTransferMode, ...
        regParamTransferScale, ...
        CTransferScale ...
    ] = ...
        normalizeParameterTransferMode( ...
            parameterTransferMode);

end


% =============================================================
% 2. Input checks
% =============================================================

if ~isnumeric(Xtrain) || isempty(Xtrain) || ~ismatrix(Xtrain) || ...
   ~isreal(Xtrain) || any(~isfinite(Xtrain(:)))
    error('Xtrain must be a nonempty finite real numeric matrix.');
end

if ~isnumeric(Xquery) || isempty(Xquery) || ~ismatrix(Xquery) || ...
   ~isreal(Xquery) || any(~isfinite(Xquery(:)))
    error('Xquery must be a nonempty finite real numeric matrix.');
end

[nTrain,p] = size(Xtrain);
[nQuery,pQuery] = size(Xquery);

if pQuery ~= p
    error('Xtrain and Xquery must have the same coordinate dimension.');
end

if ~isstruct(FinalSettings)
    error('FinalSettings must be a structure.');
end

if isfield(FinalSettings,'success') && ~FinalSettings.success
    error('FinalSettings does not contain a successful final model.');
end

% =============================================================
% 3. Transfer-distance parameters
% =============================================================

if ~isnumeric(near_threshold) || ~isscalar(near_threshold) || ...
   ~isreal(near_threshold) || ~isfinite(near_threshold) || ...
   near_threshold <= 0
    error('near_threshold must be a finite positive real scalar.');
end

if ~isnumeric(idw_radius) || ~isscalar(idw_radius) || ...
   ~isreal(idw_radius) || ~isfinite(idw_radius) || idw_radius <= 0
    error('idw_radius must be a finite positive real scalar.');
end


% =============================================================
% 4. Kernel and trained settings
% =============================================================

useShape = common.kernelHasShapeParameter(kernel);

requiredFields = {'radius','reg_param'};
for q = 1:numel(requiredFields)
    if ~isfield(FinalSettings,requiredFields{q})
        error('FinalSettings is missing required field "%s".', ...
            requiredFields{q});
    end
end

if useShape && ~isfield(FinalSettings,'C')
    error('FinalSettings.C is required for shaped kernels.');
end

radiusTrain = FinalSettings.radius(:);
regTrain = FinalSettings.reg_param(:);

if numel(radiusTrain) ~= nTrain
    error(['FinalSettings.radius must contain one value for ' ...
           'every training point.']);
end
if numel(regTrain) ~= nTrain
    error(['FinalSettings.reg_param must contain one value for ' ...
           'every training point.']);
end
if ~isnumeric(radiusTrain) || ~isreal(radiusTrain) || ...
   any(~isfinite(radiusTrain)) || any(radiusTrain <= 0)
    error(['FinalSettings.radius must contain finite positive ' ...
           'real values.']);
end
if ~isnumeric(regTrain) || ~isreal(regTrain) || ...
   any(~isfinite(regTrain)) || any(regTrain <= 0)
    error(['FinalSettings.reg_param must contain finite positive ' ...
           'real values.']);
end

if useShape
    CTrain = FinalSettings.C(:);
    if numel(CTrain) ~= nTrain
        error(['FinalSettings.C must contain one value for ' ...
               'every training point.']);
    end
    if ~isnumeric(CTrain) || ~isreal(CTrain) || ...
       any(~isfinite(CTrain)) || any(CTrain <= 0)
        error(['FinalSettings.C must contain finite positive ' ...
               'real values for shaped kernels.']);
    end
else
    CTrain = nan(nTrain,1);
end

% Validate optional radiusIndex in both modes so the storage switch cannot
% change input-validity semantics.
hasRadiusIndex = isfield(FinalSettings,'radiusIndex');
if hasRadiusIndex
    radiusIndexTrain = FinalSettings.radiusIndex(:);
    if numel(radiusIndexTrain) ~= nTrain || ...
       ~isnumeric(radiusIndexTrain) || ~isreal(radiusIndexTrain) || ...
       any(~isfinite(radiusIndexTrain)) || ...
       any(radiusIndexTrain ~= round(radiusIndexTrain)) || ...
       any(radiusIndexTrain < 1)
        error('FinalSettings.radiusIndex contains invalid indices.');
    end
else
    radiusIndexTrain = nan(nTrain,1);
end

% =============================================================
% 5. Query-to-training distances
% =============================================================

distanceMatrixProvided = nargin >= 7 && ~isempty(D_query_train);

if distanceMatrixProvided
    if ~isnumeric(D_query_train) || ~isreal(D_query_train) || ...
       ~isequal(size(D_query_train),[nQuery nTrain]) || ...
       any(~isfinite(D_query_train(:))) || any(D_query_train(:) < 0)
        error(['D_query_train must be an nQuery-by-nTrain ' ...
               'finite nonnegative real matrix.']);
    end
else
    D_query_train = pdist2(Xquery,Xtrain);
end

[nearestDistance,nearestTrainingIndex] = ...
    min(D_query_train,[],2);

% =============================================================
% 6. Compact compulsory working/output arrays
% =============================================================

radiusQuery = nan(nQuery,1);
regQuery = nan(nQuery,1);
CQuery = nan(nQuery,1);

if useShape
    epsilonQuery = nan(nQuery,1);
else
    epsilonQuery = ones(nQuery,1);
end

querySuccess = false(nQuery,1);
transferMethod = cell(nQuery,1);
messageByQuery = cell(nQuery,1);

% nTransferSources is needed for historical full output and for the source
% consistency calculation. It is small, but is only retained in output when
% detail is requested.
nTransferSources = zeros(nQuery,1);

% Full source-history containers are never allocated in compact mode.
if retainDetail
    sourceTrainingIndex = cell(nQuery,1);
    sourceDistance = cell(nQuery,1);
    sourceWeight = cell(nQuery,1);
    sourceRadius = cell(nQuery,1);
    sourceRegParam = cell(nQuery,1);
    sourceC = cell(nQuery,1);
    sourceRadiusIndex = cell(nQuery,1);
end


% Ensemble requires donor tuples/weights even in compact mode because
% LocalRBFTesting must solve each unique tuple separately and combine the
% resulting predictions. These arrays are allocated ONLY for ensemble.
isEnsembleMode = ...
    strcmp(transferMode,'ensemble');

if isEnsembleMode

    % Compulsory compact ensemble state needed later by LocalRBFTesting.
    ensembleUniqueWeight = ...
        cell(nQuery,1);

    ensembleUniqueRadius = ...
        cell(nQuery,1);

    ensembleUniqueRegParam = ...
        cell(nQuery,1);

    ensembleUniqueC = ...
        cell(nQuery,1);

    ensembleNUniqueTuples = ...
        zeros(nQuery,1);


    % Full donor provenance is DETAIL-only.
    if retainDetail

        ensembleSourceTrainingIndex = ...
            cell(nQuery,1);

        ensembleSourceDistance = ...
            cell(nQuery,1);

        ensembleSourceWeight = ...
            cell(nQuery,1);

        ensembleUniqueTrainingIndex = ...
            cell(nQuery,1);

        ensembleUniqueRadiusIndex = ...
            cell(nQuery,1);

    end

end

% Scalar consistency state is enough for compact mode.
allSuccessfulWeightsConsistent = true;
allSuccessfulSourceSelectionConsistent = true;

% =============================================================
% 7. Resolve source-selection mode
% =============================================================

switch transferMode
    case 'hybrid'
        idxNearQuery = find(nearestDistance <= near_threshold);
        idxIDWQuery = find(nearestDistance > near_threshold);

    case 'idw'
        idxNearQuery = zeros(0,1);
        idxIDWQuery = (1:nQuery).';

    case 'nearest'
        idxNearQuery = (1:nQuery).';
        idxIDWQuery = zeros(0,1);

    case 'ensemble'
        % Same near/far source-selection logic as hybrid, but distant
        % queries retain complete donor tuples instead of aggregating them.
        idxNearQuery = find(nearestDistance <= near_threshold);
        idxIDWQuery = find(nearestDistance > near_threshold);

    otherwise
        error('Unsupported transferMode: %s',transferMode);
end

% =============================================================
% 8. Direct nearest-point transfer
% =============================================================

if ~isempty(idxNearQuery)
    idxSource = nearestTrainingIndex(idxNearQuery);

    radiusQuery(idxNearQuery) = radiusTrain(idxSource);
    regQuery(idxNearQuery) = regTrain(idxSource);
    if useShape
        CQuery(idxNearQuery) = CTrain(idxSource);
    end

    querySuccess(idxNearQuery) = true;
    nTransferSources(idxNearQuery) = 1;

    for ii = 1:numel(idxNearQuery)
        iQuery = idxNearQuery(ii);
        iSource = nearestTrainingIndex(iQuery);

        transferMethod{iQuery} = 'nearest';
        messageByQuery{iQuery} = '';


        if isEnsembleMode

            ensembleUniqueWeight{iQuery} = ...
                1;

            ensembleUniqueRadius{iQuery} = ...
                radiusTrain(iSource);

            ensembleUniqueRegParam{iQuery} = ...
                regTrain(iSource);

            if useShape

                ensembleUniqueC{iQuery} = ...
                    CTrain(iSource);

            else

                ensembleUniqueC{iQuery} = ...
                    NaN;

            end

            ensembleNUniqueTuples(iQuery) = ...
                1;


            if retainDetail

                ensembleSourceTrainingIndex{iQuery} = ...
                    iSource;

                ensembleSourceDistance{iQuery} = ...
                    nearestDistance(iQuery);

                ensembleSourceWeight{iQuery} = ...
                    1;

                ensembleUniqueTrainingIndex{iQuery} = ...
                    iSource;

                ensembleUniqueRadiusIndex{iQuery} = ...
                    radiusIndexTrain(iSource);

            end

        end


        if retainDetail
            sourceTrainingIndex{iQuery} = iSource;
            sourceDistance{iQuery} = nearestDistance(iQuery);
            sourceWeight{iQuery} = 1;
            sourceRadius{iQuery} = radiusTrain(iSource);
            sourceRegParam{iQuery} = regTrain(iSource);
            if useShape
                sourceC{iQuery} = CTrain(iSource);
            else
                sourceC{iQuery} = NaN;
            end
            if hasRadiusIndex
                sourceRadiusIndex{iQuery} = radiusIndexTrain(iSource);
            else
                sourceRadiusIndex{iQuery} = NaN;
            end
        end
    end
end

% =============================================================
% 9. N-nearest inverse-square-distance transfer
% =============================================================

nUse = min(nIDWSourcesRequested,nTrain);

if ~isempty(idxIDWQuery)
    [dIDWNearest,idxIDWNearest] = ...
        mink(D_query_train(idxIDWQuery,:),nUse,2);
else
    dIDWNearest = zeros(0,nUse);
    idxIDWNearest = zeros(0,nUse);
end

for ii = 1:numel(idxIDWQuery)
    iQuery = idxIDWQuery(ii);
    idxSource = idxIDWNearest(ii,:);
    dSource = dIDWNearest(ii,:).';

    if any(dSource < 0) || any(~isfinite(dSource))
        transferMethod{iQuery} = 'failed';
        nTransferSources(iQuery) = numel(idxSource);
        messageByQuery{iQuery} = ...
            'Invalid distance encountered during N-nearest IDW transfer.';

        if retainDetail
            sourceTrainingIndex{iQuery} = idxSource(:);
            sourceDistance{iQuery} = dSource;
            sourceWeight{iQuery} = [];
            sourceRadius{iQuery} = radiusTrain(idxSource);
            sourceRegParam{iQuery} = regTrain(idxSource);
            if useShape
                sourceC{iQuery} = CTrain(idxSource);
            else
                sourceC{iQuery} = nan(numel(idxSource),1);
            end
            sourceRadiusIndex{iQuery} = radiusIndexTrain(idxSource);
        end
        continue
    end

    zeroDistance = dSource == 0;

    if any(zeroDistance)
        % Limiting IDW case for a query coincident with one or more
        % training points. Only coincident sources receive nonzero weight.
        w = zeros(size(dSource));
        w(zeroDistance) = 1 ./ nnz(zeroDistance);
        sumW = 1;
    else
        w = 1 ./ (dSource.^2);
        sumW = sum(w);
    end

    if ~isfinite(sumW) || sumW <= 0
        transferMethod{iQuery} = 'failed';
        nTransferSources(iQuery) = numel(idxSource);
        messageByQuery{iQuery} = ...
            'Invalid inverse-square-distance weights.';

        if retainDetail
            sourceTrainingIndex{iQuery} = idxSource(:);
            sourceDistance{iQuery} = dSource;
            sourceWeight{iQuery} = [];
            sourceRadius{iQuery} = radiusTrain(idxSource);
            sourceRegParam{iQuery} = regTrain(idxSource);
            if useShape
                sourceC{iQuery} = CTrain(idxSource);
            else
                sourceC{iQuery} = nan(numel(idxSource),1);
            end
            sourceRadiusIndex{iQuery} = radiusIndexTrain(idxSource);
        end
        continue
    end

    w = w ./ sumW;


    if isEnsembleMode

        % ---------------------------------------------------------
        % ENSEMBLE TRANSFER
        %
        % Do NOT aggregate Radius / RegParam / C.
        % Preserve the complete donor tuples and aggregate only
        % duplicate-tuple weights so LocalRBFTesting can solve each
        % unique tuple exactly once.
        % ---------------------------------------------------------

        [ ...
            idxUnique, ...
            wUnique, ...
            radiusUnique, ...
            regUnique, ...
            CUnique, ...
            radiusIndexUnique ...
        ] = ...
            deduplicateEnsembleTuplesLocal( ...
                idxSource(:), ...
                w, ...
                radiusTrain, ...
                regTrain, ...
                CTrain, ...
                radiusIndexTrain, ...
                useShape);


        ensembleUniqueWeight{iQuery} = ...
            wUnique;

        ensembleUniqueRadius{iQuery} = ...
            radiusUnique;

        ensembleUniqueRegParam{iQuery} = ...
            regUnique;

        ensembleUniqueC{iQuery} = ...
            CUnique;

        ensembleNUniqueTuples(iQuery) = ...
            numel(wUnique);


        if retainDetail

            ensembleSourceTrainingIndex{iQuery} = ...
                idxSource(:);

            ensembleSourceDistance{iQuery} = ...
                dSource;

            ensembleSourceWeight{iQuery} = ...
                w;

            ensembleUniqueTrainingIndex{iQuery} = ...
                idxUnique;

            ensembleUniqueRadiusIndex{iQuery} = ...
                radiusIndexUnique;

        end


        % There is intentionally no single transferred tuple.
        radiusQuery(iQuery) = ...
            NaN;

        regQuery(iQuery) = ...
            NaN;

        CQuery(iQuery) = ...
            NaN;


        querySuccess(iQuery) = ...
            all(isfinite(wUnique)) && ...
            all(wUnique >= 0) && ...
            abs(sum(wUnique)-1) <= 1e-12 && ...
            all(isfinite(radiusUnique)) && ...
            all(radiusUnique > 0) && ...
            all(isfinite(regUnique)) && ...
            all(regUnique > 0);

        if useShape

            querySuccess(iQuery) = ...
                querySuccess(iQuery) && ...
                all(isfinite(CUnique)) && ...
                all(CUnique > 0);

        end


        if querySuccess(iQuery)

            transferMethod{iQuery} = ...
                'ensemble';

            messageByQuery{iQuery} = ...
                '';

        else

            transferMethod{iQuery} = ...
                'failed';

            messageByQuery{iQuery} = ...
                'Invalid ensemble donor tuple or ensemble weight.';

        end


    else

        % ---------------------------------------------------------
        % EXISTING IDW / HYBRID PARAMETER AGGREGATION — UNCHANGED
        % ---------------------------------------------------------

        radiusQuery(iQuery) = ...
            sum(w .* radiusTrain(idxSource));

        regQuery(iQuery) = ...
            weightedPositiveParameterMean( ...
                regTrain(idxSource), ...
                w, ...
                regParamTransferScale);

        if useShape

            CQuery(iQuery) = ...
                weightedPositiveParameterMean( ...
                    CTrain(idxSource), ...
                    w, ...
                    CTransferScale);

        end

        querySuccess(iQuery) = true;
        transferMethod{iQuery} = 'idw';
        messageByQuery{iQuery} = '';

    end


    nTransferSources(iQuery) = ...
        numel(idxSource);

    currentWeightConsistent = ...
        abs(sum(w) - 1) <= 1e-12;

    currentSourceSelectionConsistent = ...
        numel(idxSource) == nUse;

    allSuccessfulWeightsConsistent = ...
        allSuccessfulWeightsConsistent && ...
        currentWeightConsistent;

    allSuccessfulSourceSelectionConsistent = ...
        allSuccessfulSourceSelectionConsistent && ...
        currentSourceSelectionConsistent;


    if retainDetail

        sourceTrainingIndex{iQuery} = ...
            idxSource(:);

        sourceDistance{iQuery} = ...
            dSource;

        sourceWeight{iQuery} = ...
            w;

        sourceRadius{iQuery} = ...
            radiusTrain(idxSource);

        sourceRegParam{iQuery} = ...
            regTrain(idxSource);

        if useShape

            sourceC{iQuery} = ...
                CTrain(idxSource);

        else

            sourceC{iQuery} = ...
                nan(numel(idxSource),1);

        end

        sourceRadiusIndex{iQuery} = ...
            radiusIndexTrain(idxSource);

    end
end

% =============================================================
% 10. Final transferred-setting validity
% =============================================================

validTransferred = ...
    isfinite(radiusQuery) & radiusQuery > 0 & ...
    isfinite(regQuery) & regQuery > 0;

if useShape
    validTransferred = validTransferred & isfinite(CQuery) & CQuery > 0;
end


if isEnsembleMode

    % Direct-nearest ensemble queries still have one scalar tuple.
    % Multi-source ensemble queries intentionally do not.
    scalarTupleRequired = ...
        strcmp(transferMethod,'nearest');

    querySuccess(scalarTupleRequired) = ...
        querySuccess(scalarTupleRequired) & ...
        validTransferred(scalarTupleRequired);

else

    querySuccess = ...
        querySuccess & validTransferred;

end

for iQuery = 1:nQuery
    if ~querySuccess(iQuery) && isempty(messageByQuery{iQuery})
        messageByQuery{iQuery} = 'Transferred settings are invalid.';
        transferMethod{iQuery} = 'failed';
    end
end

% =============================================================
% 11. Common compact output
% =============================================================

Transfer = struct();
Transfer.success = all(querySuccess);
Transfer.kernel = kernel;
Transfer.useShape = useShape;
Transfer.nTrain = nTrain;
Transfer.nQuery = nQuery;
Transfer.nDimensions = p;

Transfer.transferMode = transferMode;
Transfer.near_threshold = near_threshold;
Transfer.near_threshold_used_for_source_selection = ...
    any(strcmp(transferMode,{'hybrid','ensemble'}));
Transfer.idw_radius = idw_radius;
Transfer.idw_radius_used_for_source_selection = false;
Transfer.idw_n_nearest = nIDWSourcesRequested;
Transfer.idw_n_nearest_effective = nUse;
Transfer.idw_source_selection = 'N-nearest-training-points';
Transfer.idw_power = 2;

if isEnsembleMode

    Transfer.ensemblePower = ...
        2;

    Transfer.ensembleSourceSelection = ...
        'nearest-within-threshold-otherwise-N-nearest';

    Transfer.ensemble_n_nearest = ...
        nIDWSourcesRequested;

    Transfer.ensemble_n_nearest_effective = ...
        nUse;


    Transfer.radiusTransferScale = ...
        'not-applicable';

else

    Transfer.radiusTransferScale = ...
        'arithmetic';

end

Transfer.parameterTransferMode = ...
    parameterTransferMode;

Transfer.regParamTransferScale = ...
    regParamTransferScale;

if isEnsembleMode

    Transfer.CTransferScale = ...
        'not-applicable';

elseif useShape

    Transfer.CTransferScale = ...
        CTransferScale;

else

    Transfer.CTransferScale = ...
        'not-applicable';

end

Transfer.distanceMatrixProvided = distanceMatrixProvided;
Transfer.distanceMatrixSize = size(D_query_train);
Transfer.distanceMatrixStored = false;

Transfer.querySuccess = querySuccess;
Transfer.transferMethod = transferMethod;
Transfer.messageByQuery = messageByQuery;
Transfer.nSuccessfulQueries = sum(querySuccess);
Transfer.nFailedQueries = nQuery - Transfer.nSuccessfulQueries;
Transfer.nNearestTransfers = sum(strcmp(transferMethod,'nearest'));
Transfer.nIDWTransfers = sum(strcmp(transferMethod,'idw'));
Transfer.nEnsembleTransfers = sum(strcmp(transferMethod,'ensemble'));

Transfer.nClassifiedQueries = ...
    Transfer.nNearestTransfers + ...
    Transfer.nIDWTransfers + ...
    Transfer.nEnsembleTransfers + ...
    Transfer.nFailedQueries;

Transfer.classificationCountConsistent = ...
    Transfer.nClassifiedQueries == nQuery;

Transfer.radius = radiusQuery;
Transfer.reg_param = regQuery;
Transfer.C = CQuery;


% Ensemble tuple data are compulsory for ensemble testing, including
% compact mode. For non-ensemble modes these fields are omitted.
if isEnsembleMode

    % Compulsory ensemble prediction state.
    Transfer.ensembleUniqueWeight = ...
        ensembleUniqueWeight;

    Transfer.ensembleUniqueRadius = ...
        ensembleUniqueRadius;

    Transfer.ensembleUniqueRegParam = ...
        ensembleUniqueRegParam;

    Transfer.ensembleUniqueC = ...
        ensembleUniqueC;

    Transfer.ensembleNUniqueTuples = ...
        ensembleNUniqueTuples;

    Transfer.ensemblePredictionAggregation = ...
        'weighted-average-of-unique-tuple-predictions';

    Transfer.ensembleTupleDeduplication = ...
        true;


    % Full donor provenance only when requested.
    if retainDetail

        Transfer.ensembleSourceTrainingIndex = ...
            ensembleSourceTrainingIndex;

        Transfer.ensembleSourceDistance = ...
            ensembleSourceDistance;

        Transfer.ensembleSourceWeight = ...
            ensembleSourceWeight;

        Transfer.ensembleUniqueTrainingIndex = ...
            ensembleUniqueTrainingIndex;

        Transfer.ensembleUniqueRadiusIndex = ...
            ensembleUniqueRadiusIndex;

    end

end

% =============================================================
% 12. Historical full-detail output only
% =============================================================

if retainDetail
    Transfer.nearestTrainingIndex = nearestTrainingIndex;
    Transfer.nearestDistance = nearestDistance;
    Transfer.nTransferSources = nTransferSources;

    Transfer.sourceTrainingIndex = sourceTrainingIndex;
    Transfer.sourceDistance = sourceDistance;
    Transfer.sourceWeight = sourceWeight;
    Transfer.sourceRadius = sourceRadius;
    Transfer.sourceRegParam = sourceRegParam;
    Transfer.sourceC = sourceC;
    Transfer.sourceRadiusIndex = sourceRadiusIndex;

    Transfer.d_local = nan(nQuery,1);
    Transfer.epsilon = epsilonQuery;
    Transfer.epsilonRequiresLocalGeometry = repmat(useShape,nQuery,1);

    Transfer.weightSum = nan(nQuery,1);
    Transfer.weightsConsistent = false(nQuery,1);

    for iQuery = 1:nQuery
        if ~querySuccess(iQuery)
            continue
        end
        w = sourceWeight{iQuery};
        Transfer.weightSum(iQuery) = sum(w);
        Transfer.weightsConsistent(iQuery) = ...
            abs(Transfer.weightSum(iQuery) - 1) <= 1e-12;
    end

    Transfer.allSuccessfulWeightsConsistent = ...
        all(Transfer.weightsConsistent(querySuccess));

    Transfer.sourceSelectionConsistent = false(nQuery,1);
    for iQuery = 1:nQuery
        if ~querySuccess(iQuery)
            continue
        end
        if strcmp(transferMethod{iQuery},'nearest')
            Transfer.sourceSelectionConsistent(iQuery) = ...
                nTransferSources(iQuery) == 1;
        elseif strcmp(transferMethod{iQuery},'idw') || ...
               strcmp(transferMethod{iQuery},'ensemble')
            Transfer.sourceSelectionConsistent(iQuery) = ...
                nTransferSources(iQuery) == nUse;
        end
    end

    Transfer.allSuccessfulSourceSelectionConsistent = ...
        all(Transfer.sourceSelectionConsistent(querySuccess));

    allSuccessfulWeightsConsistent = ...
        Transfer.allSuccessfulWeightsConsistent;
    allSuccessfulSourceSelectionConsistent = ...
        Transfer.allSuccessfulSourceSelectionConsistent;
else
    % Nearest transfers are exactly one-source/unit-weight by construction.
    % IDW/ensemble consistency was checked immediately when each normalized
    % weight vector was formed, without retaining historical detail arrays.
    if any(querySuccess & strcmp(transferMethod,'nearest'))
        allSuccessfulWeightsConsistent = ...
            allSuccessfulWeightsConsistent && true;
        allSuccessfulSourceSelectionConsistent = ...
            allSuccessfulSourceSelectionConsistent && ...
            all(nTransferSources(querySuccess & ...
                strcmp(transferMethod,'nearest')) == 1);
    end
end

% =============================================================
% 13. Final status/message
% =============================================================

if Transfer.success && ...
   allSuccessfulWeightsConsistent && ...
   allSuccessfulSourceSelectionConsistent && ...
   Transfer.classificationCountConsistent
    Transfer.message = '';
elseif Transfer.success
    Transfer.success = false;
    Transfer.message = ...
        ['All query settings were transferred, but an internal ' ...
         'weight/source-selection/classification consistency check failed.'];
else
    Transfer.message = sprintf( ...
        ['Transfer succeeded for %d of %d query points; ' ...
         '%d query points failed.'], ...
        Transfer.nSuccessfulQueries,nQuery,Transfer.nFailedQueries);
end

end


% =============================================================
% LOCAL HELPER: exact ensemble-tuple deduplication
% =============================================================

function [ ...
    idxUnique,wUnique,radiusUnique,regUnique,CUnique,radiusIndexUnique ...
] = ...
    deduplicateEnsembleTuplesLocal( ...
        idxSource,w,radiusTrain,regTrain,CTrain,radiusIndexTrain,useShape)

idxSource = ...
    idxSource(:);

w = ...
    w(:);

nSource = ...
    numel(idxSource);


idxUnique = ...
    zeros(0,1);

wUnique = ...
    zeros(0,1);

radiusUnique = ...
    zeros(0,1);

regUnique = ...
    zeros(0,1);

CUnique = ...
    zeros(0,1);

radiusIndexUnique = ...
    zeros(0,1);


for i = 1:nSource

    idx = ...
        idxSource(i);

    r = ...
        radiusTrain(idx);

    reg = ...
        regTrain(idx);

    if useShape

        c = ...
            CTrain(idx);

    else

        c = ...
            NaN;

    end


    match = ...
        false(numel(wUnique),1);

    for j = 1:numel(wUnique)

        sameTuple = ...
            radiusUnique(j) == r && ...
            regUnique(j) == reg;

        if useShape

            sameTuple = ...
                sameTuple && ...
                CUnique(j) == c;

        end

        match(j) = ...
            sameTuple;

    end


    jMatch = ...
        find(match,1,'first');


    if isempty(jMatch)

        idxUnique(end+1,1) = ...
            idx; %#ok<AGROW>

        wUnique(end+1,1) = ...
            w(i); %#ok<AGROW>

        radiusUnique(end+1,1) = ...
            r; %#ok<AGROW>

        regUnique(end+1,1) = ...
            reg; %#ok<AGROW>

        CUnique(end+1,1) = ...
            c; %#ok<AGROW>

        radiusIndexUnique(end+1,1) = ...
            radiusIndexTrain(idx); %#ok<AGROW>

    else

        wUnique(jMatch) = ...
            wUnique(jMatch) + w(i);

    end

end


sumW = ...
    sum(wUnique);

if isfinite(sumW) && sumW > 0

    wUnique = ...
        wUnique ./ sumW;

end

end


% =============================================================
% LOCAL HELPER: RegParam/C aggregation mode
% =============================================================

function [mode,regScale,CScale] = ...
    normalizeParameterTransferMode(mode)

if ~(ischar(mode) || ...
     (isstring(mode) && isscalar(mode)))

    error( ...
        ['parameterTransferMode must be one of: ' ...
         '''arithmetic_arithmetic'', ''arithmetic_log'', ' ...
         '''log_arithmetic'', ''log_log''.']);

end


mode = ...
    lower( ...
        strtrim( ...
            char(mode)));


switch mode

    case {'arithmetic_arithmetic','arith_arith','m21'}

        mode = ...
            'arithmetic_arithmetic';

        regScale = ...
            'arithmetic';

        CScale = ...
            'arithmetic';


    case {'arithmetic_log','arith_log','m23'}

        mode = ...
            'arithmetic_log';

        regScale = ...
            'arithmetic';

        CScale = ...
            'log10-geometric';


    case {'log_arithmetic','log_arith','m22'}

        mode = ...
            'log_arithmetic';

        regScale = ...
            'log10-geometric';

        CScale = ...
            'arithmetic';


    case {'log_log','m24'}

        mode = ...
            'log_log';

        regScale = ...
            'log10-geometric';

        CScale = ...
            'log10-geometric';


    otherwise

        error( ...
            ['parameterTransferMode must be one of: ' ...
             '''arithmetic_arithmetic'', ''arithmetic_log'', ' ...
             '''log_arithmetic'', ''log_log''.']);

end

end


% =============================================================
% LOCAL HELPER: weighted positive parameter mean
% =============================================================

function value = weightedPositiveParameterMean( ...
    sourceValue,w,scaleMode)

sourceValue = ...
    sourceValue(:);

w = ...
    w(:);


if numel(sourceValue) ~= numel(w) || ...
   any(~isfinite(sourceValue)) || ...
   any(sourceValue <= 0) || ...
   any(~isfinite(w)) || ...
   any(w < 0)

    value = ...
        NaN;

    return

end


sumW = ...
    sum(w);


if ~isfinite(sumW) || ...
   sumW <= 0

    value = ...
        NaN;

    return

end


w = ...
    w ./ sumW;


switch scaleMode

    case 'arithmetic'

        value = ...
            sum(w .* sourceValue);


    case 'log10-geometric'

        value = ...
            10 .^ sum(w .* log10(sourceValue));


    otherwise

        error( ...
            'Unsupported parameter aggregation scale: %s', ...
            scaleMode);

end

end

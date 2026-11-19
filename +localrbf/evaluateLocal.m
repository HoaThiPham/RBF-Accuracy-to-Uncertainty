function Diagnostics = evaluateLocal( ...
    TrainingResult, options, geometryCache, evaluationMode)
%evaluateLocal
%
% Calculate numerical diagnostics for selected local RBF systems.
%
% evaluationMode = 'full'  : original behavior; diagnose every unique
%                            selected point-radius system over all caps.
%                = 'final' : diagnose only the final selected system at
%                            each training point. This mode is intended
%                            for Stage-2 compulsory output when detailed
%                            point-by-cap diagnostics are not requested.
%
% The default is 'full' for complete backward compatibility.
%
% IMPORTANT TIME-SAVING RULE
%
% At one training point, several radius caps may select exactly
% the same optimized radius. Because that selected point-radius
% system has the same:
%
%       neighbours
%       C
%       RegParam
%       epsilon
%       coefficients
%       prediction
%       numerical system
%
% its expensive diagnostics are calculated ONLY ONCE.
%
% Example:
%
%   selected radius index by cap:
%
%       [1 2 2 2 4 4]
%
% only radius systems:
%
%       [1 2 4]
%
% are reconstructed and diagnosed.
%
% The diagnostic result is then mapped back to all corresponding
% radius caps.
%
% Full selected-system diagnostics include:
%
%   - standard condition number before regularization;
%   - standard condition number after regularization;
%   - condition-number ratio and log10 change;
%   - effective condition number;
%   - effective numerical rank;
%   - retained minimum singular value;
%   - training-site evaluation condition number;
%   - target-site evaluation condition number;
%   - relative linear-system residual;
%   - normwise backward error;
%   - solver used;
%   - reconstructed target-excluded prediction;
%   - prediction residual;
%   - consistency with the stored optimization result.
%
% Residual convention:
%
%       residual = observation - prediction
%
% IMPORTANT:
%   - no optimization is performed here;
%   - no radius or cap selection is repeated;
%   - no neighbourhood search is repeated;
%   - each unique selected point-radius system is solved once;
%   - complete unique-system diagnostics are retained;
%   - point x cap diagnostic matrices are produced by mapping,
%     not by repeating diagnostics;
%   - geometryCache is reused when supplied;
%   - large RBF matrices are not retained.


% =============================================================
% 0. Evaluation mode
% =============================================================

if nargin < 4 || isempty(evaluationMode)

    evaluationMode = ...
        'full';

end


if ~(ischar(evaluationMode) || ...
     (isstring(evaluationMode) && isscalar(evaluationMode)))

    error( ...
        'evaluationMode must be ''full'' or ''final''.');

end


evaluationMode = ...
    lower( ...
        char(evaluationMode));


if ~any(strcmp(evaluationMode,{'full','final'}))

    error( ...
        'evaluationMode must be ''full'' or ''final''.');

end


% Final-only mode has an intentionally separate orchestration path because
% Stage 2 may no longer retain TrainingResult.Point(i).optimization. Both
% modes use the same common RBF system/solve/predict/diagnostic functions;
% therefore the numerical definition of one selected local system is shared.
if strcmp(evaluationMode,'final')

    Diagnostics = ...
        evaluateFinalSelectedSystems( ...
            TrainingResult, ...
            options, ...
            geometryCache);

    return

end


% =============================================================
% 1. Input checks
% =============================================================

if ~isstruct(TrainingResult)

    error( ...
        'TrainingResult must be a structure.');

end


requiredFields = { ...
    'X', ...
    'y', ...
    'kernel', ...
    'useShape', ...
    'Point', ...
    'selectedRadiusIndex_by_point_cap', ...
    'cap_success_by_point'};


for q = 1:numel(requiredFields)

    if ~isfield(TrainingResult,requiredFields{q})

        error( ...
            'TrainingResult is missing required field "%s".', ...
            requiredFields{q});

    end

end


% =============================================================
% 2. Solver options
% =============================================================

if nargin < 2 || isempty(options)

    options = ...
        struct();

end


if ~isstruct(options)

    error( ...
        'options must be a structure.');

end


if ~isfield(options,'solver') || ...
   isempty(options.solver)

    options.solver = ...
        struct();

end


if ~isfield(options.solver,'cond_lsq') || ...
   isempty(options.solver.cond_lsq)

    options.solver.cond_lsq = ...
        1e12;

end


options.solver.compute_rel_residual = ...
    true;


% =============================================================
% 3. Training data
% =============================================================

X = ...
    TrainingResult.X;


y = ...
    TrainingResult.y(:);


[n,p] = ...
    size(X);


if numel(y) ~= n

    error( ...
        'TrainingResult.X and TrainingResult.y are inconsistent.');

end


kernel = ...
    TrainingResult.kernel;


useShape = ...
    TrainingResult.useShape;


selectedRadiusIdx = ...
    TrainingResult.selectedRadiusIndex_by_point_cap;


capSuccess = ...
    TrainingResult.cap_success_by_point;


if size(selectedRadiusIdx,1) ~= n || ...
   size(capSuccess,1) ~= n || ...
   ~isequal(size(selectedRadiusIdx),size(capSuccess))

    error( ...
        ['Point-cap selection arrays are inconsistent with ' ...
         'the number of training points.']);

end


K = ...
    size(selectedRadiusIdx,2);


if numel(TrainingResult.Point) ~= n

    error( ...
        'TrainingResult.Point must contain one result per point.');

end


% =============================================================
% 4. Shared geometry
% =============================================================

tGeometry = ...
    tic;


geometryReused = ...
    nargin >= 3 && ...
    ~isempty(geometryCache);


if geometryReused

    if ~isstruct(geometryCache) || ...
       ~isfield(geometryCache,'D') || ...
       ~isfield(geometryCache,'P')

        error( ...
            'geometryCache must contain fields D and P.');

    end


    if ~isnumeric(geometryCache.D) || ...
       ~isequal(size(geometryCache.D),[n n])

        error( ...
            'geometryCache.D must be an n-by-n matrix.');

    end


    if ~isnumeric(geometryCache.P) || ...
       ~isequal(size(geometryCache.P),[n p+1])

        error( ...
            'geometryCache.P must be n-by-(p+1).');

    end


else

    geometryCache = ...
        struct();


    geometryCache.D = ...
        pdist2( ...
            X, ...
            X);


    geometryCache.P = ...
        [ones(n,1), ...
         X];

end


geometryWallTime = ...
    toc(tGeometry);


% =============================================================
% 5. Initialize output
% =============================================================

Diagnostics = ...
    struct();


Diagnostics.success = ...
    false;


Diagnostics.kernel = ...
    kernel;


Diagnostics.useShape = ...
    useShape;


Diagnostics.nPoints = ...
    n;


Diagnostics.nCaps = ...
    K;


Diagnostics.geometryReused = ...
    geometryReused;


Diagnostics.geometryWallTime = ...
    geometryWallTime;


Diagnostics.message = ...
    '';


Diagnostics.evaluationMode = ...
    'full';


% =============================================================
% 6. Retain original selection mapping
% =============================================================

Diagnostics.cap_success_by_point = ...
    capSuccess;


Diagnostics.selectedRadiusIndex_by_point_cap = ...
    selectedRadiusIdx;


% =============================================================
% 7. Point-level detailed storage
%
% Full expensive diagnostics are stored ONLY here for each
% unique selected point-radius system.
% =============================================================

Diagnostics.Point = ...
    repmat( ...
        emptyPointDiagnostic(), ...
        n,1);


% =============================================================
% 8. Point x cap mapping
%
% uniqueDiagnosticIndex_by_point_cap(i,k) identifies which
% Point(i).UniqueSelectedRadius(u) supplies the diagnostic
% information for cap k.
% =============================================================

Diagnostics.uniqueDiagnosticIndex_by_point_cap = ...
    nan(n,K);


Diagnostics.reuseGroupSize_by_point_cap = ...
    nan(n,K);


Diagnostics.diagnostic_success_by_point_cap = ...
    false(n,K);


Diagnostics.selectedSystemConsistent_by_point_cap = ...
    false(n,K);


% =============================================================
% 9. Compact condition diagnostics by point x cap
% =============================================================

Diagnostics.cond_raw_by_point_cap = ...
    nan(n,K);


Diagnostics.cond_reg_by_point_cap = ...
    nan(n,K);


Diagnostics.cond_ratio_by_point_cap = ...
    nan(n,K);


Diagnostics.delta_log10_cond_by_point_cap = ...
    nan(n,K);


Diagnostics.cond_effective_by_point_cap = ...
    nan(n,K);


% =============================================================
% 10. Rank diagnostics by point x cap
% =============================================================

Diagnostics.rank_eff_by_point_cap = ...
    nan(n,K);


Diagnostics.rank_tol_by_point_cap = ...
    nan(n,K);


Diagnostics.sigma_min_retained_by_point_cap = ...
    nan(n,K);


% =============================================================
% 11. Evaluation-condition diagnostics by point x cap
% =============================================================

Diagnostics.cond_evaluation_train_by_point_cap = ...
    nan(n,K);


Diagnostics.cond_evaluation_target_by_point_cap = ...
    nan(n,K);


% =============================================================
% 12. Linear-system diagnostics by point x cap
% =============================================================

Diagnostics.rel_residual_by_point_cap = ...
    nan(n,K);


Diagnostics.backward_error_by_point_cap = ...
    nan(n,K);


Diagnostics.solver_used_by_point_cap = ...
    cell(n,K);


% =============================================================
% 13. Recomputed prediction information by point x cap
% =============================================================

Diagnostics.yhat_recomputed_by_point_cap = ...
    nan(n,K);


Diagnostics.residual_recomputed_by_point_cap = ...
    nan(n,K);


Diagnostics.AE_recomputed_by_point_cap = ...
    nan(n,K);


% =============================================================
% 14. Consistency information by point x cap
% =============================================================

Diagnostics.yhatDifference_by_point_cap = ...
    nan(n,K);


Diagnostics.residualDifference_by_point_cap = ...
    nan(n,K);


Diagnostics.AEDifference_by_point_cap = ...
    nan(n,K);


Diagnostics.condRegDifference_by_point_cap = ...
    nan(n,K);


Diagnostics.relResidualDifference_by_point_cap = ...
    nan(n,K);


Diagnostics.yhatConsistent_by_point_cap = ...
    false(n,K);


Diagnostics.residualConsistent_by_point_cap = ...
    false(n,K);


Diagnostics.AEConsistent_by_point_cap = ...
    false(n,K);


Diagnostics.condRegConsistent_by_point_cap = ...
    false(n,K);


Diagnostics.relResidualConsistent_by_point_cap = ...
    false(n,K);


Diagnostics.solverConsistent_by_point_cap = ...
    false(n,K);


% =============================================================
% 15. Runtime / solve counts
% =============================================================

Diagnostics.nSelectedPointCapPairs = ...
    sum( ...
        capSuccess(:));


Diagnostics.nUniqueSelectedSystemsByPoint = ...
    zeros(n,1);


Diagnostics.nUniqueSelectedSystemsTotal = ...
    0;


Diagnostics.nReusedPointCapPairs = ...
    0;


Diagnostics.nRBFSolveAttempts = ...
    0;


Diagnostics.nSuccessfulRBFSolves = ...
    0;


Diagnostics.nRBFSolves = ...
    0;


Diagnostics.pointWallTime = ...
    zeros(n,1);


% =============================================================
% 16. Calculate diagnostics
% =============================================================

tDiagnostics = ...
    tic;


for iTarget = 1:n

    tPoint = ...
        tic;


    Pi = ...
        emptyPointDiagnostic();


    Pi.iTarget = ...
        iTarget;


    Pi.xTarget = ...
        X(iTarget,:);


    Pi.yTarget = ...
        y(iTarget);


    Pi.selectedRadiusIndex_by_cap = ...
        selectedRadiusIdx(iTarget,:);


    Pi.cap_success = ...
        capSuccess(iTarget,:);


    % =========================================================
    % 16.1 Retrieve point-radius optimization results
    % =========================================================

    if ~isstruct(TrainingResult.Point(iTarget)) || ...
       ~isfield(TrainingResult.Point(iTarget),'optimization') || ...
       isempty(TrainingResult.Point(iTarget).optimization)

        Pi.message = ...
            'Point optimization result is unavailable.';


        Pi.totalWallTime = ...
            toc(tPoint);


        Diagnostics.Point(iTarget) = ...
            Pi;


        Diagnostics.pointWallTime(iTarget) = ...
            Pi.totalWallTime;


        continue

    end


    PointOpt = ...
        TrainingResult.Point(iTarget).optimization;


    if ~isfield(PointOpt,'Radius') || ...
       isempty(PointOpt.Radius)

        Pi.message = ...
            'Point optimization does not contain radius results.';


        Pi.totalWallTime = ...
            toc(tPoint);


        Diagnostics.Point(iTarget) = ...
            Pi;


        Diagnostics.pointWallTime(iTarget) = ...
            Pi.totalWallTime;


        continue

    end


    nRadii = ...
        numel(PointOpt.Radius);


    % =========================================================
    % 16.2 Valid selected caps
    % =========================================================

    idxRow = ...
        selectedRadiusIdx(iTarget,:);


    validSelectedIndex = ...
        isfinite(idxRow) & ...
        idxRow == round(idxRow) & ...
        idxRow >= 1 & ...
        idxRow <= nRadii;


    invalidCaps = ...
        find( ...
            capSuccess(iTarget,:) & ...
            ~validSelectedIndex);


    Pi.invalidSelectedCaps = ...
        invalidCaps;


    validCaps = ...
        find( ...
            capSuccess(iTarget,:) & ...
            validSelectedIndex);


    Pi.validSelectedCaps = ...
        validCaps;


    Pi.nSelectedCaps = ...
        numel(validCaps);


    if isempty(validCaps)

        if isempty(invalidCaps)

            Pi.message = ...
                'No successful selected radius exists for this point.';

        else

            Pi.message = ...
                'Successful caps contain invalid selected radius indices.';

        end


        Pi.totalWallTime = ...
            toc(tPoint);


        Diagnostics.Point(iTarget) = ...
            Pi;


        Diagnostics.pointWallTime(iTarget) = ...
            Pi.totalWallTime;


        continue

    end


    % =========================================================
    % 16.3 Unique selected radius indices
    %
    % This is the core runtime-saving step.
    % =========================================================

    uniqueRadiusIdx = ...
        unique( ...
            idxRow(validCaps), ...
            'stable');


    uniqueRadiusIdx = ...
        uniqueRadiusIdx(:);


    nUnique = ...
        numel(uniqueRadiusIdx);


    Pi.uniqueSelectedRadiusIndex = ...
        uniqueRadiusIdx;


    Pi.nUniqueSelectedRadii = ...
        nUnique;


    Pi.nDiagnosticsReused = ...
        Pi.nSelectedCaps - ...
        nUnique;


    Pi.UniqueSelectedRadius = ...
        repmat( ...
            emptyUniqueDiagnostic(), ...
            nUnique,1);


    Diagnostics.nUniqueSelectedSystemsByPoint(iTarget) = ...
        nUnique;


    Diagnostics.nUniqueSelectedSystemsTotal = ...
        Diagnostics.nUniqueSelectedSystemsTotal + ...
        nUnique;


    Diagnostics.nReusedPointCapPairs = ...
        Diagnostics.nReusedPointCapPairs + ...
        Pi.nDiagnosticsReused;


    % =========================================================
    % 16.4 Diagnose each unique selected radius ONCE
    % =========================================================

    for u = 1:nUnique

        tUnique = ...
            tic;


        jRadius = ...
            uniqueRadiusIdx(u);


        capsUsingRadius = ...
            validCaps( ...
                idxRow(validCaps) == jRadius);


        U = ...
            emptyUniqueDiagnostic();


        U.uniqueDiagnosticIndex = ...
            u;


        U.radiusIndex = ...
            jRadius;


        U.capsUsingRadius = ...
            capsUsingRadius;


        U.nCapsUsingRadius = ...
            numel(capsUsingRadius);


        U.reuseCount = ...
            max( ...
                0, ...
                U.nCapsUsingRadius - 1);


        % -----------------------------------------------------
        % Map caps to this unique diagnostic immediately
        % -----------------------------------------------------

        Diagnostics. ...
            uniqueDiagnosticIndex_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            u;


        Diagnostics. ...
            reuseGroupSize_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.nCapsUsingRadius;


        % =====================================================
        % 16.5 Retrieve source optimized radius
        % =====================================================

        RStar = ...
            PointOpt.Radius(jRadius);


        U.radius = ...
            RStar.radius;


        U.C = ...
            RStar.C_best;


        U.reg_param = ...
            RStar.reg_param_best;


        U.epsilon = ...
            RStar.epsilon_best;


        U.d_local = ...
            RStar.d_local;


        U.neighbor_idx = ...
            RStar.neighbor_idx;


        U.neighbor_distances = ...
            RStar.neighbor_distances;


        U.neighbor_count = ...
            RStar.neighbor_count;


        U.yhat_stored = ...
            RStar.yhat_best;


        U.residual_stored = ...
            RStar.y_diff_best;


        U.AE_stored = ...
            RStar.AE_best;


        U.cond_reg_stored = ...
            RStar.cond_reg_best;


        U.rel_residual_stored = ...
            RStar.rel_residual_best;


        U.solver_used_stored = ...
            RStar.solver_used_best;


        % =====================================================
        % 16.6 Validate source selected system
        % =====================================================

        if ~RStar.success

            U.failureStage = ...
                'source radius';


            U.message = ...
                'Selected source radius is not successful.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        idxFit = ...
            U.neighbor_idx(:);


        if isempty(idxFit) || ...
           ~isnumeric(idxFit) || ...
           ~isreal(idxFit) || ...
           any(~isfinite(idxFit)) || ...
           any(idxFit ~= round(idxFit)) || ...
           any(idxFit < 1) || ...
           any(idxFit > n) || ...
           numel(unique(idxFit)) ~= numel(idxFit)

            U.failureStage = ...
                'selected neighbourhood';


            U.message = ...
                'Selected neighbour indices are invalid.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        if any(idxFit == iTarget)

            U.failureStage = ...
                'selected neighbourhood';


            U.message = ...
                'Target observation is present in its fitting neighbourhood.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        if U.neighbor_count ~= numel(idxFit)

            U.failureStage = ...
                'selected neighbourhood';


            U.message = ...
                'Stored neighbour count is inconsistent with indices.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        if ~isscalar(U.reg_param) || ...
           ~isreal(U.reg_param) || ...
           ~isfinite(U.reg_param) || ...
           U.reg_param <= 0

            U.failureStage = ...
                'selected parameters';


            U.message = ...
                'Selected RegParam is invalid.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        if ~isscalar(U.epsilon) || ...
           ~isreal(U.epsilon) || ...
           ~isfinite(U.epsilon) || ...
           U.epsilon <= 0

            U.failureStage = ...
                'selected parameters';


            U.message = ...
                'Selected epsilon is invalid.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        if useShape && ...
           (~isscalar(U.C) || ...
            ~isreal(U.C) || ...
            ~isfinite(U.C) || ...
            U.C <= 0)

            U.failureStage = ...
                'selected parameters';


            U.message = ...
                'Selected C is invalid for a shaped kernel.';


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        % =====================================================
        % 16.7 Reconstruct fixed selected-system geometry
        % =====================================================

        Xfit = ...
            X(idxFit,:);


        yfit = ...
            y(idxFit);


        nFit = ...
            numel(idxFit);


        q = ...
            p + 1;


        geometry = ...
            struct();


        geometry.R = ...
            geometryCache.D( ...
                idxFit, ...
                idxFit);


        geometry.P = ...
            geometryCache.P( ...
                idxFit, ...
                :);


        geometry.R_eval = ...
            reshape( ...
                geometryCache.D( ...
                    iTarget, ...
                    idxFit), ...
                1,[]);


        geometry.P_eval = ...
            geometryCache.P( ...
                iTarget, ...
                :);


        U.system_size = ...
            nFit + q;


        % =====================================================
        % 16.8 Reconstruct and solve system ONCE
        % =====================================================

        try

            S = ...
                common.buildRBFSystem( ...
                    Xfit, ...
                    yfit, ...
                    kernel, ...
                    U.epsilon, ...
                    U.reg_param, ...
                    geometry);


            Diagnostics.nRBFSolveAttempts = ...
                Diagnostics.nRBFSolveAttempts + 1;


            [coeffs,solveInfo] = ...
                common.solveLinearSystem( ...
                    S, ...
                    options);


            Diagnostics.nRBFSolves = ...
                Diagnostics.nRBFSolves + 1;


            if ~solveInfo.success

                U.failureStage = ...
                    'selected system solve';


                U.message = ...
                    'Selected local RBF system solve was unsuccessful.';


                U.totalWallTime = ...
                    toc(tUnique);


                Pi.UniqueSelectedRadius(u) = ...
                    U;


                continue

            end


            Diagnostics.nSuccessfulRBFSolves = ...
                Diagnostics.nSuccessfulRBFSolves + 1;


            % =================================================
            % 16.9 Recompute target prediction
            % =================================================

            [yhat,B_target] = ...
                common.predictRBF( ...
                    X(iTarget,:), ...
                    Xfit, ...
                    coeffs, ...
                    kernel, ...
                    U.epsilon, ...
                    geometry);


            if ~isscalar(yhat) || ...
               ~isreal(yhat) || ...
               ~isfinite(yhat)

                U.failureStage = ...
                    'selected prediction';


                U.message = ...
                    'Reconstructed selected prediction is invalid.';


                U.totalWallTime = ...
                    toc(tUnique);


                Pi.UniqueSelectedRadius(u) = ...
                    U;


                continue

            end


            % =================================================
            % 16.10 Full expensive diagnostics
            % =================================================

            Di = ...
                common.calculateConditionDiagnostics( ...
                    S, ...
                    coeffs, ...
                    solveInfo, ...
                    B_target);


        catch ME

            U.failureStage = ...
                'selected diagnostics';


            U.message = ...
                ME.message;


            U.totalWallTime = ...
                toc(tUnique);


            Pi.UniqueSelectedRadius(u) = ...
                U;


            continue

        end


        % =====================================================
        % 16.11 Retain coefficients once
        % =====================================================

        coeffs = ...
            coeffs(:);


        U.coeffs = ...
            coeffs;


        U.lambda = ...
            coeffs(1:nFit);


        U.beta = ...
            coeffs(nFit+1:end);


        % =====================================================
        % 16.12 Retain complete condition diagnostic
        % =====================================================

        U.condition = ...
            Di;


        U.cond_raw = ...
            Di.cond_raw;


        U.cond_reg = ...
            Di.cond_reg;


        U.cond_ratio = ...
            Di.cond_ratio;


        U.delta_log10_cond = ...
            Di.delta_log10_cond;


        U.cond_effective = ...
            Di.cond_effective;


        U.rank_eff = ...
            Di.rank_eff;


        U.rank_tol = ...
            Di.rank_tol;


        U.sigma_min_retained = ...
            Di.sigma_min_retained;


        U.cond_evaluation_train = ...
            Di.cond_evaluation_train;


        U.cond_evaluation_target = ...
            Di.cond_evaluation_test;


        U.rel_residual = ...
            Di.rel_residual;


        U.backward_error = ...
            Di.backward_error;


        U.solver_used = ...
            Di.solver_used;


        % =====================================================
        % 16.13 Prediction residual
        %
        % residual = observation - prediction
        % =====================================================

        U.yhat_recomputed = ...
            yhat;


        U.residual_recomputed = ...
            y(iTarget) - ...
            yhat;


        U.AE_recomputed = ...
            abs( ...
                U.residual_recomputed);


        % =====================================================
        % 16.14 Consistency checks
        % =====================================================

        [ ...
            U.yhatDifference, ...
            U.yhatConsistencyTolerance, ...
            U.yhatConsistent] = ...
            compareScalar( ...
                U.yhat_recomputed, ...
                U.yhat_stored);


        [ ...
            U.residualDifference, ...
            U.residualConsistencyTolerance, ...
            U.residualConsistent] = ...
            compareScalar( ...
                U.residual_recomputed, ...
                U.residual_stored);


        [ ...
            U.AEDifference, ...
            U.AEConsistencyTolerance, ...
            U.AEConsistent] = ...
            compareScalar( ...
                U.AE_recomputed, ...
                U.AE_stored);


        [ ...
            U.condRegDifference, ...
            U.condRegConsistencyTolerance, ...
            U.condRegConsistent] = ...
            compareScalar( ...
                U.cond_reg, ...
                U.cond_reg_stored);


        [ ...
            U.relResidualDifference, ...
            U.relResidualConsistencyTolerance, ...
            U.relResidualConsistent] = ...
            compareScalar( ...
                U.rel_residual, ...
                U.rel_residual_stored);


        U.solverConsistent = ...
            strcmp( ...
                U.solver_used, ...
                U.solver_used_stored);


        U.selectedSystemConsistent = ...
            U.yhatConsistent && ...
            U.residualConsistent && ...
            U.AEConsistent && ...
            U.condRegConsistent && ...
            U.relResidualConsistent && ...
            U.solverConsistent;


        % =====================================================
        % 16.15 Success
        % =====================================================

        U.success = ...
            true;


        U.failureStage = ...
            '';


        U.message = ...
            '';


        U.totalWallTime = ...
            toc(tUnique);


        Pi.UniqueSelectedRadius(u) = ...
            U;


        % =====================================================
        % 16.16 Map ONE unique result to ALL caps using it
        %
        % No new RBF solve or diagnostic calculation occurs.
        % =====================================================

        Diagnostics. ...
            diagnostic_success_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            true;


        Diagnostics. ...
            selectedSystemConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.selectedSystemConsistent;


        % -----------------------------------------------------
        % Condition diagnostics
        % -----------------------------------------------------

        Diagnostics. ...
            cond_raw_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.cond_raw;


        Diagnostics. ...
            cond_reg_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.cond_reg;


        Diagnostics. ...
            cond_ratio_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.cond_ratio;


        Diagnostics. ...
            delta_log10_cond_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.delta_log10_cond;


        Diagnostics. ...
            cond_effective_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.cond_effective;


        % -----------------------------------------------------
        % Rank diagnostics
        % -----------------------------------------------------

        Diagnostics. ...
            rank_eff_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.rank_eff;


        Diagnostics. ...
            rank_tol_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.rank_tol;


        Diagnostics. ...
            sigma_min_retained_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.sigma_min_retained;


        % -----------------------------------------------------
        % Evaluation condition
        % -----------------------------------------------------

        Diagnostics. ...
            cond_evaluation_train_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.cond_evaluation_train;


        Diagnostics. ...
            cond_evaluation_target_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.cond_evaluation_target;


        % -----------------------------------------------------
        % Linear-system diagnostics
        % -----------------------------------------------------

        Diagnostics. ...
            rel_residual_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.rel_residual;


        Diagnostics. ...
            backward_error_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.backward_error;


        for kCap = capsUsingRadius

            Diagnostics. ...
                solver_used_by_point_cap{ ...
                    iTarget, ...
                    kCap} = ...
                U.solver_used;

        end


        % -----------------------------------------------------
        % Prediction diagnostics
        % -----------------------------------------------------

        Diagnostics. ...
            yhat_recomputed_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.yhat_recomputed;


        Diagnostics. ...
            residual_recomputed_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.residual_recomputed;


        Diagnostics. ...
            AE_recomputed_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.AE_recomputed;


        % -----------------------------------------------------
        % Consistency differences
        % -----------------------------------------------------

        Diagnostics. ...
            yhatDifference_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.yhatDifference;


        Diagnostics. ...
            residualDifference_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.residualDifference;


        Diagnostics. ...
            AEDifference_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.AEDifference;


        Diagnostics. ...
            condRegDifference_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.condRegDifference;


        Diagnostics. ...
            relResidualDifference_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.relResidualDifference;


        % -----------------------------------------------------
        % Consistency status
        % -----------------------------------------------------

        Diagnostics. ...
            yhatConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.yhatConsistent;


        Diagnostics. ...
            residualConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.residualConsistent;


        Diagnostics. ...
            AEConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.AEConsistent;


        Diagnostics. ...
            condRegConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.condRegConsistent;


        Diagnostics. ...
            relResidualConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.relResidualConsistent;


        Diagnostics. ...
            solverConsistent_by_point_cap( ...
                iTarget, ...
                capsUsingRadius) = ...
            U.solverConsistent;

    end


    % =========================================================
    % 16.17 Point status
    % =========================================================

    uniqueSuccess = ...
        false(nUnique,1);


    for u = 1:nUnique

        uniqueSuccess(u) = ...
            Pi.UniqueSelectedRadius(u).success;

    end


    Pi.nSuccessfulUniqueDiagnostics = ...
        sum(uniqueSuccess);


    Pi.nFailedUniqueDiagnostics = ...
        nUnique - ...
        Pi.nSuccessfulUniqueDiagnostics;


    Pi.success = ...
        isempty(invalidCaps) && ...
        Pi.nSuccessfulUniqueDiagnostics == nUnique;


    if Pi.success

        Pi.message = ...
            '';

    elseif ~isempty(invalidCaps)

        Pi.message = ...
            'One or more successful caps have invalid radius indices.';

    else

        Pi.message = ...
            sprintf( ...
                ['Diagnostics succeeded for %d of %d unique ' ...
                 'selected radius systems.'], ...
                Pi.nSuccessfulUniqueDiagnostics, ...
                nUnique);

    end


    Pi.totalWallTime = ...
        toc(tPoint);


    Diagnostics.Point(iTarget) = ...
        Pi;


    Diagnostics.pointWallTime(iTarget) = ...
        Pi.totalWallTime;

end


Diagnostics.diagnosticsWallTime = ...
    toc(tDiagnostics);


% =============================================================
% 17. Overall diagnostic counts
% =============================================================

Diagnostics.nSuccessfulUniqueSystems = ...
    0;


for iTarget = 1:n

    if isempty(Diagnostics.Point(iTarget).UniqueSelectedRadius)

        continue

    end


    Diagnostics.nSuccessfulUniqueSystems = ...
        Diagnostics.nSuccessfulUniqueSystems + ...
        sum( ...
            [Diagnostics.Point(iTarget). ...
             UniqueSelectedRadius.success]);

end


Diagnostics.nFailedUniqueSystems = ...
    Diagnostics.nUniqueSelectedSystemsTotal - ...
    Diagnostics.nSuccessfulUniqueSystems;


Diagnostics.nSuccessfulPointCapDiagnostics = ...
    sum( ...
        Diagnostics.diagnostic_success_by_point_cap(:));


Diagnostics.nFailedSelectedPointCapDiagnostics = ...
    Diagnostics.nSelectedPointCapPairs - ...
    Diagnostics.nSuccessfulPointCapDiagnostics;


% =============================================================
% 18. Runtime-saving statistics
%
% Without unique-radius reuse:
%
%       one diagnostic calculation per successful point-cap pair
%
% With reuse:
%
%       one diagnostic calculation per unique selected
%       point-radius system
% =============================================================

Diagnostics.nDiagnosticCalculationsAvoided = ...
    Diagnostics.nSelectedPointCapPairs - ...
    Diagnostics.nUniqueSelectedSystemsTotal;


if Diagnostics.nSelectedPointCapPairs > 0

    Diagnostics.diagnosticReuseFraction = ...
        Diagnostics.nDiagnosticCalculationsAvoided / ...
        Diagnostics.nSelectedPointCapPairs;

else

    Diagnostics.diagnosticReuseFraction = ...
        NaN;

end


% =============================================================
% 19. Final selected-cap diagnostics
%
% Extracted by mapping only.
% NO additional system reconstruction or solve is performed.
% =============================================================

Diagnostics.Final = ...
    emptyFinalDiagnostic();


if isfield(TrainingResult,'Final') && ...
   isstruct(TrainingResult.Final) && ...
   isfield(TrainingResult.Final,'success') && ...
   TrainingResult.Final.success && ...
   isfield(TrainingResult.Final,'capIndex')


    kStar = ...
        TrainingResult.Final.capIndex;


    if isscalar(kStar) && ...
       isfinite(kStar) && ...
       kStar == round(kStar) && ...
       kStar >= 1 && ...
       kStar <= K

        Diagnostics.Final.capIndex = ...
            kStar;


        Diagnostics.Final.capRadius = ...
            TrainingResult.Final.capRadius;


        Diagnostics.Final.trainingRMSE = ...
            TrainingResult.Final.RMSE;


        Diagnostics.Final.uniqueDiagnosticIndex = ...
            Diagnostics. ...
            uniqueDiagnosticIndex_by_point_cap(:,kStar);


        Diagnostics.Final.diagnostic_success = ...
            Diagnostics. ...
            diagnostic_success_by_point_cap(:,kStar);


        Diagnostics.Final.selectedSystemConsistent = ...
            Diagnostics. ...
            selectedSystemConsistent_by_point_cap(:,kStar);


        Diagnostics.Final.cond_raw = ...
            Diagnostics. ...
            cond_raw_by_point_cap(:,kStar);


        Diagnostics.Final.cond_reg = ...
            Diagnostics. ...
            cond_reg_by_point_cap(:,kStar);


        Diagnostics.Final.cond_ratio = ...
            Diagnostics. ...
            cond_ratio_by_point_cap(:,kStar);


        Diagnostics.Final.delta_log10_cond = ...
            Diagnostics. ...
            delta_log10_cond_by_point_cap(:,kStar);


        Diagnostics.Final.cond_effective = ...
            Diagnostics. ...
            cond_effective_by_point_cap(:,kStar);


        Diagnostics.Final.rank_eff = ...
            Diagnostics. ...
            rank_eff_by_point_cap(:,kStar);


        Diagnostics.Final.rank_tol = ...
            Diagnostics. ...
            rank_tol_by_point_cap(:,kStar);


        Diagnostics.Final.sigma_min_retained = ...
            Diagnostics. ...
            sigma_min_retained_by_point_cap(:,kStar);


        Diagnostics.Final.cond_evaluation_train = ...
            Diagnostics. ...
            cond_evaluation_train_by_point_cap(:,kStar);


        Diagnostics.Final.cond_evaluation_target = ...
            Diagnostics. ...
            cond_evaluation_target_by_point_cap(:,kStar);


        Diagnostics.Final.rel_residual = ...
            Diagnostics. ...
            rel_residual_by_point_cap(:,kStar);


        Diagnostics.Final.backward_error = ...
            Diagnostics. ...
            backward_error_by_point_cap(:,kStar);


        Diagnostics.Final.solver_used = ...
            Diagnostics. ...
            solver_used_by_point_cap(:,kStar);


        Diagnostics.Final.yhat_recomputed = ...
            Diagnostics. ...
            yhat_recomputed_by_point_cap(:,kStar);


        Diagnostics.Final.residual_recomputed = ...
            Diagnostics. ...
            residual_recomputed_by_point_cap(:,kStar);


        Diagnostics.Final.AE_recomputed = ...
            Diagnostics. ...
            AE_recomputed_by_point_cap(:,kStar);


        Diagnostics.Final.RMSE_recomputed = ...
            NaN;


        Diagnostics.Final.RMSEDifference = ...
            NaN;


        Diagnostics.Final.RMSEConsistencyTolerance = ...
            NaN;


        Diagnostics.Final.RMSEConsistent = ...
            false;


        if all(Diagnostics.Final.diagnostic_success) && ...
           all(isfinite( ...
               Diagnostics.Final.residual_recomputed))

            Diagnostics.Final.RMSE_recomputed = ...
                sqrt( ...
                    mean( ...
                        Diagnostics.Final. ...
                        residual_recomputed.^2));


            [ ...
                Diagnostics.Final.RMSEDifference, ...
                Diagnostics.Final.RMSEConsistencyTolerance, ...
                Diagnostics.Final.RMSEConsistent] = ...
                compareScalar( ...
                    Diagnostics.Final.RMSE_recomputed, ...
                    Diagnostics.Final.trainingRMSE);

        end


        Diagnostics.Final.success = ...
            all( ...
                Diagnostics.Final.diagnostic_success);


        if Diagnostics.Final.success

            Diagnostics.Final.message = ...
                '';

        else

            Diagnostics.Final.message = ...
                ['Diagnostics were not successful for every ' ...
                 'point at the final selected cap.'];

        end

    end

end


% =============================================================
% 19.1 Canonical FINAL-system numerical aliases
%
% These compact vectors are authoritative for compulsory Stage-2 numerical
% storage. They are aliases only; no new system is reconstructed or solved.
% Keeping them at top level also makes FULL and FINAL evaluation modes expose
% the same compulsory numerical interface.
% =============================================================

Diagnostics.point_success = ...
    Diagnostics.Final.diagnostic_success;


Diagnostics.cond_raw = ...
    Diagnostics.Final.cond_raw;


Diagnostics.cond_reg = ...
    Diagnostics.Final.cond_reg;


Diagnostics.cond_effective = ...
    Diagnostics.Final.cond_effective;


Diagnostics.solver_used = ...
    Diagnostics.Final.solver_used;


% =============================================================
% 20. Overall status
% =============================================================

Diagnostics.success = ...
    Diagnostics.nFailedSelectedPointCapDiagnostics == 0;


if Diagnostics.success

    Diagnostics.message = ...
        '';

else

    Diagnostics.message = ...
        sprintf( ...
            ['Diagnostics succeeded for %d of %d selected ' ...
             'point-cap results using %d unique systems.'], ...
            Diagnostics.nSuccessfulPointCapDiagnostics, ...
            Diagnostics.nSelectedPointCapPairs, ...
            Diagnostics.nUniqueSelectedSystemsTotal);

end


% =============================================================
% 21. Total runtime
% =============================================================

Diagnostics.totalWallTime = ...
    Diagnostics.geometryWallTime + ...
    Diagnostics.diagnosticsWallTime;

end


% =============================================================
% Local helper
% =============================================================

function Diagnostics = evaluateFinalSelectedSystems( ...
    TrainingResult, options, geometryCache)
%EVALUATEFINALSELECTEDSYSTEMS Diagnose only final selected systems.
%
% Compact Stage-2 path. It constructs only the compulsory final-system
% numerical interface plus small aggregate counters. It intentionally does
% NOT construct Point detail, point-by-cap diagnostic matrices, coefficients,
% condition-detail containers, consistency-detail arrays, series, or maps.
%
% The actual selected local RBF system uses exactly the same common
% build/solve/predict/condition functions as FULL mode.


% =============================================================
% F1. Input and solver checks
% =============================================================

if ~isstruct(TrainingResult)
    error('TrainingResult must be a structure.');
end


requiredFields = { ...
    'X', ...
    'y', ...
    'kernel', ...
    'useShape', ...
    'minNeighbors', ...
    'Final'};


for q = 1:numel(requiredFields)

    if ~isfield(TrainingResult,requiredFields{q})
        error( ...
            'TrainingResult is missing required field "%s".', ...
            requiredFields{q});
    end

end


if nargin < 2 || isempty(options)
    options = struct();
end


if ~isstruct(options)
    error('options must be a structure.');
end


if ~isfield(options,'solver') || isempty(options.solver)
    options.solver = struct();
end


if ~isfield(options.solver,'cond_lsq') || ...
   isempty(options.solver.cond_lsq)
    options.solver.cond_lsq = 1e12;
end


% Cond1/Cond2/Cond3 are compulsory Stage-2 numerical outputs. The existing
% condition-diagnostic routine also needs the solve relative residual.
options.solver.compute_rel_residual = true;


X = TrainingResult.X;
y = TrainingResult.y(:);
[n,p] = size(X);


if numel(y) ~= n
    error('TrainingResult.X and TrainingResult.y are inconsistent.');
end


kernel = TrainingResult.kernel;
useShape = TrainingResult.useShape;
F = TrainingResult.Final;


if ~isstruct(F) || ~isscalar(F) || ...
   ~isfield(F,'success') || ~F.success
    error('TrainingResult.Final does not contain a successful final model.');
end


% Only fields required to reconstruct the already-selected final system are
% required in compact mode. The variable-length neighbourhood itself is not
% retained: it is rebuilt point-locally from radius + minNeighbors + cached
% distances. Stored prediction/diagnostic copies are DETAIL and are not read.
requiredFinalFields = { ...
    'point_success', ...
    'radius', ...
    'C', ...
    'reg_param', ...
    'epsilon', ...
    'neighbor_count'};


for q = 1:numel(requiredFinalFields)

    if ~isfield(F,requiredFinalFields{q})
        error( ...
            'TrainingResult.Final is missing required field "%s".', ...
            requiredFinalFields{q});
    end

end


validateFinalLogicalVector(F.point_success,n,'point_success');
validateFinalVectorLength(F.radius,n,'radius');
validateFinalVectorLength(F.C,n,'C');
validateFinalVectorLength(F.reg_param,n,'reg_param');
validateFinalVectorLength(F.epsilon,n,'epsilon');
validateFinalVectorLength(F.neighbor_count,n,'neighbor_count');


minNeighbors = TrainingResult.minNeighbors;


if ~isnumeric(minNeighbors) || ...
   ~isscalar(minNeighbors) || ...
   ~isreal(minNeighbors) || ...
   ~isfinite(minNeighbors) || ...
   minNeighbors < 1 || ...
   minNeighbors ~= round(minNeighbors)

    error('TrainingResult.minNeighbors must be a positive integer.');

end


% =============================================================
% F2. Shared geometry
% =============================================================

tGeometry = tic;


geometryReused = ...
    nargin >= 3 && ...
    ~isempty(geometryCache);


if geometryReused

    if ~isstruct(geometryCache) || ...
       ~isfield(geometryCache,'D') || ...
       ~isfield(geometryCache,'P')
        error('geometryCache must contain fields D and P.');
    end


    if ~isnumeric(geometryCache.D) || ...
       ~isequal(size(geometryCache.D),[n n])
        error('geometryCache.D must be an n-by-n matrix.');
    end


    if ~isnumeric(geometryCache.P) || ...
       ~isequal(size(geometryCache.P),[n p+1])
        error('geometryCache.P must be n-by-(p+1).');
    end

else

    geometryCache = struct();
    geometryCache.D = pdist2(X,X);
    geometryCache.P = [ones(n,1),X];

end


geometryWallTime = toc(tGeometry);


% =============================================================
% F3. Compulsory output only
% =============================================================

Diagnostics = struct();
Diagnostics.success = false;
Diagnostics.kernel = kernel;
Diagnostics.useShape = useShape;
Diagnostics.nPoints = n;
Diagnostics.nCaps = getNumericFieldOrNaN(TrainingResult,'nCaps');
Diagnostics.geometryReused = geometryReused;
Diagnostics.geometryWallTime = geometryWallTime;
Diagnostics.message = '';
Diagnostics.evaluationMode = 'final';


% Canonical Stage-2 NUMERICAL interface.
Diagnostics.point_success = false(n,1);
Diagnostics.cond_raw = nan(n,1);
Diagnostics.cond_reg = nan(n,1);
Diagnostics.cond_effective = nan(n,1);
Diagnostics.solver_used = cell(n,1);


Diagnostics.nRBFSolveAttempts = 0;
Diagnostics.nSuccessfulRBFSolves = 0;
Diagnostics.nRBFSolves = 0;


% Preserve all-cap reuse counters without evaluating non-final systems.
[Diagnostics.nSelectedPointCapPairs, ...
 Diagnostics.nUniqueSelectedSystemsTotal] = ...
    selectionReuseCounts(TrainingResult,n);


if isfinite(Diagnostics.nSelectedPointCapPairs) && ...
   isfinite(Diagnostics.nUniqueSelectedSystemsTotal)

    Diagnostics.nDiagnosticCalculationsAvoided = ...
        Diagnostics.nSelectedPointCapPairs - ...
        Diagnostics.nUniqueSelectedSystemsTotal;


    if Diagnostics.nSelectedPointCapPairs > 0
        Diagnostics.diagnosticReuseFraction = ...
            Diagnostics.nDiagnosticCalculationsAvoided ./ ...
            Diagnostics.nSelectedPointCapPairs;
    else
        Diagnostics.diagnosticReuseFraction = NaN;
    end

else

    Diagnostics.nDiagnosticCalculationsAvoided = NaN;
    Diagnostics.diagnosticReuseFraction = NaN;

end


% =============================================================
% F4. Diagnose exactly one final selected system per point
% =============================================================

tDiagnostics = tic;


for iTarget = 1:n

    if ~F.point_success(iTarget)
        continue
    end


    % Reconstruct the ONE final selected neighbourhood transiently.
    % This is the same getNeighborhood definition used by training, with
    % the same scaled coordinates, target exclusion, minimum-neighbour
    % fallback, selected radius, and cached distance vector. Nothing is
    % retained after this point iteration.
    radius = F.radius(iTarget);


    if ~isscalar(radius) || ~isreal(radius) || ...
       ~isfinite(radius) || radius <= 0
        continue
    end


    try

        Nbh = ...
            localrbf.getNeighborhood( ...
                X, ...
                X(iTarget,:), ...
                radius, ...
                minNeighbors, ...
                iTarget, ...
                geometryCache.D(iTarget,:).');

    catch

        continue

    end


    if ~isstruct(Nbh) || ...
       ~isfield(Nbh,'success') || ...
       ~Nbh.success
        continue
    end


    idxFit = Nbh.idx(:);


    if isempty(idxFit) || ...
       ~isnumeric(idxFit) || ...
       ~isreal(idxFit) || ...
       any(~isfinite(idxFit)) || ...
       any(idxFit ~= round(idxFit)) || ...
       any(idxFit < 1) || ...
       any(idxFit > n) || ...
       numel(unique(idxFit)) ~= numel(idxFit) || ...
       any(idxFit == iTarget)
        continue
    end


    if ~isscalar(F.neighbor_count(iTarget)) || ...
       ~isfinite(F.neighbor_count(iTarget)) || ...
       F.neighbor_count(iTarget) ~= Nbh.count || ...
       Nbh.count ~= numel(idxFit)
        continue
    end


    regParam = F.reg_param(iTarget);
    epsilon = F.epsilon(iTarget);
    C = F.C(iTarget);


    if ~isscalar(regParam) || ~isreal(regParam) || ...
       ~isfinite(regParam) || regParam <= 0 || ...
       ~isscalar(epsilon) || ~isreal(epsilon) || ...
       ~isfinite(epsilon) || epsilon <= 0
        continue
    end


    if useShape && ...
       (~isscalar(C) || ~isreal(C) || ~isfinite(C) || C <= 0)
        continue
    end


    Xfit = X(idxFit,:);
    yfit = y(idxFit);


    geometry = struct();
    geometry.R = geometryCache.D(idxFit,idxFit);
    geometry.P = geometryCache.P(idxFit,:);
    geometry.R_eval = reshape(geometryCache.D(iTarget,idxFit),1,[]);
    geometry.P_eval = geometryCache.P(iTarget,:);


    try

        S = common.buildRBFSystem( ...
            Xfit, ...
            yfit, ...
            kernel, ...
            epsilon, ...
            regParam, ...
            geometry);


        Diagnostics.nRBFSolveAttempts = ...
            Diagnostics.nRBFSolveAttempts + 1;


        [coeffs,solveInfo] = ...
            common.solveLinearSystem( ...
                S, ...
                options);


        Diagnostics.nRBFSolves = ...
            Diagnostics.nRBFSolves + 1;


        % Match FULL mode exactly: solveInfo.success is the authoritative
        % solver-success decision. Any downstream coefficient/prediction
        % problem is handled by the same predict/diagnostic path and catch.
        if ~solveInfo.success
            continue
        end


        Diagnostics.nSuccessfulRBFSolves = ...
            Diagnostics.nSuccessfulRBFSolves + 1;


        [yhat,B_target] = ...
            common.predictRBF( ...
                X(iTarget,:), ...
                Xfit, ...
                coeffs, ...
                kernel, ...
                epsilon, ...
                geometry);


        % Match FULL mode's selected-prediction validity gate before
        % calculating condition diagnostics.
        if ~isscalar(yhat) || ...
           ~isreal(yhat) || ...
           ~isfinite(yhat)
            continue
        end


        Di = ...
            common.calculateConditionDiagnostics( ...
                S, ...
                coeffs, ...
                solveInfo, ...
                B_target);

    catch
        continue
    end


    % Retain only compulsory numerical fields. All other fields in Di and
    % all matrices/coefficients are point-local temporaries and die here.
    Diagnostics.point_success(iTarget) = true;
    Diagnostics.cond_raw(iTarget) = Di.cond_raw;
    Diagnostics.cond_reg(iTarget) = Di.cond_reg;
    Diagnostics.cond_effective(iTarget) = Di.cond_effective;
    Diagnostics.solver_used{iTarget} = Di.solver_used;

end


Diagnostics.diagnosticsWallTime = toc(tDiagnostics);


% =============================================================
% F5. Final status
% =============================================================

Diagnostics.success = all(Diagnostics.point_success);


if Diagnostics.success
    Diagnostics.message = '';
else
    Diagnostics.message = sprintf( ...
        'Final-system diagnostics succeeded for %d of %d training points.', ...
        sum(Diagnostics.point_success), ...
        n);
end


Diagnostics.totalWallTime = ...
    Diagnostics.geometryWallTime + ...
    Diagnostics.diagnosticsWallTime;

end


% =============================================================
% Local helper
% =============================================================

function validateFinalVectorLength(value,n,fieldName)
%VALIDATEFINALVECTORLENGTH Require one scalar entry per training point.

if ~isnumeric(value) || ...
   ~isreal(value) || ...
   ~isvector(value) || ...
   numel(value) ~= n

    error( ...
        'TrainingResult.Final.%s must contain one real numeric value per point.', ...
        fieldName);

end

end


% =============================================================
% Local helper
% =============================================================

function validateFinalLogicalVector(value,n,fieldName)
%VALIDATEFINALLOGICALVECTOR Require one valid logical/0-1 entry per point.

validType = islogical(value) || isnumeric(value);
validValues = islogical(value) || ...
    (isreal(value) && all(isfinite(value(:))) && ...
     all(value(:) == 0 | value(:) == 1));


if ~validType || ~validValues || ...
   ~isvector(value) || numel(value) ~= n

    error( ...
        'TrainingResult.Final.%s must contain one logical/0-1 value per point.', ...
        fieldName);

end

end


% =============================================================
% Local helper
% =============================================================

function [nPairs,nUnique] = selectionReuseCounts(T,n)
%SELECTIONREUSECOUNTS Recover original all-cap reuse counters without solves.

nPairs = NaN;
nUnique = NaN;


if ~isfield(T,'cap_success_by_point') || ...
   ~isfield(T,'selectedRadiusIndex_by_point_cap')
    return
end


capSuccess = T.cap_success_by_point;
selectedIdx = T.selectedRadiusIndex_by_point_cap;


if ~isequal(size(capSuccess),size(selectedIdx)) || ...
   size(capSuccess,1) ~= n || ...
   ~(islogical(capSuccess) || isnumeric(capSuccess)) || ...
   ~isnumeric(selectedIdx)
    return
end


if isnumeric(capSuccess) && ...
   (~isreal(capSuccess) || ...
    any(~isfinite(capSuccess(:))) || ...
    any(capSuccess(:) ~= 0 & capSuccess(:) ~= 1))
    return
end


capSuccess = logical(capSuccess);
nPairs = sum(capSuccess(:));
nUnique = 0;


for i = 1:n

    row = selectedIdx(i,:);
    mask = capSuccess(i,:) & ...
           isfinite(row) & ...
           row == round(row) & ...
           row >= 1;


    if any(mask)
        nUnique = nUnique + numel(unique(row(mask)));
    end

end

end


% =============================================================
% Local helper
% =============================================================

function value = getNumericFieldOrNaN(S,fieldName)
%GETNUMERICFIELDORNAN Compact scalar metadata lookup.

value = NaN;


if isstruct(S) && isfield(S,fieldName) && ...
   isnumeric(S.(fieldName)) && isscalar(S.(fieldName)) && ...
   isreal(S.(fieldName)) && isfinite(S.(fieldName))
    value = S.(fieldName);
end

end

% =============================================================
% Local helper
% =============================================================

function P = emptyPointDiagnostic()
%EMPTYPOINTDIAGNOSTIC


P = ...
    struct();


P.iTarget = ...
    NaN;


P.xTarget = ...
    [];


P.yTarget = ...
    NaN;


P.selectedRadiusIndex_by_cap = ...
    [];


P.cap_success = ...
    [];


P.validSelectedCaps = ...
    [];


P.invalidSelectedCaps = ...
    [];


P.uniqueSelectedRadiusIndex = ...
    [];


P.nSelectedCaps = ...
    0;


P.nUniqueSelectedRadii = ...
    0;


P.nDiagnosticsReused = ...
    0;


P.nSuccessfulUniqueDiagnostics = ...
    0;


P.nFailedUniqueDiagnostics = ...
    0;


P.UniqueSelectedRadius = ...
    [];


P.success = ...
    false;


P.message = ...
    '';


P.totalWallTime = ...
    0;

end


% =============================================================
% Local helper
% =============================================================

function U = emptyUniqueDiagnostic()
%EMPTYUNIQUEDIAGNOSTIC
%
% One expensive diagnostic result for one unique selected
% point-radius system.


U = ...
    struct();


% -------------------------------------------------------------
% Source / reuse mapping
% -------------------------------------------------------------

U.uniqueDiagnosticIndex = ...
    NaN;


U.radiusIndex = ...
    NaN;


U.radius = ...
    NaN;


U.capsUsingRadius = ...
    [];


U.nCapsUsingRadius = ...
    0;


U.reuseCount = ...
    0;


% -------------------------------------------------------------
% Optimized parameters
% -------------------------------------------------------------

U.C = ...
    NaN;


U.reg_param = ...
    NaN;


U.epsilon = ...
    NaN;


U.d_local = ...
    NaN;


% -------------------------------------------------------------
% Neighbourhood
% -------------------------------------------------------------

U.neighbor_idx = ...
    [];


U.neighbor_distances = ...
    [];


U.neighbor_count = ...
    0;


U.system_size = ...
    NaN;


% -------------------------------------------------------------
% Stored selected optimization values
% -------------------------------------------------------------

U.yhat_stored = ...
    NaN;


U.residual_stored = ...
    NaN;


U.AE_stored = ...
    NaN;


U.cond_reg_stored = ...
    NaN;


U.rel_residual_stored = ...
    NaN;


U.solver_used_stored = ...
    '';


% -------------------------------------------------------------
% Coefficients
% -------------------------------------------------------------

U.coeffs = ...
    [];


U.lambda = ...
    [];


U.beta = ...
    [];


% -------------------------------------------------------------
% Complete common diagnostic
% -------------------------------------------------------------

U.condition = ...
    [];


% -------------------------------------------------------------
% Condition diagnostics
% -------------------------------------------------------------

U.cond_raw = ...
    NaN;


U.cond_reg = ...
    NaN;


U.cond_ratio = ...
    NaN;


U.delta_log10_cond = ...
    NaN;


U.cond_effective = ...
    NaN;


% -------------------------------------------------------------
% Rank diagnostics
% -------------------------------------------------------------

U.rank_eff = ...
    NaN;


U.rank_tol = ...
    NaN;


U.sigma_min_retained = ...
    NaN;


% -------------------------------------------------------------
% Evaluation-condition diagnostics
% -------------------------------------------------------------

U.cond_evaluation_train = ...
    NaN;


U.cond_evaluation_target = ...
    NaN;


% -------------------------------------------------------------
% Linear-system diagnostics
% -------------------------------------------------------------

U.rel_residual = ...
    NaN;


U.backward_error = ...
    NaN;


U.solver_used = ...
    '';


% -------------------------------------------------------------
% Recomputed prediction
% -------------------------------------------------------------

U.yhat_recomputed = ...
    NaN;


U.residual_recomputed = ...
    NaN;


U.AE_recomputed = ...
    NaN;


% -------------------------------------------------------------
% Consistency differences
% -------------------------------------------------------------

U.yhatDifference = ...
    NaN;


U.residualDifference = ...
    NaN;


U.AEDifference = ...
    NaN;


U.condRegDifference = ...
    NaN;


U.relResidualDifference = ...
    NaN;


% -------------------------------------------------------------
% Consistency tolerances
% -------------------------------------------------------------

U.yhatConsistencyTolerance = ...
    NaN;


U.residualConsistencyTolerance = ...
    NaN;


U.AEConsistencyTolerance = ...
    NaN;


U.condRegConsistencyTolerance = ...
    NaN;


U.relResidualConsistencyTolerance = ...
    NaN;


% -------------------------------------------------------------
% Consistency status
% -------------------------------------------------------------

U.yhatConsistent = ...
    false;


U.residualConsistent = ...
    false;


U.AEConsistent = ...
    false;


U.condRegConsistent = ...
    false;


U.relResidualConsistent = ...
    false;


U.solverConsistent = ...
    false;


U.selectedSystemConsistent = ...
    false;


% -------------------------------------------------------------
% Status
% -------------------------------------------------------------

U.success = ...
    false;


U.failureStage = ...
    '';


U.message = ...
    '';


U.totalWallTime = ...
    0;

end


% =============================================================
% Local helper
% =============================================================

function F = emptyFinalDiagnostic()
%EMPTYFINALDIAGNOSTIC


F = ...
    struct();


F.success = ...
    false;


F.capIndex = ...
    NaN;


F.capRadius = ...
    NaN;


F.trainingRMSE = ...
    NaN;


F.uniqueDiagnosticIndex = ...
    [];


F.diagnostic_success = ...
    [];


F.selectedSystemConsistent = ...
    [];


F.cond_raw = ...
    [];


F.cond_reg = ...
    [];


F.cond_ratio = ...
    [];


F.delta_log10_cond = ...
    [];


F.cond_effective = ...
    [];


F.rank_eff = ...
    [];


F.rank_tol = ...
    [];


F.sigma_min_retained = ...
    [];


F.cond_evaluation_train = ...
    [];


F.cond_evaluation_target = ...
    [];


F.rel_residual = ...
    [];


F.backward_error = ...
    [];


F.solver_used = ...
    {};


F.yhat_recomputed = ...
    [];


F.residual_recomputed = ...
    [];


F.AE_recomputed = ...
    [];


F.RMSE_recomputed = ...
    NaN;


F.RMSEDifference = ...
    NaN;


F.RMSEConsistencyTolerance = ...
    NaN;


F.RMSEConsistent = ...
    false;


F.message = ...
    '';

end


% =============================================================
% Local helper
% =============================================================

function [difference,tolerance,isConsistent] = ...
    compareScalar(a,b)
%COMPARESCALAR
%
% Relative-scale consistency comparison.
%
% Handles matching infinite values explicitly.


difference = ...
    NaN;


tolerance = ...
    NaN;


isConsistent = ...
    false;


if isscalar(a) && ...
   isscalar(b) && ...
   isreal(a) && ...
   isreal(b)


    if isfinite(a) && ...
       isfinite(b)

        difference = ...
            abs(a-b);


        tolerance = ...
            1e-10 * ...
            max( ...
                [1,abs(a),abs(b)]);


        isConsistent = ...
            difference <= tolerance;


    elseif isinf(a) && ...
           isinf(b) && ...
           sign(a) == sign(b)

        difference = ...
            0;


        tolerance = ...
            0;


        isConsistent = ...
            true;

    end

end

end
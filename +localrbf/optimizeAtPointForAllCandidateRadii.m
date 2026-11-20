function PointResult = optimizeAtPointForAllCandidateRadii( ...
    X, y, iTarget, radii, kernel, ...
    min_neighbors, ...
    C_min, C_max, ...
    Reg_min, Reg_max, ...
    options, geometryCache, method, nGrid)
%OPTIMIZEATPOINTFORALLCANDIDATERADII
%
% Optimize ONE training point over ALL candidate radii.
%
% For each radius:
%   1. exclude the target observation;
%   2. construct the local neighbourhood;
%   3. enforce the minimum-neighbour requirement;
%   4. check polynomial admissibility;
%   5. calculate the local geometry scale once;
%   6. extract fixed R, P, R_eval, and P_eval from the
%      precomputed training geometry;
%   7. optimize C and RegParam jointly for shaped kernels,
%      or RegParam only for shape-free kernels;
%   8. retain the complete result for every radius.
%
% IMPORTANT:
%   - one call processes one target point;
%   - all candidate radii are evaluated;
%   - minimum fitting observations default to 15;
%   - target exclusion is by observation index;
%   - the full training distance matrix is precomputed once
%     outside this function and reused;
%   - consecutive radii producing exactly the same fitting-index
%     set reuse the completed successful optimization result;
%   - neighbourhood reuse is based on exact index equality, not
%     only on neighbour count;
%   - no radius or cap selection is performed here;
%   - no RNG reset is performed here;
%   - no full CN/ECN/EVN diagnostics are calculated here;
%   - complete parameter-search histories are retained;
%   - large temporary RBF matrices are not duplicated in the
%     returned structure.


% =============================================================
% 1. Defaults
% =============================================================

if nargin < 6 || isempty(min_neighbors)

    min_neighbors = ...
        15;

end


if nargin < 11 || isempty(options)

    options = ...
        struct();

end


if nargin < 12 || isempty(geometryCache)

    error( ...
        ['geometryCache is required and must contain the ' ...
         'precomputed fields D and P.']);

end

if nargin < 13 || isempty(method)
    method = 'ga';
end

if nargin < 14 || isempty(nGrid)
    nGrid = 25;
end
% =============================================================
% 2. Basic input checks
% =============================================================

if ~isnumeric(X) || ...
   isempty(X) || ...
   ~ismatrix(X) || ...
   ~isreal(X) || ...
   any(~isfinite(X(:)))

    error( ...
        'X must be a nonempty finite real numeric matrix.');

end


if ~isnumeric(y) || ...
   isempty(y) || ...
   ~isvector(y) || ...
   ~isreal(y) || ...
   any(~isfinite(y(:)))

    error( ...
        'y must be a nonempty finite real numeric vector.');

end


y = ...
    y(:);


[n,p] = ...
    size(X);


if numel(y) ~= n

    error( ...
        'X and y must contain the same number of observations.');

end


if ~isnumeric(iTarget) || ...
   ~isscalar(iTarget) || ...
   ~isreal(iTarget) || ...
   ~isfinite(iTarget) || ...
   iTarget ~= round(iTarget) || ...
   iTarget < 1 || ...
   iTarget > n

    error( ...
        'iTarget must be a valid observation index.');

end


if ~isnumeric(radii) || ...
   isempty(radii) || ...
   ~isvector(radii) || ...
   ~isreal(radii) || ...
   any(~isfinite(radii(:))) || ...
   any(radii(:) <= 0)

    error( ...
        'radii must be a nonempty finite positive numeric vector.');

end


radii = ...
    radii(:);


if any(diff(radii) <= 0)

    error( ...
        'Candidate radii must be strictly increasing.');

end


if ~isnumeric(min_neighbors) || ...
   ~isscalar(min_neighbors) || ...
   ~isreal(min_neighbors) || ...
   ~isfinite(min_neighbors) || ...
   min_neighbors < 1 || ...
   min_neighbors ~= round(min_neighbors)

    error( ...
        'min_neighbors must be a positive integer.');

end


if n - 1 < min_neighbors

    error( ...
        ['At least %d observations must remain after target ' ...
         'exclusion, but only %d are available.'], ...
        min_neighbors, ...
        n - 1);

end


% =============================================================
% 3. Geometry-cache checks
%
% Only dimensions are checked here.
% Expensive full-matrix validation should be performed once
% when geometryCache is created at the training level.
% =============================================================

if ~isstruct(geometryCache) || ...
   ~isfield(geometryCache,'D') || ...
   ~isfield(geometryCache,'P')

    error( ...
        'geometryCache must contain fields D and P.');

end


if ~isnumeric(geometryCache.D) || ...
   ~isequal(size(geometryCache.D),[n n])

    error( ...
        'geometryCache.D must be an n-by-n distance matrix.');

end


if ~isnumeric(geometryCache.P) || ...
   ~isequal(size(geometryCache.P),[n p+1])

    error( ...
        'geometryCache.P must be n-by-(p+1).');

end


% =============================================================
% 4. Fixed point information
% =============================================================

useShape = ...
    common.kernelHasShapeParameter(kernel);


xTarget = ...
    X(iTarget,:);


yTarget = ...
    y(iTarget);


M = ...
    numel(radii);


% =============================================================
% 5. Initialize point result
% =============================================================

PointResult = ...
    struct();


PointResult.success = ...
    false;


PointResult.iTarget = ...
    iTarget;


PointResult.xTarget = ...
    xTarget;


PointResult.yTarget = ...
    yTarget;


PointResult.kernel = ...
    kernel;


PointResult.useShape = ...
    useShape;


PointResult.minNeighbors = ...
    min_neighbors;


PointResult.radii = ...
    radii;


PointResult.nRadii = ...
    M;


PointResult.geometryCacheUsed = ...
    true;


PointResult.Radius = ...
    repmat( ...
        emptyRadiusResult(), ...
        M,1);

PointResult.method = ...
    method;

PointResult.nGrid = ...
    nGrid;
% =============================================================
% 6. Point-level counters
% =============================================================

PointResult.nValidRadii = ...
    0;


PointResult.nFailedRadii = ...
    0;


PointResult.nNeighborhoodFailures = ...
    0;


PointResult.nPolynomialFailures = ...
    0;


PointResult.nGeometryFailures = ...
    0;


PointResult.nOptimizationFailures = ...
    0;


PointResult.nUniqueNeighborhoodOptimizations = ...
    0;


PointResult.nReusedRadii = ...
    0;


PointResult.nEvaluations = ...
    0;


PointResult.nGAEvaluations = ...
    0;


PointResult.nFallbackEvaluations = ...
    0;


PointResult.nGridEvaluations = ...
    0;


PointResult.nRefinementEvaluations = ...
    0;
PointResult.nValidEvaluations = ...
    0;


PointResult.nFailedEvaluations = ...
    0;


PointResult.nRBFSolves = ...
    0;


PointResult.message = ...
    '';


% =============================================================
% 7. Point-level timing
% =============================================================

tPoint = ...
    tic;


% =============================================================
% 8. Target-to-training distances
%
% Extracted ONCE from the precomputed full distance matrix.
% No pdist2 call is required here.
% =============================================================

tDistance = ...
    tic;


distAll = ...
    geometryCache.D(:,iTarget);


distAll = ...
    distAll(:);


PointResult.distanceWallTime = ...
    toc(tDistance);


PointResult.distanceLookupWallTime = ...
    PointResult.distanceWallTime;


% -------------------------------------------------------------
% Successful exact-neighbourhood reuse cache
%
% Candidate radii are strictly increasing, so identical fitting
% sets can only occur consecutively. Reuse only a previously
% successful optimization; failed searches are allowed a fresh
% attempt.
% -------------------------------------------------------------

hasReusableOptimization = ...
    false;

reusableIdxFit = ...
    [];

reusableRadiusIndex = ...
    NaN;


% =============================================================
% 9. Process every candidate radius
% =============================================================

for j = 1:M

    tRadius = ...
        tic;


    radius = ...
        radii(j);


    Rj = ...
        emptyRadiusResult();


    Rj.radiusIndex = ...
        j;


    Rj.radius = ...
        radius;


    % =========================================================
    % 9.1 Target-excluded neighbourhood
    % =========================================================

    tNeighborhood = ...
        tic;


    Nbh = ...
        localrbf.getNeighborhood( ...
            X, ...
            xTarget, ...
            radius, ...
            min_neighbors, ...
            iTarget, ...
            distAll);


    Rj.neighborhoodWallTime = ...
        toc(tNeighborhood);


    % ---------------------------------------------------------
    % Retain complete neighbourhood information
    % ---------------------------------------------------------

    Rj.neighborhoodSuccess = ...
        Nbh.success;


    Rj.neighbor_idx = ...
        Nbh.idx;


    Rj.neighbor_distances = ...
        Nbh.distances;


    Rj.neighbor_count = ...
        Nbh.count;


    Rj.requestedRadius = ...
        Nbh.requestedRadius;


    Rj.effectiveRadius = ...
        Nbh.effectiveRadius;


    Rj.selectedMaxDistance = ...
        Nbh.selectedMaxDistance;


    Rj.nAvailable = ...
        Nbh.nAvailable;


    Rj.nWithinRadius = ...
        Nbh.nWithinRadius;


    Rj.nAddedForMinimum = ...
        Nbh.nAddedForMinimum;


    Rj.usedMinNeighborFallback = ...
        Nbh.usedMinNeighborFallback;


    if ~Nbh.success

        Rj.failureStage = ...
            'neighborhood';


        Rj.message = ...
            Nbh.message;


        Rj.totalWallTime = ...
            toc(tRadius);


        PointResult.Radius(j) = ...
            Rj;


        PointResult.nNeighborhoodFailures = ...
            PointResult.nNeighborhoodFailures + 1;


        continue

    end


    % =========================================================
    % 9.2 Local fitting data
    % =========================================================

    idxFit = ...
        Nbh.idx;


    % ---------------------------------------------------------
    % Exact-neighbourhood reuse
    %
    % If the fitting observation indices are unchanged, then
    % Xfit, yfit, polynomial admissibility, d_local, R, P,
    % R_eval, P_eval, and the parameter-optimization objective
    % are all unchanged. Reuse the previous successful result
    % without starting another GA/fallback/refinement search.
    % ---------------------------------------------------------

    canReuseOptimization = ...
        hasReusableOptimization && ...
        isequal( ...
            idxFit(:), ...
            reusableIdxFit(:));


    if canReuseOptimization

        Rsrc = ...
            PointResult.Radius( ...
                reusableRadiusIndex);


        Rj.polynomialAdmissible = ...
            Rsrc.polynomialAdmissible;

        Rj.polynomialInfo = ...
            Rsrc.polynomialInfo;

        Rj.d_local = ...
            Rsrc.d_local;


        Rj.optimization = ...
            Rsrc.optimization;

        Rj.optimizationSuccess = ...
            Rsrc.optimizationSuccess;


        Rj.C_best = ...
            Rsrc.C_best;

        Rj.reg_param_best = ...
            Rsrc.reg_param_best;

        Rj.epsilon_best = ...
            Rsrc.epsilon_best;

        Rj.AE_best = ...
            Rsrc.AE_best;

        Rj.yhat_best = ...
            Rsrc.yhat_best;

        Rj.y_diff_best = ...
            Rsrc.y_diff_best;

        Rj.cond_reg_best = ...
            Rsrc.cond_reg_best;

        Rj.rel_residual_best = ...
            Rsrc.rel_residual_best;

        Rj.solver_used_best = ...
            Rsrc.solver_used_best;


        % Preserve the complete search information associated
        % with this mathematically identical radius. Point-level
        % computational totals are not incremented again.
        
      % No new parameter search or RBF solve is performed for
% this reused radius. The complete original optimization
% information remains available in Rj.optimization.

Rj.nEvaluations = ...
    0;

Rj.nGAEvaluations = ...
    0;

Rj.nFallbackEvaluations = ...
    0;

Rj.nGridEvaluations = ...
    0;

Rj.nRefinementEvaluations = ...
    0;

Rj.nValidEvaluations = ...
    0;

Rj.nFailedEvaluations = ...
    0;

Rj.nRBFSolves = ...
    0;


        Rj.optimizationReused = ...
            true;

        Rj.reusedFromRadiusIndex = ...
            reusableRadiusIndex;


        Rj.success = ...
            true;

        Rj.failureStage = ...
            '';

        Rj.message = ...
            '';


        Rj.totalWallTime = ...
            toc(tRadius);


        PointResult.Radius(j) = ...
            Rj;


        PointResult.nValidRadii = ...
            PointResult.nValidRadii + 1;

        PointResult.nReusedRadii = ...
            PointResult.nReusedRadii + 1;


        continue

    end


    Xfit = ...
        X(idxFit,:);


    yfit = ...
        y(idxFit);


    % =========================================================
    % 9.3 Polynomial admissibility
    % =========================================================

    tAdmissibility = ...
        tic;


    [isOK,polyInfo] = ...
        common.checkPolynomialAdmissibility( ...
            Xfit);


    Rj.admissibilityWallTime = ...
        toc(tAdmissibility);


    Rj.polynomialAdmissible = ...
        isOK;


    Rj.polynomialInfo = ...
        polyInfo;


    if ~isOK

        Rj.failureStage = ...
            'polynomial admissibility';


        Rj.message = ...
            'Polynomial admissibility failed.';


        Rj.totalWallTime = ...
            toc(tRadius);


        PointResult.Radius(j) = ...
            Rj;


        PointResult.nPolynomialFailures = ...
            PointResult.nPolynomialFailures + 1;


        continue

    end


    % =========================================================
    % 9.4 Fixed local geometry
    %
    % d_local is calculated ONCE for this point-radius pair.
    %
    % R, P, R_eval, and P_eval are extracted from the
    % precomputed training geometry.
    % =========================================================

    tGeometry = ...
        tic;


    if useShape

        try

            [~,d_local] = ...
                common.buildEpsilon( ...
                    Xfit, ...
                    1);

        catch ME

            Rj.failureStage = ...
                'local geometry';


            Rj.message = ...
                ME.message;


            Rj.geometryWallTime = ...
                toc(tGeometry);


            Rj.totalWallTime = ...
                toc(tRadius);


            PointResult.Radius(j) = ...
                Rj;


            PointResult.nGeometryFailures = ...
                PointResult.nGeometryFailures + 1;


            continue

        end

    else

        d_local = ...
            NaN;

    end


    % ---------------------------------------------------------
    % Fixed local training matrix geometry
    % ---------------------------------------------------------

    Rmat = ...
        geometryCache.D( ...
            idxFit, ...
            idxFit);


    Pmat = ...
        geometryCache.P( ...
            idxFit, ...
            :);


    % ---------------------------------------------------------
    % Fixed target-evaluation geometry
    % ---------------------------------------------------------

    R_eval = ...
        geometryCache.D( ...
            iTarget, ...
            idxFit);


    R_eval = ...
        reshape( ...
            R_eval, ...
            1,[]);


    P_eval = ...
        geometryCache.P( ...
            iTarget, ...
            :);


    Rj.geometryWallTime = ...
        toc(tGeometry);


    Rj.d_local = ...
        d_local;


    % =========================================================
    % 9.5 Temporary fixed cache for this point-radius pair
    %
    % Reused throughout the ONE parameter optimization.
    % =========================================================

    cache = ...
        struct();


    cache.Xfit = ...
        Xfit;


    cache.yfit = ...
        yfit;


    cache.Xtarget = ...
        xTarget;


    cache.ytarget = ...
        yTarget;


    cache.R = ...
        Rmat;


    cache.P = ...
        Pmat;


    cache.R_eval = ...
        R_eval;


    cache.P_eval = ...
        P_eval;


    cache.d = ...
        d_local;


    % =========================================================
    % 9.6 Parameter optimization
    %
    % Shaped kernels:
    %       joint C + RegParam
    %
    % Shape-free kernels:
    %       RegParam only
    % =========================================================

    tOptimization = ...
        tic;


    PointResult.nUniqueNeighborhoodOptimizations = ...
        PointResult.nUniqueNeighborhoodOptimizations + 1;


    try
Opt = ...
    localrbf.optimizeParameters( ...
        cache, ...
        kernel, ...
        C_min, ...
        C_max, ...
        Reg_min, ...
        Reg_max, ...
        options, ...
        method, ...
        nGrid);

    catch ME

        Opt = ...
            [];


        Rj.optimizationSuccess = ...
            false;


        Rj.failureStage = ...
            'parameter optimization';


        Rj.message = ...
            ME.message;

    end


    Rj.optimizationWallTime = ...
        toc(tOptimization);


    % =========================================================
    % 9.7 Optimization exception
    % =========================================================

    if isempty(Opt)

        Rj.totalWallTime = ...
            toc(tRadius);


        PointResult.Radius(j) = ...
            Rj;


        PointResult.nOptimizationFailures = ...
            PointResult.nOptimizationFailures + 1;


        continue

    end


    % =========================================================
    % 9.8 Retain COMPLETE optimization result
    % =========================================================

    Rj.optimization = ...
        Opt;


    Rj.optimizationSuccess = ...
        Opt.success;


    % ---------------------------------------------------------
    % Selected optimum for this radius
    % ---------------------------------------------------------

    Rj.C_best = ...
        Opt.C;


    Rj.reg_param_best = ...
        Opt.reg_param;


    Rj.epsilon_best = ...
        Opt.epsilon;


    Rj.AE_best = ...
        Opt.AE;


    Rj.yhat_best = ...
        Opt.yhat;


    Rj.y_diff_best = ...
        Opt.y_diff;


    Rj.cond_reg_best = ...
        Opt.cond_reg;


    Rj.rel_residual_best = ...
        Opt.rel_residual;


    Rj.solver_used_best = ...
        Opt.solver_used;


    % ---------------------------------------------------------
    % Search and computational counts
    % ---------------------------------------------------------

    Rj.nEvaluations = ...
        Opt.searchInfo.nEvaluations;


    Rj.nGAEvaluations = ...
        Opt.searchInfo.nGAEvaluations;


    Rj.nFallbackEvaluations = ...
        Opt.searchInfo.nFallbackEvaluations;

    if isfield( ...
        Opt.searchInfo, ...
        'nGridEvaluations')

    Rj.nGridEvaluations = ...
        Opt.searchInfo.nGridEvaluations;

else

    Rj.nGridEvaluations = ...
        0;

end


    if isfield( ...
            Opt.searchInfo, ...
            'nRefinementEvaluations')

        Rj.nRefinementEvaluations = ...
            Opt.searchInfo.nRefinementEvaluations;

    else

        Rj.nRefinementEvaluations = ...
            0;

    end


    Rj.nValidEvaluations = ...
        Opt.searchInfo.nValidEvaluations;


    Rj.nFailedEvaluations = ...
        Opt.searchInfo.nFailedEvaluations;


    Rj.nRBFSolves = ...
        Opt.searchInfo.nRBFSolves;


    % =========================================================
    % 9.9 Radius validity
    % =========================================================

    if Opt.success && ...
       isscalar(Opt.AE) && ...
       isreal(Opt.AE) && ...
       isfinite(Opt.AE)

        Rj.success = ...
            true;


        Rj.failureStage = ...
            '';


        Rj.message = ...
            '';


        PointResult.nValidRadii = ...
            PointResult.nValidRadii + 1;


    else

        Rj.success = ...
            false;


        Rj.failureStage = ...
            'parameter optimization';


        if isfield(Opt,'searchInfo') && ...
           isfield(Opt.searchInfo,'message')

            Rj.message = ...
                Opt.searchInfo.message;

        else

            Rj.message = ...
                'Parameter optimization failed.';

        end


        PointResult.nOptimizationFailures = ...
            PointResult.nOptimizationFailures + 1;

    end


    % ---------------------------------------------------------
    % Update exact-neighbourhood reuse cache only after a
    % successful complete optimization.
    % ---------------------------------------------------------

    if Rj.success

        hasReusableOptimization = ...
            true;

        reusableIdxFit = ...
            idxFit(:);

        reusableRadiusIndex = ...
            j;

    end


    % =========================================================
    % 9.10 Radius runtime
    % =========================================================

    Rj.totalWallTime = ...
        toc(tRadius);


    % =========================================================
    % 9.11 Store complete radius result
    % =========================================================

    PointResult.Radius(j) = ...
        Rj;


    % =========================================================
    % 9.12 Aggregate point-level search information
    % =========================================================

    PointResult.nEvaluations = ...
        PointResult.nEvaluations + ...
        Rj.nEvaluations;


    PointResult.nGAEvaluations = ...
        PointResult.nGAEvaluations + ...
        Rj.nGAEvaluations;
PointResult.nGridEvaluations = ...
    PointResult.nGridEvaluations + ...
    Rj.nGridEvaluations;

    PointResult.nFallbackEvaluations = ...
        PointResult.nFallbackEvaluations + ...
        Rj.nFallbackEvaluations;


    PointResult.nRefinementEvaluations = ...
        PointResult.nRefinementEvaluations + ...
        Rj.nRefinementEvaluations;


    PointResult.nValidEvaluations = ...
        PointResult.nValidEvaluations + ...
        Rj.nValidEvaluations;


    PointResult.nFailedEvaluations = ...
        PointResult.nFailedEvaluations + ...
        Rj.nFailedEvaluations;


    PointResult.nRBFSolves = ...
        PointResult.nRBFSolves + ...
        Rj.nRBFSolves;

end


% =============================================================
% 10. Final point-level status
% =============================================================

PointResult.nFailedRadii = ...
    M - ...
    PointResult.nValidRadii;


PointResult.success = ...
    PointResult.nValidRadii > 0;


if PointResult.success

    PointResult.message = ...
        '';

else

    PointResult.message = ...
        'No candidate radius produced a valid local RBF solution.';

end


% =============================================================
% 11. Compact radius-level arrays
%
% Detailed structures above remain authoritative.
% These arrays support fast later selection and analysis.
% =============================================================

PointResult.radius_all = ...
    radii;


PointResult.radius_success = ...
    false(M,1);


PointResult.polynomial_admissible_all = ...
    false(M,1);


PointResult.C_best_all = ...
    nan(M,1);


PointResult.Reg_best_all = ...
    nan(M,1);


PointResult.epsilon_best_all = ...
    nan(M,1);


PointResult.d_local_all = ...
    nan(M,1);


PointResult.AE_best_all = ...
    nan(M,1);


PointResult.yhat_best_all = ...
    nan(M,1);


PointResult.y_diff_best_all = ...
    nan(M,1);


PointResult.cond_reg_best_all = ...
    nan(M,1);


PointResult.rel_residual_best_all = ...
    nan(M,1);


PointResult.solver_used_best_all = ...
    cell(M,1);


PointResult.neighbor_count_all = ...
    nan(M,1);


PointResult.n_within_radius_all = ...
    nan(M,1);


PointResult.n_added_for_minimum_all = ...
    nan(M,1);


PointResult.effective_radius_all = ...
    nan(M,1);


PointResult.selected_max_distance_all = ...
    nan(M,1);


PointResult.usedMinNeighborFallback_all = ...
    false(M,1);


PointResult.nEvaluations_all = ...
    zeros(M,1);


PointResult.nGridEvaluations_all = ...
    zeros(M,1);


PointResult.nRefinementEvaluations_all = ...
    zeros(M,1);


PointResult.optimizationReused_all = ...
    false(M,1);


PointResult.reusedFromRadiusIndex_all = ...
    nan(M,1);


PointResult.nRBFSolves_all = ...
    zeros(M,1);


PointResult.optimizationWallTime_all = ...
    zeros(M,1);


PointResult.radiusWallTime_all = ...
    zeros(M,1);


for j = 1:M

    Rj = ...
        PointResult.Radius(j);


    PointResult.radius_success(j) = ...
        Rj.success;


    PointResult.polynomial_admissible_all(j) = ...
        Rj.polynomialAdmissible;


    PointResult.C_best_all(j) = ...
        Rj.C_best;


    PointResult.Reg_best_all(j) = ...
        Rj.reg_param_best;


    PointResult.epsilon_best_all(j) = ...
        Rj.epsilon_best;


    PointResult.d_local_all(j) = ...
        Rj.d_local;


    PointResult.AE_best_all(j) = ...
        Rj.AE_best;


    PointResult.yhat_best_all(j) = ...
        Rj.yhat_best;


    PointResult.y_diff_best_all(j) = ...
        Rj.y_diff_best;


    PointResult.cond_reg_best_all(j) = ...
        Rj.cond_reg_best;


    PointResult.rel_residual_best_all(j) = ...
        Rj.rel_residual_best;


    PointResult.solver_used_best_all{j} = ...
        Rj.solver_used_best;


    PointResult.neighbor_count_all(j) = ...
        Rj.neighbor_count;


    PointResult.n_within_radius_all(j) = ...
        Rj.nWithinRadius;


    PointResult.n_added_for_minimum_all(j) = ...
        Rj.nAddedForMinimum;


    PointResult.effective_radius_all(j) = ...
        Rj.effectiveRadius;


    PointResult.selected_max_distance_all(j) = ...
        Rj.selectedMaxDistance;


    PointResult.usedMinNeighborFallback_all(j) = ...
        Rj.usedMinNeighborFallback;


    PointResult.nEvaluations_all(j) = ...
        Rj.nEvaluations;


    PointResult.nGridEvaluations_all(j) = ...
        Rj.nGridEvaluations;


    PointResult.nRefinementEvaluations_all(j) = ...
        Rj.nRefinementEvaluations;


    PointResult.optimizationReused_all(j) = ...
        Rj.optimizationReused;


    PointResult.reusedFromRadiusIndex_all(j) = ...
        Rj.reusedFromRadiusIndex;


    PointResult.nRBFSolves_all(j) = ...
        Rj.nRBFSolves;


    PointResult.optimizationWallTime_all(j) = ...
        Rj.optimizationWallTime;


    PointResult.radiusWallTime_all(j) = ...
        Rj.totalWallTime;



end


% =============================================================
% 12. Point runtime
% =============================================================

PointResult.totalWallTime = ...
    toc(tPoint);

end


% =============================================================
% Local helper
% =============================================================

function R = emptyRadiusResult()
%EMPTYRADIUSRESULT
% Uniform result structure for every candidate radius.


R = ...
    struct();


% =============================================================
% Radius
% =============================================================

R.radiusIndex = ...
    NaN;


R.radius = ...
    NaN;


R.requestedRadius = ...
    NaN;


R.effectiveRadius = ...
    NaN;


R.selectedMaxDistance = ...
    NaN;


% =============================================================
% Neighbourhood
% =============================================================

R.neighborhoodSuccess = ...
    false;


R.neighbor_idx = ...
    [];


R.neighbor_distances = ...
    [];


R.neighbor_count = ...
    0;


R.nAvailable = ...
    0;


R.nWithinRadius = ...
    0;


R.nAddedForMinimum = ...
    0;


R.usedMinNeighborFallback = ...
    false;


% =============================================================
% Polynomial admissibility
% =============================================================

R.polynomialAdmissible = ...
    false;


R.polynomialInfo = ...
    [];


% =============================================================
% Local geometry
% =============================================================

R.d_local = ...
    NaN;


% =============================================================
% Optimization
% =============================================================

R.optimizationSuccess = ...
    false;


R.optimization = ...
    [];


R.optimizationReused = ...
    false;


R.reusedFromRadiusIndex = ...
    NaN;


% =============================================================
% Best configuration at this radius
% =============================================================

R.C_best = ...
    NaN;


R.reg_param_best = ...
    NaN;


R.epsilon_best = ...
    NaN;


R.AE_best = ...
    NaN;


R.yhat_best = ...
    NaN;


R.y_diff_best = ...
    NaN;


R.cond_reg_best = ...
    NaN;


R.rel_residual_best = ...
    NaN;


R.solver_used_best = ...
    '';


% =============================================================
% Search counts
% =============================================================

R.nEvaluations = ...
    0;


R.nGAEvaluations = ...
    0;


R.nFallbackEvaluations = ...
    0;


R.nGridEvaluations = ...
    0;


R.nRefinementEvaluations = ...
    0;

R.nValidEvaluations = ...
    0;


R.nFailedEvaluations = ...
    0;


R.nRBFSolves = ...
    0;


% =============================================================
% Timing
% =============================================================

R.neighborhoodWallTime = ...
    0;


R.admissibilityWallTime = ...
    0;


R.geometryWallTime = ...
    0;


R.optimizationWallTime = ...
    0;


R.totalWallTime = ...
    0;


% =============================================================
% Status
% =============================================================

R.success = ...
    false;


R.failureStage = ...
    '';


R.message = ...
    '';

end

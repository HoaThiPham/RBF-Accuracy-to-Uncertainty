function Result = optimizeParameters( ...
    cache, kernel, C_min, C_max, ...
    Reg_min, Reg_max, options, method, nGrid)

%OPTIMIZEPARAMETERS
%
% Optimize RBF parameters for ONE fixed target point and
% ONE fixed candidate radius.
%
% Shape-parameter kernels:
%
%       jointly optimize C and RegParam
%
% Shape-free kernels:
%
%       optimize RegParam only
%
% Objective:
%
%       absolute target-excluded prediction error
%
% IMPORTANT:
%   - point and radius are fixed before entering;
%   - neighbourhood and geometry are fixed and reused;
%   - no radius selection is performed here;
%   - no RNG reset is performed here;
%   - prediction error alone selects the parameters;
%   - conditioning is numerical/diagnostic only;
%   - every search evaluation is retained;
%   - failed evaluations use Inf for GA and NaN in scientific traces;
%   - full CN/ECN/EVN diagnostics are calculated later;
%   - inner GA parallelism is disabled.


% =============================================================
% 1. Options
% =============================================================

if nargin < 7 || isempty(options)
    options = struct();
end

if nargin < 8 || isempty(method)
    method = 'ga';
end

if nargin < 9 || isempty(nGrid)
    nGrid = 25;
end

if isstring(method) && isscalar(method)
    method = char(method);
end

if ~ischar(method) || size(method,1) ~= 1
    error('method must be ''ga'' or ''grid''.');
end

method = lower(strtrim(method));

if ~ismember(method,{'ga','grid'})
    error('method must be ''ga'' or ''grid''.');
end

if ~isnumeric(nGrid) || ...
   isempty(nGrid) || ...
   ~isvector(nGrid) || ...
   numel(nGrid) > 2 || ...
   ~isreal(nGrid) || ...
   any(~isfinite(nGrid(:))) || ...
   any(nGrid(:) < 1) || ...
   any(nGrid(:) ~= round(nGrid(:)))

    error( ...
        ['nGrid must be a positive integer scalar or ' ...
         'two-element vector [nC nReg].']);
end

nGrid = nGrid(:).';

if isscalar(nGrid)
    nC_requested = nGrid;
    nR_requested = nGrid;
else
    nC_requested = nGrid(1);
    nR_requested = nGrid(2);
end

if ~isfield(options,'solver') || ...
   ~isstruct(options.solver)

    options.solver = struct();

end


if ~isfield(options.solver,'cond_lsq') || ...
   isempty(options.solver.cond_lsq)

    options.solver.cond_lsq = ...
        1e12;

end


% Relative residual is unnecessary during parameter search.
searchOptions = ...
    options;

searchOptions.solver.compute_rel_residual = ...
    false;


% Relative residual is calculated for the final selected solution.
finalOptions = ...
    options;

finalOptions.solver.compute_rel_residual = ...
    true;


% =============================================================
% 2. Kernel type
% =============================================================

useShape = ...
    common.kernelHasShapeParameter(kernel);


%Initialization
nGridEvaluations = 0;
gridSucceeded = false;
gridWallTime = 0;

% =============================================================
% 3. Parameter validation
% =============================================================

if ~isnumeric(Reg_min) || ...
   ~isscalar(Reg_min) || ...
   ~isreal(Reg_min) || ...
   ~isfinite(Reg_min) || ...
   Reg_min <= 0

    error( ...
        'Reg_min must be a finite positive real numeric scalar.');

end


if ~isnumeric(Reg_max) || ...
   ~isscalar(Reg_max) || ...
   ~isreal(Reg_max) || ...
   ~isfinite(Reg_max) || ...
   Reg_max < Reg_min

    error( ...
        'Reg_max must be a finite real numeric scalar >= Reg_min.');

end


if useShape

    if ~isnumeric(C_min) || ...
       ~isscalar(C_min) || ...
       ~isreal(C_min) || ...
       ~isfinite(C_min) || ...
       C_min <= 0

        error( ...
            'C_min must be a finite positive real numeric scalar.');

    end


    if ~isnumeric(C_max) || ...
       ~isscalar(C_max) || ...
       ~isreal(C_max) || ...
       ~isfinite(C_max) || ...
       C_max < C_min

        error( ...
            'C_max must be a finite real numeric scalar >= C_min.');

    end

else

    % Shape-free kernels do not use C.
    C_min = ...
        NaN;

    C_max = ...
        NaN;

end



% =============================================================
% 4. Log-space bounds
% =============================================================

zRmin = ...
    log10(Reg_min);

zRmax = ...
    log10(Reg_max);


Reg_fixed = ...
    (Reg_min == Reg_max);


if useShape

    zCmin = ...
        log10(C_min);

    zCmax = ...
        log10(C_max);


    C_fixed = ...
        (C_min == C_max);

else

    zCmin = ...
        NaN;

    zCmax = ...
        NaN;

    C_fixed = ...
        true;

end


% =============================================================
% 5. Search variables
% =============================================================

if useShape

    if ~C_fixed && ~Reg_fixed

        nvars = ...
            2;

        lb = ...
            [zCmin zRmin];

        ub = ...
            [zCmax zRmax];


    elseif ~C_fixed

        nvars = ...
            1;

        lb = ...
            zCmin;

        ub = ...
            zCmax;


    elseif ~Reg_fixed

        nvars = ...
            1;

        lb = ...
            zRmin;

        ub = ...
            zRmax;


    else

        nvars = ...
            0;

        lb = [];
        ub = [];

    end

else

    if Reg_fixed

        nvars = ...
            0;

        lb = [];
        ub = [];

    else

        nvars = ...
            1;

        lb = ...
            zRmin;

        ub = ...
            zRmax;

    end

end


allFixed = ...
    (nvars == 0);


% =============================================================
% 6. Complete scientific evaluation history
%
% Every search evaluation is retained in actual evaluation order.
% No unique(), merging, averaging, or removal of repeated values.
% =============================================================

C_eval_log = ...
    [];

Reg_eval_log = ...
    [];

epsilon_eval_log = ...
    [];

AE_eval_log = ...
    [];

yhat_eval_log = ...
    [];

ydiff_eval_log = ...
    [];

cond_reg_eval_log = ...
    [];

solver_eval_log = ...
    {};

success_eval_log = ...
    [];

stage_eval_log = ...
    [];

% stage:
% 1 = GA
% 2 = GA fallback grid
% 3 = fixed
% 4 = GA local refinement
% 5 = direct grid


% =============================================================
% 7. Failed-objective penalty
% =============================================================

FAIL_PENALTY = ...
    Inf;


% =============================================================
% 8. Fallback-grid settings
% =============================================================

if useShape && ~C_fixed

    if nC_requested < 2
        error('nGrid for C must be at least 2 when C is free.');
    end

    nCgrid = nC_requested;

else

    nCgrid = 1;

end


if ~Reg_fixed

    if nR_requested < 2
        error( ...
            'nGrid for RegParam must be at least 2 when RegParam is free.');
    end

    nRgrid = nR_requested;

else

    nRgrid = 1;

end





% -------------------------------------------------------------
% Local-refinement settings
%
% Retains the useful post-GA refinement idea from the previous
% local optimizer, generalized to the current log-space search.
% A 2-D refinement uses a small tensor grid to limit extra cost.
% -------------------------------------------------------------

nRefine1D = ...
    11;

nRefinePerDim2D = ...
    5;


% =============================================================
% 9. GA settings
%
% Same settings as global TuneReg.
% =============================================================

popSize = ...
    40;


elite = ...
    max(2,ceil(0.05*popSize));


if strcmp(method,'ga') && ~allFixed

    gaOptions = ...
        optimoptions( ...
            'ga', ...
            'Display',              'off', ...
            'UseParallel',          false, ...
            'UseVectorized',        false, ...
            'PopulationSize',       popSize, ...
            'EliteCount',           elite, ...
            'CrossoverFraction',    0.8, ...
            'SelectionFcn',         @selectionstochunif, ...
            'MutationFcn',          @mutationadaptfeasible, ...
            'FitnessScalingFcn',    @fitscalingrank, ...
            'FunctionTolerance',    1e-8, ...
            'MaxGenerations',       60, ...
            'MaxStallGenerations',  20);
else
    gaOptions = [];
end

% -------------------------------------------------------------
% GA information retained for reproducibility
% -------------------------------------------------------------

GAInfo = ...
    struct();


GAInfo.Display = ...
    'off';

GAInfo.PopulationSize = ...
    popSize;

GAInfo.EliteCount = ...
    elite;

GAInfo.CrossoverFraction = ...
    0.8;

GAInfo.SelectionFcn = ...
    'selectionstochunif';

GAInfo.MutationFcn = ...
    'mutationadaptfeasible';

GAInfo.FitnessScalingFcn = ...
    'fitscalingrank';

GAInfo.FunctionTolerance = ...
    1e-8;

GAInfo.MaxGenerations = ...
    60;

GAInfo.MaxStallGenerations = ...
    20;

GAInfo.UseParallel = ...
    false;

GAInfo.UseVectorized = ...
    false;

GAInfo.nvars = ...
    nvars;

GAInfo.Skipped = allFixed || strcmp(method,'grid');

GAInfo.FallbackGridSize = ...
    [nCgrid nRgrid];


GAInfo.LocalRefinement1DPoints = ...
    nRefine1D;

GAInfo.LocalRefinement2DPointsPerDimension = ...
    nRefinePerDim2D;

GAInfo.LocalRefinementSpanRule = ...
    'max(0.5 decades, one-twelfth of full log10 range)';

GAInfo.FailurePenalty = ...
    FAIL_PENALTY;


if useShape

    if ~C_fixed && ~Reg_fixed

        GAInfo.SearchVariable = ...
            {'log10(C)','log10(RegParam)'};

    elseif ~C_fixed

        GAInfo.SearchVariable = ...
            {'log10(C)'};

    elseif ~Reg_fixed

        GAInfo.SearchVariable = ...
            {'log10(RegParam)'};

    else

        GAInfo.SearchVariable = ...
            {};

    end

else

    if Reg_fixed

        GAInfo.SearchVariable = ...
            {};

    else

        GAInfo.SearchVariable = ...
            {'log10(RegParam)'};

    end

end


% =============================================================
% 10. Initialize search information
% =============================================================

usedFallback = ...
    false;

fallbackSucceeded = ...
    false;

refinementSucceeded = ...
    false;

refinementSource = ...
    '';


gaMessage = ...
    '';

firstFailureMessage = ...
    '';


ga_best_ae = ...
    NaN;

search_best_ae = ...
    NaN;


selectedC = ...
    NaN;

selectedReg = ...
    NaN;


exitflag = ...
    NaN;

output = ...
    struct();


nGAEvaluations = ...
    0;

nFallbackEvaluations = ...
    0;

nRefinementEvaluations = ...
    0;


nRBFSolvesSearch = ...
    0;

nRBFSolvesFinal = ...
    0;


currentStage = ...
    1;


% -------------------------------------------------------------
% Timing
% -------------------------------------------------------------

gaWallTime = ...
    0;

fallbackWallTime = ...
    0;

refinementWallTime = ...
    0;

finalWallTime = ...
    0;


tTotal = ...
    tic;


% -------------------------------------------------------------
% Diagnostics initialized for both success and failure paths
% -------------------------------------------------------------

BOUND_TOL = ...
    0.02;


CRelativePosition = ...
    NaN;

CNearLowerBound = ...
    false;

CNearUpperBound = ...
    false;


RegRelativePosition = ...
    NaN;

RegNearLowerBound = ...
    false;

RegNearUpperBound = ...
    false;


finalSearchAEDifference = ...
    NaN;

searchConsistent = ...
    false;


% =============================================================
% 11. Parameter optimization
% =============================================================

if allFixed

    % ---------------------------------------------------------
    % All relevant parameters are fixed.
    % Do not start GA.
    % ---------------------------------------------------------
  % fixed parameters
    if useShape
        selectedC = C_min;
    else
        selectedC = NaN;
    end

    selectedReg = Reg_min;


elseif strcmp(method,'grid')

    % direct logarithmic-grid search
    currentStage = 5;

    tGrid = tic;

    evaluate_log_grid();

    nGridEvaluations = numel(AE_eval_log);
gridSucceeded = any(isfinite(AE_eval_log));
    gridWallTime = toc(tGrid);

    validAll = isfinite(AE_eval_log);

    if any(validAll)

        validIdx = find(validAll);

        [search_best_ae,jbest] = ...
            min(AE_eval_log(validIdx));

        ibest = validIdx(jbest);

        selectedReg = Reg_eval_log(ibest);

        if useShape
            selectedC = C_eval_log(ibest);
        else
            selectedC = NaN;
        end

    end

else

    % ---------------------------------------------------------
    % ONE GA for this fixed point-radius problem
    % ---------------------------------------------------------

    currentStage = ...
        1;


    tGA = ...
        tic;


    try

        [~,best_ae,exitflag,output] = ...
            ga( ...
                @ga_fitness_log_z, ...
                nvars, ...
                [],[],[],[], ...
                lb,ub, ...
                [], ...
                gaOptions);


        ga_best_ae = ...
            best_ae;


    catch ME

        usedFallback = ...
            true;


        gaMessage = ...
            ME.message;


        if isempty(firstFailureMessage)

            firstFailureMessage = ...
                ME.message;

        end

    end


    gaWallTime = ...
        toc(tGA);


    % ---------------------------------------------------------
    % Number of objective evaluations made during GA
    % ---------------------------------------------------------

    nGAEvaluations = ...
        numel(AE_eval_log);


    idxGA = ...
        (1:nGAEvaluations).';


    if nGAEvaluations > 0

        validGA = ...
            isfinite(AE_eval_log(idxGA));

    else

        validGA = ...
            false(0,1);

    end


    % ---------------------------------------------------------
    % Determine whether fallback is required
    %
    % exitflag == 0 remains accepted.
    % ---------------------------------------------------------

    if ~usedFallback

        if ~isfinite(exitflag) || ...
           exitflag < 0 || ...
           ~any(validGA)

            usedFallback = ...
                true;


            gaMessage = ...
                ['GA did not return a valid scientific ' ...
                 'target-excluded solution.'];

        end

    end


    % ---------------------------------------------------------
    % Select best genuine GA evaluation
    % ---------------------------------------------------------

    if ~usedFallback

        validIdx = ...
            idxGA(validGA);


        [search_best_ae,jbest] = ...
            min(AE_eval_log(validIdx));


        ibest = ...
            validIdx(jbest);


        selectedReg = ...
            Reg_eval_log(ibest);


        if useShape

            selectedC = ...
                C_eval_log(ibest);

        else

            selectedC = ...
                NaN;

        end

    end


    % =========================================================
    % 12. Invalid GA -> logarithmic-grid fallback
    % =========================================================

    if usedFallback

        currentStage = ...
            2;


        tFallback = ...
            tic;


        % -----------------------------------------------------
        % Construct fallback grids
        % -----------------------------------------------------

        if useShape

            if C_fixed

                C_coarse = ...
                    C_min;

            else

                C_coarse = ...
                    logspace( ...
                        zCmin, ...
                        zCmax, ...
                        nCgrid);

            end

        end


        if Reg_fixed

            Reg_coarse = ...
                Reg_min;

        else

            Reg_coarse = ...
                logspace( ...
                    zRmin, ...
                    zRmax, ...
                    nRgrid);

        end


        % -----------------------------------------------------
        % Evaluate fallback using the same objective
        % -----------------------------------------------------

        if useShape

            if ~C_fixed && ~Reg_fixed

                for iC = 1:numel(C_coarse)

                    for iR = 1:numel(Reg_coarse)

                        ga_fitness_log_z( ...
                            [log10(C_coarse(iC)), ...
                             log10(Reg_coarse(iR))]);

                    end

                end


            elseif ~C_fixed

                for iC = 1:numel(C_coarse)

                    ga_fitness_log_z( ...
                        log10(C_coarse(iC)));

                end


            elseif ~Reg_fixed

                for iR = 1:numel(Reg_coarse)

                    ga_fitness_log_z( ...
                        log10(Reg_coarse(iR)));

                end

            end


        else

            for iR = 1:numel(Reg_coarse)

                ga_fitness_log_z( ...
                    log10(Reg_coarse(iR)));

            end

        end


        fallbackWallTime = ...
            toc(tFallback);


        nFallbackEvaluations = ...
            numel(AE_eval_log) - ...
            nGAEvaluations;


        if nFallbackEvaluations > 0

            fallbackIdx = ...
                nGAEvaluations + ...
                (1:nFallbackEvaluations);


            fallbackSucceeded = ...
                any(isfinite( ...
                    AE_eval_log(fallbackIdx)));

        end


        % -----------------------------------------------------
        % Select best genuine evaluation from COMPLETE search
        % -----------------------------------------------------

        validAll = ...
            isfinite(AE_eval_log);


        if any(validAll)

            validIdx = ...
                find(validAll);


            [search_best_ae,jbest] = ...
                min(AE_eval_log(validIdx));


            ibest = ...
                validIdx(jbest);


            selectedReg = ...
                Reg_eval_log(ibest);


            if useShape

                selectedC = ...
                    C_eval_log(ibest);

            else

                selectedC = ...
                    NaN;

            end

        end

    end


    % =========================================================
    % 13. Local refinement around the best valid search point
    
    % Local refinement is applied only after a successful GA search.
% No refinement is applied after fallback or direct-grid search.
    % =========================================================

  if strcmp(method,'ga') && ...
   ~usedFallback && ...
   isfinite(search_best_ae)

     refinementSource = ...
    'GA';

        currentStage = ...
            4;


        nBeforeRefinement = ...
            numel(AE_eval_log);


        tRefinement = ...
            tic;


        if useShape && ...
           ~C_fixed && ...
           ~Reg_fixed

            zC0 = ...
                log10(selectedC);

            zR0 = ...
                log10(selectedReg);


            spanC = ...
                max(0.5,(zCmax-zCmin)/12);

            spanR = ...
                max(0.5,(zRmax-zRmin)/12);


            zClo = ...
                max(zCmin,zC0-spanC);

            zChi = ...
                min(zCmax,zC0+spanC);

            zRlo = ...
                max(zRmin,zR0-spanR);

            zRhi = ...
                min(zRmax,zR0+spanR);


            zCref = ...
                linspace( ...
                    zClo, ...
                    zChi, ...
                    nRefinePerDim2D);

            zRref = ...
                linspace( ...
                    zRlo, ...
                    zRhi, ...
                    nRefinePerDim2D);


            for iC = 1:numel(zCref)

                for iR = 1:numel(zRref)

                    ga_fitness_log_z( ...
                        [zCref(iC),zRref(iR)]);

                end

            end


        elseif useShape && ...
               ~C_fixed

            z0 = ...
                log10(selectedC);


            span = ...
                max(0.5,(zCmax-zCmin)/12);


            zlo = ...
                max(zCmin,z0-span);

            zhi = ...
                min(zCmax,z0+span);


            zref = ...
                linspace( ...
                    zlo, ...
                    zhi, ...
                    nRefine1D);


            for iz = 1:numel(zref)

                ga_fitness_log_z( ...
                    zref(iz));

            end


        else

            % RegParam is the only free parameter. This covers
            % both shaped kernels with fixed C and shape-free kernels.
            z0 = ...
                log10(selectedReg);


            span = ...
                max(0.5,(zRmax-zRmin)/12);


            zlo = ...
                max(zRmin,z0-span);

            zhi = ...
                min(zRmax,z0+span);


            zref = ...
                linspace( ...
                    zlo, ...
                    zhi, ...
                    nRefine1D);


            for iz = 1:numel(zref)

                ga_fitness_log_z( ...
                    zref(iz));

            end

        end


        refinementWallTime = ...
            toc(tRefinement);


        nRefinementEvaluations = ...
            numel(AE_eval_log) - ...
            nBeforeRefinement;


        if nRefinementEvaluations > 0

            refinementIdx = ...
                nBeforeRefinement + ...
                (1:nRefinementEvaluations);


            refinementSucceeded = ...
                any(isfinite( ...
                    AE_eval_log(refinementIdx)));

        end


        % -----------------------------------------------------
        % Re-select the best genuine evaluation from the entire
        % search history, including local refinement.
        % -----------------------------------------------------

        validAll = ...
            isfinite(AE_eval_log);


        if any(validAll)

            validIdx = ...
                find(validAll);


            [search_best_ae,jbest] = ...
                min(AE_eval_log(validIdx));


            ibest = ...
                validIdx(jbest);


            selectedReg = ...
                Reg_eval_log(ibest);


            if useShape

                selectedC = ...
                    C_eval_log(ibest);

            else

                selectedC = ...
                    NaN;

            end

        end

    end

end


% =============================================================
% 14. Complete search failure
%
% Failure of one point-radius combination does not terminate
% optimization of the remaining radii.
% =============================================================

if ~allFixed && ...
   ~isfinite(search_best_ae)

    Result = ...
        makeResult( ...
            false, ...
            'No valid parameter configuration was found.', ...
            []);

    return

end


% =============================================================
% 15. Final evaluation ONCE at selected parameters
%
% Relative residual is calculated only here.
% =============================================================

tFinal = ...
    tic;


try

    finalEval = ...
        localrbf.evaluateTargetExcludedError( ...
            cache, ...
            kernel, ...
            useShape, ...
            selectedC, ...
            selectedReg, ...
            finalOptions);


catch ME

    finalEval = ...
        makeExceptionEvaluation( ...
            selectedC, ...
            selectedReg, ...
            ME.message);


    if isempty(firstFailureMessage)

        firstFailureMessage = ...
            ME.message;

    end

end


finalWallTime = ...
    toc(tFinal);


nRBFSolvesFinal = ...
    finalEval.nRBFSolves;


% =============================================================
% 16. Fixed-parameter run
%
% The final evaluation is also the complete search trace.
% =============================================================

if allFixed

    currentStage = ...
        3;


    appendEvaluation(finalEval);


    if finalEval.success && ...
       isfinite(finalEval.AE)

        search_best_ae = ...
            finalEval.AE;

    else

        Result = ...
            makeResult( ...
                false, ...
                'Fixed-parameter evaluation failed.', ...
                finalEval);

        return

    end

end


% =============================================================
% 17. Final validity
% =============================================================

if ~finalEval.success || ...
   ~isscalar(finalEval.AE) || ...
   ~isreal(finalEval.AE) || ...
   ~isfinite(finalEval.AE) || ...
   ~isscalar(finalEval.yhat) || ...
   ~isreal(finalEval.yhat) || ...
   ~isfinite(finalEval.yhat)

    Result = ...
        makeResult( ...
            false, ...
            'Final selected-parameter evaluation failed.', ...
            finalEval);

    return

end


% =============================================================
% 18. Search-trace consistency
% =============================================================

nTrace = ...
    numel(AE_eval_log);


if numel(C_eval_log) ~= nTrace || ...
   numel(Reg_eval_log) ~= nTrace || ...
   numel(epsilon_eval_log) ~= nTrace || ...
   numel(yhat_eval_log) ~= nTrace || ...
   numel(ydiff_eval_log) ~= nTrace || ...
   numel(cond_reg_eval_log) ~= nTrace || ...
   numel(solver_eval_log) ~= nTrace || ...
   numel(success_eval_log) ~= nTrace || ...
   numel(stage_eval_log) ~= nTrace

    error( ...
        'Local parameter-search traces have inconsistent lengths.');

end


if allFixed

    if nTrace ~= 1
        error( ...
            ['Fixed-parameter local run must contain exactly ' ...
             'one evaluation.']);
    end

elseif strcmp(method,'grid')

    if nTrace ~= nGridEvaluations
        error( ...
            ['Search-trace length does not match direct-grid ' ...
             'evaluation count.']);
    end

else

    if nTrace ~= ...
            (nGAEvaluations + ...
             nFallbackEvaluations + ...
             nRefinementEvaluations)

        error( ...
            ['Search-trace length does not match GA + fallback + ' ...
             'local-refinement evaluation counts.']);
    end

end


% =============================================================
% 19. Search-versus-final consistency
% =============================================================

finalSearchAEDifference = ...
    abs(finalEval.AE-search_best_ae);


aeTol = ...
    1e-10 * ...
    max(1,abs(finalEval.AE));


searchConsistent = ...
    finalSearchAEDifference <= ...
    aeTol;


% =============================================================
% 20. Boundary diagnostics
%
% Diagnostic only.
% =============================================================

if useShape && ...
   ~C_fixed

    zbestC = ...
        log10(selectedC);


    CRelativePosition = ...
        (zbestC-zCmin) / ...
        (zCmax-zCmin);


    CNearLowerBound = ...
        CRelativePosition <= ...
        BOUND_TOL;


    CNearUpperBound = ...
        CRelativePosition >= ...
        (1-BOUND_TOL);

end


if ~Reg_fixed

    zbestReg = ...
        log10(selectedReg);


    RegRelativePosition = ...
        (zbestReg-zRmin) / ...
        (zRmax-zRmin);


    RegNearLowerBound = ...
        RegRelativePosition <= ...
        BOUND_TOL;


    RegNearUpperBound = ...
        RegRelativePosition >= ...
        (1-BOUND_TOL);

end


% =============================================================
% 21. Successful result
% =============================================================

Result = ...
    makeResult( ...
        true, ...
        '', ...
        finalEval);


% =============================================================
% 22. Nested GA objective
% =============================================================

    function f = ga_fitness_log_z(zvec)

        % -----------------------------------------------------
        % Log-space variables -> physical parameters
        % -----------------------------------------------------

        if useShape

            if ~C_fixed && ~Reg_fixed

                C_candidate = ...
                    10.^zvec(1);


                Reg_candidate = ...
                    10.^zvec(2);


            elseif ~C_fixed

                C_candidate = ...
                    10.^zvec(1);


                Reg_candidate = ...
                    Reg_min;


            elseif ~Reg_fixed

                C_candidate = ...
                    C_min;


                Reg_candidate = ...
                    10.^zvec(1);


            else

                C_candidate = ...
                    C_min;


                Reg_candidate = ...
                    Reg_min;

            end

        else

            C_candidate = ...
                NaN;


            Reg_candidate = ...
                10.^zvec(1);

        end


        % -----------------------------------------------------
        % Evaluate candidate using fixed point-radius geometry
        % -----------------------------------------------------

        try

            E = ...
                localrbf.evaluateTargetExcludedError( ...
                    cache, ...
                    kernel, ...
                    useShape, ...
                    C_candidate, ...
                    Reg_candidate, ...
                    searchOptions);


            nRBFSolvesSearch = ...
                nRBFSolvesSearch + ...
                E.nRBFSolves;


        catch ME

            E = ...
                makeExceptionEvaluation( ...
                    C_candidate, ...
                    Reg_candidate, ...
                    ME.message);


            if isempty(firstFailureMessage)

                firstFailureMessage = ...
                    ME.message;

            end

        end


        % -----------------------------------------------------
        % Objective supplied to GA
        % -----------------------------------------------------

        ae_actual = ...
            E.AE;


        if E.success && ...
           isfinite(ae_actual)

            f = ...
                ae_actual;

        else

            f = ...
                FAIL_PENALTY;


            E.AE = ...
                NaN;


            if isempty(firstFailureMessage) && ...
               ~isempty(E.failureMessage)

                firstFailureMessage = ...
                    E.failureMessage;

            end

        end


        % -----------------------------------------------------
        % Preserve every scientific evaluation
        % -----------------------------------------------------

        appendEvaluation(E);

    end


% =============================================================
% 23. Evaluation-trace helper
% =============================================================

    function appendEvaluation(E)

        C_eval_log(end+1,1) = ...
            E.C;


        Reg_eval_log(end+1,1) = ...
            E.reg_param;


        epsilon_eval_log(end+1,1) = ...
            E.epsilon;


        AE_eval_log(end+1,1) = ...
            E.AE;


        yhat_eval_log(end+1,1) = ...
            E.yhat;


        ydiff_eval_log(end+1,1) = ...
            E.y_diff;


        cond_reg_eval_log(end+1,1) = ...
            E.cond_reg;


        solver_eval_log{end+1,1} = ...
            E.solver_used;


        success_eval_log(end+1,1) = ...
            E.success;


        stage_eval_log(end+1,1) = ...
            currentStage;

    end


% =============================================================
% 24. Exception-evaluation helper
%
% Preserves the attempted parameter values even when the
% low-level evaluator throws an exception.
% =============================================================

    function E = makeExceptionEvaluation( ...
            Cv,Regv,message)

        E = ...
            struct();


        if useShape

            E.C = ...
                Cv;


            E.epsilon = ...
                max(Cv*cache.d,1e-12);

        else

            E.C = ...
                NaN;


            E.epsilon = ...
                1;

        end


        E.reg_param = ...
            Regv;


        E.yhat = ...
            NaN;


        E.y_diff = ...
            NaN;


        E.AE = ...
            NaN;


        E.cond_reg = ...
            NaN;


        E.rel_residual = ...
            NaN;


        E.solver_used = ...
            '';


        E.success = ...
            false;


        % No returned evaluator result exists from which an exact
        % completed solve count can be recovered.
        E.nRBFSolves = ...
            0;


        E.failureMessage = ...
            message;

    end


% =============================================================
% 25. Unified result helper
%
% Success and failure results use the same major field structure.
% =============================================================

    function R = makeResult( ...
            successFlag,message,selectedEval)

        R = ...
            struct();


        R.success = ...
            successFlag;


        % -----------------------------------------------------
        % Preserve selected/attempted parameters whenever known
        % -----------------------------------------------------

        R.C = ...
            selectedC;


        R.reg_param = ...
            selectedReg;


        if isempty(selectedEval)

            R.epsilon = ...
                NaN;

            R.AE = ...
                NaN;

            R.yhat = ...
                NaN;

            R.y_diff = ...
                NaN;

            R.cond_reg = ...
                NaN;

            R.rel_residual = ...
                NaN;

            R.solver_used = ...
                '';

            R.selected = ...
                [];

        else

            R.epsilon = ...
                selectedEval.epsilon;

            R.AE = ...
                selectedEval.AE;

            R.yhat = ...
                selectedEval.yhat;

            R.y_diff = ...
                selectedEval.y_diff;

            R.cond_reg = ...
                selectedEval.cond_reg;

            R.rel_residual = ...
                selectedEval.rel_residual;

            R.solver_used = ...
                selectedEval.solver_used;

            R.selected = ...
                selectedEval;

        end


        % -----------------------------------------------------
        % Complete scientific search trace
        % -----------------------------------------------------

        R.trace = ...
            struct();


        R.trace.C = ...
            C_eval_log;


        R.trace.RegParam = ...
            Reg_eval_log;


        R.trace.epsilon = ...
            epsilon_eval_log;


        R.trace.AE = ...
            AE_eval_log;


        R.trace.yhat = ...
            yhat_eval_log;


        R.trace.y_diff = ...
            ydiff_eval_log;


        R.trace.cond_reg = ...
            cond_reg_eval_log;


        R.trace.solver_used = ...
            solver_eval_log;


        R.trace.success = ...
            success_eval_log;


        R.trace.stage = ...
            stage_eval_log;


        % -----------------------------------------------------
        % Search information
        % -----------------------------------------------------

        R.searchInfo = ...
            struct();


        R.searchInfo.success = ...
            successFlag;


        R.searchInfo.useShape = ...
            useShape;


        if useShape

            R.searchInfo.C_fixed = ...
                C_fixed;

            R.searchInfo.C_min = ...
                C_min;

            R.searchInfo.C_max = ...
                C_max;

        else

            R.searchInfo.C_fixed = ...
                NaN;

            R.searchInfo.C_min = ...
                NaN;

            R.searchInfo.C_max = ...
                NaN;

        end


        R.searchInfo.Reg_fixed = ...
            Reg_fixed;


        R.searchInfo.Reg_min = ...
            Reg_min;


        R.searchInfo.Reg_max = ...
            Reg_max;


        R.searchInfo.nvars = ...
            nvars;


        R.searchInfo.usedFallback = ...
            usedFallback;


        R.searchInfo.fallbackSucceeded = ...
            fallbackSucceeded;


        R.searchInfo.refinementSucceeded = ...
            refinementSucceeded;


        R.searchInfo.refinementSource = ...
            refinementSource;


        R.searchInfo.exitflag = ...
            exitflag;


        R.searchInfo.gaBestAE = ...
            ga_best_ae;


        R.searchInfo.searchBestAE = ...
            search_best_ae;


        if isempty(selectedEval)

            R.searchInfo.finalAE = ...
                NaN;

        else

            R.searchInfo.finalAE = ...
                selectedEval.AE;

        end


        % -----------------------------------------------------
        % Evaluation counts
        % -----------------------------------------------------

        R.searchInfo.nEvaluations = ...
            numel(AE_eval_log);


        R.searchInfo.nGAEvaluations = ...
            nGAEvaluations;


        R.searchInfo.nFallbackEvaluations = ...
            nFallbackEvaluations;


        R.searchInfo.nRefinementEvaluations = ...
            nRefinementEvaluations;


        R.searchInfo.nValidEvaluations = ...
            sum(isfinite(AE_eval_log));


        R.searchInfo.nFailedEvaluations = ...
            sum(~isfinite(AE_eval_log));

R.searchInfo.requestedMethod = ...
    method;

R.searchInfo.nGrid = ...
    nGrid;

R.searchInfo.nGridEvaluations = ...
    nGridEvaluations;

R.searchInfo.gridSucceeded = ...
    gridSucceeded;

R.searchInfo.gridWallTime = ...
    gridWallTime;
        % -----------------------------------------------------
        % RBF-solve counts
        % -----------------------------------------------------

        R.searchInfo.nRBFSolvesSearch = ...
            nRBFSolvesSearch;


        R.searchInfo.nRBFSolvesFinal = ...
            nRBFSolvesFinal;


        R.searchInfo.nRBFSolves = ...
            nRBFSolvesSearch + ...
            nRBFSolvesFinal;


        % -----------------------------------------------------
        % MATLAB GA information
        % -----------------------------------------------------

        if isfield(output,'generations')

            R.searchInfo.generations = ...
                output.generations;

        else

            R.searchInfo.generations = ...
                NaN;

        end


        if isfield(output,'funccount')

            R.searchInfo.funccount = ...
                output.funccount;

        else

            R.searchInfo.funccount = ...
                NaN;

        end


        R.searchInfo.gaConverged = ...
            ~allFixed && ...
            ~usedFallback && ...
            isfinite(exitflag) && ...
            exitflag > 0;


        R.searchInfo.acceptedLimitStop = ...
            ~allFixed && ...
            ~usedFallback && ...
            isfinite(exitflag) && ...
            exitflag == 0;

R.searchInfo.converged = ...
    allFixed || ...
    (strcmp(method,'grid') && gridSucceeded) || ...
    R.searchInfo.gaConverged;


        if allFixed

            R.searchInfo.method = ...
                'fixed parameters';

        elseif usedFallback

            R.searchInfo.method = ...
       'GA + coarse-grid fallback';

            elseif strcmp(method,'grid')
    R.searchInfo.method = 'logarithmic grid';

        else

            R.searchInfo.method = ...
                'GA + local refinement';

        end


        % -----------------------------------------------------
        % Messages
        % -----------------------------------------------------

        if successFlag

            if usedFallback

                R.searchInfo.message = ...
                    gaMessage;

            elseif isfield(output,'message')

                R.searchInfo.message = ...
                    output.message;

            else

                R.searchInfo.message = ...
                    '';

            end

        else

            R.searchInfo.message = ...
                message;

        end


        R.searchInfo.gaMessage = ...
            gaMessage;


        R.searchInfo.firstFailureMessage = ...
            firstFailureMessage;


        % -----------------------------------------------------
        % Boundary diagnostics
        % -----------------------------------------------------

        R.searchInfo.boundaryTolerance = ...
            BOUND_TOL;


        R.searchInfo.CRelativePosition = ...
            CRelativePosition;


        R.searchInfo.CNearLowerBound = ...
            CNearLowerBound;


        R.searchInfo.CNearUpperBound = ...
            CNearUpperBound;


        R.searchInfo.RegRelativePosition = ...
            RegRelativePosition;


        R.searchInfo.RegNearLowerBound = ...
            RegNearLowerBound;


        R.searchInfo.RegNearUpperBound = ...
            RegNearUpperBound;


        % -----------------------------------------------------
        % Search-versus-final consistency
        % -----------------------------------------------------

        R.searchInfo.finalSearchAEDifference = ...
            finalSearchAEDifference;


        R.searchInfo.searchConsistent = ...
            searchConsistent;


        % -----------------------------------------------------
        % Timing
        % -----------------------------------------------------

        R.searchInfo.gaWallTime = ...
            gaWallTime;


        R.searchInfo.fallbackWallTime = ...
            fallbackWallTime;


        R.searchInfo.refinementWallTime = ...
            refinementWallTime;


        R.searchInfo.finalWallTime = ...
            finalWallTime;


        R.searchInfo.totalWallTime = ...
            toc(tTotal);


        % -----------------------------------------------------
        % GA configuration
        % -----------------------------------------------------

        R.searchInfo.GA = ...
            GAInfo;

    end


function evaluate_log_grid()

    if useShape

        if C_fixed
            C_grid = C_min;
        else
            C_grid = logspace(zCmin,zCmax,nCgrid);
        end

    else

        C_grid = NaN;

    end


    if Reg_fixed
        Reg_grid = Reg_min;
    else
        Reg_grid = logspace(zRmin,zRmax,nRgrid);
    end


    if useShape

        if ~C_fixed && ~Reg_fixed

            for iCgrid = 1:numel(C_grid)

                for iRgrid = 1:numel(Reg_grid)

                    ga_fitness_log_z( ...
                        [log10(C_grid(iCgrid)), ...
                         log10(Reg_grid(iRgrid))]);

                end

            end


        elseif ~C_fixed

            for iCgrid = 1:numel(C_grid)

                ga_fitness_log_z( ...
                    log10(C_grid(iCgrid)));

            end


        elseif ~Reg_fixed

            for iRgrid = 1:numel(Reg_grid)

                ga_fitness_log_z( ...
                    log10(Reg_grid(iRgrid)));

            end

        end

    else

        for iRgrid = 1:numel(Reg_grid)

            ga_fitness_log_z( ...
                log10(Reg_grid(iRgrid)));

        end

    end

end

end

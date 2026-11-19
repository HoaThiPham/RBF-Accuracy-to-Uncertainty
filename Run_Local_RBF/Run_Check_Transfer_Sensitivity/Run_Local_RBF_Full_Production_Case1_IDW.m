%% ============================================================
% LOCAL RBF FULL PRODUCTION BATCH
%% ============================================================

clearvars
clc

% Refresh function resolution on the client.
% Parallel-pool startup/restart is controlled explicitly below.
clear functions
rehash

OutputDir = ...
    'OutPut_RBF_2026_2';

% =============================================================
% 1. Load COMPLETE train/test dataset
% =============================================================

load( ...
    'OutPutSplitTrainTest\InputData_Case1.mat', ...
    'InputData_Case1');

InputData = ...
    InputData_Case1;

CaseName = ...
    'Case1';


% =============================================================
% 1.1 Result-file model/kernel tag
%
% USER INPUT:
%   Define only the model/kernel part of the output name here.
%
% Examples:
%   'E08_MG'
%   'E08_G'
%   'XG19_LCT'
%
% The scientific transfer/cap part is appended automatically later from:
%
%       TRANSFER_METHOD             % GGM x Kernel matrix
%       N_IDW_SOURCES               % GGM x Kernel matrix
%       PARAMETER_TRANSFER_MODE      % GGM x Kernel matrix
%       near_threshold              % GGM x Kernel matrix
%       CAP_RMSE_SOURCE             % common
%
% The parameter-transfer tag is included only when numerically active
% (IDW/hybrid). For nearest/ensemble, parameter aggregation is inactive.
%
% Examples:
%   E08_MG_HYBRID_N7_M24_CAPTRAIN
%   E08_MG_IDW_N7_M21_CAPPSEUDO
%   E08_MG_NEAREST_CAPTRAIN
%   E08_MG_ENSEMBLE_N7_CAPPSEUDO
% =============================================================

MODEL_KERNEL_TAG = ...
    'IDW';


% =============================================================
% 2. Models
% =============================================================

ListOfModel = {
    'EGM2008'
       'ggmplus'
    'eigen6c4'
    'geco'
    'sgg2'
        'xgm19'
};


% =============================================================
% 3. RBF kernels
% =============================================================

NameFunction_RBF = {
    'gaussian'
    'multiquadric'
       'thinplate'
    'cubic'
     'linear'
    };




% =============================================================
% 4. Candidate local-radius grids
%
% IMPORTANT:
%   - supplied in ORIGINAL coordinate units;
%   - LocalRBFTraining applies S_global internally;
%   - current XYZ coordinates are ECEF-type coordinates in metres;
%   - therefore the values below are in metres.
%
% All kernels in the current production configuration:
%       50 km : 10 km : 200 km
% =============================================================

Radii = ...
    struct();


Radii.gaussian = ...
    (50000:10000:200000).';


Radii.multiquadric = ...
    (50000:10000:200000).';


Radii.linear = ...
    (50000:10000:200000).';


Radii.cubic = ...
    (50000:10000:200000).';


Radii.thinplate = ...
    (50000:10000:200000).';


% =============================================================
% 5. Minimum local fitting observations
% =============================================================

min_neighbors = ...
    15;


% =============================================================
% 6. Shape-parameter search bounds
%
% Shape kernels:
%   Gaussian
%   Multiquadric
%
% Shape-free kernels do not use C.
% Their dummy C fields are retained only for a uniform batch
% settings structure.
% =============================================================

CMin = ...
    struct();

CMax = ...
    struct();


CMin.gaussian = ...
    1e-1;
% CMin.gaussian = ...
%     1e-3;

CMax.gaussian = ...
    1.5;


CMin.multiquadric = ...
    1e-3;

CMax.multiquadric = ...
    1.5;


% Shape-free kernels
CMin.linear = ...
    1;

CMax.linear = ...
    1;


CMin.cubic = ...
    1;

CMax.cubic = ...
    1;


CMin.thinplate = ...
    1;

CMax.thinplate = ...
    1;



% =============================================================
% Tuning method
% =============================================================

Method = ...
    struct();

Method.gaussian = ...
    'ga';

Method.multiquadric = ...
    'ga';

Method.linear = ...
    'grid';

Method.cubic = ...
    'grid';

Method.thinplate = ...
    'grid';


% =============================================================
% Number of grid points
% =============================================================

% nGrid_Value=20;

nGrid_Value=10;

nGrid = ...
    struct();

nGrid.gaussian = [20 20];

nGrid.multiquadric =[20 20];

nGrid.linear = ...
    nGrid_Value;

nGrid.cubic = ...
    nGrid_Value;

nGrid.thinplate = ...
    nGrid_Value;
% =============================================================
% 7. Regularization search bounds
% =============================================================

RegMin = ...
    struct();

RegMax = ...
    struct();



% RegMin.gaussian = ...
%      2*1e-2;
% 
% RegMax.gaussian = ...
%     2*1e-1;
% 
% 
% RegMin.multiquadric = ...
%     1e-6;
% 
% RegMax.multiquadric = ...
%      1e-4;


RegMin.gaussian = ...
     5*1e-2;

RegMax.gaussian = ...
    1.5*1e-1;


RegMin.multiquadric = ...
    5*1e-6;

RegMax.multiquadric = ...
     1.5*1e-5;





% RegMin.linear = ...
%     2*1e-3;
% 
% RegMax.linear = ...
%         2*1e-2;
% 
% 
% RegMin.cubic = ...
%     2*1e-3;
% 
% RegMax.cubic = ...
%    2*1e-2;
% 
% 
% RegMin.thinplate = ...
%     2*1e-3;
% 
% RegMax.thinplate = ...
%  2*1e-2;


RegMin.linear = ...
    5*1e-3;

RegMax.linear = ...
       1.5*1e-2;


RegMin.cubic = ...
 5*1e-3;

RegMax.cubic = ...
   1.5*1e-2;


RegMin.thinplate = ...
   5*1e-3;

RegMax.thinplate = ...
   1.5*1e-2;


% =============================================================
% 8. Radius-selection tolerances
%
% MATRIX FORMAT:
%
%   rows    = ListOfModel
%   columns = NameFunction_RBF
%
% Therefore:
%
%   size(tau_loc) = ...
%       [numel(ListOfModel), numel(NameFunction_RBF)]
%
%   size(tau_cap) = ...
%       [numel(ListOfModel), numel(NameFunction_RBF)]
%
% Point level:
%
%   choose the smallest radius satisfying
%
%       AE(r) <= min(AE) + tau_loc(iModel,iKernel)
%
% Training-set cap level:
%
%   choose the smallest cap satisfying
%
%       RMSE(k) <= min(RMSE) + tau_cap(iModel,iKernel)
%
% CURRENT ACTIVE CONFIGURATION:
%
%   ListOfModel:
%       row 1 = EGM2008
%
%   NameFunction_RBF:
%       col 1 = gaussian
%       col 2 = multiquadric
% =============================================================

tau_loc = ...
    [ ...
        0.000,   0.000,  0.000  0.000,  0.000 ;... 
    0.000,   0.000,  0.000  0.000,  0.000 ;... 
      0.000,   0.000,  0.000  0.000,  0.000 ;... 
    0.000,   0.000,  0.000  0.000,  0.000 ;...
        0.000,   0.000,  0.000  0.000,  0.000 ;... 
    0.000,   0.000,  0.000  0.000,  0.000 ;...           
    ];


tau_cap = ...
    [ ...
        0.000,   0.000,  0.000  0.000,  0.000 ;... 
    0.000,   0.000,  0.000  0.000,  0.000 ;... 
      0.000,   0.000,  0.000  0.000,  0.000 ;... 
    0.000,   0.000,  0.000  0.000,  0.000 ;...
        0.000,   0.000,  0.000  0.000,  0.000 ;... 
    0.000,   0.000,  0.000  0.000,  0.000 ;...           
    ];



% CURRENT ACTIVE tolerance matrices must match:
%   numel(ListOfModel)      = 1  -> EGM2008
%   numel(NameFunction_RBF) = 2  -> gaussian, multiquadric
%
% Therefore the active size is exactly 1 x 2.

% -------------------------------------------------------------
% Example when EGM2008 and xgm19 are both active:
%
% ListOfModel = {
%     'EGM2008'
%     'xgm19'
%     };
%
% NameFunction_RBF = {
%     'gaussian'
%     'multiquadric'
%     };
%
% tau_loc = [
%     0.0010, 0.0020;    % EGM2008
%     0.0015, 0.0025     % xgm19
%     ];
%
% tau_cap = [
%     0.0030, 0.0040;    % EGM2008
%     0.0035, 0.0045     % xgm19
%     ];
%
% IMPORTANT:
% If ListOfModel or NameFunction_RBF changes, update the matrix rows/
% columns so their order remains exactly aligned with those two lists.
% -------------------------------------------------------------

% =============================================================
% 9. Train-to-test transfer settings
%
% Public distance values are in ORIGINAL coordinate units.
% Current XYZ units are metres.
%
% TRANSFER_METHOD  [GGM x Kernel matrix]
% ----------------------------------------
% 'nearest'
%   Always use the nearest training donor.
%   N_IDW_SOURCES, near_threshold and PARAMETER_TRANSFER_MODE are inactive.
%
% 'idw'
%   Always use N nearest donors with inverse-square weights.
%   Radius = weighted arithmetic mean.
%   RegParam/C aggregation follows PARAMETER_TRANSFER_MODE.
%
% 'hybrid'
%   nearest when nearest distance <= near_threshold;
%   otherwise N-nearest inverse-square IDW.
%   Radius = weighted arithmetic mean in the IDW branch.
%   RegParam/C aggregation follows PARAMETER_TRANSFER_MODE.
%
% 'ensemble'
%   nearest when nearest distance <= near_threshold;
%   otherwise retain N nearest COMPLETE donor tuples:
%       (Radius, RegParam, C)
%   deduplicate identical tuples, solve every unique tuple once, then
%   combine tuple predictions using donor weights.
%
%   PARAMETER_TRANSFER_MODE is inactive for ensemble because complete
%   tuples are solved separately rather than parameter-averaged.
%
% N_IDW_SOURCES  [GGM x Kernel numeric matrix]
% ----------------------------------------------
% Number of nearest donors used by IDW/hybrid/ensemble far transfer.
% Each active Model x Kernel combination may use its own positive integer N.
%
% PARAMETER_TRANSFER_MODE  [GGM x Kernel matrix]
% ------------------------------------------------
% M21 = 'arithmetic_arithmetic'
%       RegParam arithmetic, C arithmetic
%
% M22 = 'log_arithmetic'
%       RegParam log10-geometric, C arithmetic
%
% M23 = 'arithmetic_log'
%       RegParam arithmetic, C log10-geometric
%
% M24 = 'log_log'   [current/default production choice]
%       RegParam log10-geometric, C log10-geometric
%
% For shape-free kernels C is inactive:
%       M21 == M23
%       M22 == M24
%
% CAP_RMSE_SOURCE
% ---------------
% 'training'
%   Final radius cap is selected from ordinary training RMSE.
%
% 'pseudo'
%   LocalRBFTraining internally calculates strict target-excluded
%   pseudo-training RMSE at each candidate cap using the SAME transfer
%   configuration. Each pseudo target is excluded from:
%       (1) transfer donors;
%       (2) local fitting observations.
%
%   Stage-1 donor settings are reused. No nested re-optimization occurs.
%
% near_threshold  [GGM x Kernel numeric matrix]
% ------------------------------------------------
% Raw-unit threshold used only by hybrid/ensemble near/far logic.
% Each active Model x Kernel combination may use its own positive threshold.
% The batch copies this exact value into:
%       options.transfer.nearThresholdRaw
% for internal pseudo-training and also passes the same positional value
% to LocalRBFTesting. Therefore pseudo-training and test transfer use the
% same threshold without maintaining two independent user inputs.
%
% idw_radius
% ----------
% Retained only for backward compatibility/metadata.
% It is NOT used for donor/source selection.
% =============================================================

% -------------------------------------------------------------
% GGM x Kernel matrix contract
%
% Rows    = ListOfModel
% Columns = NameFunction_RBF
%
% Current active order:
%   row 1 = EGM2008
%   col 1 = gaussian
%   col 2 = multiquadric
%
% Example with different transfer choices:
%
%   TRANSFER_METHOD = {
%       'hybrid',  'ensemble'
%   };
%
%   PARAMETER_TRANSFER_MODE = {
%       'log_log', 'not-applicable'
%   };
%
%   N_IDW_SOURCES = [
%       7, 9
%   ];
%
%   near_threshold = [
%       10000, 15000
%   ];
%
% For nearest/ensemble, PARAMETER_TRANSFER_MODE is numerically inactive
% and will be normalized to 'not-applicable'.
%
% The active values below preserve the previous numerical behavior while
% changing the interface to GGM x Kernel matrices.
% -------------------------------------------------------------



TRANSFER_METHOD = {
        'idw',  'idw', 'idw', 'idw', 'idw';
        'idw',  'idw', 'idw', 'idw', 'idw';
     'idw',  'idw', 'idw', 'idw', 'idw';
        'idw',  'idw', 'idw', 'idw', 'idw';
             'idw',  'idw', 'idw', 'idw', 'idw';
        'idw',  'idw', 'idw', 'idw', 'idw';
};

% Rows    = ListOfModel
% Columns = NameFunction_RBF
N_IDW_SOURCES = [
  10,   10,   10,   10,  10;
  10,   10,   10,   10,  10;
  10,   10,   10,   10,  10;
  10,   10,   10,   10,  10;
  10,   10,   10,   10,  10;
  10,   10,   10,   10,  10;
];


PARAMETER_TRANSFER_MODE = {
         'arithmetic_arithmetic', 'log_log', 'arithmetic_arithmetic',  'arithmetic_arithmetic', 'arithmetic_arithmetic';
     'arithmetic_arithmetic', 'log_log', 'arithmetic_arithmetic',  'arithmetic_arithmetic', 'arithmetic_arithmetic';
         'arithmetic_arithmetic', 'log_log', 'arithmetic_arithmetic',  'arithmetic_arithmetic', 'arithmetic_arithmetic';
     'arithmetic_arithmetic', 'log_log', 'arithmetic_arithmetic',  'arithmetic_arithmetic', 'arithmetic_arithmetic';
            'arithmetic_arithmetic', 'log_log', 'arithmetic_arithmetic',  'arithmetic_arithmetic', 'arithmetic_arithmetic';
     'arithmetic_arithmetic', 'log_log', 'arithmetic_arithmetic',  'arithmetic_arithmetic', 'arithmetic_arithmetic';
};


CAP_RMSE_SOURCE = ...
   'pseudo';                % common: 'training' | 'pseudo'


% Rows    = ListOfModel
% Columns = NameFunction_RBF
% Units   = raw coordinate units (metres for current ECEF XYZ)
near_threshold = [
      5000,5000,  5000,5000, 5000;
   5000,5000,  5000,5000, 5000;
       5000,5000,  5000,5000, 5000;
   5000,5000,  5000,5000, 5000;
       5000,5000,  5000,5000, 5000;
   5000,5000,  5000,5000, 5000;
];


idw_radius = ...
    30000;                     % compatibility/metadata only


% =============================================================
% 10. Reproducibility
% =============================================================

verbose = ...
    true;


RNG_SEED = ...
    11;


% =============================================================
% 11. Solver settings
% =============================================================

options = ...
    struct();


options.solver = ...
    struct();


% Production threshold -- keep validated value.
options.solver.cond_lsq = ...
    1e12;


% OFF during GA/fallback search.
% Selected-system diagnostics enable it internally.
options.solver.compute_rel_residual = ...
    false;


% =============================================================
% 11.1 Transfer settings
% =============================================================

options.transfer = ...
    struct();


options.transfer.method = ...
    TRANSFER_METHOD;


options.transfer.nIDWSources = ...
    N_IDW_SOURCES;


options.transfer.parameterTransferMode = ...
    PARAMETER_TRANSFER_MODE;


options.transfer.capRMSESource = ...
    CAP_RMSE_SOURCE;


% Do not manually maintain options.transfer.nearThresholdRaw here.
% The batch function copies the authoritative positional near_threshold
% into that field before LocalRBFTraining is called.


% =============================================================
% 11.2 Automatic result-file scientific tag
%
% Because TRANSFER_METHOD and PARAMETER_TRANSFER_MODE now vary over the
% GGM x Kernel matrix, a single readable transfer label is insufficient.
%
% The script validates/canonicalizes TRANSFER_METHOD,
% PARAMETER_TRANSFER_MODE, N_IDW_SOURCES and near_threshold, then creates
% one compact deterministic checksum for filenames. The full matrices saved
% in Results.Settings remain the authoritative scientific provenance.
% =============================================================

nModelActive = ...
    numel(ListOfModel);


nKernelActive = ...
    numel(NameFunction_RBF);


expectedTransferSize = ...
    [nModelActive,nKernelActive];


if ~isequal(size(TRANSFER_METHOD),expectedTransferSize)

    error( ...
        ['TRANSFER_METHOD must have size %d x %d. Rows follow ' ...
         'ListOfModel; columns follow NameFunction_RBF.'], ...
        nModelActive,nKernelActive);

end


if ~isequal(size(PARAMETER_TRANSFER_MODE),expectedTransferSize)

    error( ...
        ['PARAMETER_TRANSFER_MODE must have size %d x %d. Rows follow ' ...
         'ListOfModel; columns follow NameFunction_RBF.'], ...
        nModelActive,nKernelActive);

end


if ~isequal(size(N_IDW_SOURCES),expectedTransferSize)

    error( ...
        ['N_IDW_SOURCES must have size %d x %d. Rows follow ' ...
         'ListOfModel; columns follow NameFunction_RBF.'], ...
        nModelActive,nKernelActive);

end


if ~isnumeric(N_IDW_SOURCES) || ...
   ~isreal(N_IDW_SOURCES) || ...
   any(~isfinite(N_IDW_SOURCES(:))) || ...
   any(N_IDW_SOURCES(:) < 1) || ...
   any(N_IDW_SOURCES(:) ~= round(N_IDW_SOURCES(:)))

    error('Every N_IDW_SOURCES entry must be a positive integer.');

end


if ~isequal(size(near_threshold),expectedTransferSize)

    error( ...
        ['near_threshold must have size %d x %d. Rows follow ' ...
         'ListOfModel; columns follow NameFunction_RBF.'], ...
        nModelActive,nKernelActive);

end


if ~isnumeric(near_threshold) || ...
   ~isreal(near_threshold) || ...
   any(~isfinite(near_threshold(:))) || ...
   any(near_threshold(:) <= 0)

    error('Every near_threshold entry must be finite and positive.');

end


transferMethodCanonical = ...
    strings(expectedTransferSize);


parameterModeCanonical = ...
    strings(expectedTransferSize);


parameterTagMatrix = ...
    strings(expectedTransferSize);


for iModelConfig = 1:nModelActive

    for iKernelConfig = 1:nKernelActive

        methodValue = ...
            lower(strtrim(string( ...
                TRANSFER_METHOD{iModelConfig,iKernelConfig})));


        if ~any(methodValue == ["nearest","idw","hybrid","ensemble"])

            error( ...
                ['TRANSFER_METHOD{%d,%d} must be nearest, idw, ' ...
                 'hybrid, or ensemble.'], ...
                iModelConfig,iKernelConfig);

        end


        transferMethodCanonical(iModelConfig,iKernelConfig) = ...
            methodValue;


        if any(methodValue == ["nearest","ensemble"])

            parameterModeCanonical(iModelConfig,iKernelConfig) = ...
                "not-applicable";


            parameterTagMatrix(iModelConfig,iKernelConfig) = ...
                "NA";

        else

            parameterValue = ...
                lower(strtrim(string( ...
                    PARAMETER_TRANSFER_MODE{iModelConfig,iKernelConfig})));


            if any(parameterValue == ...
                    ["arithmetic_arithmetic","arith_arith","m21"])

                parameterModeCanonical(iModelConfig,iKernelConfig) = ...
                    "arithmetic_arithmetic";

                parameterTagMatrix(iModelConfig,iKernelConfig) = ...
                    "M21";


            elseif any(parameterValue == ...
                    ["log_arithmetic","log_arith","m22"])

                parameterModeCanonical(iModelConfig,iKernelConfig) = ...
                    "log_arithmetic";

                parameterTagMatrix(iModelConfig,iKernelConfig) = ...
                    "M22";


            elseif any(parameterValue == ...
                    ["arithmetic_log","arith_log","m23"])

                parameterModeCanonical(iModelConfig,iKernelConfig) = ...
                    "arithmetic_log";

                parameterTagMatrix(iModelConfig,iKernelConfig) = ...
                    "M23";


            elseif any(parameterValue == ["log_log","m24"])

                parameterModeCanonical(iModelConfig,iKernelConfig) = ...
                    "log_log";

                parameterTagMatrix(iModelConfig,iKernelConfig) = ...
                    "M24";


            else

                error( ...
                    ['PARAMETER_TRANSFER_MODE{%d,%d} must be M21-M24 ' ...
                     'for IDW/hybrid.'], ...
                    iModelConfig,iKernelConfig);

            end

        end

    end

end


TRANSFER_METHOD = ...
    cellstr(transferMethodCanonical);


PARAMETER_TRANSFER_MODE = ...
    cellstr(parameterModeCanonical);


capRMSESourceForFile = ...
    lower(strtrim(char(CAP_RMSE_SOURCE)));


switch capRMSESourceForFile

    case {'training','original','origin'}
        CAP_FILE_TAG = 'CAPTRAIN';

    case {'pseudo','pseudo-training','pseudotraining'}
        CAP_FILE_TAG = 'CAPPSEUDO';

    otherwise
        error('CAP_RMSE_SOURCE must be ''training'' or ''pseudo''.');
end


configTokenMatrix = ...
    strings(expectedTransferSize);


for iModelConfig = 1:nModelActive

    for iKernelConfig = 1:nKernelActive

        methodValue = ...
            transferMethodCanonical(iModelConfig,iKernelConfig);


        switch methodValue

            case "nearest"
                methodTag = "N";

            case "idw"
                methodTag = "I";

            case "hybrid"
                methodTag = "H";

            case "ensemble"
                methodTag = "E";
        end


        nTag = ...
            sprintf( ...
                '%d', ...
                N_IDW_SOURCES(iModelConfig,iKernelConfig));


        thresholdTag = ...
            sprintf( ...
                '%.12g', ...
                near_threshold(iModelConfig,iKernelConfig));


        configTokenMatrix(iModelConfig,iKernelConfig) = ...
            methodTag + "-" + ...
            parameterTagMatrix(iModelConfig,iKernelConfig) + ...
            "-N" + string(nTag) + ...
            "-T" + string(thresholdTag);

    end

end


configText = ...
    strjoin(cellstr(configTokenMatrix(:).'),'|');


configCodes = ...
    uint64(double(char(configText)));


configWeights = ...
    uint64(1:numel(configCodes));


configChecksum = ...
    mod( ...
        sum(configCodes .* configWeights), ...
        uint64(100000000));


TRANSFER_FILE_TAG = ...
    sprintf( ...
        'TMX_%dx%d_%08d_%s', ...
        nModelActive, ...
        nKernelActive, ...
        double(configChecksum), ...
        CAP_FILE_TAG);


% IMPORTANT:
%   Never concatenate MODEL_KERNEL_TAG with a literal underscore when the
%   model/kernel tag is empty.  Doing so would create a leading underscore
%   in RESULT_FILE_TAG and therefore a double underscore in output names,
%   e.g.:
%
%       Case2_local_stage2_checkpoints__HYBRID_N7
%
%   The branch below guarantees:
%
%       MODEL_KERNEL_TAG = ''
%           -> RESULT_FILE_TAG = TRANSFER_FILE_TAG
%
%       MODEL_KERNEL_TAG = 'E08_MG'
%           -> RESULT_FILE_TAG = 'E08_MG_<TRANSFER_FILE_TAG>'
%
%   Do not put a leading/trailing underscore in MODEL_KERNEL_TAG.
if isempty(strtrim(MODEL_KERNEL_TAG))

    RESULT_FILE_TAG = ...
        TRANSFER_FILE_TAG;

else

    RESULT_FILE_TAG = ...
        sprintf( ...
            '%s_%s', ...
            strtrim(MODEL_KERNEL_TAG), ...
            TRANSFER_FILE_TAG);

end


% Defensive validation: RESULT_FILE_TAG itself must not begin or end with
% an underscore.  This catches accidental user-entered separators before
% any previous Stage-2 output is deleted by FRESH_STAGE2.
if startsWith(RESULT_FILE_TAG,'_') || ...
   endsWith(RESULT_FILE_TAG,'_')

    error( ...
        ['RESULT_FILE_TAG must not begin or end with an underscore. ' ...
         'Check MODEL_KERNEL_TAG.']);

end


% =============================================================
% 12. Two-stage execution control
%
% RUN_MODE:
%   'stage1' : optimizeAtPointForAllCandidateRadii only
%   'stage2' : everything after Stage 1, including testing
%   'all'    : Stage 1 followed immediately by Stage 2
% =============================================================

RUN_MODE = ...
    'all'


FRESH_STAGE1 = ...
    false;


FRESH_STAGE2 = ...
    true;


% =============================================================
% 12.1 Stage-1 TRACE storage
%
% false -> save CORE + DIAG only (recommended production mode)
% true  -> also save TRACE_SEARCH + TRACE_OPT + TRACE_NUMERIC
% =============================================================

SAVE_TRACE = ...
    false;


% Save the optional compact Stage-1 batch summary MAT file.
SAVE_STAGE1_SUMMARY = ...
    false;


% Save optional FULL Stage-2 DETAIL in addition to compulsory CORE.
%
% false -> CORE only
% true  -> CORE + full DETAIL
SAVE_STAGE2_ALL = ...
    false;


% -------------------------------------------------------------
% Ensemble diagnostic storage when SAVE_STAGE2_ALL = false
%
% STORAGE ONLY: this option does not change prediction or diagnostics.
%
% false:
%   Minimum-memory CORE. Retain only canonical ensemble point-level maxima:
%
%       cond_raw       = max_j(cond_raw_j)
%       cond_reg       = max_j(cond_reg_j)
%       cond_effective = max_j(cond_effective_j)
%       rel_residual   = max_j(rel_residual_j)
%       backward_error = max_j(backward_error_j)
%
%   Max-tuple indices and compact unique tuple settings required for
%   reproducibility/fixed-setting reuse are still retained.
%
% true:
%   CORE additionally retains lightweight numerical diagnostics for every
%   unique tuple at every far-ensemble test point:
%
%       condition
%       solver_used
%       success
%       failureStage
%       message
%
%   It intentionally does NOT duplicate heavy neighbour indices/distances,
%   coefficients, or full neighbourhood provenance.
%
% If SAVE_STAGE2_ALL=true, full DETAIL already contains complete tuple
% diagnostics. The batch/testing code automatically suppresses this
% lightweight CORE-only copy to avoid duplication.
% -------------------------------------------------------------

SAVE_ENSEMBLE_POINT_DIAGNOSTICS = ...
    false;


% =============================================================
% 12.2 Outer Model x Kernel parallel execution
%
% One worker handles one Model x Kernel combination.
% Inner GA remains serial.
% =============================================================

USE_PARALLEL = ...
    true;


NUM_WORKERS = ...
2;


% For full production, stop if the requested process pool cannot start
% rather than silently continuing as a much slower serial run.
ALLOW_SERIAL_FALLBACK = ...
    false;


% Restart an existing pool once after +localrbf code changes.
% The batch function creates the requested Processes pool when needed.
RESTART_PARALLEL_POOL = ...
    true;


% =============================================================
% 13. Output/checkpoint locations
%
% OUTPUT-NAME RULE:
%   RESULT_FILE_TAG is already separator-safe.
%
%   If MODEL_KERNEL_TAG = '':
%       Stage-2 folder becomes
%           <Case>_local_stage2_checkpoints_<TRANSFER_FILE_TAG>
%
%       NOT
%           <Case>_local_stage2_checkpoints__<TRANSFER_FILE_TAG>
%
%   No automatic timestamp/run tail is added here.  This preserves the
%   existing resume/restart behavior and avoids creating a new checkpoint
%   folder every time the same scientific configuration is rerun.
%
% FRESH_STAGE2=true still intentionally deletes only the Stage-2 file/folder
% addressed by FinalOutputFile and Stage2CheckpointDir below. Therefore a
% different result identity must have a different RESULT_FILE_TAG if both
% Stage-2 runs need to coexist.
% =============================================================




if ~isfolder(OutputDir)
    mkdir(OutputDir);
end


Stage1OutputFile = ...
    fullfile( ...
        OutputDir, ...
        sprintf( ...
            'LocalRBF_%s_Stage1_OptimizeAllRadii_%s.mat', ...
            CaseName, ...
            RESULT_FILE_TAG));


FinalOutputFile = ...
    fullfile( ...
        OutputDir, ...
        sprintf( ...
            'LocalRBF_%s_Full_TuneReg_IDW.mat', ...
            CaseName));


Stage1CheckpointDir = ...
    fullfile( ...
        OutputDir, ...
        [CaseName '_local_stage1_checkpoints']);


Stage2CheckpointDir = ...
    fullfile( ...
        OutputDir, ...
        sprintf( ...
            '%s_local_stage2_checkpoints_%s', ...
            CaseName, ...
            RESULT_FILE_TAG));


% =============================================================
% 14. Validate active scientific/storage configuration
% =============================================================

expectedToleranceSize = ...
    [numel(ListOfModel), numel(NameFunction_RBF)];


if ~isequal(size(tau_loc),expectedToleranceSize)

    error( ...
        ['tau_loc must have size %d x %d for the active ' ...
         'ListOfModel and NameFunction_RBF.'], ...
        expectedToleranceSize(1), ...
        expectedToleranceSize(2));

end


if ~isequal(size(tau_cap),expectedToleranceSize)

    error( ...
        ['tau_cap must have size %d x %d for the active ' ...
         'ListOfModel and NameFunction_RBF.'], ...
        expectedToleranceSize(1), ...
        expectedToleranceSize(2));

end


if ~isequal(size(TRANSFER_METHOD),expectedToleranceSize)

    error( ...
        ['TRANSFER_METHOD must have size %d x %d for the active ' ...
         'GGM x Kernel configuration.'], ...
        expectedToleranceSize(1),expectedToleranceSize(2));

end


if ~isequal(size(PARAMETER_TRANSFER_MODE),expectedToleranceSize)

    error( ...
        ['PARAMETER_TRANSFER_MODE must have size %d x %d for the active ' ...
         'GGM x Kernel configuration.'], ...
        expectedToleranceSize(1),expectedToleranceSize(2));

end


if ~isequal(size(N_IDW_SOURCES),expectedToleranceSize)

    error( ...
        ['N_IDW_SOURCES must have size %d x %d for the active ' ...
         'GGM x Kernel configuration.'], ...
        expectedToleranceSize(1),expectedToleranceSize(2));

end


if ~isequal(size(near_threshold),expectedToleranceSize)

    error( ...
        ['near_threshold must have size %d x %d for the active ' ...
         'GGM x Kernel configuration.'], ...
        expectedToleranceSize(1),expectedToleranceSize(2));

end


if SAVE_STAGE2_ALL && ...
   SAVE_ENSEMBLE_POINT_DIAGNOSTICS

    fprintf([ ...
        'NOTE: SAVE_STAGE2_ALL=true already stores full ensemble DETAIL.\n' ...
        '      The lightweight ensemble diagnostic copy is suppressed.\n']);

end


% =============================================================
% 14.1 Validate run mode
% =============================================================

RUN_MODE = ...
    lower(char(RUN_MODE));


if ~any(strcmp(RUN_MODE,{'stage1','stage2','all'}))
    error('RUN_MODE must be ''stage1'', ''stage2'', or ''all''.');
end


% =============================================================
% 15. Fresh-run cleanup
%
% Stage-2 cleanup NEVER removes Stage-1 optimization checkpoints.
% =============================================================

if any(strcmp(RUN_MODE,{'stage1','all'})) && ...
   FRESH_STAGE1

    if isfile(Stage1OutputFile)
        delete(Stage1OutputFile);
    end

    if isfolder(Stage1CheckpointDir)
        rmdir(Stage1CheckpointDir,'s');
    end

end


if any(strcmp(RUN_MODE,{'stage2','all'})) && ...
   FRESH_STAGE2

    if isfile(FinalOutputFile)
        delete(FinalOutputFile);
    end

    % Remove only Stage-2 split checkpoints. Stage-1 PointOpt
    % checkpoints are preserved.
    if isfolder(Stage2CheckpointDir)
        rmdir(Stage2CheckpointDir,'s');
    end

end


% =============================================================
% 16. Shared execution options
% =============================================================

if ~isfield(options,'execution') || ...
   ~isstruct(options.execution)

    options.execution = ...
        struct();

end


options.execution.stage1CheckpointDir = ...
    Stage1CheckpointDir;


options.execution.stage2CheckpointDir = ...
    Stage2CheckpointDir;


% Reuse completed Stage-1 model x kernel checkpoints by default.
% FRESH_STAGE1=true is the explicit way to force recomputation.
options.execution.reuseCompletedStage1 = ...
    true;


options.execution.saveTrace = ...
    SAVE_TRACE;


options.execution.saveStage1Summary = ...
    SAVE_STAGE1_SUMMARY;


options.execution.saveStage2All = ...
    SAVE_STAGE2_ALL;


options.execution.saveEnsemblePointDiagnostics = ...
    SAVE_ENSEMBLE_POINT_DIAGNOSTICS;


options.execution.useParallel = ...
    USE_PARALLEL;


options.execution.numWorkers = ...
    NUM_WORKERS;


options.execution.allowSerialFallback = ...
    ALLOW_SERIAL_FALLBACK;


if RESTART_PARALLEL_POOL

    existingPool = ...
        gcp('nocreate');


    if ~isempty(existingPool)

        delete(existingPool);

    end

end


% =============================================================
% 16.1 Runtime and computational-cost recording
%
% Runtime is already measured inside the revised batch/training/testing
% pipeline for every Model x Kernel run. No extra RBF computation is needed.
%
% Training side:
%   trainWallTime
%       Wall time of LocalRBFTraining for that run.
%
%   trainCPUTime
%       CPU time of LocalRBFTraining.
%
%   pseudoTrainingWallTime
%       Time used specifically by strict pseudo-training RMSE calculation.
%       When CAP_RMSE_SOURCE='pseudo', this is a SUBSET of trainWallTime.
%       Do NOT add it again to trainWallTime.
%
%   trainSearchRBFSolves
%       Search/Stage-1 solve count where applicable.
%
%   trainPseudoRBFSolves
%       Strict pseudo-training solve count.
%
%   trainDiagnosticRBFSolves
%       Final selected-system diagnostic/evaluation solve count.
%
%   trainRBFSolves
%       Total training-side RBF solves already including the relevant
%       search/pseudo/diagnostic contributions.
%
% Testing side:
%   testWallTime
%   testCPUTime
%   testRBFSolves
%
% Total for one batch call:
%   totalWallTime  = trainWallTime + testWallTime
%   totalCPUTime   = trainCPUTime  + testCPUTime
%   totalRBFSolves = trainRBFSolves + testRBFSolves
%
% TWO-STAGE IMPORTANT NOTE:
%   This production runner launches Stage 1 and Stage 2 as separate batch
%   calls when RUN_MODE='all'. Therefore:
%
%       ResultsStage1
%           preserves Stage-1 optimization runtime/cost.
%
%       Results
%           preserves Stage-2 cap-selection/evaluation/pseudo/testing
%           runtime/cost.
%
%   Stage-1 time is intentionally NOT silently added to Stage-2
%   Results.totalWallTime. This prevents double counting and keeps the
%   provenance of optimization versus final evaluation/testing explicit.
%
% These timings are MATLAB runtime outputs produced when YOU execute this
% script in MATLAB. They are independent of whether an external code-review
% environment can execute MATLAB.
% =============================================================


% =============================================================
% 17. STAGE 1
%
% Only optimizeAtPointForAllCandidateRadii is executed.
% No cap selection, selected-system evaluation, transfer, or testing.
% =============================================================

if any(strcmp(RUN_MODE,{'stage1','all'}))

    fprintf('\n=============================================\n');
    fprintf(' LOCAL RBF PRODUCTION - STAGE 1\n');
    fprintf(' OPTIMIZE ALL CANDIDATE RADII\n');
    fprintf('=============================================\n');


    optionsStage1 = ...
        options;


    optionsStage1.execution.stage = ...
        'stage1';


    ResultsStage1 = ...
        localrbf.localRBFBatchSummaryParallel_TuneReg( ...
            InputData, ...
            CaseName, ...
            ListOfModel, ...
            NameFunction_RBF, ...
            Radii, ...
            min_neighbors, ...
            CMin, ...
            CMax, ...
            RegMin, ...
            RegMax, ...
            tau_loc, ...
            tau_cap, ...
            near_threshold, ...
            idw_radius, ...
            verbose, ...
            RNG_SEED, ...
            optionsStage1, ...
            Stage1OutputFile, ...
            Method, ...
            nGrid);


    
    fprintf('\n=============================================\n');
    fprintf(' LOCAL RBF STAGE 1 FINISHED\n');
    fprintf('=============================================\n');

end


% =============================================================
% 18. STAGE 2
%
% Loads the exact Stage-1 PointOpt results and continues from
% selectBestRadiusWithToleranceAtEachRadiusCapOverAllCaps.
% No optimizeAtPointForAllCandidateRadii call occurs in Stage 2.
% =============================================================

if any(strcmp(RUN_MODE,{'stage2','all'}))

    if ~isfolder(Stage1CheckpointDir)
        error(['Stage-1 checkpoint directory was not found:\n%s\n' ...
               'Run RUN_MODE = ''stage1'' first.'], ...
              Stage1CheckpointDir);
    end


    fprintf('\n=============================================\n');
    fprintf(' LOCAL RBF PRODUCTION - STAGE 2\n');
    fprintf(' CAP SELECTION + EVALUATION + TESTING\n');
    fprintf(' Cap RMSE source : %s\n',CAP_RMSE_SOURCE);
    fprintf(' Transfer config : GGM x Kernel matrices (%d x %d)\n', ...
        numel(ListOfModel),numel(NameFunction_RBF));
    fprintf('=============================================\n');


    optionsStage2 = ...
        options;


    optionsStage2.execution.stage = ...
        'stage2';


    Results = ...
        localrbf.localRBFBatchSummaryParallel_TuneReg( ...
            InputData, ...
            CaseName, ...
            ListOfModel, ...
            NameFunction_RBF, ...
            Radii, ...
            min_neighbors, ...
            CMin, ...
            CMax, ...
            RegMin, ...
            RegMax, ...
            tau_loc, ...
            tau_cap, ...
            near_threshold, ...
            idw_radius, ...
            verbose, ...
            RNG_SEED, ...
            optionsStage2, ...
            FinalOutputFile, ...
            Method, ...
            nGrid);


    fprintf('\n=============================================\n');
    fprintf(' LOCAL RBF STAGE 2 FINISHED\n');
    fprintf('=============================================\n');

end

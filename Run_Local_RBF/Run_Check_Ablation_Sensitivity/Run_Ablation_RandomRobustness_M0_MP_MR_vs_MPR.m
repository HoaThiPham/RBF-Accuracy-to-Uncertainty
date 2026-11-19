function Outputs = Run_Ablation_RandomRobustness_M0_MP_MR_vs_MPR()
%RUN_ABLATION_RANDOMROBUSTNESS_M0_MP_MR_VS_MPR
%
% ONE master runner for the final Local-RBF ablation + random-reference
% robustness experiment.
%
% MODE DEFINITIONS
% ----------------
%
% Manuscript notation:
%   M_0  = both neighbourhood and kernel parameters fixed      [legacy M11]
%   M_P  = adaptive kernel parameters only                     [legacy M8]
%   M_R  = adaptive neighbourhood/radius only                  [legacy M12]
%   M_PR = both neighbourhood and kernel parameters adaptive   [Model 1]

% M_0:
%   fixed random R0 + fixed random C/RegParam
%   (both neighbourhood and parameters fixed)
%
% M_P:
%   same fixed random R0 + point-specific parameter re-optimization at R0
%   (adaptive parameters only)
%   using the same C/RegParam domains as random fixed-reference generation
%
% M_R:
%   adaptive radius/cap + same fixed random C/RegParam as M_0
%   (adaptive neighbourhood only)
%
% M_PR:
%   authoritative completed adaptive local-RBF model; NEVER rerun here
%   (both neighbourhood and parameters adaptive)
%
% RANDOM REFERENCE
% ----------------
% Across the five pre-specified realizations:
%   1) fixed radii use a STRATIFIED-RANDOM design over the complete
%      {50,60,...,200} km grid. The five non-overlapping strata are
%      {50,60,70}, {80,90,100}, {110,120,130}, {140,150,160}, and
%      {170,180,190,200} km. One radius is drawn randomly within each
%      stratum using that realization's pre-specified seed;
%   2) RegParam values are sampled independently log-uniformly from the
%      GLOBAL-RBF input range [1e-6, 1] for every kernel;
%   3) Gaussian / Multiquadric C values are sampled independently
%      log-uniformly from ONE COMMON sensitivity range spanning the two
%      M_PR local investigated C ranges.
%
% The M_PR kernel-specific LOCAL RegParam ranges are retained as provenance
% only. They are not used as random-reference RegParam bounds. The random
% RegParam robustness reference deliberately uses the broader GLOBAL-RBF
% input range, whereas the C reference continues to use the common envelope
% spanning the investigated local Gaussian/MQ C ranges.
%
% All five reference realizations have EQUAL scientific status; there is no
% privileged primary reference. Strata are assigned deterministically in
% seed-list order only to guarantee complete radius-domain coverage, while
% the exact radius within each stratum is random. No train/pseudo/test RMSE
% is used to generate, resolve, rank, or choose any reference realization.
%
% IMPORTANT SCIENTIFIC CONSISTENCY
% --------------------------------
%   * The SAME realization is used in both Case 1 and Case 2.
%   * MP and M0 use the SAME fixed R0 for a given realization.
%   * MP tuning uses the SAME C/RegParam domains used to generate M0/MR
%     random fixed-parameter references.
%   * M0 and MR use the SAME fixed C/RegParam for a given realization.
%   * Test errors are NEVER used for radius/parameter selection.
%   * Comparison against M_PR uses independent TEST RMSE only.
%
% COMPARISON CONVENTIONS
% ----------------------
% To remain consistent with the user's existing comparison exporter:
%
% Alternative_Improvement_vs_Model1_pct
%   = 100 * (RMSE_Model1 - RMSE_Alternative) / RMSE_Model1
%
%   positive => alternative lower RMSE than M_PR
%   negative => M_PR lower RMSE
%
% A second reviewer-facing quantity is also reported:
%
% Model1_Reduction_vs_Alternative_pct
%   = 100 * (RMSE_Alternative - RMSE_Model1) / RMSE_Alternative
%
%   positive => M_PR lower RMSE
%
% OUTPUTS
% -------
% Full per-model MAT files are saved by the three case runners.
% For every reference seed, M0, MP, and MR are stored together under
% Seed_<seed>\M0, Seed_<seed>\MP, and Seed_<seed>\MR, respectively.
% This master additionally writes:
%   RandomReference_Configurations.xlsx
%   Ablation_vs_MPR_RandomRobustness.xlsx
%   Ablation_vs_MPR_RandomRobustness.mat
%   Ablation_vs_MPR_AllReferences.tex
%   Ablation_vs_MPR_Robustness.tex
%
% The Excel workbook contains:
%   ReferenceConfigurations
%   AllComparisons
%   SeedGroupSummary
%   RobustnessByGGMKernel
%   RobustnessStability
%   RobustnessGroupSummary
%
% ========================================================================

clc

%% ========================================================================
% 1. USER SETTINGS -- EDIT ONLY THIS SECTION
% ========================================================================

RootDir = '.';

% Process workers. Four is the recommended starting point for the revised
% memory-safe implementation.
NUM_WORKERS = 4;

% One-time maintenance option for MATLAB's Processes profile.
% Keep FALSE normally so no old job/crash information is deleted.
% If parpool startup is blocked by preserved crash-dump jobs, set TRUE for
% ONE run, then return it to FALSE.
CLEAN_STALE_PROCESS_JOBS = false;

% Start/reuse one shared process pool in the master before Models 8/11/12.
START_SHARED_PROCESS_POOL = true;

% Five pre-specified seeds with equal scientific status. Seed-list order
% assigns the five radius strata; no seed is a primary/reference winner.
REFERENCE_SEEDS = [12345 23456 34567 45678 56789];

% Fixed-reference RegParam sampling range. This MUST match the input range
% used by the GLOBAL RBF optimization. RegParam is sampled log-uniformly
% from this same range for all five kernels in M0/MR.
GLOBAL_REGPARAM_INPUT_RANGE = [1e-6 1];

% Reuse completed per-model results only when reference provenance and
% the model-specific schema match. The RegParam-range revision increments
% the M0/MR schemas, so stale legacy M11/M12 files are not reused.
REUSE_EXISTING_RESULTS = true;

% Save large all-radius pointwise matrices from M_R?
% false preserves all final scientific outputs, selected-radius arrays,
% cap/pseudo-unseen curves, and final test outputs while avoiding large
% redundant all-radius pointwise matrices in memory/files.
STORE_MR_POINTWISE_INTERMEDIATES = false;

% Strict comparison tie threshold in percentage points.
TIE_TOL_PCT = 1e-10;

%% ========================================================================
% 2. VALIDATE USER SETTINGS / PATHS
% ========================================================================

assert(isfolder(RootDir),'RootDir does not exist:\n%s',RootDir);
assert(isnumeric(NUM_WORKERS) && isscalar(NUM_WORKERS) && isfinite(NUM_WORKERS) && ...
       NUM_WORKERS>=1 && NUM_WORKERS==round(NUM_WORKERS), ...
    'NUM_WORKERS must be a positive integer.');

REFERENCE_SEEDS = double(REFERENCE_SEEDS(:).');
assert(~isempty(REFERENCE_SEEDS) && all(isfinite(REFERENCE_SEEDS)) && ...
       all(REFERENCE_SEEDS>=0) && all(REFERENCE_SEEDS==round(REFERENCE_SEEDS)), ...
    'REFERENCE_SEEDS must contain nonnegative integers.');
assert(numel(unique(REFERENCE_SEEDS))==numel(REFERENCE_SEEDS), ...
    'REFERENCE_SEEDS must be unique.');
assert(numel(REFERENCE_SEEDS)==5, ...
    'The stratified-radius design requires exactly five reference seeds.');
CLEAN_STALE_PROCESS_JOBS = logical(CLEAN_STALE_PROCESS_JOBS);
START_SHARED_PROCESS_POOL = logical(START_SHARED_PROCESS_POOL);
GLOBAL_REGPARAM_INPUT_RANGE = double(GLOBAL_REGPARAM_INPUT_RANGE(:).');
assert(numel(GLOBAL_REGPARAM_INPUT_RANGE)==2 && ...
       all(isfinite(GLOBAL_REGPARAM_INPUT_RANGE)) && ...
       GLOBAL_REGPARAM_INPUT_RANGE(1)>0 && ...
       GLOBAL_REGPARAM_INPUT_RANGE(2)>=GLOBAL_REGPARAM_INPUT_RANGE(1), ...
    'GLOBAL_REGPARAM_INPUT_RANGE must be [positiveMin positiveMax].');

thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir), addpath(thisDir); end

requiredFunctions = { ...
    'Run_Ablation_MP_Case', ...
    'Run_Ablation_M0_Case', ...
    'Run_Ablation_MR_Case'};
for k = 1:numel(requiredFunctions)
    assert(exist(requiredFunctions{k},'file')==2, ...
        'Required ablation function is not on the MATLAB path: %s',requiredFunctions{k});
end

OUTPUT_ROOT = fullfile(RootDir,'OutPut_RBF_2026_2', ...
    'Sensitivity_Ablation_RandomRobustness');
if ~isfolder(OUTPUT_ROOT), mkdir(OUTPUT_ROOT); end

OUTPUT_XLSX = fullfile(OUTPUT_ROOT,'Ablation_vs_MPR_RandomRobustness.xlsx');
OUTPUT_MAT = fullfile(OUTPUT_ROOT,'Ablation_vs_MPR_RandomRobustness.mat');
OUTPUT_REF_XLSX = fullfile(OUTPUT_ROOT,'RandomReference_Configurations.xlsx');
OUTPUT_TEX_ALL = fullfile(OUTPUT_ROOT,'Ablation_vs_MPR_AllReferences.tex');
OUTPUT_TEX_ROBUST = fullfile(OUTPUT_ROOT,'Ablation_vs_MPR_Robustness.tex');

%% ========================================================================
% 3. LOAD + VALIDATE AUTHORITATIVE MODEL-1 SETTINGS FOR BOTH CASES
% ========================================================================

Model1File = cell(2,1);
Model1Settings = cell(2,1);
Model1Kernels = cell(2,1);

for c = 1:2
    caseToken = sprintf('Case%d',c);
    Model1File{c} = resolveMainResultsFileMaster(RootDir,caseToken);
    A = load(Model1File{c},'Results');
    assert(isfield(A,'Results') && isstruct(A.Results), ...
        'M_PR file does not contain Results:\n%s',Model1File{c});
    assert(isfield(A.Results,'Settings') && isstruct(A.Results.Settings), ...
        'M_PR Results.Settings missing:\n%s',Model1File{c});
    Model1Settings{c} = A.Results.Settings;
    Model1Kernels{c} = cellstr(string(A.Results.NameFunction_RBF(:)));
end

[ReferenceRanges,RadiusGridRaw] = validateAndExtractSharedReferenceRanges( ...
    Model1Settings,Model1Kernels,GLOBAL_REGPARAM_INPUT_RANGE);

assert(isequal(RadiusGridRaw(:),(50000:10000:200000).'), ...
    'Shared M_PR local radius grid must be exactly 50:10:200 km.');
RadiusStrataRaw = buildRadiusStrataMaster(RadiusGridRaw);

%% ========================================================================
% 4. GENERATE ALL STRATIFIED-RANDOM REFERENCES BEFORE ANY MODEL RUN
% ========================================================================

References = generateRandomReferences( ...
    REFERENCE_SEEDS,ReferenceRanges,RadiusGridRaw,RadiusStrataRaw);

ReferenceConfigurations = referencesToTable(References);

if isfile(OUTPUT_REF_XLSX), delete(OUTPUT_REF_XLSX); end
writetable(ReferenceConfigurations,OUTPUT_REF_XLSX, ...
    'Sheet','ReferenceConfigurations','WriteMode','overwritesheet');

ReferenceFile = fullfile(OUTPUT_ROOT,'RandomReference_Configurations.mat');
saveAtomicVariablesMaster(ReferenceFile,struct( ...
    'References',References, ...
    'ReferenceConfigurations',ReferenceConfigurations, ...
    'ReferenceRanges',ReferenceRanges, ...
    'RadiusGridRaw',RadiusGridRaw, ...
    'RadiusStrataRaw',{RadiusStrataRaw}));

fprintf('\n=============================================================\n');
fprintf(' STRATIFIED-RANDOM REFERENCES GENERATED BEFORE MODEL RUNS\n');
fprintf(' Number of equal-status realizations = %d\n',numel(References));
fprintf(' Radius sampling design = one random draw per non-overlapping stratum\n');
fprintf(' Fixed radii by seed    = %s km\n',mat2str([References.FixedRadiusKm]));
fprintf(' Global-input Reg range = [%.12g, %.12g]\n', ...
    ReferenceRanges.ReferenceRegMin,ReferenceRanges.ReferenceRegMax);
fprintf(' Common C range       = [%.12g, %.12g]\n', ...
    ReferenceRanges.CommonCMin,ReferenceRanges.CommonCMax);
fprintf(' No predictive error used in generation.\n');
fprintf('=============================================================\n');
disp(ReferenceConfigurations);

%% ========================================================================
% 5. LOAD AUTHORITATIVE MODEL-1 LOCAL PSEUDO + INDEPENDENT TEST RMSE
% ========================================================================

TModel1 = [ ...
    readModel1FullTuneRegMaster(Model1File{1}); ...
    readModel1FullTuneRegMaster(Model1File{2})];

assert(height(TModel1)==60, ...
    'Expected 60 M_PR rows (2 cases x 6 GGMs x 5 kernels); found %d.', ...
    height(TModel1));
assertNoDuplicateKeysMaster(TModel1,{'Case','GGM','Kernel'},'M_PR');
assert(all(isfinite(TModel1.Model1_Test)) && all(TModel1.Model1_Test>0), ...
    'M_PR independent-test RMSE contains invalid values.');

%% ========================================================================
% 6. SHARED MODEL OPTIONS
% ========================================================================

MPOptions = struct();
MPOptions.ReuseExistingResults = REUSE_EXISTING_RESULTS;
MPOptions.UseParallel = true;
MPOptions.AllowSerialFallback = false;
% MP must tune over exactly the same parameter domains from which the
% random fixed references are generated.
MPOptions.TuningCMin = ReferenceRanges.CommonCMin;
MPOptions.TuningCMax = ReferenceRanges.CommonCMax;
MPOptions.TuningRegMin = ReferenceRanges.ReferenceRegMin;
MPOptions.TuningRegMax = ReferenceRanges.ReferenceRegMax;

M0Options = struct();
M0Options.ReuseExistingResults = REUSE_EXISTING_RESULTS;
M0Options.UseParallel = true;
M0Options.AllowSerialFallback = false;

MROptions = struct();
MROptions.ReuseExistingResults = REUSE_EXISTING_RESULTS;
MROptions.UseParallel = true;
MROptions.AllowSerialFallback = false;
MROptions.StorePointwise = STORE_MR_POINTWISE_INTERMEDIATES;

%% ========================================================================
% 7. START/REUSE ONE SHARED PROCESS POOL
% ========================================================================

if CLEAN_STALE_PROCESS_JOBS
    oldPool = gcp('nocreate');
    if ~isempty(oldPool)
        delete(oldPool);
    end
    processCluster = parcluster('Processes');
    oldJobs = processCluster.Jobs;
    if ~isempty(oldJobs)
        fprintf('\nDeleting %d old job(s) from Processes profile ...\n',numel(oldJobs));
        delete(oldJobs);
    end
end

if START_SHARED_PROCESS_POOL
    pool = gcp('nocreate');
    if ~isempty(pool) && ...
       (~isa(pool,'parallel.ProcessPool') || pool.NumWorkers~=NUM_WORKERS)
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        fprintf('\nStarting ONE shared Processes pool with %d workers ...\n',NUM_WORKERS);
        pool = parpool('Processes',NUM_WORKERS);
    else
        fprintf('\nReusing existing Processes pool with %d workers.\n',pool.NumWorkers);
    end
end

%% ========================================================================
% 8. RUN ALL RANDOM REFERENCES
% ========================================================================

AllComparisons = table();

% In-memory MP memoization: M_P depends on R0 but not on the random
% fixed C/RegParam values. The non-overlapping strata guarantee distinct R0
% values, but memoization is retained as a defensive runtime safeguard.
MPMemo = struct();

for s = 1:numel(References)
    Ref = References(s);

    fprintf('\n\n#############################################################\n');
    fprintf(' RANDOM REFERENCE %d/%d\n',s,numel(References));
    fprintf(' Seed = %d | Stratum %d = [%.0f, %.0f] km | R0 = %.0f km\n', ...
        Ref.Seed,Ref.RadiusStratumIndex,Ref.RadiusStratumMinKm, ...
        Ref.RadiusStratumMaxKm,Ref.FixedRadiusRaw/1000);
    fprintf('#############################################################\n');

    for c = 1:2
        caseToken = sprintf('Case%d',c);

        % ---------------------------------------------------------------
        % M_P
        % ---------------------------------------------------------------
        memoField = matlab.lang.makeValidName(sprintf('Case%d_R%.0f', ...
            c,Ref.FixedRadiusRaw));
        if isfield(MPMemo,memoField)
            S_MP = MPMemo.(memoField);
            fprintf('\nM_P %s R=%.0f km: reusing in-memory summary.\n', ...
                caseToken,Ref.FixedRadiusRaw/1000);
        else
            MPOptionsThis = MPOptions;
            MPOptionsThis.ReferenceSeed = Ref.Seed;
            R_MP = Run_Ablation_MP_Case( ...
                c,RootDir,NUM_WORKERS,Ref.FixedRadiusRaw,MPOptionsThis);
            S_MP = validateAblationSummary(R_MP.Summary,"MP",caseToken);
            MPMemo.(memoField) = S_MP;
            clear R_MP
        end

        % ---------------------------------------------------------------
        % M_0
        % ---------------------------------------------------------------
        R_M0 = Run_Ablation_M0_Case( ...
            c,RootDir,NUM_WORKERS,Ref,M0Options);
        S_M0 = validateAblationSummary(R_M0.Summary,"M0",caseToken);
        clear R_M0

        % ---------------------------------------------------------------
        % M_R
        % ---------------------------------------------------------------
        R_MR = Run_Ablation_MR_Case( ...
            c,RootDir,NUM_WORKERS,Ref,MROptions);
        S_MR = validateAblationSummary(R_MR.Summary,"MR",caseToken);
        clear R_MR

        % ---------------------------------------------------------------
        % COMPARE ONLY AGAINST M_PR, USING INDEPENDENT TEST RMSE
        % ---------------------------------------------------------------
        M1Case = TModel1(TModel1.Case==string(caseToken),:);
        assert(height(M1Case)==30,'Expected 30 M_PR rows for %s.',caseToken);

        AllComparisons = [AllComparisons; ...
            buildComparisonRowsMaster(M1Case,S_MP,Ref,"MP",TIE_TOL_PCT); ...
            buildComparisonRowsMaster(M1Case,S_M0,Ref,"M0",TIE_TOL_PCT); ...
            buildComparisonRowsMaster(M1Case,S_MR,Ref,"MR",TIE_TOL_PCT)]; %#ok<AGROW>
    end
end

assert(height(AllComparisons)==numel(References)*2*3*30, ...
    'Unexpected total comparison-row count.');
assert(all(isfinite(AllComparisons.Model1_TestRMSE)) && ...
       all(isfinite(AllComparisons.Alternative_TestRMSE)), ...
    'Comparison table contains nonfinite independent-test RMSE.');

AllComparisons = sortAllComparisonsMaster(AllComparisons);

%% ========================================================================
% 8. BUILD EQUAL-STATUS REFERENCE + ROBUSTNESS SUMMARIES
% ========================================================================

SeedGroupSummary = summarizeByGroupMaster( ...
    AllComparisons,{'ReferenceSeed','Case','Alternative','KernelGroup'},TIE_TOL_PCT);

RobustnessByGGMKernel = summarizeByGroupMaster( ...
    AllComparisons,{'Case','Alternative','GGM','Kernel'},TIE_TOL_PCT);
RobustnessByGGMKernel.KernelGroup = classifyKernelGroupMaster(RobustnessByGGMKernel.Kernel);

% Reviewer-facing stability: each GGM x kernel is one unit and asks whether
% $M_{PR}$ wins across ALL pre-specified random realizations.
RobustnessStabilitySummary = buildStabilitySummaryMaster( ...
    RobustnessByGGMKernel,numel(References));

RobustnessGroupSummary = summarizeByGroupMaster( ...
    AllComparisons,{'Case','Alternative','KernelGroup'},TIE_TOL_PCT);

% Keep shape-parameter and shape-free groups explicit in manuscript-facing
% summaries. Do not pool them into an "All kernels" median because their
% effect scales differ materially.

%% ========================================================================
% 9. FINAL CONSISTENCY AUDIT
% ========================================================================

auditCrossModelReferenceConsistency(AllComparisons,References);
auditNoTestSelection(References);

fprintf('\n=============================================================\n');
fprintf(' GROUP SUMMARY ACROSS ALL FIVE EQUAL-STATUS REFERENCES\n');
fprintf('=============================================================\n');
disp(RobustnessGroupSummary);

fprintf('\n=============================================================\n');
fprintf(' ROBUSTNESS STABILITY BY GGM x KERNEL ACROSS ALL SEEDS\n');
fprintf('=============================================================\n');
disp(RobustnessStabilitySummary);

%% ========================================================================
% 10. WRITE EXCEL / MAT / LATEX
% ========================================================================

if isfile(OUTPUT_XLSX), delete(OUTPUT_XLSX); end

writetable(ReferenceConfigurations,OUTPUT_XLSX, ...
    'Sheet','ReferenceConfigurations','WriteMode','overwritesheet');
writetable(AllComparisons,OUTPUT_XLSX, ...
    'Sheet','AllComparisons','WriteMode','overwritesheet');
writetable(SeedGroupSummary,OUTPUT_XLSX, ...
    'Sheet','SeedGroupSummary','WriteMode','overwritesheet');
writetable(RobustnessByGGMKernel,OUTPUT_XLSX, ...
    'Sheet','RobustnessByGGMKernel','WriteMode','overwritesheet');
writetable(RobustnessStabilitySummary,OUTPUT_XLSX, ...
    'Sheet','RobustnessStability','WriteMode','overwritesheet');
writetable(RobustnessGroupSummary,OUTPUT_XLSX, ...
    'Sheet','RobustnessGroupSummary','WriteMode','overwritesheet');

saveAtomicVariablesMaster(OUTPUT_MAT,struct( ...
    'References',References, ...
    'ReferenceConfigurations',ReferenceConfigurations, ...
    'RadiusStrataRaw',{RadiusStrataRaw}, ...
    'Model1',TModel1, ...
    'AllComparisons',AllComparisons, ...
    'SeedGroupSummary',SeedGroupSummary, ...
    'RobustnessByGGMKernel',RobustnessByGGMKernel, ...
    'RobustnessStabilitySummary',RobustnessStabilitySummary, ...
    'RobustnessGroupSummary',RobustnessGroupSummary));

writeAllReferencesLatexMaster(RobustnessGroupSummary,numel(References),OUTPUT_TEX_ALL);
writeRobustnessLatexMaster(RobustnessStabilitySummary,numel(References),OUTPUT_TEX_ROBUST);

assert(isfile(OUTPUT_XLSX),'Excel output was not created.');
assert(isfile(OUTPUT_MAT),'MAT output was not created.');
assert(isfile(OUTPUT_TEX_ALL),'All-reference LaTeX output was not created.');
assert(isfile(OUTPUT_TEX_ROBUST),'Robustness LaTeX output was not created.');

Outputs = struct();
Outputs.ReferenceConfigurations = ReferenceConfigurations;
Outputs.RadiusStrataRaw = RadiusStrataRaw;
Outputs.AllComparisons = AllComparisons;
Outputs.SeedGroupSummary = SeedGroupSummary;
Outputs.RobustnessByGGMKernel = RobustnessByGGMKernel;
Outputs.RobustnessStabilitySummary = RobustnessStabilitySummary;
Outputs.RobustnessGroupSummary = RobustnessGroupSummary;
Outputs.ExcelFile = OUTPUT_XLSX;
Outputs.MatFile = OUTPUT_MAT;
Outputs.AllReferencesLatexFile = OUTPUT_TEX_ALL;
Outputs.RobustnessLatexFile = OUTPUT_TEX_ROBUST;

fprintf('\n=============================================================\n');
fprintf(' RANDOM-REFERENCE ABLATION COMPLETED\n');
fprintf('=============================================================\n');
fprintf('Excel       : %s\n',OUTPUT_XLSX);
fprintf('MAT         : %s\n',OUTPUT_MAT);
fprintf('All-ref TeX : %s\n',OUTPUT_TEX_ALL);
fprintf('Robust TeX  : %s\n',OUTPUT_TEX_ROBUST);
fprintf('=============================================================\n');

end

%% ========================================================================
function [Ranges,RadiusGridRaw] = validateAndExtractSharedReferenceRanges(SCase,KernelCase,globalRegRange)
%VALIDATEANDEXTRACTSHAREDREFERENCERANGES
%
% Read the actual kernel-specific LOCAL search/investigation ranges from
% M_PR and verify that they are identical between Cases 1 and 2. These
% local ranges are retained as provenance.
%
% RANDOM-REFERENCE SAMPLING:
%   RegParam -> the supplied GLOBAL-RBF input range (same for all kernels)
%   C        -> one common envelope spanning the local Gaussian/MQ C ranges
%
% The global RegParam input range is deliberately independent of the narrower
% local kernel-specific M_PR RegParam search windows.

REG_ORDER={'gaussian','multiquadric','linear','cubic','thinplate'};
C_ORDER={'gaussian','multiquadric'};
requiredR=(50000:10000:200000).';

Rng=cell(2,1);
for c=1:2
    S=SCase{c}; kernels=KernelCase{c}; nK=numel(kernels);
    CMin=getSettingMaster(S,{'CMin'});
    CMax=getSettingMaster(S,{'CMax'});
    RegMin=getSettingMaster(S,{'RegMin'});
    RegMax=getSettingMaster(S,{'RegMax'});
    Radii=getSettingMaster(S,{'Radii'});

    Q=struct();
    Q.RegMin=nan(1,5); Q.RegMax=nan(1,5);
    for k=1:5
        Q.RegMin(k)=extractKernelBoundMaster(RegMin,REG_ORDER{k},kernels,nK,'RegMin');
        Q.RegMax(k)=extractKernelBoundMaster(RegMax,REG_ORDER{k},kernels,nK,'RegMax');
        assert(Q.RegMin(k)>0 && Q.RegMax(k)>=Q.RegMin(k), ...
            'Invalid Reg range for %s Case %d.',REG_ORDER{k},c);
    end

    Q.CMin=nan(1,2); Q.CMax=nan(1,2);
    for k=1:2
        Q.CMin(k)=extractKernelBoundMaster(CMin,C_ORDER{k},kernels,nK,'CMin');
        Q.CMax(k)=extractKernelBoundMaster(CMax,C_ORDER{k},kernels,nK,'CMax');
        assert(Q.CMin(k)>0 && Q.CMax(k)>=Q.CMin(k), ...
            'Invalid C range for %s Case %d.',C_ORDER{k},c);
    end

    for j=1:nK
        key=lower(char(string(kernels{j})));
        if isstruct(Radii)
            assert(isfield(Radii,key),'Radii.%s missing.',key);
            r=double(Radii.(key)(:));
        else
            r=double(Radii(:));
        end
        assert(numel(r)==numel(requiredR) && ...
            max(abs(r-requiredR))<=1e-10*max(1,max(abs(requiredR))), ...
            'M_PR local radius grid for %s Case %d is not 50:10:200 km.',key,c);
    end
    Rng{c}=Q;
end

assert(maxRelativeDifferenceMaster(Rng{1}.RegMin,Rng{2}.RegMin)<=1e-12 && ...
       maxRelativeDifferenceMaster(Rng{1}.RegMax,Rng{2}.RegMax)<=1e-12, ...
    'M_PR investigated RegParam ranges differ between Case 1 and Case 2.');
assert(maxRelativeDifferenceMaster(Rng{1}.CMin,Rng{2}.CMin)<=1e-12 && ...
       maxRelativeDifferenceMaster(Rng{1}.CMax,Rng{2}.CMax)<=1e-12, ...
    'M_PR investigated C ranges differ between Case 1 and Case 2.');

% Preserve the actual M_PR kernel-specific investigated ranges.
Ranges=struct();
Ranges.KernelOrderReg=REG_ORDER;
Ranges.KernelOrderC=C_ORDER;
Ranges.Model1RegMinByKernel=Rng{1}.RegMin;
Ranges.Model1RegMaxByKernel=Rng{1}.RegMax;
Ranges.Model1CMinByKernel=Rng{1}.CMin;
Ranges.Model1CMaxByKernel=Rng{1}.CMax;

% Fixed-reference RegParam range = GLOBAL-RBF input range.
globalRegRange=double(globalRegRange(:).');
assert(numel(globalRegRange)==2 && all(isfinite(globalRegRange)) && ...
       globalRegRange(1)>0 && globalRegRange(2)>=globalRegRange(1), ...
    'Invalid GLOBAL-RBF RegParam input range.');
Ranges.ReferenceRegMin=globalRegRange(1);
Ranges.ReferenceRegMax=globalRegRange(2);
Ranges.ReferenceRegRangeSource='GLOBAL-RBF input range';

% C remains sampled from the common envelope spanning the investigated
% LOCAL Gaussian/MQ C ranges.
Ranges.CommonCMin=min(Ranges.Model1CMinByKernel);
Ranges.CommonCMax=max(Ranges.Model1CMaxByKernel);
assert(isfinite(Ranges.CommonCMin) && isfinite(Ranges.CommonCMax) && ...
       Ranges.CommonCMin>0 && Ranges.CommonCMax>=Ranges.CommonCMin, ...
    'Invalid common C sensitivity envelope.');

% Backward-compatible aliases retained in provenance/output tables. For Reg,
% these aliases now refer to the GLOBAL-RBF input range, not a local envelope.
Ranges.CommonRegMin=Ranges.ReferenceRegMin;
Ranges.CommonRegMax=Ranges.ReferenceRegMax;

% Repeated arrays are retained for compact downstream metadata only.
Ranges.RegMin=repmat(Ranges.ReferenceRegMin,1,5);
Ranges.RegMax=repmat(Ranges.ReferenceRegMax,1,5);
Ranges.CMin=repmat(Ranges.CommonCMin,1,2);
Ranges.CMax=repmat(Ranges.CommonCMax,1,2);

Ranges.SamplingRule = ...
    'RegParam: GLOBAL-RBF input range; C: common envelope of local Gaussian/MQ investigated ranges';

RadiusGridRaw=requiredR;
end

%% ========================================================================
function RadiusStrataRaw = buildRadiusStrataMaster(RadiusGridRaw)
%BUILDRADIUSSTRATAMASTER Five non-overlapping strata covering 50:10:200 km.
RadiusStrataRaw = { ...
    (50000:10000:70000).', ...
    (80000:10000:100000).', ...
    (110000:10000:130000).', ...
    (140000:10000:160000).', ...
    (170000:10000:200000).'};

allStrata = vertcat(RadiusStrataRaw{:});
assert(isequal(allStrata(:),RadiusGridRaw(:)), ...
    'Radius strata must exactly partition the authoritative 50:10:200 km grid.');
assert(numel(unique(allStrata))==numel(allStrata), ...
    'Radius strata must be non-overlapping.');
end

%% ========================================================================
function References = generateRandomReferences(seeds,Ranges,RadiusGridRaw,RadiusStrataRaw)
%GENERATERANDOMREFERENCES
%
% STRATIFIED-RANDOM radius design:
%   * exactly five pre-specified seeds with equal scientific status;
%   * seed-list position s is assigned radius stratum s;
%   * one radius is drawn randomly within that non-overlapping stratum using
%     the same pre-specified seed;
%   * therefore the complete 50:10:200 km domain is represented while every
%     exact radius remains randomly generated before any model run.
%
% C and RegParam are realization-specific independent log-uniform draws.
% All RegParam values draw from the SAME GLOBAL-RBF input range; Gaussian/MQ
% C values draw from the SAME common local-C sensitivity envelope.
%
% Parameter-stream compatibility:
% A separate per-seed parameter stream deliberately consumes the legacy
% full-grid radius-index draw before uReg and uC. Thus changing the radius
% design does NOT change the already-defined C/RegParam realization for a
% given seed. The legacy consumed draw is retained for audit only.
%
% No predictive error is used anywhere in reference generation.

n=numel(seeds);
assert(n==5,'Stratified-radius design requires exactly five seeds.');
assert(numel(RadiusStrataRaw)==n,'One radius stratum is required per seed.');
nRadius=numel(RadiusGridRaw);

template=struct( ...
    'ReferenceIndex',NaN,'Seed',NaN,'ID','','RandomGenerator','mt19937ar', ...
    'SamplingRule',['radius stratified-random: one draw per pre-assigned ' ...
                    'non-overlapping stratum; RegParam from GLOBAL-RBF input ' ...
                    'range; C from common local envelope; no predictive-error selection'], ...
    'UsesPredictiveError',false,'RandomStreamIsolation',true, ...
    'RadiusSamplingDesign','stratified random', ...
    'RadiusSamplingWithoutReplacement',false, ...
    'RadiusSelectionSeed',NaN,'RadiusStratumIndex',NaN, ...
    'RadiusStratumRaw',[],'RadiusStratumMinKm',NaN,'RadiusStratumMaxKm',NaN, ...
    'FixedRadiusRaw',NaN,'FixedRadiusKm',NaN, ...
    'RadiusGridRaw',RadiusGridRaw,'FixedRegParamByKernel',nan(1,5), ...
    'FixedCByKernel',nan(1,2),'UniformDrawReg',nan(1,5), ...
    'UniformDrawC',nan(1,2),'RadiusDrawIndex',NaN, ...
    'RadiusDrawIndexWithinStratum',NaN, ...
    'OriginalWithReplacementRadiusDrawIndex',NaN, ...
    'KernelOrderReg',{Ranges.KernelOrderReg},'KernelOrderC',{Ranges.KernelOrderC}, ...
    'RegMinByKernel',Ranges.RegMin,'RegMaxByKernel',Ranges.RegMax, ...
    'CMinByKernel',Ranges.CMin,'CMaxByKernel',Ranges.CMax, ...
    'FixedReferenceRegMin',Ranges.ReferenceRegMin, ...
    'FixedReferenceRegMax',Ranges.ReferenceRegMax, ...
    'RegRangeSource',Ranges.ReferenceRegRangeSource, ...
    'CommonSensitivityRegMin',Ranges.CommonRegMin, ...
    'CommonSensitivityRegMax',Ranges.CommonRegMax, ...
    'CommonSensitivityCMin',Ranges.CommonCMin, ...
    'CommonSensitivityCMax',Ranges.CommonCMax, ...
    'Model1InvestigatedRegMinByKernel',Ranges.Model1RegMinByKernel, ...
    'Model1InvestigatedRegMaxByKernel',Ranges.Model1RegMaxByKernel, ...
    'Model1InvestigatedCMinByKernel',Ranges.Model1CMinByKernel, ...
    'Model1InvestigatedCMaxByKernel',Ranges.Model1CMaxByKernel, ...
    'Created',datetime.empty);
References=repmat(template,n,1);

for s=1:n
    seed=double(seeds(s));
    stratumRaw=double(RadiusStrataRaw{s}(:));
    assert(~isempty(stratumRaw) && all(ismember(stratumRaw,RadiusGridRaw)), ...
        'Radius stratum %d is invalid.',s);

    % Dedicated radius stream: exact radius is random within assigned stratum.
    radiusStream=RandStream('mt19937ar','Seed',seed);
    kWithin=randi(radiusStream,numel(stratumRaw));
    fixedRadiusRaw=stratumRaw(kWithin);
    kFull=find(RadiusGridRaw==fixedRadiusRaw,1,'first');
    assert(~isempty(kFull),'Selected radius is not on authoritative radius grid.');

    % Dedicated parameter stream: preserve the preceding C/Reg draw sequence.
    parameterStream=RandStream('mt19937ar','Seed',seed);
    legacyRadiusDrawIndex=randi(parameterStream,nRadius);
    uReg=rand(parameterStream,1,5);
    uC=rand(parameterStream,1,2);

    reg=10.^(log10(Ranges.ReferenceRegMin) + ...
        uReg.*(log10(Ranges.ReferenceRegMax)-log10(Ranges.ReferenceRegMin)));
    c=10.^(log10(Ranges.CommonCMin) + ...
        uC.*(log10(Ranges.CommonCMax)-log10(Ranges.CommonCMin)));

    References(s).ReferenceIndex=s;
    References(s).Seed=seed;
    References(s).ID=sprintf('Seed%d_Stratum%d',seed,s);
    References(s).RadiusSelectionSeed=seed;
    References(s).RadiusStratumIndex=s;
    References(s).RadiusStratumRaw=stratumRaw;
    References(s).RadiusStratumMinKm=min(stratumRaw)/1000;
    References(s).RadiusStratumMaxKm=max(stratumRaw)/1000;
    References(s).FixedRadiusRaw=fixedRadiusRaw;
    References(s).FixedRadiusKm=fixedRadiusRaw/1000;
    References(s).RadiusDrawIndex=kFull;
    References(s).RadiusDrawIndexWithinStratum=kWithin;
    References(s).OriginalWithReplacementRadiusDrawIndex=legacyRadiusDrawIndex;
    References(s).UniformDrawReg=uReg;
    References(s).UniformDrawC=uC;
    References(s).FixedRegParamByKernel=reg;
    References(s).FixedCByKernel=c;
    References(s).Created=datetime('now');
end

assert(numel(unique([References.FixedRadiusRaw]))==n, ...
    'Internal error: stratified radii are not unique.');
assert(all(~[References.UsesPredictiveError]), ...
    'Reference generation must not use predictive error.');
end

%% ========================================================================
function T = referencesToTable(R)

n=numel(R);
ReferenceIndex=nan(n,1); Seed=nan(n,1); RadiusKm=nan(n,1); RadiusRaw=nan(n,1);
RadiusSamplingDesign=strings(n,1); RadiusSelectionSeed=nan(n,1);
RadiusStratumIndex=nan(n,1); RadiusStratumMinKm=nan(n,1); RadiusStratumMaxKm=nan(n,1);
RadiusStratumGridKm=strings(n,1); RadiusDrawIndexWithinStratum=nan(n,1);
OriginalWithReplacementRadiusDrawIndex=nan(n,1);
ReferenceRegMin=nan(n,1); ReferenceRegMax=nan(n,1); RegRangeSource=strings(n,1);
CommonRegMin=nan(n,1); CommonRegMax=nan(n,1);
CommonCMin=nan(n,1); CommonCMax=nan(n,1);
Model1RegRanges=strings(n,1); Model1CRanges=strings(n,1);
Reg_G=nan(n,1); Reg_MQ=nan(n,1); Reg_L=nan(n,1); Reg_C=nan(n,1); Reg_TP=nan(n,1);
C_G=nan(n,1); C_MQ=nan(n,1);
uRindex=nan(n,1);
uReg_G=nan(n,1);uReg_MQ=nan(n,1);uReg_L=nan(n,1);uReg_C=nan(n,1);uReg_TP=nan(n,1);
uC_G=nan(n,1);uC_MQ=nan(n,1);
for i=1:n
    ReferenceIndex(i)=R(i).ReferenceIndex;
    Seed(i)=R(i).Seed;
    RadiusKm(i)=R(i).FixedRadiusKm; RadiusRaw(i)=R(i).FixedRadiusRaw;
    RadiusSamplingDesign(i)=string(R(i).RadiusSamplingDesign);
    RadiusSelectionSeed(i)=R(i).RadiusSelectionSeed;
    RadiusStratumIndex(i)=R(i).RadiusStratumIndex;
    RadiusStratumMinKm(i)=R(i).RadiusStratumMinKm;
    RadiusStratumMaxKm(i)=R(i).RadiusStratumMaxKm;
    RadiusStratumGridKm(i)=mat2str(double(R(i).RadiusStratumRaw(:).')/1000);
    RadiusDrawIndexWithinStratum(i)=R(i).RadiusDrawIndexWithinStratum;
    OriginalWithReplacementRadiusDrawIndex(i)=R(i).OriginalWithReplacementRadiusDrawIndex;
    ReferenceRegMin(i)=R(i).FixedReferenceRegMin;
    ReferenceRegMax(i)=R(i).FixedReferenceRegMax;
    RegRangeSource(i)=string(R(i).RegRangeSource);
    CommonRegMin(i)=R(i).CommonSensitivityRegMin;
    CommonRegMax(i)=R(i).CommonSensitivityRegMax;
    CommonCMin(i)=R(i).CommonSensitivityCMin;
    CommonCMax(i)=R(i).CommonSensitivityCMax;
    Model1RegRanges(i)=strjoin(compose('[%.12g,%.12g]', ...
        R(i).Model1InvestigatedRegMinByKernel(:), ...
        R(i).Model1InvestigatedRegMaxByKernel(:)),'; ');
    Model1CRanges(i)=strjoin(compose('[%.12g,%.12g]', ...
        R(i).Model1InvestigatedCMinByKernel(:), ...
        R(i).Model1InvestigatedCMaxByKernel(:)),'; ');
    rr=R(i).FixedRegParamByKernel; cc=R(i).FixedCByKernel;
    Reg_G(i)=rr(1);Reg_MQ(i)=rr(2);Reg_L(i)=rr(3);Reg_C(i)=rr(4);Reg_TP(i)=rr(5);
    C_G(i)=cc(1);C_MQ(i)=cc(2);
    ur=R(i).UniformDrawReg; uc=R(i).UniformDrawC;
    uRindex(i)=R(i).RadiusDrawIndex;
    uReg_G(i)=ur(1);uReg_MQ(i)=ur(2);uReg_L(i)=ur(3);uReg_C(i)=ur(4);uReg_TP(i)=ur(5);
    uC_G(i)=uc(1);uC_MQ(i)=uc(2);
end

T=table(ReferenceIndex,Seed,RadiusKm,RadiusRaw,RadiusSamplingDesign, ...
    RadiusSelectionSeed,RadiusStratumIndex,RadiusStratumMinKm,RadiusStratumMaxKm, ...
    RadiusStratumGridKm,RadiusDrawIndexWithinStratum, ...
    OriginalWithReplacementRadiusDrawIndex, ...
    ReferenceRegMin,ReferenceRegMax,RegRangeSource, ...
    CommonRegMin,CommonRegMax,CommonCMin,CommonCMax, ...
    Model1RegRanges,Model1CRanges, ...
    Reg_G,Reg_MQ,Reg_L,Reg_C,Reg_TP,C_G,C_MQ, ...
    uRindex,uReg_G,uReg_MQ,uReg_L,uReg_C,uReg_TP,uC_G,uC_MQ);
end

%% ========================================================================
function S = validateAblationSummary(S,modelLabel,caseToken)
assert(istable(S) && height(S)==30, ...
    '%s %s summary must contain 30 rows.',char(string(modelLabel)),caseToken);
assert(all(ismember({'Model','Kernel','TestRMSE','Success'},S.Properties.VariableNames)), ...
    '%s %s summary lacks required columns.',char(string(modelLabel)),caseToken);
assert(all(logical(S.Success)), ...
    '%s %s contains one or more failed runs.',char(string(modelLabel)),caseToken);
assert(all(isfinite(double(S.TestRMSE))) && all(double(S.TestRMSE)>0), ...
    '%s %s TestRMSE contains invalid values.',char(string(modelLabel)),caseToken);
assertNoDuplicateKeysMaster(renamevarsForSummary(S),{'GGM','Kernel'}, ...
    sprintf('%s %s',char(string(modelLabel)),caseToken));
end

function T = renamevarsForSummary(S)
T=table(strtrim(string(S.Model)),strtrim(string(S.Kernel)), ...
    'VariableNames',{'GGM','Kernel'});
end

%% ========================================================================
function D = buildComparisonRowsMaster(M1,S,Ref,alternative,tieTol)

n=height(S);
Case=strings(n,1);
GGM=strtrim(string(S.Model));
Kernel=strtrim(string(S.Kernel));
Alternative=repmat(string(alternative),n,1);
ReferenceSeed=repmat(double(Ref.Seed),n,1);
ReferenceID=repmat(string(Ref.ID),n,1);
ReferenceRadiusRaw=repmat(double(Ref.FixedRadiusRaw),n,1);
ReferenceRadiusKm=ReferenceRadiusRaw./1000;
ReferenceIndex=repmat(double(Ref.ReferenceIndex),n,1);
RadiusStratumIndex=repmat(double(Ref.RadiusStratumIndex),n,1);
RadiusStratumMinKm=repmat(double(Ref.RadiusStratumMinKm),n,1);
RadiusStratumMaxKm=repmat(double(Ref.RadiusStratumMaxKm),n,1);
UsesReferenceFixedRadius=repmat(any(alternative==["MP","M0"]),n,1);
UsesReferenceFixedParameters=repmat(any(alternative==["M0","MR"]),n,1);

Model1_TestRMSE=nan(n,1);
Model1_PseudoUnseenRMSE=nan(n,1);
Alternative_TestRMSE=double(S.TestRMSE);
Alternative_TrainRMSE=getOptionalNumericColumn(S,'TrainRMSE');
Alternative_PseudoRMSE=getOptionalNumericColumn(S,'PseudoRMSE');

for i=1:n
    idx=find(canonicalTextMaster(M1.GGM)==canonicalTextMaster(GGM(i)) & ...
             canonicalTextMaster(M1.Kernel)==canonicalTextMaster(Kernel(i)));
    assert(numel(idx)==1,'M_PR row matching failed for %s / %s.',GGM(i),Kernel(i));
    Case(i)=M1.Case(idx);
    Model1_TestRMSE(i)=M1.Model1_Test(idx);
    Model1_PseudoUnseenRMSE(i)=M1.Model1_PseudoUnseen(idx);
end

Alternative_Improvement_vs_Model1_pct = ...
    100.*(Model1_TestRMSE-Alternative_TestRMSE)./Model1_TestRMSE;

Model1_Reduction_vs_Alternative_pct = ...
    100.*(Alternative_TestRMSE-Model1_TestRMSE)./Alternative_TestRMSE;

Winner=strings(n,1);
Winner(Alternative_Improvement_vs_Model1_pct>tieTol)="Alternative";
Winner(Alternative_Improvement_vs_Model1_pct<-tieTol)="MPR";
Winner(abs(Alternative_Improvement_vs_Model1_pct)<=tieTol)="Tie";

KernelGroup=classifyKernelGroupMaster(Kernel);

Reg_G=repmat(Ref.FixedRegParamByKernel(1),n,1);
Reg_MQ=repmat(Ref.FixedRegParamByKernel(2),n,1);
Reg_L=repmat(Ref.FixedRegParamByKernel(3),n,1);
Reg_C=repmat(Ref.FixedRegParamByKernel(4),n,1);
Reg_TP=repmat(Ref.FixedRegParamByKernel(5),n,1);
C_G=repmat(Ref.FixedCByKernel(1),n,1);
C_MQ=repmat(Ref.FixedCByKernel(2),n,1);

D=table(ReferenceIndex,ReferenceSeed,ReferenceID,RadiusStratumIndex, ...
    RadiusStratumMinKm,RadiusStratumMaxKm,Case,Alternative,GGM,Kernel,KernelGroup, ...
    UsesReferenceFixedRadius,UsesReferenceFixedParameters, ...
    ReferenceRadiusKm,ReferenceRadiusRaw,Reg_G,Reg_MQ,Reg_L,Reg_C,Reg_TP,C_G,C_MQ, ...
    Model1_PseudoUnseenRMSE,Alternative_TrainRMSE,Alternative_PseudoRMSE, ...
    Model1_TestRMSE,Alternative_TestRMSE, ...
    Alternative_Improvement_vs_Model1_pct,Model1_Reduction_vs_Alternative_pct,Winner);
end

%% ========================================================================
function x = getOptionalNumericColumn(T,name)
if ismember(name,T.Properties.VariableNames)
    x=double(T.(name));
else
    x=nan(height(T),1);
end
end

function group = classifyKernelGroupMaster(kernel)
k=canonicalTextMaster(kernel);
group=strings(numel(k),1);
group(k=="gaussian" | k=="multiquadric")="Shape-parameter";
group(k=="linear" | k=="cubic" | k=="thinplate" | k=="thinplatespline")="Shape-free";
assert(all(strlength(group)>0),'Unclassified RBF kernel encountered.');
end

%% ========================================================================
function S = summarizeByGroupMaster(D,keyVars,tieTol)

assert(~isempty(D),'Cannot summarize an empty comparison table.');
K=unique(D(:,keyVars),'rows','stable');
rows=cell(height(K),1);

for i=1:height(K)
    mask=true(height(D),1);
    for j=1:numel(keyVars)
        v=keyVars{j};
        mask=mask & valuesEqualMaster(D.(v),K.(v)(i));
    end
    X=D(mask,:);
    x=double(X.Model1_Reduction_vs_Alternative_pct);
    y=double(X.Alternative_Improvement_vs_Model1_pct);
    assert(all(isfinite(x)) && all(isfinite(y)),'Nonfinite comparison percentages.');

    N=numel(x);
    m1=x>tieTol;
    alt=y>tieTol;
    tie=~m1 & ~alt;

    row=K(i,:);
    row.N=N;
    row.Model1_Better_n=sum(m1);
    row.Alternative_Better_n=sum(alt);
    row.Tie_n=sum(tie);
    row.Model1_Better_pct=100*sum(m1)/N;
    row.Alternative_Better_pct=100*sum(alt)/N;
    row.Model1Reduction_Min_pct=min(x);
    row.Model1Reduction_Median_pct=median(x);
    row.Model1Reduction_Max_pct=max(x);
    row.AlternativeImprovement_Min_pct=min(y);
    row.AlternativeImprovement_Median_pct=median(y);
    row.AlternativeImprovement_Max_pct=max(y);
    rows{i}=row;
end
S=vertcat(rows{:});
end

function tf = valuesEqualMaster(a,b)
if iscell(a), a=string(a); end
if iscell(b), b=string(b); end
if isstring(a) || ischar(a) || iscategorical(a)
    tf=string(a)==string(b);
else
    tf=a==b;
end
end

%% ========================================================================
function S = buildStabilitySummaryMaster(R,nSeeds)
% Each row of R is one Case x Alternative x GGM x Kernel summary across seeds.
assert(all(R.N==nSeeds), ...
    'RobustnessByGGMKernel must contain exactly one comparison per seed.');

groups=["Shape-parameter","Shape-free"];
keys=unique(R(:,{'Case','Alternative'}),'rows','stable');
rows=cell(height(keys)*numel(groups),1);
z=0;

for i=1:height(keys)
    base=R.Case==keys.Case(i) & R.Alternative==keys.Alternative(i);
    for g=groups
        z=z+1;
        mask=base & R.KernelGroup==g;
        X=R(mask,:);
        nComb=height(X);
        assert(nComb>0,'Empty stability group.');

        stableM1 = X.Model1_Better_n==nSeeds;
        stableAlt = X.Alternative_Better_n==nSeeds;
        mixed = ~(stableM1 | stableAlt);

        med=double(X.Model1Reduction_Median_pct);
        rows{z}=table(keys.Case(i),keys.Alternative(i),g, ...
            nComb,sum(stableM1),sum(stableAlt),sum(mixed), ...
            100*sum(stableM1)/nComb,100*sum(stableAlt)/nComb, ...
            min(med),median(med),max(med), ...
            'VariableNames',{'Case','Alternative','KernelGroup', ...
            'N_GGMKernel','Model1_Better_AllSeeds_n','Alternative_Better_AllSeeds_n', ...
            'MixedOrTie_n','Model1_Better_AllSeeds_pct','Alternative_Better_AllSeeds_pct', ...
            'MedianReductionAcrossSeeds_Min_pct', ...
            'MedianReductionAcrossSeeds_Median_pct', ...
            'MedianReductionAcrossSeeds_Max_pct'});
    end
end
S=vertcat(rows{:});
end

%% ========================================================================
function T = sortAllComparisonsMaster(T)

alt=canonicalTextMaster(T.Alternative);
altOrder=nan(height(T),1);
altOrder(alt=="mp")=1;
altOrder(alt=="m0")=2;
altOrder(alt=="mr")=3;

caseNorm=canonicalTextMaster(T.Case);
caseOrder=nan(height(T),1);
caseOrder(caseNorm=="case1")=1;
caseOrder(caseNorm=="case2")=2;

T.AltOrder_tmp=altOrder;
T.CaseOrder_tmp=caseOrder;
T=sortrows(T,{'ReferenceSeed','CaseOrder_tmp','AltOrder_tmp','GGM','Kernel'});
T(:,{'AltOrder_tmp','CaseOrder_tmp'})=[];
end

%% ========================================================================
function auditCrossModelReferenceConsistency(D,References)

for s=1:numel(References)
    Ref=References(s);
    X=D(D.ReferenceSeed==Ref.Seed,:);
    assert(~isempty(X),'No comparison rows for seed %d.',Ref.Seed);

    % MP + M0 must report the same fixed R0.
    Y=X(X.Alternative=="MP" | X.Alternative=="M0",:);
    assert(all(Y.ReferenceRadiusRaw==Ref.FixedRadiusRaw), ...
        'MP/M0 fixed-radius reference inconsistency for seed %d.',Ref.Seed);

    % M0 + MR must carry the same fixed parameter reference.
    Z=X(X.Alternative=="M0" | X.Alternative=="MR",:);
    assert(all(abs(Z.Reg_G-Ref.FixedRegParamByKernel(1))<=1e-12*max(1,abs(Ref.FixedRegParamByKernel(1)))) && ...
           all(abs(Z.Reg_MQ-Ref.FixedRegParamByKernel(2))<=1e-12*max(1,abs(Ref.FixedRegParamByKernel(2)))) && ...
           all(abs(Z.Reg_L-Ref.FixedRegParamByKernel(3))<=1e-12*max(1,abs(Ref.FixedRegParamByKernel(3)))) && ...
           all(abs(Z.Reg_C-Ref.FixedRegParamByKernel(4))<=1e-12*max(1,abs(Ref.FixedRegParamByKernel(4)))) && ...
           all(abs(Z.Reg_TP-Ref.FixedRegParamByKernel(5))<=1e-12*max(1,abs(Ref.FixedRegParamByKernel(5)))) && ...
           all(abs(Z.C_G-Ref.FixedCByKernel(1))<=1e-12*max(1,abs(Ref.FixedCByKernel(1)))) && ...
           all(abs(Z.C_MQ-Ref.FixedCByKernel(2))<=1e-12*max(1,abs(Ref.FixedCByKernel(2)))), ...
        'M0/MR fixed-parameter reference inconsistency for seed %d.',Ref.Seed);
end
end

function auditNoTestSelection(References)
assert(all(~[References.UsesPredictiveError]), ...
    'A random reference is incorrectly marked as using predictive error.');
end

%% ========================================================================
function T = readModel1FullTuneRegMaster(fileName)
% Robust saved-value reader adapted from the user's established comparison
% exporter.  No pseudo-unseen recomputation and no test-based selection.

A=load(fileName,'Results');
assert(isfield(A,'Results') && isstruct(A.Results), ...
    'Results structure missing in:\n%s',fileName);
Results=A.Results;

assert(isfield(Results,'Run') && isstruct(Results.Run), ...
    'Results.Run missing in M_PR file:\n%s',fileName);
assert(isfield(Results,'ListOfModel') && isfield(Results,'NameFunction_RBF'), ...
    'Model/kernel labels missing in M_PR file:\n%s',fileName);

models=string(Results.ListOfModel(:));
kernels=string(Results.NameFunction_RBF(:));
nM=numel(models); nK=numel(kernels);
n=nM*nK;

if isfield(Results,'CaseName')
    caseName=normalizeCaseStringsMaster(string(Results.CaseName));
else
    caseName=inferCaseFromFileMaster(fileName);
end

Case=repmat(caseName,n,1);
GGM=strings(n,1); Kernel=strings(n,1);
Model1_PseudoUnseen=nan(n,1); Model1_Test=nan(n,1);

r=0;
for i=1:nM
    for j=1:nK
        r=r+1;
        model=models(i); kernel=kernels(j);
        runName=char(model+"_"+kernel);
        runField=matlab.lang.makeValidName(runName);

        if isfield(Results.Run,runName)
            R=Results.Run.(runName);
        elseif isfield(Results.Run,runField)
            R=Results.Run.(runField);
        else
            [R,found]=findRunByLabelsMaster(Results.Run,model,kernel);
            assert(found,'M_PR run not found for %s / %s.',model,kernel);
        end

        pseudo=directSummaryMetricMaster(Results,model,kernel, ...
            {'PseudoRMSE','PseudoTrainRMSE','Model1_PseudoUnseen'});
        if ~isfinite(pseudo), pseudo=directMatrixMetricMaster(Results,{'Pseudo','RMSE'},i,j); end
        if ~isfinite(pseudo), pseudo=directMatrixMetricMaster(Results,{'Accuracy','Pseudo','RMSE'},i,j); end
        if ~isfinite(pseudo), pseudo=directNumericMatrixFieldMaster(Results,'PseudoRMSE',i,j); end
        if ~isfinite(pseudo), pseudo=directNumericMatrixFieldMaster(Results,'PseudoTrainRMSE',i,j); end
        if ~isfinite(pseudo) && isPseudoCapSourceMaster(Results,i,j)
            pseudo=directNumericMatrixFieldMaster(Results,'capSelectionRMSE',i,j);
        end
        if ~isfinite(pseudo), pseudo=directPseudoRMSEMaster(R); end

        test=directSummaryMetricMaster(Results,model,kernel,{'TestRMSE'});
        if ~isfinite(test), test=directMatrixMetricMaster(Results,{'Test','RMSE'},i,j); end
        if ~isfinite(test), test=directNumericMatrixFieldMaster(Results,'test_rmse',i,j); end
        if ~isfinite(test), test=directTestRMSEMaster(R); end

        assert(isfinite(pseudo),'Saved M_PR pseudo RMSE not found for %s / %s.',model,kernel);
        assert(isfinite(test) && test>0,'Saved M_PR test RMSE invalid for %s / %s.',model,kernel);

        GGM(r)=model; Kernel(r)=kernel;
        Model1_PseudoUnseen(r)=pseudo; Model1_Test(r)=test;
    end
end

T=table(Case,GGM,Kernel,Model1_PseudoUnseen,Model1_Test);
T=sortrows(T,{'Case','GGM','Kernel'});
end

%% ========================================================================
function value=directSummaryMetricMaster(Results,model,kernel,candidateNames)
value=NaN;
if ~isfield(Results,'Summary') || ~istable(Results.Summary) || isempty(Results.Summary), return, end
S=Results.Summary; vars=S.Properties.VariableNames;
modelVar=''; kernelVar='';
if ismember('Model',vars), modelVar='Model'; elseif ismember('GGM',vars), modelVar='GGM'; end
if ismember('Kernel',vars), kernelVar='Kernel'; end
if isempty(modelVar)||isempty(kernelVar), return, end
row=strcmpi(strtrim(string(S.(modelVar))),strtrim(string(model))) & ...
    strcmpi(strtrim(string(S.(kernelVar))),strtrim(string(kernel)));
if nnz(row)~=1, return, end
for q=1:numel(candidateNames)
    name=candidateNames{q};
    if ismember(name,vars)
        x=S.(name)(row);
        if isnumeric(x)&&isscalar(x)&&isreal(x)&&isfinite(x), value=double(x); return, end
    end
end
end

function value=directNumericMatrixFieldMaster(S,fieldName,i,j)
value=NaN;
if ~isstruct(S)||~isscalar(S)||~isfield(S,fieldName), return, end
X=S.(fieldName);
if isnumeric(X)&&isreal(X)&&size(X,1)>=i&&size(X,2)>=j
    x=X(i,j); if isscalar(x)&&isfinite(x), value=double(x); end
end
end

function tf=isPseudoCapSourceMaster(Results,i,j)
tf=false;
fieldCandidates={'capRMSESource','CapRMSESource'};
for q=1:numel(fieldCandidates)
    fn=fieldCandidates{q}; if ~isfield(Results,fn), continue, end
    X=Results.(fn);
    try
        if iscell(X)&&size(X,1)>=i&&size(X,2)>=j
            s=string(X{i,j});
        elseif isstring(X)&&size(X,1)>=i&&size(X,2)>=j
            s=X(i,j);
        elseif ischar(X)&&isvector(X)
            s=string(X);
        else
            continue
        end
        if any(strcmpi(strtrim(s),["pseudo","pseudo-training","pseudotraining"]))
            tf=true; return
        end
    catch
    end
end

% Compatibility with production files that store the source label only in
% the exact Results.Summary row.
if isfield(Results,'Summary') && istable(Results.Summary) && ...
   isfield(Results,'ListOfModel') && isfield(Results,'NameFunction_RBF')
    S=Results.Summary;
    vars=S.Properties.VariableNames;
    if all(ismember({'Model','Kernel','CapRMSESource'},vars))
        model=string(Results.ListOfModel{i});
        kernel=string(Results.NameFunction_RBF{j});
        row=strcmpi(strtrim(string(S.Model)),strtrim(model)) & ...
            strcmpi(strtrim(string(S.Kernel)),strtrim(kernel));
        if nnz(row)==1
            s=strtrim(string(S.CapRMSESource(row)));
            tf=any(strcmpi(s,["pseudo","pseudo-training","pseudotraining"]));
        end
    end
end
end

function value=directMatrixMetricMaster(S,path,i,j)
value=NaN; X=S;
for k=1:numel(path)
    if ~isstruct(X)||~isscalar(X)||~isfield(X,path{k}), return, end
    X=X.(path{k});
end
if isnumeric(X)&&isreal(X)&&size(X,1)>=i&&size(X,2)>=j
    q=X(i,j); if isscalar(q)&&isfinite(q), value=double(q); end
end
end

function value=directPseudoRMSEMaster(R)
value=NaN;
paths={{'Pseudo','RMSE'},{'Accuracy','Pseudo','RMSE'},{'Train','Pseudo','RMSE'}, ...
    {'TrainingResult','Pseudo','RMSE'},{'Train','TrainingResult','Pseudo','RMSE'}, ...
    {'Stage2Core','Accuracy','Pseudo','RMSE'},{'RunInfo','Stage2Core','Accuracy','Pseudo','RMSE'}};
for k=1:numel(paths)
    value=getFiniteScalarPathMaster(R,paths{k}); if isfinite(value), return, end
end
end

function value=directTestRMSEMaster(R)
value=NaN;
paths={{'Test','RMSE'},{'Test','test_rmse'},{'TestingResult','RMSE'}, ...
    {'TestingResult','test_rmse'},{'Test','TestingResult','RMSE'}, ...
    {'Test','TestingResult','test_rmse'}};
for k=1:numel(paths)
    value=getFiniteScalarPathMaster(R,paths{k}); if isfinite(value), return, end
end
end

function value=getFiniteScalarPathMaster(S,path)
value=NaN; X=S;
for k=1:numel(path)
    if ~isstruct(X)||~isscalar(X)||~isfield(X,path{k}), return, end
    X=X.(path{k});
end
if isnumeric(X)&&isscalar(X)&&isreal(X)&&isfinite(X), value=double(X); end
end

function [R,found]=findRunByLabelsMaster(Runs,model,kernel)
R=struct(); found=false; F=fieldnames(Runs);
for k=1:numel(F)
    C=Runs.(F{k}); if ~isstruct(C)||~isscalar(C), continue, end
    m=""; q="";
    if isfield(C,'model'), m=string(C.model); elseif isfield(C,'Model'), m=string(C.Model); end
    if isfield(C,'kernel'), q=string(C.kernel); elseif isfield(C,'Kernel'), q=string(C.Kernel); end
    if strcmpi(strtrim(m),strtrim(model))&&strcmpi(strtrim(q),strtrim(kernel))
        R=C; found=true; return
    end
end
end

%% ========================================================================
function value=getSettingMaster(S,names)
for k=1:numel(names)
    if isfield(S,names{k})&&~isempty(S.(names{k})), value=S.(names{k}); return, end
end
error('Required setting not found: %s',strjoin(names,', '));
end

function v=extractKernelBoundMaster(A,key,kernels,nKernel,name)
if isnumeric(A)&&isscalar(A)
    v=double(A);
elseif isstruct(A)&&isfield(A,key)
    x=double(A.(key)); assert(isscalar(x),'%s.%s must be scalar.',name,key); v=x;
elseif isnumeric(A)
    A=double(A); j=find(strcmpi(kernels,key),1); assert(~isempty(j),'Kernel %s not found.',key);
    if isvector(A)&&numel(A)==nKernel
        x=A(:); v=x(j);
    elseif size(A,2)==nKernel
        x=A(:,j); assert(all(isfinite(x)),'%s nonfinite for %s.',name,key);
        assert(max(x)-min(x)<=1e-12*max(1,max(abs(x))), ...
            '%s varies across GGMs for %s.',name,key);
        v=x(1);
    else
        error('%s has unsupported numeric shape.',name);
    end
else
    error('%s has unsupported representation.',name);
end
assert(isscalar(v)&&isfinite(v)&&v>0,'%s for %s must be positive finite.',name,key);
end

function d=maxRelativeDifferenceMaster(a,b)
a=double(a(:)); b=double(b(:)); den=max(1,max(abs([a;b]))); d=max(abs(a-b))/den;
end

function fileName=resolveMainResultsFileMaster(RootDir,caseToken)
baseDirs={fullfile(RootDir,'OutPut_RBF_2026_2'),fullfile(RootDir,'OutPut_RBF_2026_1')};
hits={};
for b=1:numel(baseDirs)
    exact=fullfile(baseDirs{b},sprintf('LocalRBF_%s_Full_TuneReg.mat',caseToken));
    if isfile(exact), hits{end+1,1}=exact; end %#ok<AGROW>
end
if numel(hits)==1, fileName=hits{1}; return, end
assert(isempty(hits),'More than one exact M_PR result found for %s.',caseToken);
for b=1:numel(baseDirs)
    if ~isfolder(baseDirs{b}), continue, end
    F=dir(fullfile(baseDirs{b},sprintf('LocalRBF_%s*TuneReg*.mat',caseToken)));
    for k=1:numel(F)
        if F(k).isdir, continue, end
        nm=lower(F(k).name);
        if contains(nm,'stage1')||contains(nm,'sensitivity'), continue, end
        hits{end+1,1}=fullfile(F(k).folder,F(k).name); %#ok<AGROW>
    end
end
assert(numel(hits)==1,'Could not uniquely resolve M_PR result for %s.',caseToken);
fileName=hits{1};
end

%% ========================================================================
function assertNoDuplicateKeysMaster(T,keyVars,label)
K=strings(height(T),1);
for i=1:height(T)
    parts=strings(1,numel(keyVars));
    for j=1:numel(keyVars)
        parts(j)=canonicalTextMaster(T.(keyVars{j})(i));
    end
    K(i)=strjoin(parts,'|');
end
assert(numel(unique(K))==height(T),'%s contains duplicate key rows.',label);
end

function s=canonicalTextMaster(x)
s=lower(strtrim(string(x)));
s=regexprep(s,'[^a-z0-9]','');
end

function s=normalizeCaseStringsMaster(s)
s=strtrim(string(s));
for i=1:numel(s)
    token=regexprep(lower(s(i)),'[^a-z0-9]','');
    if token=="case1"||token=="1", s(i)="Case1";
    elseif token=="case2"||token=="2", s(i)="Case2";
    end
end
end

function c=inferCaseFromFileMaster(fileName)
[~,name]=fileparts(fileName);
if contains(name,'Case1','IgnoreCase',true), c="Case1";
elseif contains(name,'Case2','IgnoreCase',true), c="Case2";
else, error('Cannot infer case from file:\n%s',fileName);
end
end

%% ========================================================================
function saveAtomicVariablesMaster(fileName,V)
folder=fileparts(fileName); if ~isfolder(folder), mkdir(folder); end
tmp=[tempname(folder) '.mat']; cleanup=onCleanup(@() cleanupTempMaster(tmp));
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

function cleanupTempMaster(f)
if isfile(f), try, delete(f); catch, end, end
end

%% ========================================================================
function writeAllReferencesLatexMaster(T,nSeeds,fileName)
fid=fopen(fileName,'wt'); if fid<0, error('Cannot open %s',fileName); end
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%% Auto-generated equal-status stratified-random ablation summary\n');
fprintf(fid,'\\begin{table*}[htbp]\n\\centering\n\\scriptsize\n');
fprintf(fid,'\\caption{Ablation comparison across all %d equal-status stratified-random reference realizations on the independent test set.}\n',nSeeds);
fprintf(fid,'\\label{tab:ablation_all_stratified_random}\n');
fprintf(fid,'\\begin{tabular}{lllrrrrrr}\n\\hline\n');
fprintf(fid,'Case & Alternative & Kernel group & $N$ & $M_{PR}$ better & Alt. better & Tie & Median $M_{PR}$ reduction (\\%%) & Range (\\%%) \\\\\n\\hline\n');
T=sortSummaryForLatex(T);
for i=1:height(T)
    fprintf(fid,'%s & %s & %s & %d & %d & %d & %d & %.2f & [%.2f, %.2f] \\\\\n', ...
        latexEscapeMaster(T.Case(i)),latexModelLabelMaster(T.Alternative(i)), ...
        latexEscapeMaster(T.KernelGroup(i)),T.N(i),T.Model1_Better_n(i), ...
        T.Alternative_Better_n(i),T.Tie_n(i), ...
        T.Model1Reduction_Median_pct(i),T.Model1Reduction_Min_pct(i), ...
        T.Model1Reduction_Max_pct(i));
end
fprintf(fid,'\\hline\n\\end{tabular}\n');
fprintf(fid,['\\begin{flushleft}\\footnotesize All five reference realizations have equal scientific status. ' ...
    'One fixed radius is drawn within each non-overlapping radius stratum before any predictive error is evaluated. ' ...
    'Positive $M_{PR}$ reduction denotes lower independent-test RMSE for $M_{PR}$.\\end{flushleft}\n']);
fprintf(fid,'\\end{table*}\n');
clear cleanup
end

function writeRobustnessLatexMaster(T,nSeeds,fileName)
fid=fopen(fileName,'wt'); if fid<0, error('Cannot open %s',fileName); end
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%% Auto-generated random-reference robustness stability summary\n');
fprintf(fid,'\\begin{table*}[htbp]\n\\centering\n\\scriptsize\n');
fprintf(fid,'\\caption{Stability of the M_PR comparison across %d equal-status stratified-random reference realizations.}\n',nSeeds);
fprintf(fid,'\\label{tab:ablation_random_robustness}\n');
fprintf(fid,'\\begin{tabular}{lllrrrrr}\n\\hline\n');
fprintf(fid,'Case & Alternative & Kernel group & Combinations & $M_{PR}$ better all seeds & Alt. better all seeds & Mixed/tie & Median $M_{PR}$ reduction (\\%%) \\\\\n\\hline\n');
T=sortSummaryForLatex(T);
for i=1:height(T)
    fprintf(fid,'%s & %s & %s & %d & %d & %d & %d & %.2f \\\\\n', ...
        latexEscapeMaster(T.Case(i)),latexModelLabelMaster(T.Alternative(i)), ...
        latexEscapeMaster(T.KernelGroup(i)),T.N_GGMKernel(i), ...
        T.Model1_Better_AllSeeds_n(i),T.Alternative_Better_AllSeeds_n(i), ...
        T.MixedOrTie_n(i),T.MedianReductionAcrossSeeds_Median_pct(i));
end
fprintf(fid,'\\hline\n\\end{tabular}\n');
fprintf(fid,'\\begin{flushleft}\\footnotesize A stable $M_{PR}$ win means that $M_{PR}$ had lower independent-test RMSE for the same GGM--kernel combination under every pre-specified stratified-random reference. $M_P$ shares the fixed radius with $M_0$; $M_0$ and $M_R$ share the fixed kernel parameters.\\end{flushleft}\n');
fprintf(fid,'\\end{table*}\n');
clear cleanup
end

function T=sortSummaryForLatex(T)
caseN=canonicalTextMaster(T.Case); c=nan(height(T),1); c(caseN=="case1")=1;c(caseN=="case2")=2;
altN=canonicalTextMaster(T.Alternative); a=nan(height(T),1); a(altN=="mp")=1;a(altN=="m0")=2;a(altN=="mr")=3;
g=string(T.KernelGroup); go=nan(height(T),1); go(g=="Shape-parameter")=1;go(g=="Shape-free")=2;go(g=="All")=3;
T.c_tmp=c;T.a_tmp=a;T.g_tmp=go;T=sortrows(T,{'c_tmp','a_tmp','g_tmp'});T(:,{'c_tmp','a_tmp','g_tmp'})=[];
end

function s=latexModelLabelMaster(x)
t=upper(strtrim(char(string(x))));
switch t
    case 'M0'
        s='$M_0$';
    case 'MP'
        s='$M_P$';
    case 'MR'
        s='$M_R$';
    case 'MPR'
        s='$M_{PR}$';
    otherwise
        s=latexEscapeMaster(x);
end
end

function s=latexEscapeMaster(x)
s=char(string(x)); s=strrep(s,'_','\_'); s=strrep(s,'&','\&'); s=strrep(s,'%','\%');
end

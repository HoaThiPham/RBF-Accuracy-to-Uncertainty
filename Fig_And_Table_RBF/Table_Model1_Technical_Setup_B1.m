function Table_Model1_Technical_Setup_B1()
% =========================================================================
% COLLECT_MODEL1_TECHNICAL_SETUP
%
% PURPOSE
%   Create a compact manuscript-ready Technical Setup table for QS3_10.
%   The table contains ONLY numerical/computational reproducibility settings.
%
%   Removed from the main table by design:
%     - outer train/test partition
%     - Case definitions / preprocessing
%     - GGM list
%     - RBF kernel list
%     - local cap-selection method
%     - bootstrap-design description
%     - calibration-data-source description
%     - tau_loc / tau_cap
%     - implementation-only storage details
%
%   Hybrid is the main Local-RBF transfer method; nearest and IDW alternatives are reported only as sensitivity analyses.
%
% OUTPUT
%   <RootDir>\OutPut_RBF_2026_2\ComputationalSettings\
%       Model1_Technical_Setup_Table.xlsx
%       Model1_Technical_Setup_Table.tex
%
% Excel sheets
%   Technical Setup : compact 3-column manuscript table
%   Audit Trail     : same rows + exact source provenance
%
% LaTeX
%   Includes caption and label.
%   Requires:
%       \usepackage{booktabs}
%       \usepackage{tabularx}
%       \usepackage{array}
%
% IMPORTANT
%   - Saved Results.Settings are the primary source.
%   - For fixed implementation settings not stored in MAT files
%     (e.g. GA operators / RNG algorithm), the active production source
%     is read without executing any scientific calculation.
%   - Does NOT rerun RBF, bootstrap, or calibration.
%   - Case 1 / Case 2 values are combined when identical.
%   - If values differ, both case-specific values are written.
%
% RUN
%   Collect_Model1_Technical_Setup_v17
% =========================================================================

clc;


%% ========================================================================
% 1. USER INPUT
% ========================================================================
RootDir = '.';

GlobalDir = ...
    fullfile(RootDir,'OutPut_RBF_2026');

LocalDir = ...
    fullfile(RootDir,'OutPut_RBF_2026_2');

OutputDir = ...
    fullfile(LocalDir,'ComputationalSettings_1');

if ~isfolder(OutputDir)
    mkdir(OutputDir);
end


% ------------------------------------------------------------------------
% 1.1 GLOBAL FINAL FIT
% ------------------------------------------------------------------------

GlobalFitCase1File = ...
    fullfile(GlobalDir,'GlobalRBF_Case1_Full_TuneReg_4.mat');

GlobalFitCase2File = ...
    fullfile(GlobalDir,'GlobalRBF_Case2_Full_TuneReg_4.mat');


% ------------------------------------------------------------------------
% 1.2 GLOBAL MAIN (NORMAL) BOOTSTRAP
%
% Spatial bootstrap is a sensitivity check only and is intentionally NOT
% included in this main Technical Setup table.
% ------------------------------------------------------------------------

GlobalBootstrapCase1File = ...
    fullfile(GlobalDir,'Case1_NonSpatialBootstrap.mat');

GlobalBootstrapCase2File = ...
    fullfile(GlobalDir,'Case2_NonSpatialBootstrap.mat');


% ------------------------------------------------------------------------
% 1.3 GLOBAL MAIN CALIBRATION
%
% Resolve the current non-spatial/main calibration file without silently
% selecting a spatial-sensitivity result.
% ------------------------------------------------------------------------

GlobalCalibrationCase1File = ...
    resolveGlobalCalibrationFile(GlobalDir,1);

GlobalCalibrationCase2File = ...
    resolveGlobalCalibrationFile(GlobalDir,2);


% ------------------------------------------------------------------------
% 1.4 LOCAL FINAL BOOTSTRAP
%
% Results.Source.Settings stores the exact upstream Local-RBF settings.
% ------------------------------------------------------------------------

LocalBootstrapDir = ...
    fullfile(LocalDir,'LocalBootstrap_pseudo_unseen_B200_S0');

LocalBootstrapCase1File = ...
    fullfile(LocalBootstrapDir,'Case1_Local_Bootstrap.mat');

LocalBootstrapCase2File = ...
    fullfile(LocalBootstrapDir,'Case2_Local_Bootstrap.mat');


% ------------------------------------------------------------------------
% 1.5 LOCAL FINAL CALIBRATION
% ------------------------------------------------------------------------

LocalCalibrationDir = ...
    fullfile(LocalDir,'LocalVarianceCalibration_pseudo_unseen_B200_S0');

LocalCalibrationCase1File = ...
    fullfile(LocalCalibrationDir,'Case1_Local_VarianceCalibration.mat');

LocalCalibrationCase2File = ...
    fullfile(LocalCalibrationDir,'Case2_Local_VarianceCalibration.mat');


% ------------------------------------------------------------------------
% 1.6 OUTPUT FILES
% ------------------------------------------------------------------------

ExcelFile = ...
    fullfile(OutputDir,'Model1_Technical_Setup_Table.xlsx');

LatexFile = ...
    fullfile(OutputDir,'Model1_Technical_Setup_Table.tex');

Caption = ...
    ['Computational settings used for the Global and Local RBF frameworks. ' ...
     'Values are identical for Cases 1 and 2 unless case-specific values ' ...
     'are explicitly reported.'];

LatexLabel = ...
    'tab:technical_setup';

% ------------------------------------------------------------------------
% 1.7 COMPUTING ENVIRONMENT
%
% Reported in the manuscript Results and used for reproducibility.
% ------------------------------------------------------------------------

ComputingEnvironmentText = ...
    ['MATLAB R2025a (Update 1); 64-bit Windows; Intel Core i5-10300H ' ...
     '2.50 GHz; 15.8 GB RAM'];


%% ========================================================================
% 2. VALIDATE FILES
% ========================================================================

Files = { ...
    GlobalFitCase1File
    GlobalFitCase2File
    GlobalBootstrapCase1File
    GlobalBootstrapCase2File
    GlobalCalibrationCase1File
    GlobalCalibrationCase2File
    LocalBootstrapCase1File
    LocalBootstrapCase2File
    LocalCalibrationCase1File
    LocalCalibrationCase2File};

Labels = { ...
    'Global fit Case 1'
    'Global fit Case 2'
    'Global bootstrap Case 1'
    'Global bootstrap Case 2'
    'Global calibration Case 1'
    'Global calibration Case 2'
    'Local bootstrap Case 1'
    'Local bootstrap Case 2'
    'Local calibration Case 1'
    'Local calibration Case 2'};

for i = 1:numel(Files)
    if ~isfile(Files{i})
        error('Missing %s:\n%s',Labels{i},Files{i});
    end
end


%% ========================================================================
% 3. LOAD AUTHORITATIVE SAVED SETTINGS
% ========================================================================

G1 = loadResults(GlobalFitCase1File);
G2 = loadResults(GlobalFitCase2File);

GB1 = loadResults(GlobalBootstrapCase1File);
GB2 = loadResults(GlobalBootstrapCase2File);

GC1 = loadResults(GlobalCalibrationCase1File);
GC2 = loadResults(GlobalCalibrationCase2File);

LB1 = loadResults(LocalBootstrapCase1File);
LB2 = loadResults(LocalBootstrapCase2File);

LC1 = loadResults(LocalCalibrationCase1File);
LC2 = loadResults(LocalCalibrationCase2File);

requireSettings(G1,'Global fit Case 1');
requireSettings(G2,'Global fit Case 2');
requireSettings(GB1,'Global bootstrap Case 1');
requireSettings(GB2,'Global bootstrap Case 2');
requireSettings(GC1,'Global calibration Case 1');
requireSettings(GC2,'Global calibration Case 2');
requireSettings(LB1,'Local bootstrap Case 1');
requireSettings(LB2,'Local bootstrap Case 2');
requireSettings(LC1,'Local calibration Case 1');
requireSettings(LC2,'Local calibration Case 2');

Sg1 = G1.Settings;
Sg2 = G2.Settings;

Sgb1 = GB1.Settings;
Sgb2 = GB2.Settings;

Sgc1 = GC1.Settings;
Sgc2 = GC2.Settings;

Sl1 = getBootstrapSourceSettings(LB1,'Local bootstrap Case 1');
Sl2 = getBootstrapSourceSettings(LB2,'Local bootstrap Case 2');

Slb1 = LB1.Settings;
Slb2 = LB2.Settings;

Slc1 = LC1.Settings;
Slc2 = LC2.Settings;


%% ========================================================================
% 4. BUILD COMPACT MANUSCRIPT TABLE
%
% Columns:
%   Computational setting | Global RBF | Local RBF
% ========================================================================

Rows = cell(0,5);
% columns:
% 1 Setting
% 2 Global RBF
% 3 Local RBF
% 4 Global source
% 5 Local source


% ------------------------------------------------------------------------
% 4.1 Radius search
% ------------------------------------------------------------------------

localRadius = compareCaseText( ...
    formatField(Sl1,'Radii',@formatRadii), ...
    formatField(Sl2,'Radii',@formatRadii));

Rows = addTechnicalRow( ...
    Rows, ...
    'Radius search', ...
    'Not applicable', ...
    localRadius, ...
    'Global RBF formulation', ...
    'Local bootstrap Results.Source.Settings.Radii');


% ------------------------------------------------------------------------
% 4.2 C search range
% ------------------------------------------------------------------------

globalC = compareCaseText( ...
    formatBoundsFromSettings(Sg1,'CMin','CMax',true), ...
    formatBoundsFromSettings(Sg2,'CMin','CMax',true));

localC = compareCaseText( ...
    formatBoundsFromSettings(Sl1,'CMin','CMax',true), ...
    formatBoundsFromSettings(Sl2,'CMin','CMax',true));

Rows = addTechnicalRow( ...
    Rows, ...
    'C search range', ...
    globalC, ...
    localC, ...
    'Global Results.Settings.CMin/CMax', ...
    'Local bootstrap Results.Source.Settings.CMin/CMax');


% ------------------------------------------------------------------------
% 4.3 Regularisation parameter eta search range
% ------------------------------------------------------------------------

globalReg = compareCaseText( ...
    formatBoundsFromSettings(Sg1,'RegMin','RegMax',false), ...
    formatBoundsFromSettings(Sg2,'RegMin','RegMax',false));

localReg = compareCaseText( ...
    formatBoundsFromSettings(Sl1,'RegMin','RegMax',false), ...
    formatBoundsFromSettings(Sl2,'RegMin','RegMax',false));

Rows = addTechnicalRow( ...
    Rows, ...
    'Regularisation parameter η search range', ...
    globalReg, ...
    localReg, ...
    'Global Results.Settings.RegMin/RegMax', ...
    'Local bootstrap Results.Source.Settings.RegMin/RegMax');


% ------------------------------------------------------------------------
% 4.4 Minimum local observations
% ------------------------------------------------------------------------

localMinObs = compareCaseText( ...
    formatFieldAny(Sl1,{'minNeighbors','MinNeighbors'},@valueText), ...
    formatFieldAny(Sl2,{'minNeighbors','MinNeighbors'},@valueText));

Rows = addTechnicalRow( ...
    Rows, ...
    'Minimum local observations', ...
    'Not applicable', ...
    localMinObs, ...
    'Global RBF formulation', ...
    'Local bootstrap Results.Source.Settings.minNeighbors');


% ------------------------------------------------------------------------
% 4.5 Hyperparameter optimization
% ------------------------------------------------------------------------

% Global RBF uses GA as the optimizer for all kernels:
%   - Gaussian / multiquadric: jointly optimize C and RegParam.
%   - Shape-free kernels: optimize RegParam only.
% The fixed K-fold CV is the GA objective evaluation, NOT a second
% optimization method. A logarithmic-grid search is used only if GA fails.
globalOpt = compareCaseText( ...
    formatGlobalOptimizationSettings(G1), ...
    formatGlobalOptimizationSettings(G2));

localOpt = compareCaseText( ...
    formatLocalOptimizationSettings(Sl1), ...
    formatLocalOptimizationSettings(Sl2));

Rows = addTechnicalRow( ...
    Rows, ...
    'Hyperparameter optimization', ...
    globalOpt, ...
    localOpt, ...
    ['Global Results.Settings + Results.k_used + ' ...
     'globalrbf.globalRBFTrainingCV_TuneReg'], ...
    ['Local bootstrap Results.Source.Settings.Method/nGrid + ' ...
     'localrbf.optimizeParameters']);


% ------------------------------------------------------------------------
% 4.6 Parameter-search RNG seed
%
% This is NOT the seed that created the inherited outer train/test split.
% ------------------------------------------------------------------------

globalSeed = compareCaseText( ...
    formatFieldAny(Sg1,{'RNG_SEED'},@valueText), ...
    formatFieldAny(Sg2,{'RNG_SEED'},@valueText));

localSeed = compareCaseText( ...
    formatFieldAny(Sl1,{'RNG_SEED'},@valueText), ...
    formatFieldAny(Sl2,{'RNG_SEED'},@valueText));

Rows = addTechnicalRow( ...
    Rows, ...
    'Parameter-search RNG seed', ...
    globalSeed, ...
    localSeed, ...
    'Global Results.Settings.RNG_SEED', ...
    'Local bootstrap Results.Source.Settings.RNG_SEED');


% ------------------------------------------------------------------------
% 4.7 GA settings
%
% Prefer saved settings. If absent, parse the currently active optimizer
% source without executing the optimizer.
% ------------------------------------------------------------------------

globalGA = compareCaseText( ...
    formatGASettings(Sg1,'globalrbf.globalRBFTrainingCV_TuneReg'), ...
    formatGASettings(Sg2,'globalrbf.globalRBFTrainingCV_TuneReg'));

localGA = compareCaseText( ...
    formatGASettings(Sl1,'localrbf.optimizeParameters'), ...
    formatGASettings(Sl2,'localrbf.optimizeParameters'));

if ~strcmp(localGA,'Not stored')
    localGA = ['Gaussian/multiquadric only: ' localGA];
end

Rows = addTechnicalRow( ...
    Rows, ...
    'GA settings', ...
    globalGA, ...
    localGA, ...
    'Global saved settings or globalrbf.globalRBFTrainingCV_TuneReg source', ...
    'Local saved settings or localrbf.optimizeParameters source');


% ------------------------------------------------------------------------
% 4.8 Numerical solver control
% ------------------------------------------------------------------------

globalSolver = compareCaseText( ...
    formatNestedField(Sg1,{'options','solver','cond_lsq'},@valueText), ...
    formatNestedField(Sg2,{'options','solver','cond_lsq'},@valueText));

localSolver = compareCaseText( ...
    formatNestedField(Sl1,{'options','solver','cond_lsq'},@valueText), ...
    formatNestedField(Sl2,{'options','solver','cond_lsq'},@valueText));

Rows = addTechnicalRow( ...
    Rows, ...
    'Least-squares condition threshold', ...
    globalSolver, ...
    localSolver, ...
    'Global Results.Settings.options.solver.cond_lsq', ...
    'Local bootstrap Results.Source.Settings.options.solver.cond_lsq');


% ------------------------------------------------------------------------
% 4.9 Transfer method
%
% Hybrid is the main Local transfer method.
% Nearest-only and IDW-only alternatives are sensitivity analyses.
% ------------------------------------------------------------------------

localTransferSensitivity = compareCaseText( ...
    formatHybridSensitivity(Sl1), ...
    formatHybridSensitivity(Sl2));

Rows = addTechnicalRow( ...
    Rows, ...
    'Transfer method', ...
    'Not applicable', ...
    localTransferSensitivity, ...
    'Not applicable', ...
    'Local production transfer settings; nearest/IDW alternatives are sensitivity analyses');


% ------------------------------------------------------------------------
% 4.10 Bootstrap settings
%
% Only numerical bootstrap controls are reported here.
% Bootstrap design/methodology is intentionally described in Methods.
% ------------------------------------------------------------------------

globalBootstrap = compareCaseText( ...
    formatBootstrapCompact(Sgb1,'MT19937'), ...
    formatBootstrapCompact(Sgb2,'MT19937'));

localBootstrap = compareCaseText( ...
    formatBootstrapCompact(Slb1,'MT19937'), ...
    formatBootstrapCompact(Slb2,'MT19937'));

Rows = addTechnicalRow( ...
    Rows, ...
    'Bootstrap settings', ...
    globalBootstrap, ...
    localBootstrap, ...
    'Global bootstrap Results.Settings', ...
    'Local bootstrap Results.Settings');


% ------------------------------------------------------------------------
% 4.11 Calibration parameter bounds
% ------------------------------------------------------------------------

globalCalBounds = compareCaseText( ...
    formatCalibrationBounds(Sgc1), ...
    formatCalibrationBounds(Sgc2));

localCalBounds = compareCaseText( ...
    formatCalibrationBounds(Slc1), ...
    formatCalibrationBounds(Slc2));

Rows = addTechnicalRow( ...
    Rows, ...
    'Calibration parameter bounds', ...
    globalCalBounds, ...
    localCalBounds, ...
    'Global calibration Results.Settings.alpha_bounds/beta_bounds', ...
    'Local calibration Results.Settings.alpha_bounds/beta_bounds');


% ------------------------------------------------------------------------
% 4.12 Calibration GA settings
% ------------------------------------------------------------------------

globalCalGA = compareCaseText( ...
    formatCalibrationGA(Sgc1), ...
    formatCalibrationGA(Sgc2));

localCalGA = compareCaseText( ...
    formatCalibrationGA(Slc1), ...
    formatCalibrationGA(Slc2));

Rows = addTechnicalRow( ...
    Rows, ...
    'Calibration GA settings', ...
    globalCalGA, ...
    localCalGA, ...
    ['Global calibration Results.Settings + ' ...
     'common.calibrateBootstrapVariance fixed GA settings'], ...
    ['Local calibration Results.Settings + ' ...
     'common.calibrateBootstrapVariance fixed GA settings']);


% ------------------------------------------------------------------------
% 4.13 Parallel execution
%
% Stage-specific worker counts are read from the saved settings whenever
% available.  For Local RBF, the final archived settings are expected to
% report optimisation=2, bootstrap=3, calibration=4; inner GA is serial.
% No universal worker count is assumed across stages.
% ------------------------------------------------------------------------

globalParallel = compareCaseText( ...
    formatStageParallelSettings(Sg1,Sgb1,Sgc1), ...
    formatStageParallelSettings(Sg2,Sgb2,Sgc2));

localParallel = compareCaseText( ...
    formatStageParallelSettings(Sl1,Slb1,Slc1), ...
    formatStageParallelSettings(Sl2,Slb2,Slc2));

Rows = addTechnicalRow( ...
    Rows, ...
    'Parallel execution', ...
    globalParallel, ...
    localParallel, ...
    ['Global fit/bootstrap/calibration Results.Settings.NumWorkers ' ...
     'or options.execution.numWorkers'], ...
    ['Local source settings/bootstrap/calibration Results.Settings.NumWorkers ' ...
     'or options.execution.numWorkers']);


% ------------------------------------------------------------------------
% 4.14 Computing environment
% ------------------------------------------------------------------------

Rows = addTechnicalRow( ...
    Rows, ...
    'Computing environment', ...
    ComputingEnvironmentText, ...
    ComputingEnvironmentText, ...
    'Recorded computing environment', ...
    'Recorded computing environment');


%% ========================================================================
% 5. CREATE TABLES
% ========================================================================

TManuscript = cell2table( ...
    Rows(:,1:3), ...
    'VariableNames', ...
    {'Computational_Setting','Global_RBF','Local_RBF'});

TAudit = cell2table( ...
    Rows, ...
    'VariableNames', ...
    {'Computational_Setting','Global_RBF','Local_RBF', ...
     'Global_Source','Local_Source'});


%% ========================================================================
% 6. EXPORT EXCEL
% ========================================================================

if isfile(ExcelFile)
    delete(ExcelFile);
end

writecell( ...
    {Caption}, ...
    ExcelFile, ...
    'Sheet','Technical Setup', ...
    'Range','A1');

writetable( ...
    TManuscript, ...
    ExcelFile, ...
    'Sheet','Technical Setup', ...
    'Range','A3');

writetable( ...
    TAudit, ...
    ExcelFile, ...
    'Sheet','Audit Trail', ...
    'Range','A1');


%% ========================================================================
% 7. EXPORT LATEX
% ========================================================================

writeLatexTable( ...
    TManuscript, ...
    LatexFile, ...
    Caption, ...
    LatexLabel);


%% ========================================================================
% 8. FINISHED
% ========================================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 1 TECHNICAL SETUP TABLE CREATED\n');
fprintf('============================================================\n');
fprintf('Excel : %s\n',ExcelFile);
fprintf('LaTeX : %s\n',LatexFile);
fprintf('============================================================\n\n');

disp(TManuscript);

end


%% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function R = loadResults(fileName)

info = whos('-file',fileName);
names = {info.name};

if ~any(strcmp(names,'Results'))
    error('Variable "Results" was not found in:\n%s',fileName);
end

S = load(fileName,'Results');
R = S.Results;

end


function requireSettings(R,label)

if ~isstruct(R) || ...
   ~isfield(R,'Settings') || ...
   ~isstruct(R.Settings)

    error('%s does not contain a valid Results.Settings structure.',label);
end

end


function S = getBootstrapSourceSettings(B,label)

if ~isstruct(B) || ...
   ~isfield(B,'Source') || ...
   ~isstruct(B.Source) || ...
   ~isfield(B.Source,'Settings') || ...
   ~isstruct(B.Source.Settings)

    error([ ...
        '%s does not contain Results.Source.Settings.\n' ...
        'Use the final Local-RBF bootstrap file generated from Model 1.'], ...
        label);
end

S = B.Source.Settings;

end


function fileName = resolveGlobalCalibrationFile(globalDir,caseNumber)

exactNames = { ...
    sprintf('Case%d_NonSpatialVarianceCalibration.mat',caseNumber)
    sprintf('Case%d_VarianceCalibration.mat',caseNumber)
    sprintf('Case%d_Global_VarianceCalibration.mat',caseNumber)};

for i = 1:numel(exactNames)
    f = fullfile(globalDir,exactNames{i});
    if isfile(f)
        fileName = f;
        return
    end
end

pattern = sprintf('Case%d*VarianceCalibration*.mat',caseNumber);
F = dir(fullfile(globalDir,pattern));
F = F(~[F.isdir]);

keep = true(numel(F),1);

for i = 1:numel(F)
    nameLower = lower(F(i).name);

    if contains(nameLower,'spatial') || ...
       contains(nameLower,'local')

        keep(i) = false;
    end
end

F = F(keep);

if numel(F) == 1
    fileName = fullfile(F(1).folder,F(1).name);
    return
end

if isempty(F)
    error([ ...
        'Could not resolve the main Global-RBF calibration file for Case %d.\n' ...
        'Expected a non-spatial/main variance-calibration MAT file in:\n%s'], ...
        caseNumber,globalDir);
end

candidateNames = strjoin({F.name},', ');

error([ ...
    'Multiple non-spatial Global-RBF calibration files were found for Case %d:\n%s\n' ...
    'Please keep/rename the authoritative file or add its exact name to ' ...
    'resolveGlobalCalibrationFile().'], ...
    caseNumber,candidateNames);

end


function Rows = addTechnicalRow( ...
    Rows,setting,globalText,localText,globalSource,localSource)

Rows(end+1,:) = { ...
    char(string(setting)), ...
    nonemptyText(globalText), ...
    nonemptyText(localText), ...
    char(string(globalSource)), ...
    char(string(localSource))};

end


function txt = nonemptyText(txt)

txt = char(string(txt));

if isempty(strtrim(txt))
    txt = 'Not stored';
end

end


function txt = compareCaseText(txt1,txt2)

txt1 = nonemptyText(txt1);
txt2 = nonemptyText(txt2);

if strcmp(txt1,txt2)
    txt = txt1;
else
    txt = sprintf('Case 1: %s; Case 2: %s',txt1,txt2);
end

end


function txt = formatField(S,fieldName,formatter)

if isstruct(S) && isfield(S,fieldName)
    txt = formatter(S.(fieldName));
else
    txt = 'Not stored';
end

end


function txt = formatFieldAny(S,fieldNames,formatter)

txt = 'Not stored';

if ~isstruct(S)
    return
end

for i = 1:numel(fieldNames)
    if isfield(S,fieldNames{i})
        txt = formatter(S.(fieldNames{i}));
        return
    end
end

end


function txt = formatNestedField(S,pathParts,formatter)

[tf,v] = getNestedField(S,pathParts);

if tf
    txt = formatter(v);
else
    txt = 'Not stored';
end

end


function [tf,value] = getNestedField(S,pathParts)

tf = true;
value = [];
current = S;

for i = 1:numel(pathParts)

    if ~isstruct(current) || ...
       ~isfield(current,pathParts{i})

        tf = false;
        value = [];
        return
    end

    current = current.(pathParts{i});
end

value = current;

end


function txt = formatBoundsFromSettings(S,minField,maxField,isShape)

if ~isstruct(S) || ...
   ~isfield(S,minField) || ...
   ~isfield(S,maxField)

    txt = 'Not stored';
    return
end

txt = formatBounds(S.(minField),S.(maxField),isShape);

end


function txt = formatBounds(Smin,Smax,isShape)

if ~isstruct(Smin) || ~isstruct(Smax)

    txt = sprintf('min=%s; max=%s',valueText(Smin),valueText(Smax));
    return
end

fields = intersect(fieldnames(Smin),fieldnames(Smax),'stable');
pieces = strings(0,1);

for i = 1:numel(fields)

    f = fields{i};
    lo = Smin.(f);
    hi = Smax.(f);

    if isShape && any(strcmpi(f,{'linear','cubic','thinplate'}))

        pieces(end+1,1) = string(kernelDisplayName(f)) + ...
            ": not applicable"; %#ok<AGROW>

    elseif isnumeric(lo) && isscalar(lo) && ...
           isnumeric(hi) && isscalar(hi)

        pieces(end+1,1) = sprintf( ...
            '%s: %.10g-%.10g', ...
            kernelDisplayName(f),lo,hi); %#ok<AGROW>

    else

        pieces(end+1,1) = string(kernelDisplayName(f)) + ...
            ": " + string(valueText(lo)) + "-" + string(valueText(hi)); %#ok<AGROW>
    end
end

txt = char(strjoin(pieces,'; '));

end


function name = kernelDisplayName(fieldName)

switch lower(char(fieldName))
    case 'gaussian'
        name = 'Gaussian';
    case 'multiquadric'
        name = 'multiquadric';
    case 'linear'
        name = 'linear';
    case 'cubic'
        name = 'cubic';
    case 'thinplate'
        name = 'thin-plate';
    otherwise
        name = char(fieldName);
end

end


function txt = formatRadii(R)

if ~isstruct(R)
    txt = valueText(R);
    return
end

fields = fieldnames(R);
pieces = strings(numel(fields),1);

for i = 1:numel(fields)

    f = fields{i};
    x = R.(f);

    if ~isnumeric(x) || isempty(x)
        pieces(i) = string(kernelDisplayName(f)) + ": " + string(valueText(x));
        continue
    end

    x = x(:);

    if all(abs(x) >= 1000)
        xd = x/1000;
        unit = ' km';
    else
        xd = x;
        unit = '';
    end

    if numel(xd) >= 2
        dx = diff(xd);

        if all(abs(dx-dx(1)) < 1e-10)

            pieces(i) = sprintf( ...
                '%s: %.10g-%.10g%s; increment %.10g%s', ...
                kernelDisplayName(f), ...
                xd(1),xd(end),unit,dx(1),unit);

        else

            pieces(i) = string(kernelDisplayName(f)) + ": " + ...
                string(mat2str(xd.')) + string(unit);
        end

    else

        pieces(i) = sprintf( ...
            '%s: %.10g%s', ...
            kernelDisplayName(f),xd(1),unit);
    end
end

rhs = extractAfter(pieces,": ");

if ~isempty(rhs) && numel(unique(rhs)) == 1
    txt = char(rhs(1));
else
    txt = char(strjoin(pieces,'; '));
end

end


function txt = formatGlobalOptimizationSettings(R)

% GLOBAL RBF:
%   GA is the optimizer.
%   A single fixed K-fold CV partition supplies the RMSE objective.
%   GA variables are represented in log10 parameter space.
%   A logarithmic grid is used only if GA fails.

if isstruct(R) && isfield(R,'Settings')
    S = R.Settings;
else
    S = R;
end

methodText = formatFieldAny(S,{'method','Method'},@valueText);

if ~strcmpi(strtrim(methodText),'ga')
    txt = sprintf('Stored method: %s',methodText);
    return
end

kRequestedText = formatFieldAny( ...
    S, ...
    {'kfolds','k_folds','KFold'}, ...
    @valueText);

if strcmp(kRequestedText,'Not stored')
    kRequestedText = '10';
end

kUsed = [];

if isstruct(R) && isfield(R,'k_used') && isnumeric(R.k_used)
    kUsed = R.k_used(isfinite(R.k_used));
elseif isstruct(R) && isfield(R,'CVInfo') && ...
       isstruct(R.CVInfo) && isfield(R.CVInfo,'k_used') && ...
       isnumeric(R.CVInfo.k_used)
    kUsed = R.CVInfo.k_used(isfinite(R.CVInfo.k_used));
end

if isempty(kUsed)
    cvText = sprintf('%s-fold CV RMSE',kRequestedText);
else
    u = unique(kUsed(:).');
    if numel(u)==1 && strcmp(valueText(u(1)),kRequestedText)
        cvText = sprintf('%s-fold CV RMSE',kRequestedText);
    elseif numel(u)==1
        cvText = sprintf('requested %s-fold CV; effective K=%g; RMSE objective', ...
            kRequestedText,u(1));
    else
        cvText = sprintf('requested %s-fold CV; effective K=%s; RMSE objective', ...
            kRequestedText,mat2str(u));
    end
end

txt = sprintf([ ...
    'GA in log10 parameter space (objective: %s; ' ...
    '10-point-per-free-parameter logarithmic-grid fallback only if GA fails)'], ...
    cvText);

end


function txt = formatLocalOptimizationSettings(S)

% LOCAL RBF:
%   Gaussian / multiquadric: GA in log10(C,RegParam) parameterization;
%   nGrid gives the logarithmic-grid fallback size.
%   Shape-free kernels: direct logarithmic η grid.

if ~isstruct(S)
    txt = 'Not stored';
    return
end

method = [];
grid = [];

if isfield(S,'Method')
    method = S.Method;
elseif isfield(S,'method')
    method = S.method;
end

if isfield(S,'nGrid')
    grid = S.nGrid;
end

if ~isstruct(method)
    txt = formatOptimizationSettings(S,false);
    return
end

gaussianMethod = getStructText(method,'gaussian');
mqMethod = getStructText(method,'multiquadric');
linearMethod = getStructText(method,'linear');
cubicMethod = getStructText(method,'cubic');
thinMethod = getStructText(method,'thinplate');

gaussianGrid = getStructValue(grid,'gaussian',[]);
mqGrid = getStructValue(grid,'multiquadric',[]);
linearGrid = getStructValue(grid,'linear',[]);
cubicGrid = getStructValue(grid,'cubic',[]);
thinGrid = getStructValue(grid,'thinplate',[]);

if strcmpi(gaussianMethod,'ga') && strcmpi(mqMethod,'ga') && ...
   strcmpi(linearMethod,'grid') && strcmpi(cubicMethod,'grid') && ...
   strcmpi(thinMethod,'grid')

    gaGridText = commonGridText(gaussianGrid,mqGrid);
    sfGridText = commonGridText(linearGrid,cubicGrid,thinGrid);

    txt = sprintf([ ...
        'Gaussian/multiquadric: GA in log10(C,η) space, followed by ' ...
        '5x5 local refinement when GA succeeds; logarithmic-grid fallback ' ...
        '%s if GA fails; linear/cubic/thin-plate: direct logarithmic ' ...
        'η grid (%s)'], ...
        gaGridText,sfGridText);
else
    txt = formatOptimizationSettings(S,false);
end

end


function v = getStructValue(S,fieldName,defaultValue)

v = defaultValue;

if isstruct(S) && isfield(S,fieldName)
    v = S.(fieldName);
end

end


function txt = getStructText(S,fieldName)

txt = '';

if isstruct(S) && isfield(S,fieldName)
    txt = valueText(S.(fieldName));
end

end


function txt = commonGridText(varargin)

vals = varargin;

if isempty(vals)
    txt = 'not stored';
    return
end

formatted = strings(numel(vals),1);

for i = 1:numel(vals)
    v = vals{i};

    if isempty(v)
        formatted(i) = "not stored";
    elseif isnumeric(v) && numel(v)==2
        formatted(i) = sprintf('%gx%g',v(1),v(2));
    elseif isnumeric(v) && isscalar(v)
        formatted(i) = sprintf('%g points',v);
    else
        formatted(i) = string(valueText(v));
    end
end

if numel(unique(formatted))==1
    txt = char(formatted(1));
else
    txt = char(strjoin(formatted,', '));
end

end


function txt = formatOptimizationSettings(S,isGlobal)

methodText = formatFieldAny(S,{'Method','method'},@formatMethod);
gridText = formatFieldAny(S,{'nGrid'},@formatGrid);

parts = strings(0,1);

if ~strcmp(methodText,'Not stored')
    parts(end+1,1) = string(methodText); %#ok<AGROW>
end

if ~strcmp(gridText,'Not stored')
    parts(end+1,1) = "grid/fallback: " + string(gridText); %#ok<AGROW>
end

if isGlobal

    kText = formatFieldAny(S,{'kfolds','k_folds','KFold'},@valueText);

    if ~strcmp(kText,'Not stored')
        parts(end+1,1) = string(kText) + "-fold CV"; %#ok<AGROW>
    end
end

if isempty(parts)
    txt = 'Not stored';
else
    txt = char(strjoin(parts,'; '));
end

end


function txt = formatMethod(v)

if isstruct(v)

    f = fieldnames(v);
    pieces = strings(numel(f),1);

    for i = 1:numel(f)
        pieces(i) = string(kernelDisplayName(f{i})) + ": " + ...
            upper(string(valueText(v.(f{i}))));
    end

    txt = char(strjoin(pieces,'; '));

else

    txt = upper(valueText(v));
end

end


function txt = formatGrid(v)

if isstruct(v)

    f = fieldnames(v);
    pieces = strings(numel(f),1);

    for i = 1:numel(f)

        g = v.(f{i});

        if isnumeric(g) && numel(g) == 2
            pieces(i) = sprintf( ...
                '%s: %gx%g', ...
                kernelDisplayName(f{i}),g(1),g(2));
        else
            pieces(i) = string(kernelDisplayName(f{i})) + ": " + ...
                string(valueText(g));
        end
    end

    txt = char(strjoin(pieces,'; '));

else
    txt = valueText(v);
end

end


function txt = formatGASettings(S,functionName)

% Prefer an exact representative GA structure saved in Results.Settings.
G = representativeGAStruct(S);

% If unavailable, parse the active production source without running it.
if isempty(fieldnames(G))

    src = which(functionName);

    if isempty(src) || ~isfile(src)
        txt = 'Not stored';
        return
    end

    code = fileread(src);

    G.PopulationSize = regexNumberAny(code,{ ...
        'popSize\s*=\s*\.\.\.\s*([0-9.eE+\-]+)', ...
        'popSize\s*=\s*([0-9.eE+\-]+)', ...
        '''PopulationSize''\s*,\s*([0-9.eE+\-]+)'});

    G.MaxGenerations = regexNumberAny(code,{ ...
        '''MaxGenerations''\s*,\s*([0-9.eE+\-]+)'});

    G.MaxStallGenerations = regexNumberAny(code,{ ...
        '''MaxStallGenerations''\s*,\s*([0-9.eE+\-]+)'});

    G.CrossoverFraction = regexNumberAny(code,{ ...
        '''CrossoverFraction''\s*,\s*([0-9.eE+\-]+)'});

    G.FunctionTolerance = regexNumberAny(code,{ ...
        '''FunctionTolerance''\s*,\s*([0-9.eE+\-]+)'});

    G.UseParallel = false;
    G.SelectionFcn = 'selectionstochunif';
    G.MutationFcn = 'mutationadaptfeasible';
    G.FitnessScalingFcn = 'fitscalingrank';

    if isfiniteScalar(G.PopulationSize)
        G.EliteCount = max(2,ceil(0.05*G.PopulationSize));
    end
end

txt = formatGAStruct(G);

end


function G = representativeGAStruct(S)

G = struct();

if ~isstruct(S) || ~isfield(S,'GA') || ~isstruct(S.GA)
    return
end

A = S.GA;

% Flat saved GA structure.
if isfield(A,'PopulationSize')
    G = A;
    return
end

% Global batch stores one GA structure per kernel.
f = fieldnames(A);

for i = 1:numel(f)
    candidate = A.(f{i});

    if isstruct(candidate) && isfield(candidate,'PopulationSize')
        G = candidate;
        return
    end
end

end


function txt = formatGAStruct(G)

parts = strings(0,1);

if isfield(G,'PopulationSize') && isfiniteScalar(G.PopulationSize)
    parts(end+1,1) = sprintf('population=%g',G.PopulationSize); %#ok<AGROW>
end

if isfield(G,'EliteCount') && isfiniteScalar(G.EliteCount)
    parts(end+1,1) = sprintf('elite=%g',G.EliteCount); %#ok<AGROW>
elseif isfield(G,'PopulationSize') && isfiniteScalar(G.PopulationSize)
    parts(end+1,1) = sprintf('elite=%g',max(2,ceil(0.05*G.PopulationSize))); %#ok<AGROW>
end

if isfield(G,'MaxGenerations') && isfiniteScalar(G.MaxGenerations)
    parts(end+1,1) = sprintf('max generations=%g',G.MaxGenerations); %#ok<AGROW>
end

if isfield(G,'MaxStallGenerations') && isfiniteScalar(G.MaxStallGenerations)
    parts(end+1,1) = sprintf('max stall=%g',G.MaxStallGenerations); %#ok<AGROW>
end

if isfield(G,'CrossoverFraction') && isfiniteScalar(G.CrossoverFraction)
    parts(end+1,1) = sprintf('crossover=%.10g',G.CrossoverFraction); %#ok<AGROW>
end

if isfield(G,'FunctionTolerance') && isfiniteScalar(G.FunctionTolerance)
    parts(end+1,1) = sprintf('function tolerance=%.10g',G.FunctionTolerance); %#ok<AGROW>
end

if isfield(G,'SelectionFcn')
    parts(end+1,1) = "selection=stochastic uniform"; %#ok<AGROW>
end

if isfield(G,'MutationFcn')
    parts(end+1,1) = "mutation=adaptive feasible"; %#ok<AGROW>
end

if isfield(G,'FitnessScalingFcn')
    parts(end+1,1) = "fitness scaling=rank"; %#ok<AGROW>
end

if isfield(G,'UseParallel') && isscalar(G.UseParallel)
    parts(end+1,1) = "GA parallel=" + string(logicalText(G.UseParallel)); %#ok<AGROW>
end

if isempty(parts)
    txt = 'Not stored';
else
    txt = char(strjoin(parts,'; '));
end

end


function v = getFirstNumericField(S,names)

v = NaN;

if ~isstruct(S)
    return
end

for i = 1:numel(names)

    if isfield(S,names{i})
        candidate = S.(names{i});

        if isnumeric(candidate) && isscalar(candidate) && isfinite(candidate)
            v = candidate;
            return
        end
    end
end

end


function tf = isfiniteScalar(x)

tf = isnumeric(x) && isscalar(x) && isfinite(x);

end


function txt = formatHybridSensitivity(S)

methodText = formatFieldAny(S, ...
    {'TransferMethodByModelKernel','TransferMethod'}, ...
    @formatCellMatrix);

if strcmp(methodText,'Not stored')
    txt = 'Not stored';
    return
end

if ~contains(lower(methodText),'hybrid')
    txt = sprintf('Stored transfer method is %s (hybrid sensitivity settings not present in this source)', ...
        methodText);
    return
end

N = formatFieldAny(S, ...
    {'NIDWSourcesByModelKernel','NIDWSources'}, ...
    @formatNumericMatrix);

threshold = formatFieldAny(S, ...
    {'NearThresholdRawByModelKernel','near_threshold'}, ...
    @formatDistanceMatrix);

powerText = formatFieldAny(S, ...
    {'idw_power'}, ...
    @valueText);

parts = strings(0,1);
parts(end+1,1) = "Hybrid nearest-neighbour/IDW; nearest-only and IDW-only for sensitivity analysis";

if ~strcmp(N,'Not stored')
    parts(end+1,1) = "donors=" + string(N);
end

if ~strcmp(threshold,'Not stored')
    parts(end+1,1) = "near threshold=" + string(threshold);
end

if ~strcmp(powerText,'Not stored')
    parts(end+1,1) = "IDW power=" + string(powerText);
end

txt = char(strjoin(parts,'; '));

end



function txt = formatStageParallelSettings(Sfit,Sboot,Scal)

parts = strings(0,1);

fitWorkers = formatFieldAny(Sfit,{'NumWorkers'},@valueText);
if strcmp(fitWorkers,'Not stored')
    fitWorkers = formatNestedField(Sfit,{'options','execution','numWorkers'},@valueText);
end

bootWorkers = formatFieldAny(Sboot,{'NumWorkers'},@valueText);
if strcmp(bootWorkers,'Not stored')
    bootWorkers = formatNestedField(Sboot,{'options','execution','numWorkers'},@valueText);
end

calWorkers = formatFieldAny(Scal,{'NumWorkers'},@valueText);
if strcmp(calWorkers,'Not stored')
    calWorkers = formatNestedField(Scal,{'options','execution','numWorkers'},@valueText);
end

if ~strcmp(fitWorkers,'Not stored')
    parts(end+1,1) = "optimisation=" + string(fitWorkers) + " workers"; %#ok<AGROW>
end

if ~strcmp(bootWorkers,'Not stored')
    parts(end+1,1) = "bootstrap=" + string(bootWorkers) + " workers"; %#ok<AGROW>
end

if ~strcmp(calWorkers,'Not stored')
    parts(end+1,1) = "calibration=" + string(calWorkers) + " workers"; %#ok<AGROW>
end

innerGA = formatFieldAny(Sfit,{'InnerGAParallel'},@valueText);
if strcmp(innerGA,'Not stored')
    innerGA = formatNestedField(Sfit,{'options','execution','innerGAParallel'},@valueText);
end

if strcmpi(strtrim(innerGA),'false') || strcmpi(strtrim(innerGA),'0')
    parts(end+1,1) = "inner GA serial"; %#ok<AGROW>
elseif ~strcmp(innerGA,'Not stored')
    parts(end+1,1) = "inner GA parallel=" + string(innerGA); %#ok<AGROW>
end

if isempty(parts)
    txt = 'Not stored';
else
    txt = char(strjoin(parts,'; '));
end

end


function txt = formatBootstrapCompact(S,verifiedFallbackAlgorithm)

if nargin < 2
    verifiedFallbackAlgorithm = '';
end

parts = strings(0,1);

B = formatFieldAny(S,{'nBootstrap'},@valueText);

if ~strcmp(B,'Not stored')
    parts(end+1,1) = "B=" + string(B); %#ok<AGROW>
end

trainSeed = formatFieldAny(S, ...
    {'RNG_SEED_Train','RNG_SEED'}, ...
    @valueText);

if ~strcmp(trainSeed,'Not stored')
    parts(end+1,1) = "base training seed=" + string(trainSeed); %#ok<AGROW>
end

testSeed = formatFieldAny(S, ...
    {'RNG_SEED_Test'}, ...
    @valueText);

if ~strcmp(testSeed,'Not stored')
    parts(end+1,1) = "base test seed=" + string(testSeed); %#ok<AGROW>
end

algorithm = formatFieldAny(S, ...
    {'RNGAlgorithm'}, ...
    @valueText);

if ~strcmp(algorithm,'Not stored')
    if strcmpi(strtrim(algorithm),'mt19937ar') || strcmpi(strtrim(algorithm),'twister')
        algorithm = 'MT19937';
    end
    parts(end+1,1) = "RNG=" + string(algorithm); %#ok<AGROW>
elseif ~isempty(verifiedFallbackAlgorithm)
    % Verified directly from the corresponding production bootstrap source.
    parts(end+1,1) = "RNG=" + string(verifiedFallbackAlgorithm); %#ok<AGROW>
end

if isempty(parts)
    txt = 'Not stored';
else
    txt = char(strjoin(parts,'; '));
end

end


function txt = formatCalibrationBounds(S)

alpha = formatFieldAny(S,{'alpha_bounds'},@numericVectorText);
beta = formatFieldAny(S,{'beta_bounds'},@numericVectorText);

parts = strings(0,1);

if ~strcmp(alpha,'Not stored')
    parts(end+1,1) = "alpha in " + string(alpha); %#ok<AGROW>
end

if ~strcmp(beta,'Not stored')
    parts(end+1,1) = "beta in " + string(beta); %#ok<AGROW>
end

if isempty(parts)
    txt = 'Not stored';
else
    txt = char(strjoin(parts,'; '));
end

end


function txt = formatCalibrationGA(S)

parts = strings(0,1);

pop = formatFieldAny(S,{'pop_size','PopulationSize'},@valueText);
gen = formatFieldAny(S,{'max_gen','MaxGenerations'},@valueText);
g1 = formatFieldAny(S,{'gamma1'},@valueText);
g2 = formatFieldAny(S,{'gamma2'},@valueText);
tol = formatFieldAny(S,{'boundary_tol'},@valueText);
seed = formatFieldAny(S,{'RNG_SEED'},@valueText);

% common.calibrateBootstrapVariance defaults RNG_SEED to 1 when the
% caller does not provide it. Current Global/Local calibration runners
% rely on that default.
if strcmp(seed,'Not stored')
    seed = '1';
end

if ~strcmp(pop,'Not stored')
    parts(end+1,1) = "population=" + string(pop); %#ok<AGROW>

    popNum = str2double(pop);
    if isfinite(popNum)
        parts(end+1,1) = "elite=" + string(max(2,ceil(0.05*popNum))); %#ok<AGROW>
    end
end

if ~strcmp(gen,'Not stored')
    parts(end+1,1) = "max generations=" + string(gen); %#ok<AGROW>
end

% Fixed by common.calibrateBootstrapVariance.
parts(end+1,1) = "max stall=30"; %#ok<AGROW>
parts(end+1,1) = "crossover=0.8"; %#ok<AGROW>
parts(end+1,1) = "function tolerance=1e-8"; %#ok<AGROW>
parts(end+1,1) = "selection=stochastic uniform"; %#ok<AGROW>
parts(end+1,1) = "mutation=adaptive feasible"; %#ok<AGROW>
parts(end+1,1) = "fitness scaling=rank"; %#ok<AGROW>
parts(end+1,1) = "GA parallel=false"; %#ok<AGROW>
parts(end+1,1) = "RNG seed=" + string(seed); %#ok<AGROW>

if ~strcmp(g1,'Not stored') || ~strcmp(g2,'Not stored')
    parts(end+1,1) = "gamma1=" + string(g1) + ", gamma2=" + string(g2); %#ok<AGROW>
end

if ~strcmp(tol,'Not stored')
    parts(end+1,1) = "boundary tolerance=" + string(tol); %#ok<AGROW>
end

txt = char(strjoin(parts,'; '));

end


function txt = numericVectorText(v)

if isnumeric(v)
    txt = mat2str(v,10);
else
    txt = valueText(v);
end

end


function txt = formatNumericMatrix(x)

if isempty(x)
    txt = 'Not stored';
elseif isnumeric(x) || islogical(x)

    if all(x(:) == x(1))
        txt = valueText(x(1));
    else
        txt = mat2str(x,10);
    end
else
    txt = valueText(x);
end

end


function txt = formatDistanceMatrix(x)

if isempty(x)
    txt = 'Not stored';
    return
end

if isnumeric(x) && ...
   all(isfinite(x(:))) && ...
   all(abs(x(:)) >= 1000)

    xkm = x/1000;

    if all(xkm(:) == xkm(1))
        txt = sprintf('%.10g km',xkm(1));
    else
        txt = [mat2str(xkm,10) ' km'];
    end
else
    txt = formatNumericMatrix(x);
end

end


function txt = formatCellMatrix(x)

if isstring(x)
    values = x(:);

elseif iscell(x)

    try
        values = string(x(:));
    catch
        txt = valueText(x);
        return
    end

else
    txt = valueText(x);
    return
end

if isempty(values)
    txt = 'Not stored';
elseif numel(unique(values)) == 1
    txt = char(values(1));
else
    txt = char(strjoin(unique(values,'stable'),', '));
end

end


function value = regexNumberAny(textValue,expressions)

value = NaN;

for i = 1:numel(expressions)

    token = regexp( ...
        textValue, ...
        expressions{i}, ...
        'tokens', ...
        'once');

    if ~isempty(token)

        candidate = str2double(token{1});

        if isfinite(candidate)
            value = candidate;
            return
        end
    end
end

end


function txt = logicalText(x)

if islogical(x) && isscalar(x)

    if x
        txt = 'true';
    else
        txt = 'false';
    end

elseif isnumeric(x) && isscalar(x) && any(x == [0 1])

    if logical(x)
        txt = 'true';
    else
        txt = 'false';
    end

else
    txt = valueText(x);
end

end


function txt = valueText(v)

if isempty(v)

    txt = 'Not stored';

elseif ischar(v)

    txt = v;

elseif isstring(v)

    txt = char(strjoin(v(:),', '));

elseif islogical(v)

    if isscalar(v)
        txt = logicalText(v);
    else
        txt = mat2str(v);
    end

elseif isnumeric(v)

    if isscalar(v)
        txt = sprintf('%.10g',v);
    else
        txt = mat2str(v,10);
    end

elseif iscell(v)

    try
        txt = char(strjoin(string(v(:)),', '));
    catch
        txt = sprintf('[cell %s]',mat2str(size(v)));
    end

elseif isstruct(v)

    f = fieldnames(v);
    pieces = strings(numel(f),1);

    for i = 1:numel(f)
        pieces(i) = string(f{i}) + ": " + string(valueText(v.(f{i})));
    end

    txt = char(strjoin(pieces,'; '));

else
    txt = sprintf('[%s]',class(v));
end

end


function writeLatexTable(T,fileName,captionText,labelText)

fid = fopen(fileName,'w');

if fid < 0
    error('Could not create LaTeX file:\n%s',fileName);
end

cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid,'%% Auto-generated by Collect_Model1_Technical_Setup_v17.m\n');
fprintf(fid,'%% Required packages:\n');
fprintf(fid,'%% \\usepackage{booktabs}\n');
fprintf(fid,'%% \\usepackage{tabularx}\n');
fprintf(fid,'%% \\usepackage{array}\n\n');

fprintf(fid,'\\begin{table}[t]\n');
fprintf(fid,'\\centering\n');
fprintf(fid,'\\caption{%s}\n',latexEscape(captionText));
fprintf(fid,'\\label{%s}\n',labelText);
fprintf(fid,'\\small\n');
fprintf(fid,'\\setlength{\\tabcolsep}{4pt}\n');
fprintf(fid,'\\renewcommand{\\arraystretch}{1.12}\n');
fprintf(fid,['\\begin{tabularx}{\\linewidth}{' ...
    '>{\\raggedright\\arraybackslash}p{0.22\\linewidth}' ...
    '>{\\raggedright\\arraybackslash}X' ...
    '>{\\raggedright\\arraybackslash}X}\n']);

fprintf(fid,'\\toprule\n');
fprintf(fid,'\\textbf{Computational setting} & \\textbf{Global RBF} & \\textbf{Local RBF} \\\\\n');
fprintf(fid,'\\midrule\n');

for i = 1:height(T)

    a = latexEscape(T.Computational_Setting{i});
    b = latexEscape(T.Global_RBF{i});
    c = latexEscape(T.Local_RBF{i});

    fprintf(fid,'%s & %s & %s \\\\\n',a,b,c);
end

fprintf(fid,'\\bottomrule\n');
fprintf(fid,'\\end{tabularx}\n');
fprintf(fid,'\\end{table}\n');

end


function txt = latexEscape(value)

txt = char(string(value));

% Preserve manuscript math symbols through escaping.
etaToken = '@@ETA@@';
txt = strrep(txt,'η',etaToken);

% Escape backslash first using a temporary token.
token = '@@BACKSLASH@@';
txt = strrep(txt,'\',token);

txt = strrep(txt,'&','\&');
txt = strrep(txt,'%','\%');
txt = strrep(txt,'$','\$');
txt = strrep(txt,'#','\#');
txt = strrep(txt,'_','\_');
txt = strrep(txt,'{','\{');
txt = strrep(txt,'}','\}');
txt = strrep(txt,'~','\textasciitilde{}');
txt = strrep(txt,'^','\textasciicircum{}');

txt = strrep(txt,token,'\textbackslash{}');
txt = strrep(txt,etaToken,'$\eta$');

% Convert common multiplication symbol for robust pdfLaTeX output.
txt = strrep(txt,'×','$\times$');

end

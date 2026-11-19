function Table_Final_Hyperparameters_C2()
% =========================================================================
% Table_Final_Hyperparameters_C2
%
% PURPOSE
%   Create ONE compact manuscript table containing:
%
%     Panel A - Case 1
%     Panel B - Case 2
%
%   Columns:
%     Parameter / kernel
%     EGM2008
%     GGMPlus
%     EIGEN6C4
%     GECO
%     SGG-UGM-2
%     XGM2019e_2159
%
%   Each panel has exactly 14 DATA rows:
%     Global C       : Gaussian, Multiquadric                  = 2
%     Local C        : Gaussian, Multiquadric                  = 2
%     Global eta: Gaussian, Multiquadric, Linear,
%                      Cubic, TPS                              = 5
%     Local eta : Gaussian, Multiquadric, Linear,
%                      Cubic, TPS                              = 5
%                                                           --------
%                                                              14
%
% SCIENTIFIC DEFINITION
%   Global C / Global eta:
%       exact optimized scalar for each EGM x kernel.
%
%   Local C / Local eta:
%       median [Q1-Q3] of the FINAL pointwise TRAINING settings:
%           Stage2Core.Parameters.Train.C
%           Stage2Core.Parameters.Train.Reg
%
%   Linear / Cubic / TPS have no C parameter, so they are not included
%   in the C sections.
%
% OUTPUT
%   <RootDir>\OutPut_RBF_2026_2\ComputationalSettings\
%       Model1_Final_Hyperparameters_OneTable.xlsx
%       Model1_Final_Hyperparameters_OneTable.tex
%
% IMPORTANT
%   - Reads saved results only.
%   - Does NOT rerun optimization, fitting, transfer, bootstrap, or calibration.
%   - SGG-UGM-1 is intentionally excluded.
%   - Main Hybrid no-suffix Local-RBF files are used; IDW/Nearest
%     sensitivity files are never auto-selected.
%
% RUN
%   Collect_Final_Hyperparameters_OneTable_Panels_v7
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
    fullfile(LocalDir,'ComputationalSettings');

if ~isfolder(OutputDir)
    mkdir(OutputDir);
end


% ------------------------------------------------------------------------
% 1.1 GLOBAL FINAL FIT FILES
% ------------------------------------------------------------------------

GlobalCase1File = ...
    fullfile(GlobalDir,'GlobalRBF_Case1_Full_TuneReg_4.mat');

GlobalCase2File = ...
    fullfile(GlobalDir,'GlobalRBF_Case2_Full_TuneReg_4.mat');


% ------------------------------------------------------------------------
% 1.2 LOCAL FINAL RESULT FILES
%
% IMPORTANT:
%   The authoritative MAIN Local-RBF fit used by the final calibration is
%   the no-suffix file:
%
%       LocalRBF_Case1_Full_TuneReg
%       LocalRBF_Case2_Full_TuneReg
%
%   The helper resolveMatFile() accepts the exact base name above and uses
%   the corresponding .mat file when MATLAB stores it with the .mat suffix.
%
%   The *_IDW.mat and *_Nearest.mat files are sensitivity-analysis outputs
%   and MUST NOT be used for this main manuscript table.
% ------------------------------------------------------------------------

LocalCase1File = ...
    resolveMatFile( ...
        LocalDir, ...
        'LocalRBF_Case1_Full_TuneReg');

LocalCase2File = ...
    resolveMatFile( ...
        LocalDir, ...
        'LocalRBF_Case2_Full_TuneReg');


% ------------------------------------------------------------------------
% 1.3 EXACT EGM ORDER FOR THE MANUSCRIPT TABLE
%
% SGG-UGM-1 is intentionally NOT included.
% Canonical matching ignores spaces, "-", "_" and case.
% IMPORTANT: the production result files use short stored identifiers
% such as 'sgg2' and 'xgm19'. The table displays the manuscript labels
% 'SGG-UGM-2' and 'XGM2019e_2159' without requiring those long strings
% to exist literally in Results.ListOfModel.
% ------------------------------------------------------------------------

% Stored result names used by the current production files.
ModelsStored = { ...
    'EGM2008'
    'ggmplus'
    'eigen6c4'
    'geco'
    'sgg2'
    'xgm19'};

% Manuscript/display labels.
ModelsDisplay = { ...
    'EGM2008'
    'GGMPlus'
    'EIGEN6C4'
    'GECO'
    'SGG-UGM-2'
    'XGM2019e_2159'};


% ------------------------------------------------------------------------
% 1.4 OUTPUT FILES
% ------------------------------------------------------------------------

ExcelFile = ...
    fullfile( ...
        OutputDir, ...
        'Model1_Final_Hyperparameters_OneTable.xlsx');

LatexFile = ...
    fullfile( ...
        OutputDir, ...
        'Model1_Final_Hyperparameters_OneTable.tex');


% ------------------------------------------------------------------------
% 1.5 DISPLAY PRECISION
% ------------------------------------------------------------------------

N_SIGNIFICANT_DIGITS = ...
    3;


%% ========================================================================
% 2. RESOLVE INPUT FILES
% ========================================================================

mustExist(GlobalCase1File,'Global Case 1 result');
mustExist(GlobalCase2File,'Global Case 2 result');

LocalCase1File = ...
    resolveUserPath( ...
        LocalCase1File, ...
        RootDir, ...
        LocalDir);

LocalCase2File = ...
    resolveUserPath( ...
        LocalCase2File, ...
        RootDir, ...
        LocalDir);

mustExist( ...
    LocalCase1File, ...
    'MAIN Hybrid Local Case 1 result');

mustExist( ...
    LocalCase2File, ...
    'MAIN Hybrid Local Case 2 result');


%% ========================================================================
% 3. LOAD AUTHORITATIVE RESULTS
% ========================================================================

G1 = loadResults(GlobalCase1File);
G2 = loadResults(GlobalCase2File);

L1 = loadResults(LocalCase1File);
L2 = loadResults(LocalCase2File);

validateGlobalResult(G1,'Global Case 1');
validateGlobalResult(G2,'Global Case 2');

validateLocalResult(L1,'Local Case 1');
validateLocalResult(L2,'Local Case 2');


%% ========================================================================
% 4. VALIDATE THE SIX REQUESTED EGMs AND FIVE KERNELS
% ========================================================================

for i = 1:numel(ModelsStored)

    modelName = ModelsStored{i};

    requireModel(G1,modelName,'Global Case 1');
    requireModel(G2,modelName,'Global Case 2');
    requireModel(L1,modelName,'Local Case 1');
    requireModel(L2,modelName,'Local Case 2');

end

KernelOrder = { ...
    'gaussian'
    'multiquadric'
    'linear'
    'cubic'
    'thinplate'};

for i = 1:numel(KernelOrder)

    kernel = KernelOrder{i};

    requireKernel(G1,kernel,'Global Case 1');
    requireKernel(G2,kernel,'Global Case 2');
    requireKernel(L1,kernel,'Local Case 1');
    requireKernel(L2,kernel,'Local Case 2');

end


%% ========================================================================
% 5. BUILD PANEL A - CASE 1
% ========================================================================

PanelA = buildCasePanel( ...
    'Case 1', ...
    G1, ...
    L1, ...
    LocalCase1File, ...
    RootDir, ...
    LocalDir, ...
    ModelsStored, ...
    N_SIGNIFICANT_DIGITS);


%% ========================================================================
% 6. BUILD PANEL B - CASE 2
% ========================================================================

PanelB = buildCasePanel( ...
    'Case 2', ...
    G2, ...
    L2, ...
    LocalCase2File, ...
    RootDir, ...
    LocalDir, ...
    ModelsStored, ...
    N_SIGNIFICANT_DIGITS);


%% ========================================================================
% 7. EXPORT EXCEL
% ========================================================================

if isfile(ExcelFile)
    delete(ExcelFile);
end

captionText = [ ...
    'Final Global and Local RBF hyperparameters. C and eta denote the shape coefficient and regularisation parameter; ' ...
    'G, M, T, C, and L denote Gaussian, Multiquadric, thin-plate spline, Cubic, and Linear kernels. ' ...
    'Global entries are exact optimized values; Local entries are median [Q1-Q3] of the final pointwise training settings.'];

header = [ ...
    {'Setting'}, ...
    ModelsDisplay(:).'];


% Main one-table sheet.
writecell( ...
    {captionText}, ...
    ExcelFile, ...
    'Sheet','Final Hyperparameters', ...
    'Range','A1');

writecell( ...
    header, ...
    ExcelFile, ...
    'Sheet','Final Hyperparameters', ...
    'Range','A3');

writecell( ...
    [{'Panel A - Case 1'},repmat({''},1,numel(ModelsDisplay))], ...
    ExcelFile, ...
    'Sheet','Final Hyperparameters', ...
    'Range','A4');

writecell( ...
    PanelA, ...
    ExcelFile, ...
    'Sheet','Final Hyperparameters', ...
    'Range','A5');

panelBTitleRow = ...
    5 + size(PanelA,1);

writecell( ...
    [{'Panel B - Case 2'},repmat({''},1,numel(ModelsDisplay))], ...
    ExcelFile, ...
    'Sheet','Final Hyperparameters', ...
    'Range',sprintf('A%d',panelBTitleRow));

writecell( ...
    PanelB, ...
    ExcelFile, ...
    'Sheet','Final Hyperparameters', ...
    'Range',sprintf('A%d',panelBTitleRow+1));


% Audit sheet.
Audit = { ...
    'Item','Value'
    'Global Case 1',GlobalCase1File
    'Global Case 2',GlobalCase2File
    'Local Case 1',LocalCase1File
    'Local Case 2',LocalCase2File
    'Stored EGM names',strjoin(ModelsStored,', ')
    'Displayed EGM names',strjoin(ModelsDisplay,', ')
    'Excluded EGM','SGG-UGM-1'
    'Global summary','Exact optimized scalar'
    'Local summary','Median [Q1-Q3] of final pointwise training settings'
    'Local C source','Stage2Core.Parameters.Train.C'
    'Local eta source','Stage2Core.Parameters.Train.Reg'};

writecell( ...
    Audit, ...
    ExcelFile, ...
    'Sheet','Audit', ...
    'Range','A1');


%% ========================================================================
% 8. EXPORT LATEX
% ========================================================================

writeLatexOneTable( ...
    PanelA, ...
    PanelB, ...
    ModelsDisplay, ...
    LatexFile);


%% ========================================================================
% 9. FINISHED
% ========================================================================

fprintf('\n============================================================\n');
fprintf(' FINAL HYPERPARAMETER ONE-TABLE OUTPUT CREATED\n');
fprintf('============================================================\n');
fprintf('Excel : %s\n',ExcelFile);
fprintf('LaTeX : %s\n',LatexFile);
fprintf('Columns        : 7\n');
fprintf('Data rows      : 28 total = 14 Case 1 + 14 Case 2\n');
fprintf('Panel-title rows: 2\n');
fprintf('============================================================\n\n');

end


%% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function Panel = buildCasePanel( ...
    caseLabel,G,L,localResultFile,RootDir,LocalDir,ModelsStored,nSig)

% Exactly 14 DATA rows.
RowDefs = { ...
    'Global C - G',   'globalC',   'gaussian'
    'Global C - M',   'globalC',   'multiquadric'
    'Local C - G',    'localC',    'gaussian'
    'Local C - M',    'localC',    'multiquadric'
    'Global eta - G', 'globalReg', 'gaussian'
    'Global eta - M', 'globalReg', 'multiquadric'
    'Global eta - L', 'globalReg', 'linear'
    'Global eta - C', 'globalReg', 'cubic'
    'Global eta - T', 'globalReg', 'thinplate'
    'Local eta - G',  'localReg',  'gaussian'
    'Local eta - M',  'localReg',  'multiquadric'
    'Local eta - L',  'localReg',  'linear'
    'Local eta - C',  'localReg',  'cubic'
    'Local eta - T',  'localReg',  'thinplate'};

nRows = size(RowDefs,1);
nModels = numel(ModelsStored);

Panel = cell(nRows,nModels+1);

for r = 1:nRows

    rowLabel = RowDefs{r,1};
    quantity = RowDefs{r,2};
    kernel   = RowDefs{r,3};

    Panel{r,1} = rowLabel;

    for im = 1:nModels

        modelName = ModelsStored{im};

        switch quantity

            case 'globalC'

                value = getGlobalScalar( ...
                    G,modelName,kernel,'C_global');

                Panel{r,im+1} = ...
                    formatScalarExcel(value,nSig);

            case 'globalReg'

                value = getGlobalScalar( ...
                    G,modelName,kernel,'reg_param');

                Panel{r,im+1} = ...
                    formatScalarExcel(value,nSig);

            case {'localC','localReg'}

                Core = getLocalStage2Core( ...
                    L, ...
                    localResultFile, ...
                    RootDir, ...
                    LocalDir, ...
                    modelName, ...
                    kernel);

                requireCoreParameterBlock( ...
                    Core, ...
                    modelName, ...
                    kernel);

                if strcmp(quantity,'localC')

                    x = Core.Parameters.Train.C(:);

                else

                    x = Core.Parameters.Train.Reg(:);

                end

                Panel{r,im+1} = ...
                    formatMedianIQRExcel(x,nSig);

            otherwise

                error( ...
                    'Unknown row quantity "%s" in %s.', ...
                    quantity,caseLabel);

        end

    end

end

end


function R = loadResults(fileName)

info = whos('-file',fileName);
names = {info.name};

if ~any(strcmp(names,'Results'))
    error('Variable "Results" was not found in:\n%s',fileName);
end

S = load(fileName,'Results');
R = S.Results;

end


function validateGlobalResult(R,label)

required = { ...
    'ListOfModel', ...
    'NameFunction_RBF', ...
    'C_global', ...
    'reg_param'};

for i = 1:numel(required)

    if ~isstruct(R) || ~isfield(R,required{i})
        error('%s is missing Results.%s.',label,required{i});
    end

end

nModel  = numel(R.ListOfModel);
nKernel = numel(R.NameFunction_RBF);

if ~isequal(size(R.C_global),[nModel,nKernel])
    error('%s: Results.C_global has inconsistent dimensions.',label);
end

if ~isequal(size(R.reg_param),[nModel,nKernel])
    error('%s: Results.reg_param has inconsistent dimensions.',label);
end

end


function validateLocalResult(R,label)

required = { ...
    'ListOfModel', ...
    'NameFunction_RBF', ...
    'Run'};

for i = 1:numel(required)

    if ~isstruct(R) || ~isfield(R,required{i})
        error('%s is missing Results.%s.',label,required{i});
    end

end

if ~isstruct(R.Run) || isempty(fieldnames(R.Run))
    error('%s: Results.Run is empty.',label);
end

end


function requireModel(R,modelName,label)

names = cellstr(string(R.ListOfModel(:)));

if isempty(findNameIndex(names,modelName,false))
    error('%s does not contain model "%s".',label,modelName);
end

end


function requireKernel(R,kernelName,label)

names = cellstr(string(R.NameFunction_RBF(:)));

if isempty(findNameIndex(names,kernelName,true))
    error('%s does not contain kernel "%s".',label,kernelName);
end

end


function value = getGlobalScalar(R,modelName,kernelName,fieldName)

models = cellstr(string(R.ListOfModel(:)));
kernels = cellstr(string(R.NameFunction_RBF(:)));

im = findNameIndex(models,modelName,false);
ik = findNameIndex(kernels,kernelName,true);

if isempty(im)
    error('Global model "%s" was not found.',modelName);
end

if isempty(ik)
    error('Global kernel "%s" was not found.',kernelName);
end

M = R.(fieldName);
value = M(im,ik);

if ~(isnumeric(value) && isscalar(value) && isfinite(value))
    error( ...
        'Global %s for %s / %s is not a finite numeric scalar.', ...
        fieldName,modelName,kernelName);
end

end


function Core = getLocalStage2Core( ...
    R,localResultFile,RootDir,LocalDir,modelName,kernelName)

runNames = fieldnames(R.Run);
matchName = '';

for i = 1:numel(runNames)

    runField = runNames{i};
    Z = R.Run.(runField);

    if ~isstruct(Z)
        continue
    end

    runModel = getTextFieldAny(Z,{'model','Model'});
    runKernel = getTextFieldAny(Z,{'kernel','Kernel'});

    if isempty(runModel) || isempty(runKernel)
        continue
    end

    if strcmpi( ...
            canonicalName(runModel,false), ...
            canonicalName(modelName,false)) && ...
       strcmpi( ...
            canonicalName(runKernel,true), ...
            canonicalName(kernelName,true))

        matchName = runField;
        break

    end

end

if isempty(matchName)

    error( ...
        ['Could not find Local Results.Run entry for:\n' ...
         '  Model  = %s\n' ...
         '  Kernel = %s'], ...
        modelName,kernelName);

end

Z = R.Run.(matchName);


% Preferred: Stage2Core embedded in Results.Run.
if isfield(Z,'Stage2Core') && ...
   isstruct(Z.Stage2Core) && ...
   ~isempty(fieldnames(Z.Stage2Core))

    Core = Z.Stage2Core;
    return

end


% Normal production mode: Stage2Core saved in checkpoint file.
storedCorePath = '';

if isfield(Z,'Stage2Paths') && ...
   isstruct(Z.Stage2Paths) && ...
   isfield(Z.Stage2Paths,'core')

    storedCorePath = char(string(Z.Stage2Paths.core));

end

if isempty(strtrim(storedCorePath))

    error( ...
        ['Local run %s (%s / %s) contains neither embedded Stage2Core ' ...
         'nor Stage2Paths.core.'], ...
        matchName,modelName,kernelName);

end

coreFile = resolveStoredCorePath( ...
    storedCorePath, ...
    localResultFile, ...
    RootDir, ...
    LocalDir);

mustExist( ...
    coreFile, ...
    sprintf('Stage2 CORE for %s / %s',modelName,kernelName));

info = whos('-file',coreFile);
names = {info.name};

if ~any(strcmp(names,'Stage2Core'))
    error('Variable "Stage2Core" was not found in:\n%s',coreFile);
end

S = load(coreFile,'Stage2Core');
Core = S.Stage2Core;

end


function requireCoreParameterBlock(Core,modelName,kernelName)

if ~isstruct(Core) || ...
   ~isfield(Core,'Parameters') || ...
   ~isstruct(Core.Parameters) || ...
   ~isfield(Core.Parameters,'Train') || ...
   ~isstruct(Core.Parameters.Train)

    error( ...
        'Stage2Core.Parameters.Train is missing for %s / %s.', ...
        modelName,kernelName);

end

if ~isfield(Core.Parameters.Train,'C')
    error( ...
        'Stage2Core.Parameters.Train.C is missing for %s / %s.', ...
        modelName,kernelName);
end

if ~isfield(Core.Parameters.Train,'Reg')
    error( ...
        'Stage2Core.Parameters.Train.Reg is missing for %s / %s.', ...
        modelName,kernelName);
end

end


function txt = formatMedianIQRExcel(x,nSig)

x = x(:);
x = x(isfinite(x));

if isempty(x)
    txt = 'NA';
    return
end

q1  = quantileLinear(x,0.25);
med = quantileLinear(x,0.50);
q3  = quantileLinear(x,0.75);

txt = sprintf( ...
    '%s [%s-%s]', ...
    formatNumberPlain(med,nSig), ...
    formatNumberPlain(q1,nSig), ...
    formatNumberPlain(q3,nSig));

end


function q = quantileLinear(x,p)
% Toolbox-independent linearly interpolated sample quantile.

x = sort(x(:));
n = numel(x);

if n == 0
    q = NaN;
    return
end

if n == 1
    q = x(1);
    return
end

h = 1 + (n-1)*p;
lo = floor(h);
hi = ceil(h);

if lo == hi

    q = x(lo);

else

    w = h - lo;
    q = x(lo) + w*(x(hi)-x(lo));

end

end


function txt = formatScalarExcel(x,nSig)

if ~(isnumeric(x) && isscalar(x) && isfinite(x))
    txt = 'NA';
    return
end

txt = formatNumberPlain(x,nSig);

end


function txt = formatNumberPlain(x,nSig)

fmt = sprintf('%%.%dg',nSig);
txt = sprintf(fmt,x);

end


function writeLatexOneTable(PanelA,PanelB,ModelsDisplay,fileName)

fid = fopen(fileName,'w');

if fid < 0
    error('Could not create LaTeX file:\n%s',fileName);
end

cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid,'%% Auto-generated by Collect_Final_Hyperparameters_OneTable_Panels_v7.m\n');
fprintf(fid,'%% Required packages:\n');
fprintf(fid,'%% \\usepackage{booktabs}\n');
fprintf(fid,'%% \\usepackage{tabularx}\n');
fprintf(fid,'%% \\usepackage{array}\n\n');

fprintf(fid,'\\begin{table}[t]\n');
fprintf(fid,'\\centering\n');
fprintf(fid,['\\caption{Final Global and Local RBF hyperparameters. ' ...
    'Global entries are optimized scalar values; Local entries are median ' ...
    '[Q1--Q3] of the final pointwise training settings.}\n']);
fprintf(fid,'\\label{tab:final_rbf_hyperparameters}\n');
fprintf(fid,'\\scriptsize\n');
fprintf(fid,'\\setlength{\\tabcolsep}{2.3pt}\n');
fprintf(fid,'\\renewcommand{\\arraystretch}{0.94}\n');

fprintf(fid,['\\begin{tabularx}{\\linewidth}{' ...
    '>{\\raggedright\\arraybackslash}p{0.18\\linewidth}' ...
    '*{6}{>{\\centering\\arraybackslash}X}}\n']);

fprintf(fid,'\\toprule\n');

ModelsLatex = {'E08','GGP','EC4','GCO','SG2','X19'};
fprintf(fid,'\\textbf{Setting}');
for i = 1:numel(ModelsLatex)
    fprintf(fid,' & \\textbf{%s}',ModelsLatex{i});
end
fprintf(fid,' \\\\\n');

fprintf(fid,'\\midrule\n');

writePanelLatex(fid,'Panel A -- Case 1',PanelA);
fprintf(fid,'\\midrule\n');
writePanelLatex(fid,'Panel B -- Case 2',PanelB);

fprintf(fid,'\\bottomrule\n');
fprintf(fid,'\\end{tabularx}\n');
fprintf(fid,'\\vspace{2pt}\n');
fprintf(fid,['\\parbox{\\linewidth}{\\scriptsize \\textit{Note:} ' ...
    '$C$ and $\\eta$ denote the shape coefficient and regularisation parameter; ' ...
    'subscripts ``glob'' and ``loc'' denote the Global- and Local-RBF formulations. ' ...
    'G, M, T, C, and L denote Gaussian, Multiquadric, thin-plate spline, Cubic, and Linear kernels, respectively. ' ...
    'E08, GGP, EC4, GCO, SG2, and X19 denote EGM2008, GGMPlus, EIGEN-6C4, GECO, SGG-UGM-2, and XGM2019e\\_2159, respectively. ' ...
    'The $\\eta_{\\mathrm{loc}}$--M row is reported in units of $10^{-6}$.}\n']);
fprintf(fid,'\\end{table}\n');

end


function writePanelLatex(fid,panelTitle,Panel)

nCols = size(Panel,2);

fprintf( ...
    fid, ...
    '\\multicolumn{%d}{l}{\\textbf{%s}} \\\\\n', ...
    nCols, ...
    latexEscape(panelTitle));

for r = 1:size(Panel,1)

    fprintf(fid,'%s',rowLabelToLatex(Panel{r,1}));

    for c = 2:nCols

        value = Panel{r,c};

        if strcmp(Panel{r,1},'Local eta - M')
            txt = summaryCellToLatexScaled(value,1e6);
        elseif startsWith(Panel{r,1},'Local ')
            txt = summaryCellToLatex(value);
        else
            txt = valueCellToLatex(value);
        end

        fprintf(fid,' & %s',txt);

    end

    fprintf(fid,' \\\\\n');

end

end



function txt = rowLabelToLatex(value)

s = char(string(value));

% Compact manuscript notation; kernel abbreviations are defined in the caption.
if startsWith(s,'Global C - ')
    k = strtrim(extractAfter(s,'Global C - '));
    txt = sprintf('$C_{\\mathrm{glob}}$--%s',latexEscape(k));
elseif startsWith(s,'Local C - ')
    k = strtrim(extractAfter(s,'Local C - '));
    txt = sprintf('$C_{\\mathrm{loc}}$--%s',latexEscape(k));
elseif startsWith(s,'Global eta - ')
    k = strtrim(extractAfter(s,'Global eta - '));
    txt = sprintf('$\\eta_{\\mathrm{glob}}$--%s',latexEscape(k));
elseif strcmp(s,'Local eta - M')
    txt = '$\\eta_{\\mathrm{loc}}$--M $(10^{-6})$';
elseif startsWith(s,'Local eta - ')
    k = strtrim(extractAfter(s,'Local eta - '));
    txt = sprintf('$\\eta_{\\mathrm{loc}}$--%s',latexEscape(k));
else
    txt = latexEscape(s);
end

end

function txt = summaryCellToLatexScaled(value,scaleFactor)

s = strtrim(char(string(value)));

if strcmpi(s,'NA')
    txt = 'NA';
    return
end

numPat = ...
    '([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)';

expr = ...
    ['^\s*' numPat ...
     '\s*\[\s*' numPat ...
     '\s*-\s*' numPat ...
     '\s*\]\s*$'];

tok = regexp(s,expr,'tokens','once');

if isempty(tok)
    txt = latexEscape(s);
    return
end

med = str2double(strtrim(tok{1})) * scaleFactor;
q1  = str2double(strtrim(tok{2})) * scaleFactor;
q3  = str2double(strtrim(tok{3})) * scaleFactor;

if all(isfinite([med q1 q3]))
    txt = sprintf( ...
        '$%s$ [$%s$--$%s$]', ...
        latexNumber(med), ...
        latexNumber(q1), ...
        latexNumber(q3));
else
    txt = latexEscape(s);
end

end


function txt = valueCellToLatex(value)

s = strtrim(char(string(value)));

if strcmpi(s,'NA')
    txt = 'NA';
    return
end

x = str2double(s);

if isfinite(x)
    txt = ['$' latexNumber(x) '$'];
else
    txt = latexEscape(s);
end

end


function txt = summaryCellToLatex(value)

s = strtrim(char(string(value)));

if strcmpi(s,'NA')
    txt = 'NA';
    return
end

numPat = ...
    '([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)';

expr = ...
    ['^\s*' numPat ...
     '\s*\[\s*' numPat ...
     '\s*-\s*' numPat ...
     '\s*\]\s*$'];

tok = regexp( ...
    s, ...
    expr, ...
    'tokens', ...
    'once');

if isempty(tok)
    txt = latexEscape(s);
    return
end

med = str2double(strtrim(tok{1}));
q1  = str2double(strtrim(tok{2}));
q3  = str2double(strtrim(tok{3}));

if all(isfinite([med q1 q3]))

    txt = sprintf( ...
        '$%s$ [$%s$--$%s$]', ...
        latexNumber(med), ...
        latexNumber(q1), ...
        latexNumber(q3));

else

    txt = latexEscape(s);

end

end


function txt = latexNumber(x)

if x == 0
    txt = '0';
    return
end

ax = abs(x);

if ax < 1e-3 || ax >= 1e4

    exponent = floor(log10(ax));
    mantissa = x/(10^exponent);

    if abs(mantissa-1) < 1e-12

        txt = sprintf('10^{%d}',exponent);

    elseif abs(mantissa+1) < 1e-12

        txt = sprintf('-10^{%d}',exponent);

    else

        txt = sprintf( ...
            '%.3g\\times10^{%d}', ...
            mantissa, ...
            exponent);

    end

else

    txt = sprintf('%.3g',x);

end

end


function idx = findNameIndex(names,target,isKernel)

idx = [];
targetCanonical = canonicalName(target,isKernel);

for i = 1:numel(names)

    if strcmpi( ...
            canonicalName(names{i},isKernel), ...
            targetCanonical)

        idx = i;
        return

    end

end

end


function name = canonicalName(value,isKernel)

name = lower(strtrim(char(string(value))));
name = regexprep(name,'[^a-z0-9]','');

if isKernel

    switch name

        case {'gaussian','gauss'}
            name = 'gaussian';

        case {'multiquadric','mq'}
            name = 'multiquadric';

        case 'linear'
            name = 'linear';

        case 'cubic'
            name = 'cubic';

        case {'thinplate','thinplatespline','tps'}
            name = 'thinplate';

    end

end

end


function txt = getTextFieldAny(S,names)

txt = '';

if ~isstruct(S)
    return
end

for i = 1:numel(names)

    if isfield(S,names{i})

        value = S.(names{i});

        if ischar(value) || (isstring(value) && isscalar(value))
            txt = char(string(value));
            return
        end

    end

end

end




function fileName = resolveMatFile(folderPath,baseName)
% Resolve one exact MATLAB result file from its base name.
% No wildcard search is used, so _IDW and _Nearest files can never be chosen.

baseName = char(string(baseName));

candidateNoExt = fullfile(folderPath,baseName);
candidateMat   = fullfile(folderPath,[baseName '.mat']);

if isfile(candidateNoExt)

    fileName = candidateNoExt;
    return

end

if isfile(candidateMat)

    fileName = candidateMat;
    return

end

error( ...
    ['Missing exact Local-RBF result:\n%s\n' ...
     'or\n%s'], ...
    candidateNoExt, ...
    candidateMat);

end


function fileName = resolveUserPath(value,RootDir,LocalDir)

value = char(string(value));

candidates = { ...
    value
    fullfile(RootDir,value)
    fullfile(LocalDir,value)};

for i = 1:numel(candidates)

    if isfile(candidates{i})
        fileName = candidates{i};
        return
    end

end

fileName = value;

end


function fileName = resolveStoredCorePath( ...
    storedPath,localResultFile,RootDir,LocalDir)

storedPath = char(string(storedPath));
localResultDir = fileparts(localResultFile);

candidates = { ...
    storedPath
    fullfile(RootDir,storedPath)
    fullfile(localResultDir,storedPath)
    fullfile(LocalDir,storedPath)};

for i = 1:numel(candidates)

    if isfile(candidates{i})
        fileName = candidates{i};
        return
    end

end

error( ...
    ['Could not resolve stored Stage2 CORE path:\n%s\n\n' ...
     'Tried:\n  %s'], ...
    storedPath, ...
    strjoin(candidates,'\n  '));

end


function mustExist(fileName,label)

if ~isfile(fileName)
    error('Missing %s:\n%s',label,fileName);
end

end


function txt = latexEscape(value)

txt = char(string(value));

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

end

%% ========================================================================
% Make_NumericalCondition_AppendixTables
%
% PURPOSE
% -------
% Fast manuscript post-processor for the numerical-conditioning results.
%
% It does NOT recompute any RBF system and does NOT rerun the numerical
% collector. It reads the existing:
%
%   Current_NumericalCondition_Manuscript.xlsx
%
% and creates EXACTLY TWO manuscript outputs:
%
%   1) NumericalCondition_Appendix_Tables.xlsx
%      - Sheet "Case1"
%      - Sheet "Case2"
%
%   2) NumericalCondition_Appendix_Tables.tex
%      - Appendix table for Case 1
%      - Appendix table for Case 2
%      - captions, labels, compact notes included
%
% TABLE DESIGN
% ------------
% Rows:
%   G-Train : G M T C L
%   G-Test  : G M T C L
%   L-Train : G M T C L
%   L-Test  : G M T C L
%
% Columns:
%   Set. | K | log10(k_raw) | log10(k_reg) | log10(k_eff) |
%   log10(k_eval) | RelRes | BErr
%
% Each cell is:
%   median (maximum)
%
% Aggregation across the six GGMs:
%   reported median  = median of the six GGM-specific medians
%   reported maximum = maximum of the six GGM-specific maxima
%
% Condition-number columns are shown on log10 scale.
% Relative residual and backward error remain in the original scale.
% ========================================================================

clear
clc

fprintf('\n=============================================================\n');
fprintf(' NUMERICAL-CONDITION APPENDIX TABLE GENERATOR\n');
fprintf(' TWO OUTPUTS: XLSX + LATEX\n');
fprintf('=============================================================\n');

%% ========================================================================
% 1. PATHS
% ========================================================================

RootDir = '.';

LocalRoot = fullfile(RootDir,'OutPut_RBF_2026_2');

TableDir = fullfile(LocalRoot,'Manuscript','tables');

InputXLSX = fullfile( ...
    TableDir, ...
    'Current_NumericalCondition_Manuscript.xlsx');

OutputXLSX = fullfile( ...
    TableDir, ...
    'NumericalCondition_Appendix_Tables.xlsx');

OutputTEX = fullfile( ...
    TableDir, ...
    'NumericalCondition_Appendix_Tables.tex');

assert(isfile(InputXLSX), ...
    'Input numerical-condition workbook not found:\n%s',InputXLSX);

%% ========================================================================
% 2. READ EXISTING COMPLETE DIAGNOSTICS
% ========================================================================

fprintf('[1/4] Reading current diagnostics ... ');

T = readtable( ...
    InputXLSX, ...
    'Sheet','Diagnostics', ...
    'VariableNamingRule','preserve');

requiredColumns = { ...
    'Case','Regime','Phase','Model','ModelCode', ...
    'Kernel','KernelCode','Metric','Median','P95','Max','N'};

assert(all(ismember(requiredColumns,T.Properties.VariableNames)), ...
    'The Diagnostics sheet does not have the expected v9 columns.');

fprintf('done\n');

%% ========================================================================
% 3. BUILD TWO COMPACT TABLES
% ========================================================================

fprintf('[2/4] Building Case 1 / Case 2 compact tables ... ');

Case1Table = buildCompactCaseTable(T,'Case1');
Case2Table = buildCompactCaseTable(T,'Case2');

assert(height(Case1Table)==20,'Case1 compact table must have 20 rows.');
assert(height(Case2Table)==20,'Case2 compact table must have 20 rows.');

fprintf('done\n');

%% ========================================================================
% 4. WRITE EXCEL
% ========================================================================

fprintf('[3/4] Writing Excel ... ');

if isfile(OutputXLSX)
    delete(OutputXLSX);
end

writetable( ...
    Case1Table, ...
    OutputXLSX, ...
    'Sheet','Case1', ...
    'WriteMode','overwritesheet');

writetable( ...
    Case2Table, ...
    OutputXLSX, ...
    'Sheet','Case2', ...
    'WriteMode','overwritesheet');

noteText = [ ...
    "Values are reported as median (maximum). For each row, the reported median is the median " + ...
    "of the six GGM-specific medians, while the reported maximum is the largest of the six " + ...
    "GGM-specific maxima. Condition-number columns are expressed as log_{10}\kappa; " + ...
    "RelRes and BErr are reported in their original scales. " + ...
    "G-Train and G-Test denote global training and testing, respectively; " + ...
    "L-Train and L-Test denote local training and testing. " + ...
    "Kernel codes G, M, T, C, and L denote Gaussian, multiquadric, TPS, cubic, " + ...
    "and linear RBFs, respectively." ];

writecell( ...
    {'Note',char(noteText)}, ...
    OutputXLSX, ...
    'Sheet','Case1', ...
    'Range','A23');

writecell( ...
    {'Note',char(noteText)}, ...
    OutputXLSX, ...
    'Sheet','Case2', ...
    'Range','A23');

fprintf('done\n');

%% ========================================================================
% 5. WRITE LATEX WITH TWO COMPLETE TABLE ENVIRONMENTS
% ========================================================================

fprintf('[4/4] Writing LaTeX ... ');

fid = fopen(OutputTEX,'w');

assert(fid>=0, ...
    'Cannot create LaTeX output:\n%s',OutputTEX);

cleanupFID = onCleanup(@() fclose(fid));

fprintf(fid,'%% ============================================================\n');
fprintf(fid,'%% Numerical-conditioning Appendix tables\n');
fprintf(fid,'%% Generated from Current_NumericalCondition_Manuscript.xlsx\n');
fprintf(fid,'%% Requires booktabs (already commonly used in manuscripts).\n');
fprintf(fid,'%% ============================================================\n\n');

writeLatexCaseTable( ...
    fid, ...
    Case1Table, ...
    'Numerical-conditioning diagnostics for Case~1.', ...
    'tab:numerical_condition_case1');

fprintf(fid,'\n\n');

writeLatexCaseTable( ...
    fid, ...
    Case2Table, ...
    'Numerical-conditioning diagnostics for Case~2.', ...
    'tab:numerical_condition_case2');

clear cleanupFID

fprintf('done\n');

%% ========================================================================
% 6. REPORT
% ========================================================================

fprintf('\nCreated:\n');
fprintf('  XLSX : %s\n',OutputXLSX);
fprintf('  TEX  : %s\n',OutputTEX);
fprintf('=============================================================\n');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function Tout = buildCompactCaseTable(T,caseName)

setLabels = { ...
    'G-Train','Global','Training'; ...
    'G-Test', 'Global','Testing'; ...
    'L-Train','Local', 'Training'; ...
    'L-Test', 'Local', 'Testing'};

kernelCodes = {'G','M','T','C','L'};

metricNames = { ...
    'CondRaw', ...
    'CondReg', ...
    'CondEffective', ...
    'CondEvaluation', ...
    'RelResidual', ...
    'BackwardError'};

Set = strings(20,1);
K = strings(20,1);

kRaw = strings(20,1);
kReg = strings(20,1);
kEff = strings(20,1);
kEval = strings(20,1);
RelRes = strings(20,1);
BErr = strings(20,1);

row = 0;

for iset = 1:size(setLabels,1)

    displaySet = setLabels{iset,1};
    regime = setLabels{iset,2};
    phase = setLabels{iset,3};

    for ik = 1:numel(kernelCodes)

        row = row + 1;

        if ik == 1
            Set(row) = string(displaySet);
        else
            % Blank repeated setting labels to save horizontal/visual space.
            Set(row) = "";
        end

        K(row) = string(kernelCodes{ik});

        values = strings(1,numel(metricNames));

        for im = 1:numel(metricNames)

            metric = metricNames{im};

            mask = ...
                strcmpi(string(T.Case),caseName) & ...
                strcmpi(string(T.Regime),regime) & ...
                strcmpi(string(T.Phase),phase) & ...
                strcmpi(string(T.KernelCode),kernelCodes{ik}) & ...
                strcmpi(string(T.Metric),metric);

            S = T(mask,:);

            assert(height(S)==6, ...
                ['Expected six GGM rows for %s | %s | %s | %s | %s; ' ...
                 'found %d.'], ...
                caseName,regime,phase,kernelCodes{ik},metric,height(S));

            modelMedians = double(S.Median);
            modelMaxima  = double(S.Max);

            medValue = median(modelMedians,'omitnan');
            maxValue = max(modelMaxima,[],'omitnan');

            if startsWith(metric,'Cond')
                values(im) = formatLogConditionPair(medValue,maxValue);
            else
                values(im) = formatScientificPair(medValue,maxValue);
            end
        end

        kRaw(row)  = values(1);
        kReg(row)  = values(2);
        kEff(row)  = values(3);
        kEval(row) = values(4);
        RelRes(row)= values(5);
        BErr(row)  = values(6);

    end
end

Tout = table( ...
    Set,K,kRaw,kReg,kEff,kEval,RelRes,BErr, ...
    'VariableNames',{ ...
    'Set','K', ...
    'log10_k_raw', ...
    'log10_k_reg', ...
    'log10_k_eff', ...
    'log10_k_eval', ...
    'RelRes', ...
    'BErr'});

end

% =========================================================================
function s = formatLogConditionPair(medValue,maxValue)

s = string( ...
    sprintf( ...
        '%s (%s)', ...
        formatLogNumber(medValue), ...
        formatLogNumber(maxValue)));

end

% =========================================================================
function s = formatLogNumber(x)

if isnan(x)
    s = 'NaN';
elseif isinf(x)
    s = 'Inf';
elseif x <= 0
    s = 'NA';
else
    s = sprintf('%.1f',log10(x));
end

end

% =========================================================================
function s = formatScientificPair(medValue,maxValue)

s = string( ...
    sprintf( ...
        '%s (%s)', ...
        formatScientificNumber(medValue), ...
        formatScientificNumber(maxValue)));

end

% =========================================================================
function s = formatScientificNumber(x)

if isnan(x)
    s = 'NaN';
elseif isinf(x)
    s = 'Inf';
else
    s = sprintf('%.1e',x);
end

end

% =========================================================================
function writeLatexCaseTable(fid,T,captionText,labelText)

fprintf(fid,'\\begin{table*}[!htbp]\n');
fprintf(fid,'\\centering\n');
fprintf(fid,'\\caption{%s}\n',captionText);
fprintf(fid,'\\label{%s}\n',labelText);
fprintf(fid,'\\scriptsize\n');
fprintf(fid,'\\setlength{\\tabcolsep}{3pt}\n');
fprintf(fid,'\\renewcommand{\\arraystretch}{0.95}\n');
fprintf(fid,'\\begin{tabular}{llcccccc}\n');
fprintf(fid,'\\toprule\n');
fprintf(fid, ...
    ['Set. & K & $\\log_{10}\\kappa_{\\rm raw}$ & ' ...
     '$\\log_{10}\\kappa_{\\rm reg}$ & ' ...
     '$\\log_{10}\\kappa_{\\rm eff}$ & ' ...
     '$\\log_{10}\\kappa_{\\rm eval}$ & RelRes & BErr \\\\\n']);
fprintf(fid,'\\midrule\n');

for r = 1:height(T)

    setText = char(T.Set(r));
    kernelText = char(T.K(r));

    fprintf(fid, ...
        '%s & %s & %s & %s & %s & %s & %s & %s \\\\\n', ...
        latexEscape(setText), ...
        latexEscape(kernelText), ...
        latexEscape(char(T.log10_k_raw(r))), ...
        latexEscape(char(T.log10_k_reg(r))), ...
        latexEscape(char(T.log10_k_eff(r))), ...
        latexEscape(char(T.log10_k_eval(r))), ...
        latexEscape(char(T.RelRes(r))), ...
        latexEscape(char(T.BErr(r))));

    if any(r == [5 10 15])
        fprintf(fid,'\\midrule\n');
    end
end

fprintf(fid,'\\bottomrule\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\vspace{1mm}\n');

fprintf(fid, ...
    ['\\parbox{\\textwidth}{\\footnotesize\\textit{Note:} ' ...
     'Values are reported as median (maximum). For each row, the reported ' ...
     'median is the median of the six GGM-specific medians, while the ' ...
     'reported maximum is the largest of the six GGM-specific maxima. ' ...
     'Condition-number columns are expressed as $\\log_{10}\\kappa$; ' ...
     'RelRes and BErr are reported in their original scales. ' ...
     'G-Train and G-Test denote global training and testing, respectively; ' ...
     'L-Train and L-Test denote local training and testing. ' ...
     'Kernel codes G, M, T, C, and L denote Gaussian, multiquadric, ' ...
     'TPS, cubic, and linear RBFs, respectively.}\n']);

fprintf(fid,'\\end{table*}\n');

end

% =========================================================================
function out = latexEscape(in)

out = char(in);

% Only characters that may occur in generated compact cells/labels need
% escaping here. Numeric cells do not contain LaTeX-sensitive symbols.
out = strrep(out,'_','\_');
out = strrep(out,'%','\%');
out = strrep(out,'&','\&');
out = strrep(out,'#','\#');

end

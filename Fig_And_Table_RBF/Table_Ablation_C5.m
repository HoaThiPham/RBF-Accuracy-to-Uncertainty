function Outputs = Table_Ablation_C5()
%CREATE_ABLATION_TABLE
% Fast Appendix Table A1 generator for the FINAL ablation workbook.
% Manuscript notation: M0 = former Model 11; MP = former Model 8;
% MR = former Model 12; MPR = former Model 1 (complete framework).
%
% This code DOES NOT rerun any RBF model and DOES NOT create figures.
% It reads only the existing AllComparisons sheet and creates:
%   1) Table_A1_Ablation.xlsx
%   2) Table_A1_Ablation.tex
%
% FINAL COMPACT TABLE COLUMNS
%   Reference-model
%   Practical outcome (%) C1/C2
%       Eq.: |Delta| <= 5 mm
%       >5 : Delta > 5 mm
%   Severe tail (%) C1/C2
%       >20: Delta > 20 mm
%       >50: Delta > 50 mm
%   Median Delta (mm) C1/C2
%
% SCIENTIFIC DEFINITION
%   Delta_j = 1000*(RMSE_Mj,test - RMSE_MPR,test) [mm]
%
% Positive Delta = higher test RMSE than M_PR.
% Negative Delta = lower test RMSE than M_PR.
%
% Overall rows are the median [minimum--maximum] of the five
% reference-level values, calculated separately for Case 1 and Case 2.
%
% Each reference-model-case percentage has n=30 configurations
% (all GGM-kernel combinations for that reference and case).

clc;
TIMER_ALL = tic;

%% ========================================================================
% 1. PATHS
% =========================================================================
RootDir = '.';
AblationDir = fullfile(RootDir,'OutPut_RBF_2026_2', ...
    'Sensitivity_Ablation_RandomRobustness');

InputWorkbook = fullfile(AblationDir, ...
    'Ablation_vs_MPR_RandomRobustness.xlsx');

OutputDir = fullfile(AblationDir,'Manuscript_Ablation');
if ~isfolder(OutputDir), mkdir(OutputDir); end

OutputExcel = fullfile(OutputDir,'Table_A1_Ablation.xlsx');
OutputTex   = fullfile(OutputDir,'Table_A1_Ablation.tex');

% Optional MAT cache to avoid repeated Excel reading.
USE_CACHE = true;
CacheFile = fullfile(OutputDir,'Ablation_MPR_AllComparisons_CACHE.mat');

%% ========================================================================
% 2. SCIENTIFIC SETTINGS / FINAL-DATA GUARD
% =========================================================================
EQ_MM = 5;
HARM20_MM = 20;
HARM50_MM = 50;

EXPECTED_SEEDS = [12345 23456 34567 45678 56789];
MODEL_ORDER = ["M0","MP","MR"];
CASE_ORDER = ["Case1","Case2"];

FINAL_DATA_GUARD = true;
EXPECTED_TOTAL_ROWS = 900;
EXPECTED_ROWS_PER_MODEL = 300;
EXPECTED_ABOVE_500MM = [24 70 20];
EXPECTED_MP_LINEAR_GT_0P5M = 60;
EXPECTED_MP_LINEAR_GT_1M = 47;
EXPECTED_MP_LINEAR_MAX_TEST_M = 554.6136605821222;
EXPECTED_MR_MAX_TEST_M = 11.059669252201648;
TOL = 1e-8;

%% ========================================================================
% 3. READ FINAL DATA, WITH CACHE
% =========================================================================
assert(isfile(InputWorkbook), ...
    'Input workbook not found:\n%s',InputWorkbook);

srcInfo = dir(InputWorkbook);
sourceStamp = [srcInfo.datenum, srcInfo.bytes];

loadedCache = false;
if USE_CACHE && isfile(CacheFile)
    try
        C = load(CacheFile,'D','sourceStampCached');
        if isfield(C,'D') && isfield(C,'sourceStampCached') && ...
                isequal(C.sourceStampCached,sourceStamp)
            D = C.D;
            loadedCache = true;
            fprintf('Loaded cached final data.\n');
        end
    catch
        loadedCache = false;
    end
end

if ~loadedCache
    fprintf('Reading AllComparisons from Excel... ');
    t = tic;

    D = readtable(InputWorkbook,'Sheet','AllComparisons', ...
        'VariableNamingRule','preserve');

    D = normalizeInputColumns(D);

    D.DeltaTest_mm = 1000 .* ...
        (D.Alternative_TestRMSE - D.MPR_TestRMSE);

    sourceStampCached = sourceStamp; %#ok<NASGU>
    if USE_CACHE
        save(CacheFile,'D','sourceStampCached','-v7');
    end

    fprintf('done (%.2f s)\n',toc(t));
else
    if ~ismember('DeltaTest_mm',D.Properties.VariableNames)
        D.DeltaTest_mm = 1000 .* ...
            (D.Alternative_TestRMSE - D.MPR_TestRMSE);
    end
end

validateCoverage(D,EXPECTED_SEEDS,MODEL_ORDER,CASE_ORDER);

if FINAL_DATA_GUARD
    validateFinalDatasetFingerprint(D,InputWorkbook, ...
        EXPECTED_TOTAL_ROWS,EXPECTED_ROWS_PER_MODEL,EXPECTED_ABOVE_500MM, ...
        EXPECTED_MP_LINEAR_GT_0P5M,EXPECTED_MP_LINEAR_GT_1M, ...
        EXPECTED_MP_LINEAR_MAX_TEST_M,EXPECTED_MR_MAX_TEST_M,TOL);
end

%% ========================================================================
% 4. BUILD SEED-LEVEL NUMERIC SUMMARY
% =========================================================================
fprintf('Building reference-level summaries... ');
t = tic;

nSeed = numel(EXPECTED_SEEDS);
nModel = numel(MODEL_ORDER);
nCase = numel(CASE_ORDER);

ReferenceSeed = zeros(nSeed*nModel*nCase,1);
Alternative = strings(nSeed*nModel*nCase,1);
Case = strings(nSeed*nModel*nCase,1);
N = zeros(nSeed*nModel*nCase,1);
EquivalentPct = nan(nSeed*nModel*nCase,1);
Harm5Pct = nan(nSeed*nModel*nCase,1);
Harm20Pct = nan(nSeed*nModel*nCase,1);
Harm50Pct = nan(nSeed*nModel*nCase,1);
MedianDelta_mm = nan(nSeed*nModel*nCase,1);

r = 0;
for s = EXPECTED_SEEDS
    for m = MODEL_ORDER
        for c = CASE_ORDER
            r = r + 1;
            X = D(D.ReferenceSeed==s & D.Alternative==m & D.Case==c,:);
            delta = X.DeltaTest_mm;
            delta = delta(isfinite(delta));

            ReferenceSeed(r) = s;
            Alternative(r) = m;
            Case(r) = c;
            N(r) = numel(delta);

            assert(N(r)==30, ...
                'Expected n=30 for seed %d / %s / %s; found %d.', ...
                s,m,c,N(r));

            EquivalentPct(r) = 100*mean(abs(delta)<=EQ_MM);
            Harm5Pct(r) = 100*mean(delta>EQ_MM);
            Harm20Pct(r) = 100*mean(delta>HARM20_MM);
            Harm50Pct(r) = 100*mean(delta>HARM50_MM);
            MedianDelta_mm(r) = median(delta);
        end
    end
end

SeedSummary = table(ReferenceSeed,Alternative,Case,N, ...
    EquivalentPct,Harm5Pct,Harm20Pct,Harm50Pct,MedianDelta_mm);

fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 5. BUILD COMPACT MANUSCRIPT TABLE A1
% =========================================================================
fprintf('Building compact Table A1... ');
t = tic;

ReferenceModel = strings(18,1);
PracticalOutcome = strings(18,1);
SevereTail = strings(18,1);
MedianDelta_C1_C2 = strings(18,1);

row = 0;

% ---- Overall rows: median [min--max] across the five references ----
for m = MODEL_ORDER
    row = row + 1;
    ReferenceModel(row) = "Overall " + manuscriptModelLabel(m);

    eqLine = "Eq.: " + overallPair(SeedSummary,m,'EquivalentPct');
    harm5Line = ">5: " + overallPair(SeedSummary,m,'Harm5Pct');
    PracticalOutcome(row) = eqLine + newline + harm5Line;

    harm20Line = ">20: " + overallPair(SeedSummary,m,'Harm20Pct');
    harm50Line = ">50: " + overallPair(SeedSummary,m,'Harm50Pct');
    SevereTail(row) = harm20Line + newline + harm50Line;

    MedianDelta_C1_C2(row) = overallPair(SeedSummary,m,'MedianDelta_mm');
end

% ---- Reference rows: C1/C2 values ----
for s = EXPECTED_SEEDS
    for m = MODEL_ORDER
        row = row + 1;
        ReferenceModel(row) = string(s) + "--" + manuscriptModelLabel(m);

        eqLine = "Eq.: " + referencePair(SeedSummary,s,m,'EquivalentPct');
        harm5Line = ">5: " + referencePair(SeedSummary,s,m,'Harm5Pct');
        PracticalOutcome(row) = eqLine + newline + harm5Line;

        harm20Line = ">20: " + referencePair(SeedSummary,s,m,'Harm20Pct');
        harm50Line = ">50: " + referencePair(SeedSummary,s,m,'Harm50Pct');
        SevereTail(row) = harm20Line + newline + harm50Line;

        MedianDelta_C1_C2(row) = referencePair(SeedSummary,s,m,'MedianDelta_mm');
    end
end

TableA1 = table(ReferenceModel,PracticalOutcome,SevereTail,MedianDelta_C1_C2);

TableA1.Properties.VariableNames = { ...
    'Reference_Model', ...
    'Practical_outcome_pct_C1_C2', ...
    'Severe_tail_pct_C1_C2', ...
    'Median_Delta_mm_C1_C2'};

fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 6. WRITE EXCEL
% =========================================================================
fprintf('Writing Excel Table A1... ');
t = tic;

if isfile(OutputExcel)
    delete(OutputExcel);
end

writetable(TableA1,OutputExcel, ...
    'Sheet','AppendixA1','WriteMode','overwritesheet');

% Keep a second numeric sheet for transparent checking/reuse.
writetable(SeedSummary,OutputExcel, ...
    'Sheet','ReferenceLevel_Numeric','WriteMode','overwritesheet');

fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 7. WRITE LATEX TABLE WITH CAPTION + NOTES
% =========================================================================
fprintf('Writing compact LaTeX Table A1... ');
t = tic;

fid = fopen(OutputTex,'w');
assert(fid>0,'Could not create LaTeX output: %s',OutputTex);
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid,'%% Auto-generated compact Appendix Table A1\n');
fprintf(fid,'\\begin{table*}[htbp]\n');
fprintf(fid,'\\centering\n');
fprintf(fid,['\\caption{Reference-level robustness of the ablated local-RBF configurations relative to the complete framework $M_{PR}$. ' ...
    'Overall rows report the median [minimum--maximum] across the five equal-status fixed-reference realizations, calculated separately for Case~1 and Case~2.}\n']);
fprintf(fid,'\\label{tab:ablation_appendix}\n');
fprintf(fid,'\\small\n');
fprintf(fid,'\\setlength{\\tabcolsep}{4.0pt}\n');
fprintf(fid,'\\renewcommand{\\arraystretch}{1.12}\n');

fprintf(fid,['\\begin{tabularx}{\\textwidth}{' ...
    'l' ...
    '>{\\centering\\arraybackslash}X' ...
    '>{\\centering\\arraybackslash}X' ...
    '>{\\centering\\arraybackslash}X}\n']);

fprintf(fid,'\\hline\n');
fprintf(fid,['\\textbf{Reference--model} & ' ...
    '\\textbf{Practical outcome (\\%%) C1/C2} & ' ...
    '\\textbf{Severe tail (\\%%) C1/C2} & ' ...
    '\\textbf{Median $\\boldsymbol{\\Delta}$ (mm) C1/C2} \\\\\n']);
fprintf(fid,'\\hline\n');

for i = 1:height(TableA1)

    if i==4
        fprintf(fid,'\\hline\n');
    end

    ref = latexReferenceModel(TableA1.Reference_Model(i));

    % Convert Excel line breaks into two-line LaTeX shortstacks.
    pLines = split(string(TableA1.Practical_outcome_pct_C1_C2(i)),newline);
    sLines = split(string(TableA1.Severe_tail_pct_C1_C2(i)),newline);

    p1 = latexMetricLine(pLines(1));
    p2 = latexMetricLine(pLines(2));
    s1 = latexMetricLine(sLines(1));
    s2 = latexMetricLine(sLines(2));

    practical = sprintf('\\shortstack{%s\\\\%s}',p1,p2);
    severe = sprintf('\\shortstack{%s\\\\%s}',s1,s2);
    med = char(TableA1.Median_Delta_mm_C1_C2(i));

    if i<=3
        ref = ['\textbf{' ref '}'];
    end

    fprintf(fid,'%s & %s & %s & %s \\\\\n', ...
        ref,practical,severe,med);
end

fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabularx}\n');
fprintf(fid,'\\vspace{1mm}\n');
fprintf(fid,'\\begin{minipage}{0.99\\textwidth}\n');
fprintf(fid,'\\footnotesize\n');
fprintf(fid,['\\textit{Note:} All paired values are reported as C1/C2, denoting Case~1/Case~2, respectively. ' ...
    '$\\Delta=1000(RMSE_{M_j}^{\\rm test}-RMSE_{M_{PR}}^{\\rm test})$ in millimetres. ' ...
    'Eq. denotes $|\\Delta|\\leq5$~mm, while $>5$, $>20$, and $>50$ denote the corresponding positive-degradation thresholds. ' ...
    'Positive $\\Delta$ indicates higher independent-test RMSE than $M_{PR}$, whereas negative $\\Delta$ indicates lower test RMSE. ' ...
    'The $\\pm5$~mm interval is a predefined descriptive practical-equivalence band rather than a formal statistical equivalence margin. ' ...
    'Each reference--model--case percentage is based on $n=30$ Case--GGM--kernel configurations. ' ...
    'The five reference realizations are robustness conditions rather than independent datasets.}\n']);
fprintf(fid,'\\end{minipage}\n');
fprintf(fid,'\\end{table*}\n');

clear cleanupObj;
fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 8. REPORT
% =========================================================================
fprintf('\n=============================================================\n');
fprintf(' FINAL TABLE A1 FINISHED\n');
fprintf('=============================================================\n');
fprintf('Excel : %s\n',OutputExcel);
fprintf('LaTeX : %s\n',OutputTex);
fprintf('Rows  : %d\n',height(TableA1));
fprintf('Total runtime: %.2f s\n',toc(TIMER_ALL));
fprintf('=============================================================\n');

Outputs = struct();
Outputs.TableA1 = TableA1;
Outputs.SeedSummary = SeedSummary;
Outputs.ExcelFile = OutputExcel;
Outputs.LatexFile = OutputTex;
Outputs.CacheFile = CacheFile;
Outputs.CacheUsed = loadedCache;

end

%% ========================================================================
% LOCAL FUNCTIONS
% =========================================================================
function D = normalizeInputColumns(D)

D = renameIfNeeded(D,'ReferenceSeed',{'Seed'});
D = renameIfNeeded(D,'Alternative',{'AlternativeModel','Model'});
D = renameIfNeeded(D,'MPR_TestRMSE',{'Model1_TestRMSE','Model1_Test','M1_TestRMSE','MPR_Test'});
D = renameIfNeeded(D,'Alternative_TestRMSE',{'Alternative_Test','Alt_TestRMSE'});
D = renameIfNeeded(D,'Alternative_TrainRMSE',{'Alternative_Train','Alt_TrainRMSE'});
D = renameIfNeeded(D,'Alternative_PseudoRMSE',{'Alternative_Pseudo','Alt_PseudoRMSE'});
D = renameIfNeeded(D,'MPR_PseudoUnseenRMSE',{'Model1_PseudoUnseenRMSE','Model1_PseudoUnseen','M1_PseudoRMSE','MPR_PseudoRMSE'});

required = {'ReferenceSeed','Case','Alternative','GGM','Kernel', ...
    'MPR_TestRMSE','Alternative_TestRMSE', ...
    'Alternative_TrainRMSE','Alternative_PseudoRMSE','MPR_PseudoUnseenRMSE'};

assert(all(ismember(required,D.Properties.VariableNames)), ...
    'AllComparisons is missing one or more required columns.');

D.ReferenceSeed = double(D.ReferenceSeed);
D.Case = normalizeCaseLabel(string(D.Case));
D.Alternative = normalizeAlternativeLabel(string(D.Alternative));
D.GGM = strtrim(string(D.GGM));
D.Kernel = canonicalKernel(string(D.Kernel));

numericVars = {'MPR_TestRMSE','Alternative_TestRMSE', ...
    'Alternative_TrainRMSE','Alternative_PseudoRMSE','MPR_PseudoUnseenRMSE'};

for k=1:numel(numericVars)
    D.(numericVars{k}) = double(D.(numericVars{k}));
end

assert(all(isfinite(D.MPR_TestRMSE) & D.MPR_TestRMSE>0), ...
    'MPR_TestRMSE contains invalid values.');
assert(all(isfinite(D.Alternative_TestRMSE) & D.Alternative_TestRMSE>0), ...
    'Alternative_TestRMSE contains invalid values.');
end

function D = renameIfNeeded(D,target,aliases)
if ismember(target,D.Properties.VariableNames)
    return;
end
for k=1:numel(aliases)
    if ismember(aliases{k},D.Properties.VariableNames)
        idx = strcmp(D.Properties.VariableNames,aliases{k});
        D.Properties.VariableNames{idx} = target;
        return;
    end
end
end

function validateCoverage(D,seeds,models,cases)
assert(height(D)==900,'Expected 900 rows; found %d.',height(D));

for s=seeds
    for m=models
        for c=cases
            n = sum(D.ReferenceSeed==s & D.Alternative==m & D.Case==c);
            assert(n==30, ...
                'Expected 30 rows for seed %d / %s / %s; found %d.', ...
                s,m,c,n);
        end
    end
end
end

function validateFinalDatasetFingerprint(D,InputWorkbook, ...
    expectedTotalRows,expectedRowsPerModel,expectedAbove500, ...
    expectedMPLinearGT05,expectedMPLinearGT1,expectedMPLinearMax, ...
    expectedMRMax,tol)

fprintf('\n=============================================================\n');
fprintf(' FINAL DATA SCIENTIFIC GUARD\n');
fprintf(' Workbook: %s\n',InputWorkbook);

assert(height(D)==expectedTotalRows, ...
    'Wrong workbook: expected %d rows; found %d.', ...
    expectedTotalRows,height(D));

models = ["M0","MP","MR"];
for k=1:3
    n = sum(D.Alternative==models(k));
    assert(n==expectedRowsPerModel, ...
        'Wrong workbook: %s expected %d rows; found %d.', ...
        models(k),expectedRowsPerModel,n);
end

above500 = zeros(1,3);
for k=1:3
    x = D.DeltaTest_mm(D.Alternative==models(k));
    above500(k) = sum(x>500);
end
assert(isequal(above500,expectedAbove500), ...
    'Wrong workbook: >500-mm fingerprint is [%s], expected [%s].', ...
    num2str(above500),num2str(expectedAbove500));

mpLin = D(D.Alternative=="MP" & canonicalKernel(D.Kernel)=="linear",:);
assert(height(mpLin)==60,'Wrong workbook: expected 60 M_P Linear rows.');
assert(sum(mpLin.Alternative_TestRMSE>0.5)==expectedMPLinearGT05, ...
    'Wrong M_P Linear >0.5-m count.');
assert(sum(mpLin.Alternative_TestRMSE>1.0)==expectedMPLinearGT1, ...
    'Wrong M_P Linear >1-m count.');

maxMP = max(mpLin.Alternative_TestRMSE);
assert(abs(maxMP-expectedMPLinearMax)<=tol*max(1,abs(expectedMPLinearMax)), ...
    'Wrong M_P Linear max test RMSE.');

mr = D(D.Alternative=="MR",:);
maxMR = max(mr.Alternative_TestRMSE);
assert(abs(maxMR-expectedMRMax)<=tol*max(1,abs(expectedMRMax)), ...
    'Wrong M_R max test RMSE.');

fprintf(' Rows/model       : %d / %d / %d\n', ...
    sum(D.Alternative=="M0"), ...
    sum(D.Alternative=="MP"), ...
    sum(D.Alternative=="MR"));
fprintf(' Delta >500 mm    : %d / %d / %d\n',above500);
fprintf(' M_P Linear >0.5m : %d\n',sum(mpLin.Alternative_TestRMSE>0.5));
fprintf(' M_P Linear >1m   : %d\n',sum(mpLin.Alternative_TestRMSE>1.0));
fprintf(' M_P Linear max   : %.12g m\n',maxMP);
fprintf(' M_R max test     : %.12g m\n',maxMR);
fprintf(' FINAL DATA GUARD : PASS\n');
fprintf('=============================================================\n\n');
end

function s = overallPair(T,m,varName)
a = T(T.Alternative==m & T.Case=="Case1",:).(varName);
b = T(T.Alternative==m & T.Case=="Case2",:).(varName);
s = formatMedianRange(a) + " / " + formatMedianRange(b);
end

function s = referencePair(T,seed,m,varName)
a = T(T.ReferenceSeed==seed & T.Alternative==m & T.Case=="Case1",:).(varName);
b = T(T.ReferenceSeed==seed & T.Alternative==m & T.Case=="Case2",:).(varName);
assert(numel(a)==1 && numel(b)==1,'Reference pair lookup failed.');
s = sprintf('%.1f / %.1f',a,b);
end

function s = formatMedianRange(x)
x = x(isfinite(x));
assert(numel(x)==5,'Overall summary requires exactly five reference values.');
s = sprintf('%.1f [%.1f--%.1f]',median(x),min(x),max(x));
end

function s = manuscriptModelLabel(m)
m = normalizeAlternativeLabel(string(m));
assert(isscalar(m),'manuscriptModelLabel expects one model label.');
switch m
    case "M0"
        s = "M0";
    case "MP"
        s = "MP";
    case "MR"
        s = "MR";
    otherwise
        error('Unknown model label: %s',m);
end
end

function s = latexReferenceModel(x)
s = char(string(x));
s = strrep(s,'Overall M0','Overall $M_0$');
s = strrep(s,'Overall MP','Overall $M_P$');
s = strrep(s,'Overall MR','Overall $M_R$');

s = regexprep(s,'--M0$','--$M_0$');
s = regexprep(s,'--MP$','--$M_P$');
s = regexprep(s,'--MR$','--$M_R$');
end

function s = latexMetricLine(x)
% Convert compact metric text to LaTeX.
s = char(string(x));
s = strrep(s,'Eq.: ','Eq.: ');
s = regexprep(s,'^>5: ','$>5$: ');
s = regexprep(s,'^>20: ','$>20$: ');
s = regexprep(s,'^>50: ','$>50$: ');
end

function s = normalizeCaseLabel(x)
x = strtrim(string(x));
s = strings(size(x));
for i=1:numel(x)
    z = lower(regexprep(x(i),'[^a-z0-9]',''));
    if any(z==["case1","c1","1"])
        s(i) = "Case1";
    elseif any(z==["case2","c2","2"])
        s(i) = "Case2";
    else
        s(i) = x(i);
    end
end
end

function s = normalizeAlternativeLabel(x)
x = strtrim(string(x));
s = strings(size(x));
for i=1:numel(x)
    z = lower(regexprep(x(i),'[^a-z0-9]',''));
    switch z
        case {"m0","model0","model11","m11","mode11"}
            s(i) = "M0";
        case {"mp","modelp","model8","m8","mode8"}
            s(i) = "MP";
        case {"mr","modelr","model12","m12","mode12"}
            s(i) = "MR";
        otherwise
            s(i) = x(i);
    end
end
end

function k = canonicalKernel(x)
x = strtrim(lower(string(x)));
k = strings(size(x));
for i=1:numel(x)
    z = regexprep(x(i),'[^a-z0-9]','');
    if z=="gaussian"
        k(i)="gaussian";
    elseif any(z==["multiquadric","mq"])
        k(i)="multiquadric";
    elseif z=="linear"
        k(i)="linear";
    elseif z=="cubic"
        k(i)="cubic";
    elseif any(z==["thinplate","thinplatespline","tps"])
        k(i)="thinplate";
    else
        k(i)=x(i);
    end
end
end

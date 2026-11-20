function Outputs = Fig_Ablation_Fig1()
%CREATE_ABLATION_FIGURE
% Lightweight / fast figure-only renderer for the FINAL ablation workbook.
% Manuscript notation used internally/output: M0, MP, MR, and MPR.
%
% KEY SPEED CHANGES relative to v18:
%   1) No boxchart objects.
%   2) No scatter objects with transparency.
%   3) No exportgraphics in preview mode.
%   4) No .fig saving in preview mode.
%   5) Optional MAT cache avoids rereading Excel on later runs.
%   6) Low-level patch/line/plot only.
%
% SCIENTIFIC DEFINITIONS ARE UNCHANGED:
%   M0 (former Model 11) -> M_0
%   MP (former Model 8)  -> M_P
%   MR (former Model 12) -> M_R
%   MPR (former Model 1) -> M_{PR}
%
%   Delta_test = 1000*(RMSE_alt,test - RMSE_MPR,test) [mm]
%
% The FINAL-DATA fingerprint is retained. If the workbook is not the locked
% 900-row dataset, the script stops rather than plotting a different pattern.

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

OutputFigurePNG = fullfile(OutputDir,'Figure_Ablation_Main.png');
OutputFigurePDF = fullfile(OutputDir,'Figure_Ablation_Main.pdf');
OutputFigureTex = fullfile(OutputDir,'Figure_Ablation_Main.tex');

% ------------------------------------------------------------------------
% FAST MODE:
% true  = fast preview PNG only using PRINT at 220 dpi
% false = final 600-dpi PNG + vector PDF
% ------------------------------------------------------------------------
FAST_PREVIEW = true;

% Cache the normalized 900-row table. If the source workbook changes, the
% cache is automatically rebuilt.
USE_CACHE = true;
CacheFile = fullfile(OutputDir,'Ablation_MPR_AllComparisons_CACHE.mat');

%% ========================================================================
% 2. SCIENTIFIC SETTINGS / FINAL-DATA GUARD
% =========================================================================
PRACTICAL_EQ_MM = 5;
SEVERITY_20_MM = 20;
SEVERITY_50_MM = 50;
EXTRA_DIAGNOSTIC_THRESHOLDS_MM = [10 100];

EXPECTED_SEEDS = [12345 23456 34567 45678 56789];
EXPECTED_MODELS = ["M0","MP","MR"];
ALLOW_INCOMPLETE_INPUT = false;

FINAL_DATA_GUARD = true;
EXPECTED_TOTAL_ROWS = 900;
EXPECTED_ROWS_PER_MODEL = 300;
EXPECTED_ABOVE_500MM = [24 70 20];
EXPECTED_MP_LINEAR_GT_0P5M = 60;
EXPECTED_MP_LINEAR_GT_1M = 47;
EXPECTED_MP_LINEAR_MAX_TEST_M = 554.6136605821222;
EXPECTED_MR_MAX_TEST_M = 11.059669252201648;
FINAL_DATA_TOL = 1e-8;

% Guard helper expects this limit.
FIG.A.YLim_mm = [-20 500];

%% ========================================================================
% 3. VISUAL SETTINGS -- LOW-LEVEL GRAPHICS ONLY
% =========================================================================
FontName = 'Times New Roman';
BaseFont = 11.5;
LabelFont = 12.5;
CaptionFont = 11.5;
LegendFont = 8.5;

% Keep non-interactive to avoid MATLAB Desktop ViewModel overhead.
FigVisible = 'off';
FigPosition = [100 100 1850 690];

AxesPos = [ ...
    0.055 0.270 0.255 0.665; ...
    0.382 0.270 0.255 0.665; ...
    0.709 0.270 0.255 0.665];

KernelOrder   = ["gaussian","multiquadric","linear","cubic","thinplate"];
KernelDisplay = {'Gaussian','Multiquadric','Linear','Cubic','TPS'};
KernelMarkers = {'o','s','^','d','v'};

% MATLAB default color-order palette, explicitly stored for reproducibility.
KernelColors = [ ...
    0.0000 0.4470 0.7410; ...
    0.8500 0.3250 0.0980; ...
    0.9290 0.6940 0.1250; ...
    0.4940 0.1840 0.5560; ...
    0.4660 0.6740 0.1880];

%% ========================================================================
% 4. READ FINAL DATA -- WITH OPTIONAL CACHE
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
            fprintf('Loaded cached final data: %.2f s\n',toc(TIMER_ALL));
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
    D = addDerivedColumns(D,PRACTICAL_EQ_MM,SEVERITY_20_MM,SEVERITY_50_MM, ...
        EXTRA_DIAGNOSTIC_THRESHOLDS_MM);
    validateInputCoverage(D,EXPECTED_SEEDS,EXPECTED_MODELS,ALLOW_INCOMPLETE_INPUT);

    D.Alternative = string(D.Alternative);
    D.Case = string(D.Case);
    D.GGM = string(D.GGM);
    D.Kernel = string(D.Kernel);
    D.KernelGroup = string(D.KernelGroup);
    fprintf('done (%.2f s)\n',toc(t));

    if USE_CACHE
        sourceStampCached = sourceStamp; %#ok<NASGU>
        save(CacheFile,'D','sourceStampCached','-v7');
    end
end

if FINAL_DATA_GUARD
    validateFinalDatasetFingerprint(D,InputWorkbook,FIG, ...
        EXPECTED_TOTAL_ROWS,EXPECTED_ROWS_PER_MODEL,EXPECTED_ABOVE_500MM, ...
        EXPECTED_MP_LINEAR_GT_0P5M,EXPECTED_MP_LINEAR_GT_1M, ...
        EXPECTED_MP_LINEAR_MAX_TEST_M,EXPECTED_MR_MAX_TEST_M,FINAL_DATA_TOL);
end

%% ========================================================================
% 5. PRE-COMPUTE PLOT ARRAYS BEFORE OPENING FIGURE
% =========================================================================
fprintf('Preparing plot arrays... ');
t = tic;

modelOrder = ["M0","MP","MR"];
modelTickLabels = {'$M_0$','$M_P$','$M_R$'};

A = struct();
for j = 1:3
    y = D.DeltaTest_mm(D.Alternative == modelOrder(j));
    y = y(isfinite(y));
    A(j).raw = y; %#ok<AGROW>
    A(j).display = min(max(y,-20),500); %#ok<AGROW>
    A(j).nAbove = sum(y > 500); %#ok<AGROW>
    A(j).q1 = percentile_linear(y,25); %#ok<AGROW>
    A(j).med = percentile_linear(y,50); %#ok<AGROW>
    A(j).q3 = percentile_linear(y,75); %#ok<AGROW>
    iqrVal = A(j).q3 - A(j).q1;
    A(j).lowWhisk = max(min(y), A(j).q1 - 1.5*iqrVal); %#ok<AGROW>
    A(j).highWhisk = min(max(y), A(j).q3 + 1.5*iqrVal); %#ok<AGROW>
end

MP = D(D.Alternative=="MP",:);
MR = D(D.Alternative=="MR",:);

fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 6. OPEN ONE LIGHTWEIGHT FIGURE
% =========================================================================
fprintf('Creating lightweight figure... ');
t = tic;

fig = figure('Color','w', ...
    'Visible',FigVisible, ...
    'WindowStyle','normal', ...
    'MenuBar','none', ...
    'ToolBar','none', ...
    'NumberTitle','off', ...
    'IntegerHandle','off', ...
    'Renderer','opengl', ...
    'Position',FigPosition);

axA = axes('Parent',fig,'Position',AxesPos(1,:));
axB = axes('Parent',fig,'Position',AxesPos(2,:));
axC = axes('Parent',fig,'Position',AxesPos(3,:));

set([axA axB axC], ...
    'FontName',FontName, ...
    'FontSize',BaseFont, ...
    'LineWidth',0.8, ...
    'Box','on', ...
    'Layer','top');

%% ========================================================================
% 7. PANEL (a): SIMPLE BOXES + LOW-LEVEL POINTS
% =========================================================================
hold(axA,'on');
rng(42,'twister');

rawPointColor = [0.92 0.55 0.55];
boxFill = [1.00 0.93 0.93];
boxEdge = [0.78 0.18 0.18];
medianColor = [0.60 0.08 0.08];

for j = 1:3
    % Plot actual values within the displayed range.
    yyRaw = A(j).raw;
    yy = yyRaw(yyRaw <= 500);
    xx = j + (rand(size(yy))-0.5)*0.24;
    line(axA,xx,yy, ...
        'LineStyle','none', ...
        'Marker','o', ...
        'MarkerSize',2.7, ...
        'MarkerFaceColor',rawPointColor, ...
        'MarkerEdgeColor',rawPointColor);

    % Preserve EVERY configuration above 500 mm as a top-boundary
    % overflow marker instead of silently dropping/clipping it.
    nOver = sum(yyRaw > 500);
    if nOver > 0
        xOver = j + (rand(nOver,1)-0.5)*0.30;
        yOver = 497*ones(nOver,1);
        line(axA,xOver,yOver, ...
            'LineStyle','none', ...
            'Marker','^', ...
            'MarkerSize',3.2, ...
            'MarkerFaceColor','none', ...
            'MarkerEdgeColor',boxEdge, ...
            'Clipping','off');
    end

    % Whisker
    line(axA,[j j],[A(j).lowWhisk A(j).highWhisk], ...
        'Color',boxEdge,'LineWidth',0.9);
    line(axA,[j-0.10 j+0.10],[A(j).lowWhisk A(j).lowWhisk], ...
        'Color',boxEdge,'LineWidth',0.9);
    line(axA,[j-0.10 j+0.10],[A(j).highWhisk A(j).highWhisk], ...
        'Color',boxEdge,'LineWidth',0.9);

    % Box (Q1-Q3)
    x0 = j-0.22; x1 = j+0.22;
    patch(axA,[x0 x1 x1 x0], ...
        [A(j).q1 A(j).q1 A(j).q3 A(j).q3], ...
        boxFill,'EdgeColor',boxEdge,'LineWidth',1.0);

    % Median
    line(axA,[x0 x1],[A(j).med A(j).med], ...
        'Color',medianColor,'LineWidth',1.4);

    % One direct clipped count per model.
    text(axA,j,520,sprintf('n=%d',A(j).nAbove), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontName',FontName, ...
        'FontSize',9.5, ...
        'Interpreter','tex', ...
        'Clipping','off');
end

line(axA,[0.45 3.55],[0 0],'LineStyle','-','Color',[0.25 0.25 0.25],'LineWidth',0.9);
line(axA,[0.45 3.55],[5 5],'LineStyle','--','Color',[0.45 0.45 0.45],'LineWidth',0.8);
line(axA,[0.45 3.55],[-5 -5],'LineStyle','--','Color',[0.45 0.45 0.45],'LineWidth',0.8);

set(axA,'XLim',[0.45 3.55],'YLim',[-20 500], ...
    'XTick',1:3,'XTickLabel',modelTickLabels,'TickLabelInterpreter','latex');
ylabel(axA,'$\Delta_{\rm test}$ (mm)', ...
    'Interpreter','latex','FontSize',LabelFont,'FontName',FontName);
grid(axA,'on');
axA.GridAlpha = 0.12;


%% ========================================================================
% 8. PANEL (b): M_P DIAGNOSTIC -- LOW-LEVEL PLOT MARKERS
% =========================================================================
hold(axB,'on');
hB = gobjects(numel(KernelOrder),1);

for k = 1:numel(KernelOrder)
    mask = canonicalKernel(MP.Kernel) == KernelOrder(k);
    x = MP.Alternative_TrainRMSE(mask);
    y = MP.Alternative_TestRMSE(mask);
    good = isfinite(x) & isfinite(y) & x>0 & y>0;

    hB(k) = line(axB,x(good),y(good), ...
        'LineStyle','none', ...
        'Marker',KernelMarkers{k}, ...
        'MarkerSize',4.6, ...
        'MarkerFaceColor',KernelColors(k,:), ...
        'MarkerEdgeColor',KernelColors(k,:));
end

xx = [0.012 0.12];
line(axB,xx,xx,'LineStyle','--','Color',[0.35 0.35 0.35],'LineWidth',0.9);

set(axB,'XScale','log','YScale','log', ...
    'XLim',[0.012 0.12],'YLim',[0.07 1000]);
xlabel(axB,'Training-phase optimisation RMSE (m)', ...
    'FontName',FontName,'FontSize',LabelFont);
ylabel(axB,'Independent-test RMSE (m)', ...
    'FontName',FontName,'FontSize',LabelFont);
grid(axB,'on'); axB.GridAlpha = 0.12;
legend(axB,hB,KernelDisplay,'Location','northwest', ...
    'Box','off','FontName',FontName,'FontSize',LegendFont);

%% ========================================================================
% 9. PANEL (c): M_R DIAGNOSTIC -- LOW-LEVEL PLOT MARKERS
% =========================================================================
hold(axC,'on');
hC = gobjects(numel(KernelOrder),1);

for k = 1:numel(KernelOrder)
    mask = canonicalKernel(MR.Kernel) == KernelOrder(k);
    x = MR.Alternative_PseudoRMSE(mask);
    y = MR.Alternative_TestRMSE(mask);
    good = isfinite(x) & isfinite(y) & x>0 & y>0;

    hC(k) = line(axC,x(good),y(good), ...
        'LineStyle','none', ...
        'Marker',KernelMarkers{k}, ...
        'MarkerSize',4.6, ...
        'MarkerFaceColor',KernelColors(k,:), ...
        'MarkerEdgeColor',KernelColors(k,:));
end

xx = [0.07 4];
line(axC,xx,xx,'LineStyle','--','Color',[0.35 0.35 0.35],'LineWidth',0.9);

set(axC,'XScale','log','YScale','log', ...
    'XLim',[0.07 4],'YLim',[0.07 20]);
xlabel(axC,'Pseudo-unseen RMSE (m)', ...
    'FontName',FontName,'FontSize',LabelFont);
ylabel(axC,'Independent-test RMSE (m)', ...
    'FontName',FontName,'FontSize',LabelFont);
grid(axC,'on'); axC.GridAlpha = 0.12;
legend(axC,hC,KernelDisplay,'Location','northwest', ...
    'Box','off','FontName',FontName,'FontSize',LegendFont);

%% ========================================================================
% 10. SUBFIGURE CAPTIONS BELOW PANELS -- NO IN-PLOT TITLES
% =========================================================================
annotation(fig,'textbox',[AxesPos(1,1) 0.045 AxesPos(1,3) 0.085], ...
    'String','(a) Omission-harm distribution', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName',FontName,'FontSize',CaptionFont);

annotation(fig,'textbox',[AxesPos(2,1) 0.045 AxesPos(2,3) 0.085], ...
    'String','(b) M_P: fixed neighbourhood, pointwise parameter tuning', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName',FontName,'FontSize',CaptionFont, ...
    'Interpreter','none');

annotation(fig,'textbox',[AxesPos(3,1) 0.045 AxesPos(3,3) 0.085], ...
    'String','(c) M_R: adaptive neighbourhood, fixed parameters', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName',FontName,'FontSize',CaptionFont, ...
    'Interpreter','none');

drawnow limitrate nocallbacks;
fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 10B. DEFENSIVE GRAPHICS CHECK
% =========================================================================
% Avoid export failures caused by malformed TeX strings in legacy MATLAB
% graphics objects. All remaining plot annotations are plain text or valid
% LaTeX labels.
badText = findall(fig,'Type','text');
for ii = 1:numel(badText)
    try
        s = get(badText(ii),'String');
        if ischar(s) && contains(s,'triangle: value')
            set(badText(ii),'String','');
        end
    catch
    end
end

%% ========================================================================
% 11. FAST EXPORT
% =========================================================================
fprintf('Exporting... ');
t = tic;

set(fig,'PaperPositionMode','auto');

if FAST_PREVIEW
    % Fast raster preview. OpenGL avoids the slower vector renderer.
    drawnow;
    set(fig,'Renderer','opengl');
    print(fig,OutputFigurePNG,'-dpng','-r180');
else
    % Final manuscript export.
    drawnow;
    set(fig,'Renderer','opengl');
    print(fig,OutputFigurePNG,'-dpng','-r600');

    % Use painters only for the final vector PDF.
    set(fig,'Renderer','painters');
    print(fig,OutputFigurePDF,'-dpdf','-painters','-bestfit');
end

fprintf('done (%.2f s)\n',toc(t));

%% ========================================================================
% 12. WRITE LATEX FIGURE SNIPPET WITH CAPTION
% =========================================================================
fid = fopen(OutputFigureTex,'w');
assert(fid>0,'Could not create LaTeX figure file: %s',OutputFigureTex);
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid,'%% Auto-generated manuscript figure snippet\n');
fprintf(fid,'\\begin{figure*}[htbp]\n');
fprintf(fid,'\\centering\n');
if FAST_PREVIEW
    LatexGraphicFile = 'Figure_Ablation_Main.png';
else
    LatexGraphicFile = 'Figure_Ablation_Main.pdf';
end
fprintf(fid,'\\includegraphics[width=\\textwidth]{%s}\n',LatexGraphicFile);
fprintf(fid,['\\caption{Mechanistic ablation and robustness diagnostics of the adaptive local-RBF framework. ' ...
    '(a) Distribution of independent-test degradation, $\\Delta_{\\rm test}=1000(RMSE_{M_j}^{\\rm test}-RMSE_{M_{PR}}^{\\rm test})$, ' ...
    'for $M_0$, $M_P$, and $M_R$. Circles represent individual Case--GGM--kernel--reference configurations; boxes show the interquartile range and the central line the median. ' ...
    'Values exceeding the displayed upper limit of 500~mm are represented by hollow triangles at the upper boundary, with their counts shown above the corresponding model groups; triangle position indicates overflow rather than actual magnitude. ' ...
    '(b) Independent-test RMSE versus pointwise target-excluded optimization RMSE for $M_P$. ' ...
    '(c) Independent-test RMSE versus pseudo-unseen RMSE for $M_R$. The dashed diagonal in (b)--(c) is the 1:1 line. ' ...
    'The internal quantities in (b) and (c) arise from different procedures and are shown as separate qualitative diagnostics rather than directly comparable validation estimators.}\n']);
fprintf(fid,'\\label{fig:ablation_robustness}\n');
fprintf(fid,'\\end{figure*}\n');
clear cleanupObj;

close(fig);

fprintf('\n=============================================================\n');
fprintf(' FAST LIGHTWEIGHT FIGURE FINISHED\n');
fprintf('=============================================================\n');
fprintf('PNG : %s\n',OutputFigurePNG);
fprintf('LaTeX: %s\n',OutputFigureTex);
if ~FAST_PREVIEW
    fprintf('PDF : %s\n',OutputFigurePDF);
end
fprintf('Cache used: %d\n',loadedCache);
fprintf('Total runtime: %.2f s\n',toc(TIMER_ALL));
fprintf('=============================================================\n');

Outputs = struct();
Outputs.FigurePNG = OutputFigurePNG;
Outputs.FigurePDF = OutputFigurePDF;
Outputs.FigureTex = OutputFigureTex;
Outputs.CacheFile = CacheFile;
Outputs.CacheUsed = loadedCache;
Outputs.FastPreview = FAST_PREVIEW;

end

%% ========================================================================
% FAST PERCENTILE HELPER -- no Statistics Toolbox required
% =========================================================================
function q = percentile_linear(x,p)
x = sort(x(isfinite(x)));
n = numel(x);
if n == 0
    q = NaN;
    return;
elseif n == 1
    q = x;
    return;
end
r = 1 + (n-1)*(p/100);
lo = floor(r);
hi = ceil(r);
if lo == hi
    q = x(lo);
else
    q = x(lo) + (r-lo)*(x(hi)-x(lo));
end
end

%% ========================================================================
% ORIGINAL FINAL-DATA NORMALIZATION / GUARD HELPERS FROM v18
% =========================================================================
function D = normalizeInputColumns(D)
% Accept the current master-workbook schema and a few harmless aliases.

D = renameIfNeeded(D,'ReferenceSeed',{'Seed'});
D = renameIfNeeded(D,'Alternative',{'AlternativeModel','Model'});
D = renameIfNeeded(D,'MPR_TestRMSE',{'Model1_TestRMSE','Model1_Test','M1_TestRMSE','MPR_Test'});
D = renameIfNeeded(D,'Alternative_TestRMSE',{'Alternative_Test','Alt_TestRMSE'});
D = renameIfNeeded(D,'MPR_PseudoUnseenRMSE', ...
    {'Model1_PseudoUnseenRMSE','Model1_PseudoUnseen','M1_PseudoRMSE','MPR_PseudoRMSE'});
D = renameIfNeeded(D,'Alternative_TrainRMSE', ...
    {'Alternative_Train','Alt_TrainRMSE'});
D = renameIfNeeded(D,'Alternative_PseudoRMSE', ...
    {'Alternative_Pseudo','Alt_PseudoRMSE'});

required = {'ReferenceSeed','Case','Alternative','GGM','Kernel', ...
    'MPR_TestRMSE','Alternative_TestRMSE', ...
    'MPR_PseudoUnseenRMSE','Alternative_TrainRMSE','Alternative_PseudoRMSE'};
assert(all(ismember(required,D.Properties.VariableNames)), ...
    'AllComparisons is missing one or more required manuscript columns.');

% Normalize labels.
D.Case = normalizeCaseLabel(string(D.Case));
D.Alternative = normalizeAlternativeLabel(string(D.Alternative));
D.GGM = strtrim(string(D.GGM));
D.Kernel = canonicalKernel(string(D.Kernel));

if ~ismember('KernelGroup',D.Properties.VariableNames)
    D.KernelGroup = classifyKernelGroup(D.Kernel);
else
    D.KernelGroup = classifyKernelGroup(D.Kernel); % authoritative regrouping
end

% Numeric conversion.
numericVars = {'ReferenceSeed','MPR_TestRMSE','Alternative_TestRMSE', ...
    'MPR_PseudoUnseenRMSE','Alternative_TrainRMSE','Alternative_PseudoRMSE'};
for i=1:numel(numericVars)
    D.(numericVars{i}) = double(D.(numericVars{i}));
end

assert(all(isfinite(D.MPR_TestRMSE) & D.MPR_TestRMSE>0), ...
    'MPR_TestRMSE contains invalid values.');
assert(all(isfinite(D.Alternative_TestRMSE) & D.Alternative_TestRMSE>0), ...
    'Alternative_TestRMSE contains invalid values.');
end

function D = renameIfNeeded(D,target,aliases)
if ismember(target,D.Properties.VariableNames), return; end
for k=1:numel(aliases)
    if ismember(aliases{k},D.Properties.VariableNames)
        D.Properties.VariableNames{strcmp(D.Properties.VariableNames,aliases{k})} = target;
        return
    end
end
end

function D = addDerivedColumns(D,eqMM,t20,t50,extraThresholds)
D.DeltaTest_m = D.Alternative_TestRMSE - D.MPR_TestRMSE;
D.DeltaTest_mm = 1000 .* D.DeltaTest_m;

D.Better5 = D.DeltaTest_mm < -eqMM;
D.Equivalent5 = abs(D.DeltaTest_mm) <= eqMM;
D.Harm5 = D.DeltaTest_mm > eqMM;
D.Harm20 = D.DeltaTest_mm > t20;
D.Harm50 = D.DeltaTest_mm > t50;

for t = extraThresholds(:).'
    vn = sprintf('Harm%d',round(t));
    D.(vn) = D.DeltaTest_mm > t;
end

% MR pseudo-unseen degradation relative to MPR pseudo-unseen.
D.DeltaPU_m = D.Alternative_PseudoRMSE - D.MPR_PseudoUnseenRMSE;
D.DeltaPU_mm = 1000 .* D.DeltaPU_m;

% M8 optimization-to-test amplification.  Only scientifically interpreted for MP.
D.InternalToTestGap_m = D.Alternative_TestRMSE - D.Alternative_TrainRMSE;
D.InternalToTestGap_mm = 1000 .* D.InternalToTestGap_m;
D.InternalToTestRatio = D.Alternative_TestRMSE ./ D.Alternative_TrainRMSE;
D.InternalToTestRatio(~isfinite(D.InternalToTestRatio)) = NaN;
end

function validateInputCoverage(D,seeds,models,allowIncomplete)
actualSeeds = sort(unique(D.ReferenceSeed(:))).';
actualModels = sort(unique(string(D.Alternative(:))));

if ~allowIncomplete
    assert(isequal(actualSeeds,sort(double(seeds(:).'))), ...
        'Final manuscript mode requires exactly the five expected seeds.');
    assert(all(ismember(models,actualModels)), ...
        'Final manuscript mode requires M0, MP, and MR.');

    for s=seeds(:).'
        for m=models(:).'
            n = sum(D.ReferenceSeed==s & D.Alternative==m);
            assert(n==60, ...
                'Expected 60 rows for seed %d / %s; found %d.',s,m,n);
        end
    end
end
end

function validateFinalDatasetFingerprint(D,InputWorkbook,FIG, ...
    expectedTotalRows,expectedRowsPerModel,expectedAbove500, ...
    expectedMPLinearGT05,expectedMPLinearGT1,expectedMPLinearMaxTest, ...
    expectedMRMaxTest,tol)
% Prevent a visually plausible but scientifically different figure from an
% older/different AllComparisons workbook.

fprintf('\n=============================================================\n');
fprintf(' FINAL-DATA SCIENTIFIC FINGERPRINT\n');
fprintf('=============================================================\n');
fprintf('Input workbook: %s\n',InputWorkbook);

assert(height(D)==expectedTotalRows, ...
    'FINAL DATA GUARD: expected %d rows, found %d. Wrong/old workbook.', ...
    expectedTotalRows,height(D));

models = ["M0","MP","MR"];
for i=1:3
    n = sum(D.Alternative==models(i));
    fprintf('%s rows = %d\n',char(models(i)),n);
    assert(n==expectedRowsPerModel, ...
        'FINAL DATA GUARD: %s expected %d rows, found %d.', ...
        char(models(i)),expectedRowsPerModel,n);
end

% Panel-(a) clipping fingerprint at the manuscript upper display limit.
assert(~isempty(FIG.A.YLim_mm) && abs(FIG.A.YLim_mm(2)-500)<tol, ...
    'FINAL DATA GUARD currently expects panel-(a) upper display limit = 500 mm.');
actualAbove = zeros(1,3);
for i=1:3
    actualAbove(i) = sum(D.DeltaTest_mm(D.Alternative==models(i)) > 500);
end
fprintf('Delta_test > 500 mm counts [M0 MP MR] = [%d %d %d]\n',actualAbove);
assert(isequal(actualAbove,expectedAbove500), ...
    ['FINAL DATA GUARD: expected Delta_test>500 mm counts [%d %d %d], ' ...
     'but found [%d %d %d]. This is NOT the uploaded final workbook.'], ...
    expectedAbove500(1),expectedAbove500(2),expectedAbove500(3), ...
    actualAbove(1),actualAbove(2),actualAbove(3));

% M_P Linear tail fingerprint: this is the strongest visible behavior in panel (b).
Z = D(D.Alternative=="MP" & D.Kernel=="linear",:);
assert(height(Z)==60,'FINAL DATA GUARD: expected 60 MP Linear rows, found %d.',height(Z));
nGT05 = sum(Z.Alternative_TestRMSE>0.5);
nGT1 = sum(Z.Alternative_TestRMSE>1.0);
maxMP = max(Z.Alternative_TestRMSE);
fprintf('MP Linear: >0.5 m = %d; >1 m = %d; max test = %.12g m\n', ...
    nGT05,nGT1,maxMP);
assert(nGT05==expectedMPLinearGT05 && nGT1==expectedMPLinearGT1 && ...
       abs(maxMP-expectedMPLinearMaxTest)<=max(tol,1e-10*abs(expectedMPLinearMaxTest)), ...
    'FINAL DATA GUARD: M_P Linear fingerprint mismatch. Wrong/old workbook.');

% M_R test-tail fingerprint for panel (c).
R = D(D.Alternative=="MR",:);
maxMR = max(R.Alternative_TestRMSE);
fprintf('MR max test RMSE = %.12g m\n',maxMR);
assert(abs(maxMR-expectedMRMaxTest)<=max(tol,1e-10*abs(expectedMRMaxTest)), ...
    'FINAL DATA GUARD: M_R max-test fingerprint mismatch. Wrong/old workbook.');

fprintf('FINAL DATA GUARD: PASS. Plotting scientific pattern is locked.\n');
fprintf('=============================================================\n\n');
end

function c = normalizeCaseLabel(c)
c = lower(regexprep(strtrim(c),'[^a-zA-Z0-9]',''));
out = strings(size(c));
out(c=="case1" | c=="1") = "Case1";
out(c=="case2" | c=="2") = "Case2";
assert(all(strlength(out)>0),'Unrecognized Case label encountered.');
c = out;
end

function a = normalizeAlternativeLabel(a)
x = lower(regexprep(strtrim(a),'[^a-zA-Z0-9]',''));
out = strings(size(x));
out(x=="model8" | x=="mode8" | x=="m8" | x=="mp" | x=="modelp") = "MP";
out(x=="model11" | x=="mode11" | x=="m11" | x=="m0" | x=="model0") = "M0";
out(x=="model12" | x=="mode12" | x=="m12" | x=="mr" | x=="modelr") = "MR";
assert(all(strlength(out)>0),'Unrecognized Alternative label encountered.');
a = out;
end

function k = canonicalKernel(k)
x = lower(regexprep(strtrim(k),'[^a-zA-Z0-9]',''));
out = strings(size(x));
out(x=="gaussian") = "gaussian";
out(x=="multiquadric" | x=="mq") = "multiquadric";
out(x=="linear") = "linear";
out(x=="cubic") = "cubic";
out(x=="thinplate" | x=="thinplatespline" | x=="tps") = "thinplate";
assert(all(strlength(out)>0),'Unrecognized kernel label encountered.');
k = out;
end

function g = classifyKernelGroup(k)
g = strings(size(k));
g(k=="gaussian" | k=="multiquadric") = "Shape-parameter";
g(k=="linear" | k=="cubic" | k=="thinplate") = "Shape-free";
assert(all(strlength(g)>0),'Kernel-group classification failed.');
end

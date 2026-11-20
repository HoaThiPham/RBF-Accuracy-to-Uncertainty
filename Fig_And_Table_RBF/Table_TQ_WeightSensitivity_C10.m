function Sensitivity = Table_TQ_WeightSensitivity_C10()
%SUMMARIZE_TQ_WEIGHTSENSITIVITY
%
% Reviewer-response sensitivity analysis for the Total-Quality (TQ) index.
%
% PURPOSE
% -------
% Test how sensitive the exact best model/kernel is to the user-defined
% weights theta1, theta2 and theta3.
%
% The input TQ file must have been produced by:
%   Calculate_TQ_WeightSensitivity.m
%
% Four scales are constructed independently:
%   Case 1 - Train, Case 1 - Test, Case 2 - Train, Case 2 - Test.
% Within each scale, Global/Local and RBF/Kriging are pooled.
%
% Sensitivity is evaluated on the INDEPENDENT TEST phase only. Therefore,
% all 120 theta combinations for Case 1 use the same fixed Case 1-Test
% scale, and all 120 combinations for Case 2 use the same fixed
% Case 2-Test scale. Only theta changes during sensitivity analysis.
% No TQ ranking or averaging is performed across cases or phases.
%
% For each:
%   Case x Regime x GGM
%
% this code:
%   1) identifies the best candidate under the reference weights
%      [0.50 0.25 0.25];
%   2) recomputes its rank over all 120 admissible theta combinations;
%   3) reports:
%        - same-winner stability (%)
%        - top-2 stability (%)
%        - minimum and maximum rank
%
% Candidate columns:
%   1 = RBF-G
%   2 = RBF-M
%   3 = RBF-T
%   4 = RBF-C
%   5 = RBF-L
%   6 = Kriging-S
%
% Lower TQ = better.
%
% OUTPUTS
% -------
% Excel:
%   TQ_WeightSensitivity.xlsx
%     Sheet Model_Stability
%     Sheet Regime_Summary
%     Sheet Theta_Grid
%
% LaTeX:
%   TQ_WeightSensitivity.tex
%
% ========================================================================

clc


% Run TQ
Calculate_TQ_WeightSensitivity

%% ========================================================================
% 1. USER INPUT
% ========================================================================

TQ_FILE = ...
    'OutPut_RBF_2026_2\TQ\TQ_Metrix.mat';

OUTPUT_DIR = ...
    'OutPut_RBF_2026_2\TQ';

OUTPUT_XLSX = ...
    fullfile(OUTPUT_DIR,'TQ_WeightSensitivity.xlsx');

OUTPUT_TEX = ...
    fullfile(OUTPUT_DIR,'TQ_WeightSensitivity.tex');

TIE_TOL = ...
    1e-12;

%% ========================================================================
% 2. LOAD
% ========================================================================

S = load(TQ_FILE,'TQ_Metrix');

assert(isfield(S,'TQ_Metrix'), ...
    'TQ_Metrix is missing from:\n%s',TQ_FILE);

TQ = S.TQ_Metrix;

assert(isfield(TQ,'Theta') && ...
       isfield(TQ,'MainTheta') && ...
       isfield(TQ,'MainThetaIndex'), ...
    'Theta metadata is missing from TQ_Metrix.');


assert(isfield(TQ,'Scale') && ...
       isfield(TQ.Scale,'Case1') && ...
       isfield(TQ.Scale.Case1,'Train') && ...
       isfield(TQ.Scale.Case1,'Test') && ...
       isfield(TQ.Scale,'Case2') && ...
       isfield(TQ.Scale.Case2,'Train') && ...
       isfield(TQ.Scale.Case2,'Test'), ...
    ['Four Case x Phase scales are missing. Run ' ...
     'Run Calculate_TQ_WeightSensitivity.m first.']);


if isfield(TQ,'Settings') && isfield(TQ.Settings,'Scaling')
    assert(contains(string(TQ.Settings.Scaling),'Four separate Case x Phase'), ...
        'TQ file does not appear to use four Case x Phase scales.');
end

Theta = double(TQ.Theta);
nTheta = size(Theta,1);

assert(nTheta == 120, ...
    ['Expected 120 sensitivity weight combinations. Found %d. ' ...
     'Run Calculate_TQ_WeightSensitivity.m first.'], ...
    nTheta);

assert(all(abs(sum(Theta,2)-1) < 1e-10), ...
    'At least one theta combination does not sum to 1.');

assert(all(Theta(:) >= 0.10-1e-10) && ...
       all(Theta(:) <= 0.80+1e-10), ...
    'Theta grid is outside the intended [0.10,0.80] range.');

idxMain = TQ.MainThetaIndex;

assert(all(abs(TQ.MainTheta(:).' - [0.50 0.25 0.25]) < 1e-10), ...
    'Unexpected reference theta.');

%% ========================================================================
% 3. LABELS
% ========================================================================

CaseList = {'Case1','Case2'};
RegimeList = {'Global','Local'};

if isfield(TQ,'Settings') && isfield(TQ.Settings,'Models')
    ModelLabels = string(TQ.Settings.Models(:));
else
    ModelLabels = ["E08";"GGP";"EC4";"GCO";"SG2";"X19"];
end

if isfield(TQ,'Settings') && isfield(TQ.Settings,'Columns')
    CandidateLabels = string(TQ.Settings.Columns(:));
else
    CandidateLabels = ...
        ["Rbf-G";"Rbf-M";"Rbf-T";"Rbf-C";"Rbf-L";"Kriging-S"];
end

assert(numel(ModelLabels)==6,'Expected six GGM rows.');
assert(numel(CandidateLabels)==6,'Expected six candidate columns.');

%% ========================================================================
% 4. MODEL-LEVEL STABILITY
% ========================================================================

nRows = 2*2*6;

CaseCol = strings(nRows,1);
RegimeCol = strings(nRows,1);
GGMCol = strings(nRows,1);
ReferenceBest = strings(nRows,1);

ReferenceTQ = nan(nRows,1);
SameWinner_pct = nan(nRows,1);
Top2_pct = nan(nRows,1);
MeanRank = nan(nRows,1);
MinRank = nan(nRows,1);
MaxRank = nan(nRows,1);
N_Theta = repmat(nTheta,nRows,1);

row = 0;

for ic = 1:numel(CaseList)

    CaseName = CaseList{ic};

    for ir = 1:numel(RegimeList)

        Regime = RegimeList{ir};

        QI = ...
            TQ.Test.(CaseName).(Regime).QI;

        Qmain = ...
            TQ.Test.(CaseName).(Regime).QI_Main;

        assert(isequal(size(QI),[6 6 nTheta]), ...
            '%s %s Test QI must be 6 x 6 x %d.', ...
            CaseName,Regime,nTheta);

        assert(isequal(size(Qmain),[6 6]), ...
            '%s %s QI_Main must be 6 x 6.', ...
            CaseName,Regime);

        for ig = 1:6

            row = row + 1;

            qRefRow = Qmain(ig,:);

            [bestTQ,bestIdx] = min(qRefRow,[],2);

            ranks = nan(nTheta,1);

            for it = 1:nTheta

                q = QI(ig,:,it);
                qRefCandidate = q(bestIdx);

                % Competition rank of the reference-best candidate.
                % A tie at the minimum therefore still receives rank 1.
                ranks(it) = ...
                    1 + sum(q < qRefCandidate - TIE_TOL);

            end

            CaseCol(row) = prettyCase(CaseName);
            RegimeCol(row) = string(Regime);
            GGMCol(row) = ModelLabels(ig);
            ReferenceBest(row) = CandidateLabels(bestIdx);

            ReferenceTQ(row) = bestTQ;
            SameWinner_pct(row) = 100*mean(ranks==1);
            Top2_pct(row) = 100*mean(ranks<=2);
            MeanRank(row) = mean(ranks);
            MinRank(row) = min(ranks);
            MaxRank(row) = max(ranks);

        end

    end

end

ModelStability = table( ...
    CaseCol,RegimeCol,GGMCol,ReferenceBest,ReferenceTQ, ...
    SameWinner_pct,Top2_pct,MeanRank,MinRank,MaxRank,N_Theta, ...
    'VariableNames',{ ...
    'Case','Regime','GGM','Reference_Best','Reference_TQ', ...
    'Same_Winner_pct','Top2_pct','Mean_Rank','Min_Rank','Max_Rank', ...
    'N_Theta'});

%% ========================================================================
% 5. CASE x REGIME SUMMARY
% ========================================================================

nSummary = 4;

CaseS = strings(nSummary,1);
RegimeS = strings(nSummary,1);
SameWinnerS = nan(nSummary,1);
Top2S = nan(nSummary,1);
MeanRankS = nan(nSummary,1);
WorstRankS = nan(nSummary,1);
N_GGM = repmat(6,nSummary,1);
N_ThetaS = repmat(nTheta,nSummary,1);

row = 0;

for ic = 1:numel(CaseList)

    for ir = 1:numel(RegimeList)

        row = row + 1;

        caseLabel = prettyCase(CaseList{ic});
        regimeLabel = string(RegimeList{ir});

        idx = ...
            ModelStability.Case == caseLabel & ...
            ModelStability.Regime == regimeLabel;

        CaseS(row) = caseLabel;
        RegimeS(row) = regimeLabel;

        SameWinnerS(row) = ...
            mean(ModelStability.Same_Winner_pct(idx));

        Top2S(row) = ...
            mean(ModelStability.Top2_pct(idx));

        MeanRankS(row) = ...
            mean(ModelStability.Mean_Rank(idx));

        WorstRankS(row) = ...
            max(ModelStability.Max_Rank(idx));

    end

end

RegimeSummary = table( ...
    CaseS,RegimeS,SameWinnerS,Top2S,MeanRankS,WorstRankS,N_GGM,N_ThetaS, ...
    'VariableNames',{ ...
    'Case','Regime','Same_Winner_pct','Top2_pct','Mean_Rank', ...
    'Worst_Rank','N_GGM','N_Theta'});

%% ========================================================================
% 6. STABILITY OF BROADER TQ COMPARISONS
%
% These tables do not redefine "comparable".  They report whether the
% DIRECTION of each reference-weight comparison is preserved as theta
% changes, together with the distribution of the signed TQ difference.
%
% Signed differences:
%   Kernel groups : mean(shape kernels) - mean(shape-free kernels)
%   RBF/Kriging   : mean(five RBF kernels) - Kriging
%   Local/Global  : mean(Local RBF) - mean(Global RBF)
%
% Negative = first-named group has lower/better TQ.
% ========================================================================

% ------------------------------------------------------------------------
% 6A. Shape-parameter kernels vs shape-free kernels
% -------------------------------------------------------------------------

CaseKG = strings(24,1);
RegimeKG = strings(24,1);
GGMKG = strings(24,1);
ReferenceDirectionKG = strings(24,1);
SameDirectionKG = nan(24,1);
MedianDiffKG = nan(24,1);
MinDiffKG = nan(24,1);
MaxDiffKG = nan(24,1);

row = 0;

for ic = 1:2

    CaseName = CaseList{ic};

    for ir = 1:2

        Regime = RegimeList{ir};

        QI = TQ.Test.(CaseName).(Regime).QI;
        Qmain = TQ.Test.(CaseName).(Regime).QI_Main;

        for ig = 1:6

            row = row + 1;

            dRef = ...
                mean(Qmain(ig,1:2)) - ...
                mean(Qmain(ig,3:5));

            dAll = nan(nTheta,1);

            for it = 1:nTheta
                dAll(it) = ...
                    mean(QI(ig,1:2,it)) - ...
                    mean(QI(ig,3:5,it));
            end

            refSign = signWithTol(dRef,TIE_TOL);
            same = arrayfun(@(x) signWithTol(x,TIE_TOL)==refSign,dAll);

            CaseKG(row) = prettyCase(CaseName);
            RegimeKG(row) = string(Regime);
            GGMKG(row) = ModelLabels(ig);
            ReferenceDirectionKG(row) = directionLabel( ...
                refSign,"Shape","Shape-free");
            SameDirectionKG(row) = 100*mean(same);
            MedianDiffKG(row) = median(dAll);
            MinDiffKG(row) = min(dAll);
            MaxDiffKG(row) = max(dAll);

        end

    end

end

KernelGroupStability = table( ...
    CaseKG,RegimeKG,GGMKG,ReferenceDirectionKG, ...
    SameDirectionKG,MedianDiffKG,MinDiffKG,MaxDiffKG, ...
    'VariableNames',{ ...
    'Case','Regime','GGM','Reference_Direction', ...
    'Same_Direction_pct','Median_Difference', ...
    'Min_Difference','Max_Difference'});

% ------------------------------------------------------------------------
% 6B. Mean RBF vs Kriging
% -------------------------------------------------------------------------

CaseRK = strings(24,1);
RegimeRK = strings(24,1);
GGMRK = strings(24,1);
ReferenceDirectionRK = strings(24,1);
SameDirectionRK = nan(24,1);
MedianDiffRK = nan(24,1);
MinDiffRK = nan(24,1);
MaxDiffRK = nan(24,1);

row = 0;

for ic = 1:2

    CaseName = CaseList{ic};

    for ir = 1:2

        Regime = RegimeList{ir};

        QI = TQ.Test.(CaseName).(Regime).QI;
        Qmain = TQ.Test.(CaseName).(Regime).QI_Main;

        for ig = 1:6

            row = row + 1;

            dRef = mean(Qmain(ig,1:5)) - Qmain(ig,6);
            dAll = nan(nTheta,1);

            for it = 1:nTheta
                dAll(it) = ...
                    mean(QI(ig,1:5,it)) - QI(ig,6,it);
            end

            refSign = signWithTol(dRef,TIE_TOL);
            same = arrayfun(@(x) signWithTol(x,TIE_TOL)==refSign,dAll);

            CaseRK(row) = prettyCase(CaseName);
            RegimeRK(row) = string(Regime);
            GGMRK(row) = ModelLabels(ig);
            ReferenceDirectionRK(row) = directionLabel( ...
                refSign,"RBF","Kriging");
            SameDirectionRK(row) = 100*mean(same);
            MedianDiffRK(row) = median(dAll);
            MinDiffRK(row) = min(dAll);
            MaxDiffRK(row) = max(dAll);

        end

    end

end

RBFKrigingStability = table( ...
    CaseRK,RegimeRK,GGMRK,ReferenceDirectionRK, ...
    SameDirectionRK,MedianDiffRK,MinDiffRK,MaxDiffRK, ...
    'VariableNames',{ ...
    'Case','Regime','GGM','Reference_Direction', ...
    'Same_Direction_pct','Median_RBF_minus_Kriging', ...
    'Min_RBF_minus_Kriging','Max_RBF_minus_Kriging'});

% ------------------------------------------------------------------------
% 6C. Local vs Global mean RBF
% -------------------------------------------------------------------------

CaseLG = strings(12,1);
GGMLG = strings(12,1);
ReferenceDirectionLG = strings(12,1);
SameDirectionLG = nan(12,1);
MedianDiffLG = nan(12,1);
MinDiffLG = nan(12,1);
MaxDiffLG = nan(12,1);

row = 0;

for ic = 1:2

    CaseName = CaseList{ic};

    QG = TQ.Test.(CaseName).Global.QI;
    QL = TQ.Test.(CaseName).Local.QI;

    QGmain = TQ.Test.(CaseName).Global.QI_Main;
    QLmain = TQ.Test.(CaseName).Local.QI_Main;

    for ig = 1:6

        row = row + 1;

        dRef = ...
            mean(QLmain(ig,1:5)) - ...
            mean(QGmain(ig,1:5));

        dAll = nan(nTheta,1);

        for it = 1:nTheta
            dAll(it) = ...
                mean(QL(ig,1:5,it)) - ...
                mean(QG(ig,1:5,it));
        end

        refSign = signWithTol(dRef,TIE_TOL);
        same = arrayfun(@(x) signWithTol(x,TIE_TOL)==refSign,dAll);

        CaseLG(row) = prettyCase(CaseName);
        GGMLG(row) = ModelLabels(ig);
        ReferenceDirectionLG(row) = directionLabel( ...
            refSign,"Local","Global");
        SameDirectionLG(row) = 100*mean(same);
        MedianDiffLG(row) = median(dAll);
        MinDiffLG(row) = min(dAll);
        MaxDiffLG(row) = max(dAll);

    end

end

LocalGlobalStability = table( ...
    CaseLG,GGMLG,ReferenceDirectionLG,SameDirectionLG, ...
    MedianDiffLG,MinDiffLG,MaxDiffLG, ...
    'VariableNames',{ ...
    'Case','GGM','Reference_Direction','Same_Direction_pct', ...
    'Median_Local_minus_Global','Min_Local_minus_Global', ...
    'Max_Local_minus_Global'});


%% ========================================================================
% 7. THETA GRID TABLE
% ========================================================================

ThetaGrid = array2table(Theta, ...
    'VariableNames',{'Theta1_RMSE','Theta2_ASE_RMSE','Theta3_RMSSE'});

ThetaGrid.Is_Reference = false(nTheta,1);
ThetaGrid.Is_Reference(idxMain) = true;

%% ========================================================================
% 8. WRITE EXCEL
% ========================================================================

if ~isfolder(OUTPUT_DIR)
    mkdir(OUTPUT_DIR);
end

if isfile(OUTPUT_XLSX)
    delete(OUTPUT_XLSX);
end

writetable(ModelStability,OUTPUT_XLSX, ...
    'Sheet','Model_Stability');

writetable(RegimeSummary,OUTPUT_XLSX, ...
    'Sheet','Regime_Summary');

writetable(KernelGroupStability,OUTPUT_XLSX, ...
    'Sheet','Kernel_Group_Stability');

writetable(RBFKrigingStability,OUTPUT_XLSX, ...
    'Sheet','RBF_vs_Kriging_Stability');

writetable(LocalGlobalStability,OUTPUT_XLSX, ...
    'Sheet','Local_vs_Global_Stability');

writetable(ThetaGrid,OUTPUT_XLSX, ...
    'Sheet','Theta_Grid');

%% ========================================================================
% 9. WRITE LATEX APPENDIX TABLE
% ========================================================================

if isfile(OUTPUT_TEX)
    delete(OUTPUT_TEX);
end

writeLatexSensitivity(ModelStability,OUTPUT_TEX);

%% ========================================================================
% 10. DISPLAY
% ========================================================================

fprintf('\n=============================================================\n');
fprintf(' TQ WEIGHT-SENSITIVITY ANALYSIS\n');
fprintf('=============================================================\n');
fprintf('Phase             = Test only\n');
fprintf('Scaling           = Four Case x Phase scales\n');
fprintf('Sensitivity scale = Case-specific TEST scale only\n');
fprintf('Cross-case TQ     = not compared\n');
fprintf('Cross-phase TQ    = not compared\n');
fprintf('Reference theta   = [0.50 0.25 0.25]\n');
fprintf('Sensitivity grid  = 0.10:0.05:0.80, sum(theta)=1\n');
fprintf('Theta combinations= %d\n',nTheta);
fprintf('Lower TQ          = better\n');
fprintf('=============================================================\n\n');

disp(RegimeSummary);

fprintf('\nDetailed model-selection stability:\n');
disp(ModelStability);

fprintf('\nExcel : %s\n',OUTPUT_XLSX);
fprintf('LaTeX : %s\n',OUTPUT_TEX);

%% ========================================================================
% 11. RETURN
% ========================================================================

Sensitivity = struct();
Sensitivity.ModelStability = ModelStability;
Sensitivity.RegimeSummary = RegimeSummary;
Sensitivity.KernelGroupStability = KernelGroupStability;
Sensitivity.RBFKrigingStability = RBFKrigingStability;
Sensitivity.LocalGlobalStability = LocalGlobalStability;
Sensitivity.ThetaGrid = ThetaGrid;

end


% =========================================================================
function s = prettyCase(caseName)

switch char(caseName)
    case 'Case1'
        s = "Case 1";
    case 'Case2'
        s = "Case 2";
    otherwise
        s = string(caseName);
end

end


% =========================================================================
function s = signWithTol(x,tol)

if x < -tol
    s = -1;
elseif x > tol
    s = 1;
else
    s = 0;
end

end


% =========================================================================
function label = directionLabel(s,firstName,secondName)

if s < 0
    label = firstName + " lower";
elseif s > 0
    label = secondName + " lower";
else
    label = "Tie";
end

end


% =========================================================================
function writeLatexSensitivity(T,fileName)

fid = fopen(fileName,'wt');

if fid < 0
    error('Cannot open LaTeX output:\n%s',fileName);
end

cleanup = onCleanup(@() fclose(fid));

fprintf(fid,'%% Auto-generated TQ weight-sensitivity table\n');
fprintf(fid,'\\begin{table}[htbp]\n');
fprintf(fid,'\\centering\n');
fprintf(fid,'\\small\n');
fprintf(fid,['\\caption{Sensitivity of the reference-weight best model to ' ...
             'alternative Total-Quality weight combinations on the ' ...
             'independent test data.}\\n']);
fprintf(fid,'\\label{tab:tq_weight_sensitivity}\n');
fprintf(fid,'\\begin{tabular}{lllrrrr}\n');
fprintf(fid,'\\hline\n');
fprintf(fid,['Case & Regime & GGM & Reference best & Same winner (\\%%) & ' ...
             'Top--2 (\\%%) & Rank range \\\\\n']);
fprintf(fid,'\\hline\n');

for i = 1:height(T)

    fprintf(fid,'%s & %s & %s & %s & %.1f & %.1f & %d--%d \\\\\n', ...
        escapeLatex(T.Case(i)), ...
        escapeLatex(T.Regime(i)), ...
        escapeLatex(T.GGM(i)), ...
        escapeLatex(T.Reference_Best(i)), ...
        T.Same_Winner_pct(i), ...
        T.Top2_pct(i), ...
        T.Min_Rank(i), ...
        T.Max_Rank(i));

    if i < height(T) && ...
       (T.Case(i) ~= T.Case(i+1) || T.Regime(i) ~= T.Regime(i+1))
        fprintf(fid,'\\cline{1-7}\n');
    end

end

fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fprintf(fid,'\\begin{flushleft}\n');
fprintf(fid,'\\footnotesize\n');
fprintf(fid,[ ...
    'Case~1 and Case~2 are scaled independently, with one common scale ' ...
    'within each case. Reference weights are $(\\theta_1,\\theta_2,\\theta_3)=' ...
    '(0.50,0.25,0.25)$. Sensitivity uses all 120 combinations with ' ...
    '$\\theta_j\\in[0.10,0.80]$ at increments of 0.05 and ' ...
    '$\\sum_j\\theta_j=1$. Same winner denotes the percentage of weight ' ...
    'combinations for which the reference-weight best candidate remains ' ...
    'rank~1; Top--2 denotes the percentage for which it remains within ' ...
    'the two highest-ranked candidates. Lower TQ is better.\\n']);
fprintf(fid,'\\end{flushleft}\n');
fprintf(fid,'\\end{table}\n');

clear cleanup

end


% =========================================================================
function s = escapeLatex(x)

s = char(string(x));
s = strrep(s,'\','\textbackslash{}');
s = strrep(s,'_','\_');
s = strrep(s,'&','\&');
s = strrep(s,'%','\%');
s = strrep(s,'#','\#');

end

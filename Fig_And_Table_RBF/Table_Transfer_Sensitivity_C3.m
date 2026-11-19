%% Build_Transfer_Sensitivity_Tables.m
% Compare nearest-centre, distance-weighted, and hybrid parameter-transfer runs.
% TEST results only; calibrated uncertainty (AfterAdjust).
%
% Required inputs are the final metric files produced by
% Gather_TQ_Metrics / Gather_TQ_Metrics_IDW / Gather_TQ_Metrics_Nearest:
%
%   Hybrid : RMSE_ASE_RMSSE_Metrix.mat
%   IDW    : RMSE_ASE_RMSSE_Metrix_IDW.mat
%   Nearest: RMSE_ASE_RMSSE_Metrix_Nearest.mat
%
% Metrics are read directly from:
%   RMSE_ASE_RMSSE_Metrix.Test.Case1.Local.RMSE
%   RMSE_ASE_RMSSE_Metrix.Test.Case1.Local.ASE
%   RMSE_ASE_RMSSE_Metrix.Test.Case1.Local.RMSSE
% and likewise for Case 2.
%
% Outputs:
%   Transfer_Sensitivity_Comparison.xlsx
%   Transfer_Sensitivity_Comparison.tex
%
% Compact manuscript table:
% Case | Transfer rule | RMSE | DeltaRMSE (%) | ASE | DeltaASE (%) | RMSSE
%
% Hybrid is the reference. Compact metric values are means across the
% matched model-kernel configurations. Detailed matched values are retained
% in the Excel workbook.

clear; clc;

%% Run
Collect_RMSE_ASE_RMSSE
Collect_RMSE_ASE_RMSSE_IDW
Collect_RMSE_ASE_RMSSE_Nearest
%% 1. USER SETTINGS ========================================================

RootDir = '.';

INPUT_FILE.Nearest = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'TQ_Nearest', ...
        'RMSE_ASE_RMSSE_Metrix_Nearest.mat');

INPUT_FILE.IDW = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'TQ_IDW', ...
        'RMSE_ASE_RMSSE_Metrix_IDW.mat');

INPUT_FILE.Hybrid = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'TQ', ...
        'RMSE_ASE_RMSSE_Metrix.mat');

OUT_DIR = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'Transfer_Sensitivity_Comparison');

EXCEL_OUT = fullfile(OUT_DIR,'Transfer_Sensitivity_Comparison.xlsx');

ROBUST_EXCEL_OUT = ...
    fullfile( ...
        OUT_DIR, ...
        'Cellwise_Robustness_Summary.xlsx');

LATEX_OUT = fullfile(OUT_DIR,'Transfer_Sensitivity_Comparison.tex');

% Current six-model order (SG1 removed)
MODELS6 = {'E08','GGP','EC4','GCO','SG2','X19'};

% Legacy seven-model order, used only if saved matrices still contain SG1
MODELS7 = {'E08','GGP','EC4','GCO','SG1','SG2','X19'};

KERNELS = {'Gaussian','Multiquadric','Thin plate','Cubic','Linear'};

REMOVE_SG1_IF_PRESENT = true;

if ~isfolder(OUT_DIR), mkdir(OUT_DIR); end
if isfile(EXCEL_OUT), delete(EXCEL_OUT); end
if isfile(ROBUST_EXCEL_OUT), delete(ROBUST_EXCEL_OUT); end
if isfile(LATEX_OUT), delete(LATEX_OUT); end

%% 2. LOAD GATHERED TQ METRICS FOR ALL THREE TRANSFER RUNS =================

methods = {'Nearest','IDW','Hybrid'};
D = struct();

for i = 1:numel(methods)

    mth = methods{i};

    D.(mth) = ...
        loadGatheredMetrics( ...
            INPUT_FILE.(mth), ...
            MODELS6, ...
            MODELS7, ...
            KERNELS, ...
            REMOVE_SG1_IF_PRESENT);

end

% Ensure all three runs are directly comparable
for c = {'Case1','Case2'}
    cf = c{1};
    ref = size(D.Hybrid.(cf).RMSE);
    for i = 1:numel(methods)
        mth = methods{i};
        assert(isequal(size(D.(mth).(cf).RMSE),ref), ...
            '%s %s RMSE dimensions differ from Hybrid.',mth,cf);
        assert(isequal(size(D.(mth).(cf).ASE),ref), ...
            '%s %s ASE dimensions differ from Hybrid.',mth,cf);
        assert(isequal(size(D.(mth).(cf).RMSSE),ref), ...
            '%s %s RMSSE dimensions differ from Hybrid.',mth,cf);
    end
end

%% 3. COMPACT MANUSCRIPT TABLE ============================================

rows = cell(0,7);
cases = {'Case1','Case2'};

for ic = 1:2
    cf = cases{ic};

    hybridRMSE = fmean(D.Hybrid.(cf).RMSE(:));
    hybridASE  = fmean(D.Hybrid.(cf).ASE(:));

    for im = 1:3
        mth = methods{im};

        rmse  = fmean(D.(mth).(cf).RMSE(:));
        ase   = fmean(D.(mth).(cf).ASE(:));
        rmsse = fmean(D.(mth).(cf).RMSSE(:));

        dRMSE = relpct(rmse,hybridRMSE);
        dASE  = relpct(ase,hybridASE);

        rows(end+1,:) = {caseName(cf),transferName(mth), ...
                         rmse,dRMSE,ase,dASE,rmsse}; %#ok<SAGROW>
    end
end

TCompact = cell2table(rows,'VariableNames', ...
    {'Case','TransferRule','RMSE','DeltaRMSE_pct','ASE','DeltaASE_pct','RMSSE'});

%% 4. DETAILED MATCHED TABLES =============================================

TCase1 = detailedTable(D,'Case1');
TCase2 = detailedTable(D,'Case2');

%% 5. ROBUSTNESS SUMMARY ==================================================
% Cellwise differences relative to Hybrid.
% Absolute differences are included here so opposite signed changes cannot
% cancel each other in the sensitivity assessment.

rows = cell(0,8);
for ic = 1:2
    cf = cases{ic};

    for im = 1:2  % Nearest and IDW versus Hybrid
        mth = methods{im};

        dR = relpctArray(D.(mth).(cf).RMSE,D.Hybrid.(cf).RMSE);
        dA = relpctArray(D.(mth).(cf).ASE,D.Hybrid.(cf).ASE);
        dS = D.(mth).(cf).RMSSE - D.Hybrid.(cf).RMSSE;

        rows(end+1,:) = {caseName(cf),transferName(mth), ...
            fmean(abs(dR(:))),fmax(abs(dR(:))), ...
            fmean(abs(dA(:))),fmax(abs(dA(:))), ...
            fmean(abs(dS(:))),fmax(abs(dS(:)))}; %#ok<SAGROW>
    end
end

TRobust = cell2table(rows,'VariableNames', ...
    {'Case','TransferRule', ...
     'MeanAbsDeltaRMSE_pct','MaxAbsDeltaRMSE_pct', ...
     'MeanAbsDeltaASE_pct','MaxAbsDeltaASE_pct', ...
     'MeanAbsDeltaRMSSE','MaxAbsDeltaRMSSE'});

%% 6. EXCEL EXPORT =========================================================

caption = ['Sensitivity of prediction accuracy and uncertainty to the ' ...
           'parameter-transfer rule at unseen test locations. Hybrid ' ...
           'is the reference scheme. Compact values are means ' ...
           'across matched model-kernel configurations.'];

writecell({caption},EXCEL_OUT,'Sheet','Compact','Range','A1');
writetable(TCompact,EXCEL_OUT,'Sheet','Compact','Range','A3');

writecell({ ...
    'DeltaRMSE (%) = 100*(RMSE_method - RMSE_Hybrid)/RMSE_Hybrid'; ...
    'DeltaASE (%)  = 100*(ASE_method - ASE_Hybrid)/ASE_Hybrid'; ...
    'Positive DeltaRMSE means larger RMSE than Hybrid; negative means smaller.'; ...
    'DeltaASE describes change in uncertainty magnitude; smaller ASE is not automatically better.'; ...
    'RMSSE should be interpreted by closeness to 1.'; ...
    'Distance-weighted transfer refers only to transfer of optimized local RBF parameters; it is distinct from the IDW interpolation benchmark.'; ...
    'Nearest-centre transfer refers only to parameter transfer; it is distinct from nearest-neighbour interpolation.'}, ...
    EXCEL_OUT,'Sheet','Compact','Range','A11');

% -------------------------------------------------------------------------
% Keep the cellwise robustness summary in the main Excel workbook.
% -------------------------------------------------------------------------

robustCaption = [ ...
    'Cellwise robustness relative to Hybrid transfer. MeanAbs and MaxAbs values are ' ...
    'computed from matched model-kernel cells so positive and negative ' ...
    'changes cannot cancel.'];

writecell( ...
    {robustCaption}, ...
    EXCEL_OUT, ...
    'Sheet','Cellwise Robustness', ...
    'Range','A1');

writetable( ...
    TRobust, ...
    EXCEL_OUT, ...
    'Sheet','Cellwise Robustness', ...
    'Range','A3');


% -------------------------------------------------------------------------
% Also save the same robustness summary as a separate Excel table.
% This makes it easy to use as reviewer-response / supplementary material.
% -------------------------------------------------------------------------

writecell( ...
    {robustCaption}, ...
    ROBUST_EXCEL_OUT, ...
    'Sheet','Cellwise Robustness', ...
    'Range','A1');

writetable( ...
    TRobust, ...
    ROBUST_EXCEL_OUT, ...
    'Sheet','Cellwise Robustness', ...
    'Range','A3');

writecell({ ...
    'MeanAbsDeltaRMSE_pct = mean absolute cellwise percentage RMSE change vs Hybrid'; ...
    'MaxAbsDeltaRMSE_pct  = maximum absolute cellwise percentage RMSE change vs Hybrid'; ...
    'MeanAbsDeltaASE_pct  = mean absolute cellwise percentage ASE change vs Hybrid'; ...
    'MaxAbsDeltaASE_pct   = maximum absolute cellwise percentage ASE change vs Hybrid'; ...
    'MeanAbsDeltaRMSSE    = mean absolute cellwise RMSSE difference vs Hybrid'; ...
    'MaxAbsDeltaRMSSE     = maximum absolute cellwise RMSSE difference vs Hybrid'}, ...
    ROBUST_EXCEL_OUT, ...
    'Sheet','Cellwise Robustness', ...
    'Range','A10');


% Detailed matched cells remain available in the main workbook.
writetable(TCase1,EXCEL_OUT,'Sheet','Detailed Case 1','Range','A1');
writetable(TCase2,EXCEL_OUT,'Sheet','Detailed Case 2','Range','A1');

%% 7. LATEX EXPORT =========================================================

writeLatex(LATEX_OUT,TCompact);

%% 8. DISPLAY ===============================================================

fprintf('\n=============================================================\n');
fprintf(' COMPACT TRANSFER SENSITIVITY TABLE\n');
fprintf('=============================================================\n');
disp(TCompact);

fprintf('\n=============================================================\n');
fprintf(' CELLWISE ROBUSTNESS SUMMARY VS HYBRID\n');
fprintf('=============================================================\n');
disp(TRobust);

fprintf('\nMain Excel: %s\n',EXCEL_OUT);
fprintf('Cellwise robustness Excel: %s\n',ROBUST_EXCEL_OUT);
fprintf('LaTeX: %s\n',LATEX_OUT);

%% LOCAL FUNCTIONS =========================================================

function D = loadGatheredMetrics( ...
    inputFile,models6,models7,kernels,removeSG1)

    if ~isfile(inputFile)

        error( ...
            'Gathered metric file not found:\n%s', ...
            inputFile);

    end


    S = ...
        load( ...
            inputFile, ...
            'RMSE_ASE_RMSSE_Metrix');


    if ~isfield(S,'RMSE_ASE_RMSSE_Metrix')

        error( ...
            ['Variable RMSE_ASE_RMSSE_Metrix was not found in:\n%s'], ...
            inputFile);

    end


    M = ...
        S.RMSE_ASE_RMSSE_Metrix;


    % ---------------------------------------------------------------------
    % Read final TEST LOCAL metrics directly from Gather_TQ_Metrics output.
    % ---------------------------------------------------------------------

    requiredCases = { ...
        'Case1', ...
        'Case2'};

    requiredMetrics = { ...
        'RMSE', ...
        'ASE', ...
        'RMSSE'};


    for ic = 1:numel(requiredCases)

        cf = ...
            requiredCases{ic};

        assert( ...
            isfield(M,'Test') && ...
            isfield(M.Test,cf) && ...
            isfield(M.Test.(cf),'Local'), ...
            'Missing Test.%s.Local in:\n%s', ...
            cf, ...
            inputFile);


        for imetric = 1:numel(requiredMetrics)

            metric = ...
                requiredMetrics{imetric};

            assert( ...
                isfield(M.Test.(cf).Local,metric), ...
                'Missing Test.%s.Local.%s in:\n%s', ...
                cf, ...
                metric, ...
                inputFile);

        end

    end


    D.Case1.RMSE = ...
        M.Test.Case1.Local.RMSE;

    D.Case1.ASE = ...
        M.Test.Case1.Local.ASE;

    D.Case1.RMSSE = ...
        M.Test.Case1.Local.RMSSE;


    D.Case2.RMSE = ...
        M.Test.Case2.Local.RMSE;

    D.Case2.ASE = ...
        M.Test.Case2.Local.ASE;

    D.Case2.RMSSE = ...
        M.Test.Case2.Local.RMSSE;


    % ---------------------------------------------------------------------
    % Internal consistency.
    % ---------------------------------------------------------------------

    for cf = {'Case1','Case2'}

        X = ...
            D.(cf{1});

        assert( ...
            isequal( ...
                size(X.RMSE), ...
                size(X.ASE), ...
                size(X.RMSSE)), ...
            'RMSE/ASE/RMSSE size mismatch in %s for:\n%s', ...
            cf{1}, ...
            inputFile);

    end


    % ---------------------------------------------------------------------
    % Labels.
    %
    % Prefer labels already saved by Gather_TQ_Metrics. Fall back to the
    % known six/seven model and five-kernel orders if not present.
    % ---------------------------------------------------------------------

    nr = ...
        size(D.Case1.RMSE,1);

    nk = ...
        size(D.Case1.RMSE,2);


    % ---------------------------------------------------------------------
    % Keep RBF kernels only.
    %
    % Gather_TQ_Metrics may append Kriging as an additional column after
    % the five RBF kernels.  Transfer sensitivity is an RBF-only comparison,
    % so remove any appended non-RBF column(s) here.
    % ---------------------------------------------------------------------

    nRBFKernel = ...
        numel(kernels);


    if nk > nRBFKernel

        fprintf( ...
            ['Detected %d metric columns in %s. ' ...
             'Keeping the first %d RBF-kernel columns and removing %d ' ...
             'appended non-RBF column(s) (e.g. Kriging).\n'], ...
            nk, ...
            inputFile, ...
            nRBFKernel, ...
            nk-nRBFKernel);


        keepCols = ...
            1:nRBFKernel;


        for cf = {'Case1','Case2'}

            f = ...
                cf{1};

            D.(f).RMSE = ...
                D.(f).RMSE(:,keepCols);

            D.(f).ASE = ...
                D.(f).ASE(:,keepCols);

            D.(f).RMSSE = ...
                D.(f).RMSSE(:,keepCols);

        end


        nk = ...
            nRBFKernel;

    elseif nk < nRBFKernel

        error( ...
            ['Found only %d metric columns, but %d RBF kernels are expected.\n' ...
             'Input file:\n%s'], ...
            nk, ...
            nRBFKernel, ...
            inputFile);

    end


    if isfield(M,'ModelShort') && numel(M.ModelShort) == nr

        savedModelNames = ...
            M.ModelShort;

    elseif nr == numel(models6)

        savedModelNames = ...
            models6;

    elseif nr == numel(models7)

        savedModelNames = ...
            models7;

    else

        error( ...
            'Unexpected number of model rows (%d) in:\n%s', ...
            nr, ...
            inputFile);

    end


    % After removal of any appended Kriging/non-RBF column, exactly the
    % five RBF kernel columns must remain.
    if nk ~= numel(kernels)

        error( ...
            ['After retaining RBF kernels, found %d columns, but KERNELS ' ...
             'defines %d.\nInput file:\n%s'], ...
            nk, ...
            numel(kernels), ...
            inputFile);

    end


    % ---------------------------------------------------------------------
    % Compatibility with a legacy seven-model matrix containing SG1.
    % ---------------------------------------------------------------------

    if nr == numel(models7) && removeSG1

        idxSG1 = ...
            find( ...
                strcmpi(savedModelNames,'SG1'), ...
                1);

        if isempty(idxSG1)

            % Gather_TQ_Metrics may save the full names instead of short names.
            if isfield(M,'Models')

                idxSG1 = ...
                    find( ...
                        strcmpi(M.Models,'sgg1'), ...
                        1);

            end

        end


        if isempty(idxSG1)

            % Known legacy ordering:
            % E08, GGP, EC4, GCO, SG1, SG2, X19
            idxSG1 = ...
                5;

        end


        for cf = {'Case1','Case2'}

            f = ...
                cf{1};

            D.(f).RMSE(idxSG1,:) = ...
                [];

            D.(f).ASE(idxSG1,:) = ...
                [];

            D.(f).RMSSE(idxSG1,:) = ...
                [];

        end


        D.ModelNames = ...
            models6;

    else

        D.ModelNames = ...
            savedModelNames;

    end


    D.KernelNames = ...
        kernels;

end


function T = detailedTable(D,cf)

    M = D.Hybrid.ModelNames;
    K = D.Hybrid.KernelNames;

    n = numel(M)*numel(K);

    Model = strings(n,1);
    Kernel = strings(n,1);

    NR = nan(n,1); NDR = nan(n,1);
    IR = nan(n,1); IDR = nan(n,1);
    HR = nan(n,1);

    NA = nan(n,1); NDA = nan(n,1);
    IA = nan(n,1); IDA = nan(n,1);
    HA = nan(n,1);

    NS = nan(n,1); IS = nan(n,1); HS = nan(n,1);

    q = 0;
    for i = 1:numel(M)
        for j = 1:numel(K)
            q = q+1;

            Model(q)  = string(M{i});
            Kernel(q) = string(K{j});

            NR(q) = D.Nearest.(cf).RMSE(i,j);
            IR(q) = D.IDW.(cf).RMSE(i,j);
            HR(q) = D.Hybrid.(cf).RMSE(i,j);
            NDR(q)= relpct(NR(q),HR(q));
            IDR(q)= relpct(IR(q),HR(q));

            NA(q) = D.Nearest.(cf).ASE(i,j);
            IA(q) = D.IDW.(cf).ASE(i,j);
            HA(q) = D.Hybrid.(cf).ASE(i,j);
            NDA(q)= relpct(NA(q),HA(q));
            IDA(q)= relpct(IA(q),HA(q));

            NS(q) = D.Nearest.(cf).RMSSE(i,j);
            IS(q) = D.IDW.(cf).RMSSE(i,j);
            HS(q) = D.Hybrid.(cf).RMSSE(i,j);
        end
    end

    T = table(Model,Kernel, ...
        NR,NDR,IR,IDR,HR, ...
        NA,NDA,IA,IDA,HA, ...
        NS,IS,HS, ...
        'VariableNames',{ ...
        'Model','Kernel', ...
        'Nearest_RMSE','Nearest_DeltaRMSE_pct', ...
        'IDW_RMSE','IDW_DeltaRMSE_pct','Hybrid_RMSE', ...
        'Nearest_ASE','Nearest_DeltaASE_pct', ...
        'IDW_ASE','IDW_DeltaASE_pct','Hybrid_ASE', ...
        'Nearest_RMSSE','IDW_RMSSE','Hybrid_RMSSE'});
end

function y = relpct(x,r)
    if ~(isfinite(x) && isfinite(r)) || abs(r) <= eps
        y = NaN;
    else
        y = 100*(x-r)/r;
    end
end

function Y = relpctArray(X,R)
    assert(isequal(size(X),size(R)),'Matched arrays must have equal size.');
    Y = nan(size(X));
    ok = isfinite(X) & isfinite(R) & abs(R)>eps;
    Y(ok) = 100*(X(ok)-R(ok))./R(ok);
end

function y = fmean(x)
    x = x(isfinite(x));
    if isempty(x), y = NaN; else, y = mean(x); end
end

function y = fmax(x)
    x = x(isfinite(x));
    if isempty(x), y = NaN; else, y = max(x); end
end

function s = caseName(cf)
    if strcmp(cf,'Case1'), s='Case 1'; else, s='Case 2'; end
end

function s = transferName(m)
    switch lower(m)
        case 'nearest'
            s = 'Nearest-centre transfer';
        case 'idw'
            s = 'Distance-weighted transfer';
        case 'hybrid'
            s = 'Hybrid transfer';
        otherwise
            s = m;
    end
end

function writeLatex(fileName,T)

    fid = fopen(fileName,'w');
    if fid < 0, error('Cannot create %s',fileName); end
    cc = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid,'%% Auto-generated transfer sensitivity table\n');
    fprintf(fid,'\\begin{table}[t]\n');
    fprintf(fid,'\\centering\n');
    fprintf(fid,['\\caption{Sensitivity of prediction accuracy and uncertainty ' ...
                 'to the parameter-transfer rule at unseen test locations.}\n']);
    fprintf(fid,'\\label{tab:transfer_sensitivity}\n');
    fprintf(fid,'\\footnotesize\n');
    fprintf(fid,'\\setlength{\\tabcolsep}{4pt}\n');
    fprintf(fid,'\\renewcommand{\\arraystretch}{1.08}\n');
    fprintf(fid,'\\begin{tabular}{llrrrrr}\n');
    fprintf(fid,'\\toprule\n');
    fprintf(fid,['\\textbf{Case} & \\textbf{Transfer rule} & \\textbf{RMSE} & ' ...
        '$\\boldsymbol{\\Delta}$\\textbf{RMSE (\\%%)} & \\textbf{ASE} & ' ...
        '$\\boldsymbol{\\Delta}$\\textbf{ASE (\\%%)} & \\textbf{RMSSE} \\\\\n']);
    fprintf(fid,'\\midrule\n');

    previous = '';
    for i = 1:height(T)
        thisCase = char(string(T.Case{i}));

        if i > 1 && ~strcmp(thisCase,previous)
            fprintf(fid,'\\midrule\n');
        end

        if strcmp(thisCase,previous)
            caseOut = '';
        else
            caseOut = thisCase;
        end

        tr = char(string(T.TransferRule{i}));

        fprintf(fid,'%s & %s & %.5f & %.2f & %.5f & %.2f & %.5f \\\\\n', ...
            caseOut,tr,T.RMSE(i),T.DeltaRMSE_pct(i), ...
            T.ASE(i),T.DeltaASE_pct(i),T.RMSSE(i));

        previous = thisCase;
    end

    fprintf(fid,'\\bottomrule\n');
    fprintf(fid,'\\end{tabular}\n');
    fprintf(fid,'\\vspace{2pt}\n');
    fprintf(fid,'\\begin{minipage}{0.98\\linewidth}\n');
    fprintf(fid,'\\scriptsize ');
    fprintf(fid,['Hybrid transfer is the reference transfer scheme. ' ...
        'RMSE, ASE, and RMSSE are means across matched model--kernel ' ...
        'configurations. $\\Delta\\mathrm{RMSE}=100(' ...
        '\\mathrm{RMSE}_{\\mathrm{method}}-\\mathrm{RMSE}_{\\mathrm{Hybrid}})' ...
        '/\\mathrm{RMSE}_{\\mathrm{Hybrid}}$, and $\\Delta\\mathrm{ASE}$ is ' ...
        'defined analogously. Positive $\\Delta\\mathrm{RMSE}$ indicates ' ...
        'larger RMSE than Hybrid. $\\Delta\\mathrm{ASE}$ describes a change ' ...
        'in uncertainty magnitude and is not itself a measure of improvement. ']);
    fprintf(fid,'\\end{minipage}\n');
    fprintf(fid,'\\end{table}\n');
end

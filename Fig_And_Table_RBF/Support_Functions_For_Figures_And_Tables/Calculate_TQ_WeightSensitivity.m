%% ========================================================================
% CALCULATE TOTAL QUALITY INDEX - TQ
%
% CASE x PHASE-SPECIFIC TQ SCALES:
%
%   Case 1 - Train: Global + Local + RBF + Kriging
%   Case 1 - Test : Global + Local + RBF + Kriging
%   Case 2 - Train: Global + Local + RBF + Kriging
%   Case 2 - Test : Global + Local + RBF + Kriging
%
% The four Case x Phase populations are scaled independently.
% Therefore, absolute TQ values are intended for comparison only WITHIN
% the same Case x Phase population, not across cases or across phases.
%
% Rows:
%   E08, GGP, EC4, GCO, SG2, X19
%
% Columns:
%   1 = RBF-G
%   2 = RBF-M
%   3 = RBF-T
%   4 = RBF-C
%   5 = RBF-L
%   6 = Kriging-S
%
% TQ =
%   theta1*N_RMSE ...
% + theta2*N_Delta ...
% + theta3*N_RMSSE1
%
% where:
%   Delta  = |ASE - RMSE|
%   RMSSE1 = |RMSSE - 1|
%
% Lower TQ = better.
% ========================================================================

clear
clc


%% ========================================================================
% 1. USER INPUT
% ========================================================================

InputFile = ...
    'OutPut_RBF_2026_2\TQ\RMSE_ASE_RMSSE_Metrix.mat';


OutputFile = ...
    'OutPut_RBF_2026_2\TQ\TQ_Metrix.mat';


%% ------------------------------------------------------------------------
% Fixed scaling parameters
% -------------------------------------------------------------------------

tau = ...
    0.08;


RMSEPercentile = ...
   75;


%% ------------------------------------------------------------------------
% Theta ranges
% -------------------------------------------------------------------------

% Sensitivity grid:
%   theta_j = 0.10, 0.15, ..., 0.80
%   theta1 + theta2 + theta3 = 1
%
% This gives 120 admissible combinations and includes the reference
% setting [0.50 0.25 0.25].
Theta1 = ...
    0.10:0.05:0.80;


Theta2 = ...
    0.10:0.05:0.80;


Theta3 = ...
    0.10:0.05:0.80;


%% ------------------------------------------------------------------------
% Main weights used for manuscript figure
% -------------------------------------------------------------------------

MainTheta = ...
    [0.50 0.25 0.25];


%% ========================================================================
% 2. LOAD RMSE / ASE / RMSSE
% ========================================================================

load( ...
    InputFile, ...
    'RMSE_ASE_RMSSE_Metrix');


%% ========================================================================
% 3. CREATE VALID THETA COMBINATIONS
%
% theta1 + theta2 + theta3 = 1
% ========================================================================

Theta = [];


for t1 = Theta1

    for t2 = Theta2

        for t3 = Theta3

            if abs(t1+t2+t3-1) < 1e-10

                Theta = ...
                    [Theta; t1 t2 t3];

            end

        end

    end

end


nTheta = ...
    size(Theta,1);


if nTheta == 0

    error( ...
        'No valid theta combinations.');

end


%% ========================================================================
% 4. FIND MAIN THETA
% ========================================================================

idxMainTheta = ...
    find( ...
        abs(Theta(:,1)-MainTheta(1)) < 1e-10 & ...
        abs(Theta(:,2)-MainTheta(2)) < 1e-10 & ...
        abs(Theta(:,3)-MainTheta(3)) < 1e-10, ...
        1);


if isempty(idxMainTheta)

    error( ...
        'Main theta [0.50 0.25 0.25] is not included in the theta grid.');

end


ExpectedNTheta = 120;

if nTheta ~= ExpectedNTheta
    error( ...
        'Expected %d admissible theta combinations, but found %d.', ...
        ExpectedNTheta,nTheta);
end

fprintf( ...
    '\nValid theta combinations = %d\n', ...
    nTheta);


%% ========================================================================
% 5. LISTS
% ========================================================================

PhaseList = ...
    { ...
    'Train', ...
    'Test' ...
    };


CaseList = ...
    { ...
    'Case1', ...
    'Case2' ...
    };


RegimeList = ...
    { ...
    'Global', ...
    'Local' ...
    };


%% ========================================================================
% 6. BUILD FOUR REFERENCE POPULATIONS: CASE x PHASE
%
% A separate scale is constructed for:
%
%   Case1.Train
%   Case1.Test
%   Case2.Train
%   Case2.Test
%
% Within EACH Case x Phase population, the scale includes:
%
%   Global + Local
%   RBF columns 1:5
%   Kriging column 6
%
% Global and Local Kriging are both included because they are distinct
% results in the metric matrix.
%
% IMPORTANT:
%   - Global and Local remain directly comparable within a Case x Phase.
%   - RBF and Kriging remain directly comparable within a Case x Phase.
%   - Train and Test do NOT share a scale.
%   - Case 1 and Case 2 do NOT share a scale.
%   - The same fixed Test scale is used for all 120 theta combinations
%     in the weight-sensitivity analysis; only theta changes.
% ========================================================================

StageScale = ...
    struct();


for ic = 1:numel(CaseList)

    CaseName = ...
        CaseList{ic};


    for ip = 1:numel(PhaseList)

        Phase = ...
            PhaseList{ip};


        G = ...
            RMSE_ASE_RMSSE_Metrix. ...
            (Phase). ...
            (CaseName).Global;


        L = ...
            RMSE_ASE_RMSSE_Metrix. ...
            (Phase). ...
            (CaseName).Local;


        %% ---------------------------------------------------------------
        % CHECK
        % ---------------------------------------------------------------

        if ~isequal(size(G.RMSE),[6 6]) || ...
           ~isequal(size(G.ASE),[6 6]) || ...
           ~isequal(size(G.RMSSE),[6 6]) || ...
           ~isequal(size(L.RMSE),[6 6]) || ...
           ~isequal(size(L.ASE),[6 6]) || ...
           ~isequal(size(L.RMSSE),[6 6])

            error( ...
                ['%s %s: RMSE, ASE and RMSSE must all be 6 x 6.'], ...
                Phase, ...
                CaseName);

        end


        %% ---------------------------------------------------------------
        % REFERENCE POPULATION FOR THIS CASE x PHASE
        % ---------------------------------------------------------------

        RMSE_All = ...
            [ ...
                reshape(G.RMSE(:,1:5),[],1); ...
                reshape(L.RMSE(:,1:5),[],1); ...
                G.RMSE(:,6); ...
                L.RMSE(:,6) ...
            ];


        Delta_All = ...
            [ ...
                reshape(abs(G.ASE(:,1:5)-G.RMSE(:,1:5)),[],1); ...
                reshape(abs(L.ASE(:,1:5)-L.RMSE(:,1:5)),[],1); ...
                abs(G.ASE(:,6)-G.RMSE(:,6)); ...
                abs(L.ASE(:,6)-L.RMSE(:,6)) ...
            ];


        RMSSE1_All = ...
            [ ...
                reshape(abs(G.RMSSE(:,1:5)-1),[],1); ...
                reshape(abs(L.RMSSE(:,1:5)-1),[],1); ...
                abs(G.RMSSE(:,6)-1); ...
                abs(L.RMSSE(:,6)-1) ...
            ];


        %% ---------------------------------------------------------------
        % CASE x PHASE-SPECIFIC SCALE
        % ---------------------------------------------------------------

        StageScale.(CaseName).(Phase).RMSEmin = ...
            min(RMSE_All);


        StageScale.(CaseName).(Phase).RMSEmax = ...
            max(RMSE_All);


        StageScale.(CaseName).(Phase).RMSEp = ...
            prctile( ...
                RMSE_All, ...
                RMSEPercentile);


        StageScale.(CaseName).(Phase).DeltaMin = ...
            min(Delta_All);


        StageScale.(CaseName).(Phase).DeltaMax = ...
            max(Delta_All);


        StageScale.(CaseName).(Phase).RMSSE1Min = ...
            min(RMSSE1_All);


        StageScale.(CaseName).(Phase).RMSSE1Max = ...
            max(RMSSE1_All);


        StageScale.(CaseName).(Phase).NReferenceRMSE = ...
            numel(RMSE_All);


        StageScale.(CaseName).(Phase).NReferenceDelta = ...
            numel(Delta_All);


        StageScale.(CaseName).(Phase).NReferenceRMSSE1 = ...
            numel(RMSSE1_All);


        fprintf('\n=============================================================\n');
        fprintf(' TQ SCALE: %s - %s\n',CaseName,Phase);
        fprintf('=============================================================\n');

        fprintf('RMSEmin    = %.8f\n', ...
            StageScale.(CaseName).(Phase).RMSEmin);

        fprintf('RMSE_%g    = %.8f\n', ...
            RMSEPercentile, ...
            StageScale.(CaseName).(Phase).RMSEp);

        fprintf('RMSEmax    = %.8f\n', ...
            StageScale.(CaseName).(Phase).RMSEmax);

        fprintf('DeltaMin   = %.8f\n', ...
            StageScale.(CaseName).(Phase).DeltaMin);

        fprintf('DeltaMax   = %.8f\n', ...
            StageScale.(CaseName).(Phase).DeltaMax);

        fprintf('RMSSE1Min  = %.8f\n', ...
            StageScale.(CaseName).(Phase).RMSSE1Min);

        fprintf('RMSSE1Max  = %.8f\n', ...
            StageScale.(CaseName).(Phase).RMSSE1Max);

        fprintf('=============================================================\n');

    end

end


%% ========================================================================
% 7. CHECK ALL FOUR CASE x PHASE SCALES
% ========================================================================

for ic = 1:numel(CaseList)

    CaseName = ...
        CaseList{ic};


    for ip = 1:numel(PhaseList)

        Phase = ...
            PhaseList{ip};


        S = ...
            StageScale.(CaseName).(Phase);


        assert(isfinite(S.RMSEmin) && ...
               isfinite(S.RMSEp) && ...
               isfinite(S.RMSEmax) && ...
               isfinite(S.DeltaMin) && ...
               isfinite(S.DeltaMax) && ...
               isfinite(S.RMSSE1Min) && ...
               isfinite(S.RMSSE1Max), ...
            '%s %s scale contains non-finite values.',CaseName,Phase);


        assert(S.RMSEmin <= S.RMSEp && ...
               S.RMSEp <= S.RMSEmax, ...
            '%s %s RMSE scale is inconsistent.',CaseName,Phase);


        assert(S.DeltaMin <= S.DeltaMax, ...
            '%s %s Delta scale is inconsistent.',CaseName,Phase);


        assert(S.RMSSE1Min <= S.RMSSE1Max, ...
            '%s %s RMSSE scale is inconsistent.',CaseName,Phase);

    end

end


%% ========================================================================
% 8. INITIALISE OUTPUT
% ========================================================================

TQ_Metrix = ...
    struct();


TQ_Metrix.Theta = ...
    Theta;


TQ_Metrix.MainTheta = ...
    MainTheta;


TQ_Metrix.MainThetaIndex = ...
    idxMainTheta;


TQ_Metrix.Sensitivity.ThetaMin = ...
    0.10;


TQ_Metrix.Sensitivity.ThetaMax = ...
    0.80;


TQ_Metrix.Sensitivity.ThetaStep = ...
    0.05;


TQ_Metrix.Sensitivity.NumberOfCombinations = ...
    nTheta;


%% ========================================================================
% 9. CALCULATE TQ USING THE CASE x PHASE-SPECIFIC SCALE
% ========================================================================

for ip = 1:numel(PhaseList)

    Phase = ...
        PhaseList{ip};


    for ic = 1:numel(CaseList)

        CaseName = ...
            CaseList{ic};


        % Use the scale belonging to the current Case x Phase.
        RMSEmin = ...
            StageScale.(CaseName).(Phase).RMSEmin;

        RMSEp = ...
            StageScale.(CaseName).(Phase).RMSEp;

        RMSEmax = ...
            StageScale.(CaseName).(Phase).RMSEmax;

        DeltaMin = ...
            StageScale.(CaseName).(Phase).DeltaMin;

        DeltaMax = ...
            StageScale.(CaseName).(Phase).DeltaMax;

        RMSSE1Min = ...
            StageScale.(CaseName).(Phase).RMSSE1Min;

        RMSSE1Max = ...
            StageScale.(CaseName).(Phase).RMSSE1Max;


        for ir = 1:numel(RegimeList)

            Regime = ...
                RegimeList{ir};


            %% ===========================================================
            % GET DATA
            % ===========================================================

            D = ...
                RMSE_ASE_RMSSE_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime);


            RMSE = ...
                D.RMSE;


            ASE = ...
                D.ASE;


            RMSSE = ...
                D.RMSSE;


            %% ===========================================================
            % ORIGINAL CRITERIA
            % ===========================================================

            Delta = ...
                abs(ASE-RMSE);


            RMSSE1 = ...
                abs(RMSSE-1);


            %% ===========================================================
            % N_RMSE
            %
            % RMSE remains in metres and is evaluated on the current Case x Phase scale.
            % ===========================================================

            N_RMSE = ...
                RMSE;


            %% ===========================================================
            % N_DELTA
            %
            % Case x Phase-specific mapping:
            %
            % [DeltaMin,DeltaMax]
            %
            %       ->
            %
            % [RMSEmin,RMSEmax]
            % ===========================================================

            if DeltaMax > DeltaMin

                N_Delta = ...
                    RMSEmin + ...
                    (Delta-DeltaMin) ./ ...
                    (DeltaMax-DeltaMin) .* ...
                    (RMSEmax-RMSEmin);

            else

                N_Delta = ...
                    RMSEmin .* ...
                    ones(size(Delta));

            end


            %% ===========================================================
            % N_RMSSE1
            %
            % Case x Phase-specific piecewise mapping:
            %
            % |RMSSE-1| <= tau
            %
            %       -> [RMSEmin,RMSEp]
            %
            % |RMSSE-1| > tau
            %
            %       -> [RMSEp,RMSEmax]
            % ===========================================================

            N_RMSSE1 = ...
                nan(size(RMSSE1));


            idxLow = ...
                RMSSE1 <= tau;


            idxHigh = ...
                RMSSE1 > tau;


            %% -----------------------------------------------------------
            % LOWER REGION
            % -----------------------------------------------------------

            if any(idxLow,'all')

                if tau > RMSSE1Min

                    N_RMSSE1(idxLow) = ...
                        RMSEmin + ...
                        ( ...
                            RMSSE1(idxLow) - ...
                            RMSSE1Min ...
                        ) ./ ...
                        ( ...
                            tau - ...
                            RMSSE1Min ...
                        ) .* ...
                        ( ...
                            RMSEp - ...
                            RMSEmin ...
                        );

                else

                    N_RMSSE1(idxLow) = ...
                        RMSEp;

                end

            end


            %% -----------------------------------------------------------
            % UPPER REGION
            % -----------------------------------------------------------

            if any(idxHigh,'all')

                if RMSSE1Max > tau

                    N_RMSSE1(idxHigh) = ...
                        RMSEp + ...
                        ( ...
                            RMSSE1(idxHigh) - ...
                            tau ...
                        ) ./ ...
                        ( ...
                            RMSSE1Max - ...
                            tau ...
                        ) .* ...
                        ( ...
                            RMSEmax - ...
                            RMSEp ...
                        );

                else

                    N_RMSSE1(idxHigh) = ...
                        RMSEp;

                end

            end


            %% ===========================================================
            % CALCULATE TQ FOR ALL THETA COMBINATIONS
            % ===========================================================

            QI = ...
                nan( ...
                    size(RMSE,1), ...
                    size(RMSE,2), ...
                    nTheta);


            for k = 1:nTheta

                QI(:,:,k) = ...
                    Theta(k,1).*N_RMSE + ...
                    Theta(k,2).*N_Delta + ...
                    Theta(k,3).*N_RMSSE1;

            end


            %% ===========================================================
            % SAVE ORIGINAL METRICS
            % ===========================================================

            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).RMSE = ...
                RMSE;


            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).ASE = ...
                ASE;


            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).RMSSE = ...
                RMSSE;


            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).Delta = ...
                Delta;


            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).RMSSE1 = ...
                RMSSE1;


            %% ===========================================================
            % SAVE CASE x PHASE-SPECIFIC SCALED COMPONENTS
            % ===========================================================

            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).N_RMSE = ...
                N_RMSE;


            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).N_Delta = ...
                N_Delta;


            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).N_RMSSE1 = ...
                N_RMSSE1;


            %% ===========================================================
            % SAVE ALL TQ
            % ===========================================================

            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).QI = ...
                QI;


            %% ===========================================================
            % SAVE MAIN TQ
            %
            % theta = [0.50 0.25 0.25]
            % ===========================================================

            TQ_Metrix. ...
                (Phase). ...
                (CaseName). ...
                (Regime).QI_Main = ...
                QI(:,:,idxMainTheta);

        end

    end

end


%% ========================================================================
% 10. SAVE CASE x PHASE-SPECIFIC SCALES
% ========================================================================

TQ_Metrix.Scale = ...
    StageScale;


%% ========================================================================
% 11. SETTINGS
% ========================================================================

TQ_Metrix.Settings.tau = ...
    tau;


TQ_Metrix.Settings.RMSEPercentile = ...
    RMSEPercentile;


TQ_Metrix.Settings.MainTheta = ...
    MainTheta;


TQ_Metrix.Settings.ThetaConstraint = ...
    'theta_j in [0.10,0.80], step 0.05, sum(theta)=1';


TQ_Metrix.Settings.SensitivityAnalysisPhase = ...
    'Test';


TQ_Metrix.Settings.Scaling = ...
    ['Four separate Case x Phase scales: Case1-Train, Case1-Test, ' ...
     'Case2-Train, Case2-Test; within each scale Global/Local and ' ...
     'RBF/Kriging are pooled'];


TQ_Metrix.Settings.CrossCaseTQComparison = ...
    'Not intended: absolute TQ values should not be compared across cases';


TQ_Metrix.Settings.CrossPhaseTQComparison = ...
    'Not intended: absolute TQ values should not be compared across Train/Test';


TQ_Metrix.Settings.Columns = ...
    { ...
    'Rbf-G', ...
    'Rbf-M', ...
    'Rbf-T', ...
    'Rbf-C', ...
    'Rbf-L', ...
    'Kriging-S' ...
    };


TQ_Metrix.Settings.Models = ...
    { ...
    'E08', ...
    'GGP', ...
    'EC4', ...
    'GCO', ...
    'SG2', ...
    'X19' ...
    };


%% ========================================================================
% 12. SAVE
% ========================================================================

OutputDir = ...
    fileparts(OutputFile);


if ~isfolder(OutputDir)

    mkdir(OutputDir);

end


save( ...
    OutputFile, ...
    'TQ_Metrix', ...
    '-v7.3');


%% ========================================================================
% 13. CHECK
% ========================================================================

fprintf('\n=============================================================\n');
fprintf(' TQ CALCULATION FINISHED\n');
fprintf('=============================================================\n');

fprintf('tau             = %.3f\n',tau);
fprintf('RMSE percentile = %.1f\n',RMSEPercentile);
fprintf('Theta settings  = %d (expected 120)\n',nTheta);
fprintf('Theta grid      = 0.10:0.05:0.80, sum(theta)=1\n');
fprintf('Main theta      = [0.50 0.25 0.25]\n');

fprintf( ...
    'Scaling         = FOUR CASE x PHASE SCALES\n');

fprintf( ...
    'Stages          = C1-Train, C1-Test, C2-Train, C2-Test\n');

fprintf( ...
    'Within stage    = Global/Local + RBF/Kriging\n');

fprintf( ...
    'Cross-case TQ   = NOT intended for absolute comparison\n');

fprintf( ...
    'Cross-phase TQ  = NOT intended for absolute comparison\n');

fprintf('Rows            = E08 GGP EC4 GCO SG2 X19\n');
fprintf('Columns         = G M T C L Kriging-S\n');
fprintf('Output          = %s\n',OutputFile);

fprintf('=============================================================\n');
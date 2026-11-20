%% ========================================================================
% 03_CREATE_CURRENT_SPATIALVARIANCE_FIGURES
%
% Rows:
%   E08, GGP, EC4, GCO, SG2, X19
%
% Columns:
%   RBF-G, RBF-M, RBF-T, RBF-C, RBF-L, Kri-S
%
% Normal execution:
%   silent
%
% Errors/assertions:
%   still displayed
% ========================================================================

clear
clc
close all

%% Collect Spatial Variance

Collect_SpatialVariance
%% ========================================================================
% 1. USER INPUT
% ========================================================================

RootDir = '.';

InputFile = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'SpatialVariance', ...
        'Current_SpatialVariance_Calibrated.mat');


OutputDir = ...
    fullfile( ...
        RootDir, ...
        'OutPut_RBF_2026_2', ...
        'Manuscript', ...
        'figures');


if ~isfolder(OutputDir)

    mkdir(OutputDir);

end


%% ------------------------------------------------------------------------
% TEST OPTION
% -------------------------------------------------------------------------

TestOneFigure = ...
    false;


TestFigureNumber = ...
    4;


CloseAfterSave = ...
    false;


%% ========================================================================
% 2. CHECK
% ========================================================================

assert( ...
    isfile(InputFile), ...
    'Input file not found:\n%s', ...
    InputFile);


% FigSpatialDistribution_Variance, SubFig, and compact_subplot are included as local functions below.


%% ========================================================================
% 3. LOAD
% ========================================================================

S = ...
    load(InputFile);


SpatialVariance = ...
    S.SpatialVariance;


Coordinates = ...
    S.Coordinates;


ListOfModel = ...
    S.ListOfModel;


Model_Name = ...
    S.Model_Name;


ListFunction = ...
    S.ListFunction;


ListFunction_Name = ...
    S.ListFunction_Name;


Meta = ...
    S.Meta;


clear S


assert( ...
    numel(ListOfModel) == 6, ...
    'Expected six EGMs.');


assert( ...
    numel(ListFunction) == 6, ...
    'Expected six methods.');


assert( ...
    strcmp( ...
        Meta.RBFVariance, ...
        'Final VarianceCalibrated'), ...
    'RBF input is not final calibrated variance.');


%% ========================================================================
% 4. BOUNDARY
% ========================================================================

S = ...
    load( ...
        fullfile( ...
            RootDir, ...
            'VNBoundaryLon.mat'));


VNBoundaryLon = ...
    S.VNBoundaryLon;


clear S


S = ...
    load( ...
        fullfile( ...
            RootDir, ...
            'VNBoundaryLat.mat'));


VNBoundaryLat = ...
    S.VNBoundaryLat;


clear S


%% ========================================================================
% 5. ALL FIGURE FORMAT SETTINGS
% ========================================================================

Fmt = ...
    struct();


%% ------------------------------------------------------------------------
% FIGURE
% -------------------------------------------------------------------------

Fmt.Figure.OuterPosition = ...
    [52 35 900 1500];


Fmt.Figure.BackgroundColor = ...
    [1 1 1];


%% ------------------------------------------------------------------------
% LAYOUT
% -------------------------------------------------------------------------

Fmt.Layout.Gap = ...
    [0.017 0.013];


Fmt.Layout.MarginHeight = ...
    [0.065 0.050];


Fmt.Layout.MarginWidth = ...
    [0.060 0.020];


%% ------------------------------------------------------------------------
% FONT
% -------------------------------------------------------------------------

Fmt.Font.Name = ...
    'Times New Roman';


Fmt.Font.AxisSize = ...
    12;


Fmt.Font.PanelSize = ...
10;


Fmt.Font.ColorBarSize = ...
    11;


Fmt.Font.LabelSize = ...
    14;


Fmt.Font.TitleSize = ...
    14;


%% ------------------------------------------------------------------------
% MARKER
% -------------------------------------------------------------------------

Fmt.Marker.Size = ...
    10;


%% ------------------------------------------------------------------------
% THRESHOLD
%
% Only affects figure display.
% Original variance remains unchanged.
% -------------------------------------------------------------------------

Fmt.Threshold.Value = ...
    0.05;


Fmt.Threshold.Mode = ...
    'cap';


%% ------------------------------------------------------------------------
% COLORBAR
% -------------------------------------------------------------------------

Fmt.ColorBar.CLim = ...
    [];


Fmt.ColorBar.NumberTicks = ...
    5;


Fmt.ColorBar.Decimals = ...
    3;

% Reduce only the empty gap between map and colorbar.
% Start with 0.004.
Fmt.ColorBar.GapReduction = ...
    0.001;

% Optional colorbar width.
Fmt.ColorBar.WidthFactor = ...
    0.85;


%% ------------------------------------------------------------------------
% AXIS
% -------------------------------------------------------------------------

Fmt.Axis.XLabel = ...
    'Longitude';


Fmt.Axis.YLabel = ...
    'Latitude';


Fmt.Axis.XLim = ...
    [];


Fmt.Axis.YLim = ...
    [];


%% ------------------------------------------------------------------------
% PANEL LABELS
% -------------------------------------------------------------------------

Fmt.Panel.RowLabel = { ...
    'a', ...
    'b', ...
    'c', ...
    'd', ...
    'e', ...
    'f'};


Fmt.Panel.ColLabel = { ...
    '1)', ...
    '2)', ...
    '3)', ...
    '4)', ...
    '5)', ...
    '6)'};


%% ------------------------------------------------------------------------
% PANEL TEXT POSITION
%
% Adjust ONLY HERE if required.
% -------------------------------------------------------------------------

Fmt.Panel.Units = ...
    'data';


Fmt.Panel.X = ...
    102.5;


Fmt.Panel.Y_ID = ...
    17.1;


Fmt.Panel.Y_Model = ...
    15.1;


Fmt.Panel.Y_Method = ...
    13.1;


Fmt.Panel.FontWeight = ...
    'normal';


Fmt.Panel.HorizontalAlignment = ...
    'left';


Fmt.Panel.VerticalAlignment = ...
    'middle';


%% ------------------------------------------------------------------------
% TITLE
% -------------------------------------------------------------------------

Fmt.Title.Show = ...
    true;


Fmt.Title.Position = ...
    [0 0.95 1 0.05];


Fmt.Title.HorizontalAlignment = ...
    'center';


Fmt.Title.VerticalAlignment = ...
    'middle';


Fmt.Title.FontWeight = ...
    'bold';


%% ========================================================================
% 6. FIGURE DEFINITIONS
% ========================================================================

FigDef = { ...
    'Case1','Global','Train', ...
    'Case 1 (Global-Train): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Nation_Case1_Train_1.png'; ...

    'Case1','Global','Test', ...
    'Case 1 (Global-Test): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Nation_Case1_Test_2.png'; ...

    'Case1','Local','Train', ...
    'Case 1 (Local-Train): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Local_Case1_Train_3.png'; ...

    'Case1','Local','Test', ...
    'Case 1 (Local-Test): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Local_Case1_Test_4.png'; ...

    'Case2','Global','Train', ...
    'Case 2 (Global-Train): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Nation_Case2_Train_5.png'; ...

    'Case2','Global','Test', ...
    'Case 2 (Global-Test): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Nation_Case2_Test_6.png'; ...

    'Case2','Local','Train', ...
    'Case 2 (Local-Train): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Local_Case2_Train_7.png'; ...

    'Case2','Local','Test', ...
    'Case 2 (Local-Test): Predictive variance of quasigeoid offset estimates', ...
    'Variance_SpatialDis_Local_Case2_Test_8.png'};


%% ========================================================================
% 7. SELECT FIGURES
% ========================================================================

if TestOneFigure

    assert( ...
        TestFigureNumber >= 1 && ...
        TestFigureNumber <= 8, ...
        'TestFigureNumber must be 1--8.');


    FigureIndex = ...
        TestFigureNumber;

else

    FigureIndex = ...
        1:8;

end


%% ========================================================================
% 8. CREATE ALL REQUESTED FIGURES
%
% No saving inside this loop.
% ========================================================================

CreatedFigures = ...
    gobjects(numel(FigureIndex),1);


CreatedFiles = ...
    cell(numel(FigureIndex),1);


for kk = 1:numel(FigureIndex)

    k = ...
        FigureIndex(kk);


    CaseName = ...
        FigDef{k,1};


    Regime = ...
        FigDef{k,2};


    Phase = ...
        FigDef{k,3};


    Fmt.Title.String = ...
        FigDef{k,4};


    x = ...
        Coordinates.(CaseName).(Phase).X;


    y = ...
        Coordinates.(CaseName).(Phase).Y;


    Data = ...
        SpatialVariance. ...
        (CaseName). ...
        (Regime). ...
        (Phase);


    % --------------------------------------------------------------------
    % evalc suppresses ALL normal Command Window output generated inside
    % the plotting function, SubFig, compact_subplot, colorbar, etc.
    %
    % Errors remain errors and will still stop execution.
    % --------------------------------------------------------------------

    evalc( ...
        ['fig = FigSpatialDistribution_Variance(' ...
         'x,y,Data,ListOfModel,Model_Name,ListFunction,' ...
         'ListFunction_Name,VNBoundaryLon,VNBoundaryLat,Fmt);']);


    CreatedFigures(kk) = ...
        fig;


    CreatedFiles{kk} = ...
        FigDef{k,5};

end


%% ========================================================================
% 9. SAVE
%
% Follow original successful workflow:
%
%   figure(fig)
%   saveas(fig,...)
%
% No resizing or re-render settings before save.
% ========================================================================

for kk = 1:numel(CreatedFigures)

    fig = ...
        CreatedFigures(kk);


    % Suppress any normal output generated while activating figure.
    evalc('figure(fig);');


    OutputPNG = ...
        fullfile( ...
            OutputDir, ...
            CreatedFiles{kk});


    % Suppress any normal saveas output.
    evalc('saveas(fig,OutputPNG);');

end


%% ========================================================================
% 10. OPTIONAL CLOSE
% ========================================================================

if CloseAfterSave

    close(CreatedFigures);

end


%% ========================================================================
% LOCAL FUNCTION: FigSpatialDistribution_Variance
% ========================================================================

function fig = FigSpatialDistribution_Variance( ...
    x, ...
    y, ...
    Data, ...
    ListOfModel, ...
    Model_Name, ...
    ListFunction, ...
    Function_Name, ...
    VNBoundaryLon, ...
    VNBoundaryLat, ...
    Fmt)
%% ========================================================================
% FIGSPATIALDISTRIBUTION_VARIANCE
%
% Clean publication version for spatial distributions of variance.
%
% Expected data structure:
%
%   Data.(model_method).Variance
%
% Example:
%
%   Data.EGM2008_gaussian.Variance
%
% Current arrangement:
%
%   Rows:
%       E08, GGP, EC4, GCO, SG2, X19
%
%   Columns:
%       RBF-G, RBF-M, RBF-T, RBF-C, RBF-L, Kri-S
%
% All manuscript-specific formatting is controlled externally through Fmt.
%
% Required external figure-helper functions:
%
%   None. SubFig and compact_subplot are embedded below.
%
% Normal execution:
%   no Command Window messages
%
% Errors/assertions:
%   remain active
% ========================================================================


%% ========================================================================
% 1. DIMENSIONS / BASIC CHECKS
% ========================================================================

nRow = ...
    numel(ListOfModel);


nCol = ...
    numel(ListFunction);


assert( ...
    numel(Model_Name) == nRow, ...
    'Model_Name and ListOfModel must have the same length.');


assert( ...
    numel(Function_Name) == nCol, ...
    'Function_Name and ListFunction must have the same length.');


assert( ...
    numel(Fmt.Panel.RowLabel) == nRow, ...
    'Fmt.Panel.RowLabel must contain one label per row.');


assert( ...
    numel(Fmt.Panel.ColLabel) == nCol, ...
    'Fmt.Panel.ColLabel must contain one label per column.');


x = ...
    x(:);


y = ...
    y(:);


assert( ...
    numel(x) == numel(y), ...
    'x and y must have the same number of elements.');


%% ========================================================================
% 2. CREATE FIGURE
%
% Keep same basic creation style as the original manuscript code.
% ========================================================================

fig = ...
    figure( ...
        'OuterPosition', ...
        Fmt.Figure.OuterPosition, ...
        'Color', ...
        Fmt.Figure.BackgroundColor);


%% ========================================================================
% 3. CREATE AXES
%
% Suppress any accidental output from compact_subplot.
% ========================================================================

evalc( ...
    ['ha = compact_subplot(' ...
     'nRow,nCol,' ...
     'Fmt.Layout.Gap,' ...
     'Fmt.Layout.MarginHeight,' ...
     'Fmt.Layout.MarginWidth);']);


%% ========================================================================
% 4. DRAW ALL PANELS
% ========================================================================

for i = 1:nRow

    for j = 1:nCol

        %% ----------------------------------------------------------------
        % CURRENT AXIS
        % -----------------------------------------------------------------

        k = ...
            (i-1)*nCol + j;


        ax = ...
            ha(k);


        axes(ax);


        %% ----------------------------------------------------------------
        % RUN NAME
        % -----------------------------------------------------------------

        runName = ...
            [ListOfModel{i} '_' ListFunction{j}];


        assert( ...
            isfield(Data,runName), ...
            'Missing variance data: %s', ...
            runName);


        assert( ...
            isfield(Data.(runName),'Variance'), ...
            'Missing Variance field: %s', ...
            runName);


        %% ----------------------------------------------------------------
        % ORIGINAL VARIANCE
        % -----------------------------------------------------------------

        V = ...
            Data.(runName).Variance(:);


        assert( ...
            numel(V) == numel(x), ...
            ['Variance/coordinate length mismatch: %s\n' ...
             'Variance = %d, coordinates = %d'], ...
            runName, ...
            numel(V), ...
            numel(x));


        assert( ...
            all(isfinite(V)), ...
            'Nonfinite variance detected: %s', ...
            runName);


        assert( ...
            all(V >= 0), ...
            'Negative variance detected: %s', ...
            runName);


        %% =================================================================
        % 5. PREPARE DISPLAY VALUES
        %
        % IMPORTANT:
        %
        % V remains unchanged.
        %
        % Only Vplot is modified for graphical display.
        % =================================================================

        Vplot = ...
            V;


        switch lower(Fmt.Threshold.Mode)

            %% -------------------------------------------------------------
            % CAP VALUES ABOVE THRESHOLD
            %
            % Example:
            %
            % V > 0.05 -> displayed as 0.05
            %
            % Used for manuscript spatial-distribution figures.
            % --------------------------------------------------------------

            case 'cap'

                idxAbove = ...
                    Vplot > Fmt.Threshold.Value;


                Vplot(idxAbove) = ...
                    Fmt.Threshold.Value;


                xPlot = ...
                    x;


                yPlot = ...
                    y;


            %% -------------------------------------------------------------
            % DISPLAY ONLY VALUES ABOVE THRESHOLD
            % --------------------------------------------------------------

            case 'above'

                idxAbove = ...
                    Vplot > Fmt.Threshold.Value;


                Vplot = ...
                    Vplot(idxAbove);


                xPlot = ...
                    x(idxAbove);


                yPlot = ...
                    y(idxAbove);


            %% -------------------------------------------------------------
            % NO THRESHOLD PROCESSING
            % --------------------------------------------------------------

            case 'none'

                xPlot = ...
                    x;


                yPlot = ...
                    y;


            otherwise

                error( ...
                    'Unknown Fmt.Threshold.Mode: %s', ...
                    Fmt.Threshold.Mode);

        end


        %% =================================================================
        % 6. DRAW MAP
        %
        % Legend only in first panel, as in original code.
        %
        % evalc suppresses accidental Command Window output generated
        % inside SubFig.
        % =================================================================

        Legend = ...
            (i == 1 && j == 1);


        evalc( ...
            ['SubFig(' ...
             'xPlot,yPlot,Vplot,' ...
             'VNBoundaryLon,VNBoundaryLat,' ...
             'Legend,Fmt.Marker.Size);']);


        %% =================================================================
        % 7. OPTIONAL COMMON COLOR LIMIT
        % =================================================================

        if ~isempty(Fmt.ColorBar.CLim)

            caxis( ...
                ax, ...
                Fmt.ColorBar.CLim);

        end


        %% =================================================================
        % 8. OPTIONAL MAP LIMITS
        % =================================================================

        if ~isempty(Fmt.Axis.XLim)

            xlim( ...
                ax, ...
                Fmt.Axis.XLim);

        end


        if ~isempty(Fmt.Axis.YLim)

            ylim( ...
                ax, ...
                Fmt.Axis.YLim);

        end


        %% =================================================================
        % 9. AXIS FONT
        % =================================================================

        set( ...
            ax, ...
            'FontName', ...
            Fmt.Font.Name, ...
            'FontSize', ...
            Fmt.Font.AxisSize);


        %% =================================================================
        % 10. PANEL TEXT
        %
        % Positions are controlled entirely from the main script.
        % =================================================================

        PanelID = ...
            [ ...
            Fmt.Panel.RowLabel{i}, ...
            Fmt.Panel.ColLabel{j} ...
            ];


        %% ----------------------------------------------------------------
        % PANEL ID
        %
        % a1), a2), ...
        % -----------------------------------------------------------------

        text( ...
            ax, ...
            Fmt.Panel.X, ...
            Fmt.Panel.Y_ID, ...
            PanelID, ...
            'Units', ...
            Fmt.Panel.Units, ...
            'FontName', ...
            Fmt.Font.Name, ...
            'FontSize', ...
            Fmt.Font.PanelSize, ...
            'FontWeight', ...
            Fmt.Panel.FontWeight, ...
            'HorizontalAlignment', ...
            Fmt.Panel.HorizontalAlignment, ...
            'VerticalAlignment', ...
            Fmt.Panel.VerticalAlignment, ...
            'Interpreter', ...
            'none');


        %% ----------------------------------------------------------------
        % MODEL NAME
        %
        % E08, GGP, ...
        % -----------------------------------------------------------------

        text( ...
            ax, ...
            Fmt.Panel.X, ...
            Fmt.Panel.Y_Model, ...
            Model_Name{i}, ...
            'Units', ...
            Fmt.Panel.Units, ...
            'FontName', ...
            Fmt.Font.Name, ...
            'FontSize', ...
            Fmt.Font.PanelSize, ...
            'FontWeight', ...
            Fmt.Panel.FontWeight, ...
            'HorizontalAlignment', ...
            Fmt.Panel.HorizontalAlignment, ...
            'VerticalAlignment', ...
            Fmt.Panel.VerticalAlignment, ...
            'Interpreter', ...
            'none');


        %% ----------------------------------------------------------------
        % METHOD NAME
        %
        % RBF-G, ..., Kri-S
        % -----------------------------------------------------------------

        text( ...
            ax, ...
            Fmt.Panel.X, ...
            Fmt.Panel.Y_Method, ...
            Function_Name{j}, ...
            'Units', ...
            Fmt.Panel.Units, ...
            'FontName', ...
            Fmt.Font.Name, ...
            'FontSize', ...
            Fmt.Font.PanelSize, ...
            'FontWeight', ...
            Fmt.Panel.FontWeight, ...
            'HorizontalAlignment', ...
            Fmt.Panel.HorizontalAlignment, ...
            'VerticalAlignment', ...
            Fmt.Panel.VerticalAlignment, ...
            'Interpreter', ...
            'none');


        %% =================================================================
        % 11. COLORBAR
        %
        % Important:
        %
        % 1. First create the normal MATLAB colorbar.
        % 2. Preserve MATLAB's automatically reserved position.
        % 3. Shift the bar slightly LEFT only.
        % 4. Optionally make it narrower.
        %
        % This avoids the overlap caused by recalculating the colorbar
        % position directly from ax.Position.
        % =================================================================

        cb = ...
            colorbar(ax);


        %% ----------------------------------------------------------------
        % FONT
        % -----------------------------------------------------------------

        cb.FontName = ...
            Fmt.Font.Name;


        cb.FontSize = ...
            Fmt.Font.ColorBarSize;


        %% ----------------------------------------------------------------
        % COLORBAR TICKS
        % -----------------------------------------------------------------

        if Fmt.ColorBar.NumberTicks > 1

            lim = ...
                cb.Limits;


            ticks = ...
                linspace( ...
                    lim(1), ...
                    lim(2), ...
                    Fmt.ColorBar.NumberTicks);


            cb.Ticks = ...
                ticks;


            tickFormat = ...
                [ ...
                '%.' ...
                num2str(Fmt.ColorBar.Decimals) ...
                'f' ...
                ];


            tickLabels = ...
                arrayfun( ...
                    @(z) sprintf(tickFormat,z), ...
                    ticks, ...
                    'UniformOutput', ...
                    false);


            cb.TickLabels = ...
                tickLabels;

        end


        %% ----------------------------------------------------------------
        % COLORBAR POSITION
        %
        % Default MATLAB position is used as the reference.
        %
        % Main-script controls:
        %
        % Fmt.ColorBar.GapReduction
        % Fmt.ColorBar.WidthFactor
        %
        % Recommended starting values:
        %
        % GapReduction = 0.001
        % WidthFactor = 0.85
        % -----------------------------------------------------------------

        if isfield(Fmt.ColorBar,'GapReduction')

            cb.Units = ...
                'normalized';


            cbPos = ...
                cb.Position;


            cbPos(1) = ...
                cbPos(1) - ...
                Fmt.ColorBar.GapReduction;


            if isfield(Fmt.ColorBar,'WidthFactor')

                cbPos(3) = ...
                    cbPos(3) * ...
                    Fmt.ColorBar.WidthFactor;

            end


            cb.Position = ...
                cbPos;

        end


        %% =================================================================
        % 12. AXIS TICK VISIBILITY
        %
        % Y ticks:
        %   first column only
        %
        % X ticks:
        %   last row only
        % =================================================================

        if j > 1

            ax.YTick = ...
                [];

        end


        if i < nRow

            ax.XTick = ...
                [];

        end


        %% =================================================================
        % 13. AXIS LABELS
        % =================================================================

        if j == 1

            ylabel( ...
                ax, ...
                Fmt.Axis.YLabel, ...
                'FontName', ...
                Fmt.Font.Name, ...
                'FontSize', ...
                Fmt.Font.LabelSize);

        end


        if i == nRow

            xlabel( ...
                ax, ...
                Fmt.Axis.XLabel, ...
                'FontName', ...
                Fmt.Font.Name, ...
                'FontSize', ...
                Fmt.Font.LabelSize);

        end

    end

end


%% ========================================================================
% 14. GLOBAL TITLE
% ========================================================================

if Fmt.Title.Show

    annotation( ...
        fig, ...
        'textbox', ...
        Fmt.Title.Position, ...
        'String', ...
        Fmt.Title.String, ...
        'EdgeColor', ...
        'none', ...
        'HorizontalAlignment', ...
        Fmt.Title.HorizontalAlignment, ...
        'VerticalAlignment', ...
        Fmt.Title.VerticalAlignment, ...
        'FontSize', ...
        Fmt.Font.TitleSize, ...
        'FontName', ...
        Fmt.Font.Name, ...
        'FontWeight', ...
        Fmt.Title.FontWeight, ...
        'Interpreter', ...
        'none');

end


%% ========================================================================
% 15. FINAL FONT ENFORCEMENT
% ========================================================================

fontObjects = ...
    findall( ...
        fig, ...
        '-property', ...
        'FontName');


set( ...
    fontObjects, ...
    'FontName', ...
    Fmt.Font.Name);

end

%% ========================================================================
% LOCAL FUNCTION: SubFig
%
% Draw one spatial panel.
% ========================================================================

function SubFig(x,y,Data,VNBoundaryLon,VNBoundaryLat,Legend,sizeDot) %#ok<INUSD>

scatter(x,y,sizeDot,Data,'filled');

c = ...
    colorbar; %#ok<NASGU>

hold on

plot(VNBoundaryLon,VNBoundaryLat,'r');

end

%% ========================================================================
% LOCAL FUNCTION: compact_subplot
%
% Create the compact axes grid used by the spatial-variance figures.
% ========================================================================

function [ha, pos] = compact_subplot(Nh, Nw, gap, marg_h, marg_w)
%COMPACT_SUBPLOT Create a compact grid of axes with configurable spacing.
%
%   [HA, POS] = COMPACT_SUBPLOT(NH, NW, GAP, MARG_H, MARG_W)
%
%   Inputs
%   ------
%   NH      Number of axes rows.
%   NW      Number of axes columns.
%
%   GAP     Spacing between adjacent axes, in normalized figure units.
%           A scalar applies the same spacing vertically and horizontally.
%           A two-element vector is interpreted as:
%               [vertical_gap horizontal_gap]
%
%   MARG_H  Vertical margins in normalized figure units.
%           A scalar applies the same lower and upper margin.
%           A two-element vector is interpreted as:
%               [lower_margin upper_margin]
%
%   MARG_W  Horizontal margins in normalized figure units.
%           A scalar applies the same left and right margin.
%           A two-element vector is interpreted as:
%               [left_margin right_margin]
%
%   Outputs
%   -------
%   HA      Column vector containing axes handles. Axes are ordered
%           row-by-row, starting from the upper-left corner.
%
%   POS     Axes positions returned by MATLAB when requested.
%
%   This implementation preserves the original layout logic while using
%   clearer variable names and comments.

%% Default settings

if nargin < 3
    gap = 0.02;
end

if nargin < 4 || isempty(marg_h)
    marg_h = 0.05;
end

if nargin < 5
    marg_w = 0.05;
end

%% Expand scalar settings to two-element vectors

if numel(gap) == 1
    gap = [gap gap];
end

if numel(marg_h) == 1
    marg_h = [marg_h marg_h];
end

if numel(marg_w) == 1
    marg_w = [marg_w marg_w];
end

%% Compute axes dimensions

axesHeight = ...
    (1 - sum(marg_h) - (Nh - 1) * gap(1)) / Nh;

axesWidth = ...
    (1 - sum(marg_w) - (Nw - 1) * gap(2)) / Nw;

%% Create axes from upper-left to lower-right

currentY = ...
    1 - marg_h(2) - axesHeight;

ha = ...
    gobjects(Nh * Nw,1);

panelIndex = ...
    0;

for rowIndex = 1:Nh

    currentX = ...
        marg_w(1);

    for colIndex = 1:Nw

        panelIndex = ...
            panelIndex + 1;

        ha(panelIndex) = ...
            axes( ...
                'Units','normalized', ...
                'Position',[currentX currentY axesWidth axesHeight], ...
                'XTickLabel','', ...
                'YTickLabel','');

        currentX = ...
            currentX + axesWidth + gap(2);

    end

    currentY = ...
        currentY - axesHeight - gap(1);

end

%% Return positions when requested

if nargout > 1
    pos = ...
        get(ha,'Position');
end

ha = ...
    ha(:);

end

function [summary, series, mapData] = ...
    summarizeConditionDiagnostics(diagInfo, X)
%SUMMARIZECONDITIONDIAGNOSTICS
% Summarize and retain pointwise conditioning diagnostics.
%
%   [summary, series] = summarizeConditionDiagnostics(diagInfo)
%
%   [summary, series, mapData] = ...
%       summarizeConditionDiagnostics(diagInfo, X)
%
%   diagInfo : structure array of selected-system diagnostics
%   X        : optional N-by-p coordinates for spatial mapping
%
%   summary  : distribution statistics of log10 condition values
%   series   : pointwise raw and log10 condition diagnostics
%   mapData  : optional coordinates and map-ready condition diagnosticses
% cond_raw   :    condition number before regularization
% cond_reg    :   condition number after applying RegParam
% cond_effective: effective condition measure of the solved system

fields = { ...
    'cond_raw', ...
    'cond_reg', ...
    'cond_effective'};

    N = numel(diagInfo);

    series.index = (1:N)';

    % ---------------------------------------------------------
    % Condition series and summary statistics
    % ---------------------------------------------------------
    for k = 1:numel(fields)

        name = fields{k};

        values = reshape([diagInfo.(name)], [], 1);

        % Store raw values
        series.(name) = values;

        % Store log10 values
        logValues = nan(size(values));

        valid = isfinite(values) & values > 0;
        logValues(valid) = log10(values(valid));

        series.(['log10_' name]) = logValues;

        % Summary statistics
        z = logValues(valid);

        if isempty(z)
            summary.(name) = emptySummary();
        else
            summary.(name).median = median(z);
            summary.(name).IQR    = iqr(z);
            summary.(name).P95    = prctile(z,95);
            summary.(name).max    = max(z);
            summary.(name).n      = numel(z);
        end
    end

    % ---------------------------------------------------------
    % Optional spatial map data
    % ---------------------------------------------------------
    mapData = [];

    if nargin > 1 && ~isempty(X)

        if size(X,1) ~= N
            error(['Number of coordinate rows must match the ' ...
                   'number of diagnostic systems.']);
        end

        mapData.X = X;

        for k = 1:numel(fields)

            name = fields{k};

            mapData.(name) = series.(name);
            mapData.(['log10_' name]) = ...
                series.(['log10_' name]);
        end
    end

end


function S = emptySummary()

    S.median = NaN;
    S.IQR    = NaN;
    S.P95    = NaN;
    S.max    = NaN;
    S.n      = 0;

end
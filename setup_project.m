function projectRoot = setup_project()
%SETUP_PROJECT Add the manuscript code to the MATLAB search path.
%
% Recommended use:
%   projectRoot = setup_project();
%   cd(projectRoot);
%
% This helper only configures the MATLAB search path. It does not modify
% scientific settings, input files, output files, or the current folder.

thisFile = mfilename('fullpath');
projectRoot = fileparts(thisFile);

% Package folders (+common, +globalrbf, +localrbf) are accessed through
% their parent directory.
addpath(projectRoot);

% Add top-level runners and manuscript-output folders.
addpath(fullfile(projectRoot,'Run_Global_RBF'));
addpath(genpath(fullfile(projectRoot,'Run_Local_RBF')));
addpath(genpath(fullfile(projectRoot,'Fig_And_Table_RBF')));

fprintf('\nRBF manuscript code added to the MATLAB path.\n');
fprintf('Project root: %s\n',projectRoot);
fprintf(['Set the MATLAB current folder to the project root before running ' ...
         'the manuscript workflows, e.g. cd(projectRoot).\n\n']);
end

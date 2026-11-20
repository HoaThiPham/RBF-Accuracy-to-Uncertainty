function [epsilon, d] = buildEpsilon(X, C)
%BUILDEPSILON
%
% Geometry-aware global RBF shape parameter:
%
%       epsilon = C * d
%
% where d is calculated from the supplied training geometry.

if ~isnumeric(C) || ...
   ~isscalar(C) || ...
   ~isreal(C) || ...
   ~isfinite(C) || ...
   C <= 0

    error('C must be a finite positive real numeric scalar.');

end


d = ...
    common.MeanDistant(X);


if ~isscalar(d) || ...
   ~isreal(d) || ...
   ~isfinite(d) || ...
   d <= 0

    error(['MeanDistant(X) must return a finite positive ' ...
           'geometry scale.']);

end


epsilon = ...
    max(C * d, 1e-12);

end

%% 
% 
% function [epsilon, d] = buildEpsilon(X, C)
% %BUILDEPSILON
% %
% % Geometry-aware global RBF shape parameter:
% %
% %       epsilon = C * d
% %
% % where d is calculated from the supplied training geometry.
% 
% 
% d = common.MeanDistant(X);
% 
% 
% if ~isfinite(d) || d <= 0
%     d = 1;
% end
% 
% 
% epsilon = ...
%     max(C * d, 1e-12);
% 
% end
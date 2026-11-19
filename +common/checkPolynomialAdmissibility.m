
function [isOK, info] = checkPolynomialAdmissibility(X)
%CHECKPOLYNOMIALADMISSIBILITY Check full-rank linear polynomial tail.
%
%   The RBF polynomial matrix is
%
%       P = [1, X]
%
%   and must have full column rank d + 1.

    [n,d] = ...
        size(X);

    P = ...
        [ones(n,1), X];

    rankP = ...
        rank(P);


    info.n = ...
        n;

    info.d = ...
        d;

    info.rankP = ...
        rankP;

    info.requiredRank = ...
        d + 1;


    isOK = ...
        n >= (d + 1) && ...
        rankP == (d + 1);

end

% 
% 
% function [isOK, info] = checkPolynomialAdmissibility(X)
% 
%     [n,d] = size(X);
% 
%     P = [ones(n,1), X];
% 
%     info.n       = n;
%     info.d       = d;
%     info.rankP   = rank(P);
%     info.requiredRank = d + 1;
% 
%     isOK = ...
%         n >= (d + 1) && ...
%         info.rankP >= (d + 1);
% 
% end
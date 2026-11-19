function u = applyRBFKernel(r, epsilon, kernel)
%APPLYRBFKERNEL Evaluate the selected radial basis function kernel.
%
%   u = common.applyRBFKernel(r, epsilon, kernel)
%
%   INPUTS
%   ------
%   r       : scalar, vector, or matrix of non-negative distances
%   epsilon : positive shape parameter for Gaussian and multiquadric
%             kernels; ignored for shape-free kernels
%   kernel  : kernel name:
%             "gaussian"
%             "multiquadric"
%             "linear"
%             "cubic"
%             "thinplate"
%
%   OUTPUT
%   ------
%   u       : evaluated RBF kernel values

    switch lower(string(kernel))

        case "gaussian"
            checkEpsilon(epsilon);
            u = exp(-0.5 * (r ./ epsilon).^2);

        case "multiquadric"
            checkEpsilon(epsilon);
            u = sqrt(1 + (r ./ epsilon).^2);

        case "linear"
            u = r;

        case "cubic"
            u = r.^3;

        case "thinplate"
            u = r.^2 .* log(r + 1);

        otherwise
            error('Unknown RBF kernel: %s', kernel);
    end
end


function checkEpsilon(epsilon)
%CHECKEPSILON Check validity of the RBF shape parameter.

    if isempty(epsilon) || ...
            ~isscalar(epsilon) || ...
            ~isfinite(epsilon) || ...
            epsilon <= 0

        error('epsilon must be a positive finite scalar.');
    end
end
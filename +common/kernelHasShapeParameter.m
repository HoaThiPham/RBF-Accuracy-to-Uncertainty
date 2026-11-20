function tf = kernelHasShapeParameter(kernel)
%KERNELHASSHAPEPARAMETER Return true for kernels with a shape parameter.

    switch lower(string(kernel))

        case {"gaussian", "multiquadric"}
            tf = true;

        case {"linear", "cubic", "thinplate"}
            tf = false;

        otherwise
            error('Unknown RBF kernel: %s', kernel);
    end
end
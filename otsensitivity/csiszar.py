"""
Csiszar based indices
---------------------

Da Veiga, S., Global Sensitivity Analysis with Dependence Measures, Journal of Statistical Computation and Simulation 85(7), November 2013. DOI: 10.1080/00949655.2014.945932
"""

import numpy as np
import openturns as ot

def csiszar(sample, data, f=ot.SymbolicFunction("t", "(sqrt(t)-1)^2"), Nnodes=32):
    """Dependence measure-based sensitivity indices.

    Use Csiszar divergence to compute sensitivity indices.

    :param array_like sample: Sample of parameters of Shape
      (n_samples, n_params).
    :param array_like data: Sample of realization which corresponds to the
      sample of parameters :attr:`sample` (n_samples, ).
    :param OpenTURNS Function: function R->R defining the dependence measure.
      must be positive, convex and zero at 1. The default is the KL divergence.
    :param 
    :returns: First order sensitivity indices.
    :rtype: (Csiszar, n_features).
    """
    def kernel(t):
        return t[0] * f([1 / (ot.SpecFunc.ScalarEpsilon + t[0])])
    
    g = ot.PythonFunction(1, 1, kernel)
    fullSample = ot.Sample(np.asarray(sample))
    dim = fullSample.getDimension()
    size = fullSample.getSize()
    fullSample.stack(np.asarray(data))
    fullSampleRanked = (fullSample.rank() + 0.5) / fullSample.getSize()
    algoGL = ot.GaussLegendre([Nnodes]*2)
    nodes = algoGL.getNodes()
    weights = algoGL.getWeights()
    s_indices = []
    for d in range(dim):
        marginalSample = fullSampleRanked.getMarginal([d, dim])
        copula = ot.BernsteinCopulaFactory().buildAsEmpiricalBernsteinCopula(marginalSample, "AMISE")
        integrand = g(copula.computePDF(nodes)).asPoint()
        s_indices.append(weights.dot(integrand))

    return s_indices

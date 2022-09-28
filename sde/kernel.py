import numpy as np
import tensorflow as tf
from gpflow.kernels import IsotropicStationary


class CompactMatern12(IsotropicStationary):
    """
    The compact Matern 1/2 kernel. Functions drawn from a
    GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r} * [1 - r/C]_+^\nu

    C

    where:
    r  is the Euclidean distance between the input points,
    scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    def K_r(self, r) -> tf.Tensor:
        compact_term = self.compact_term(r)
        return self.variance * tf.exp(-r) * compact_term

    def compact_term(self, r):
        term = 1 - r / 1
        term = tf.math.maximum(term, 0) ** 2
        return term


class CompactMatern32(IsotropicStationary):
    """
    The compact Matern 1/2 kernel. Functions drawn from a
    GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r} * [1 - r/C]_+^\nu

    C

    where:
    r  is the Euclidean distance between the input points,
    scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    def K_r(self, r) -> tf.Tensor:
        sqrt3 = np.sqrt(3.0)
        compact_term = self.compact_term(r)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r) * compact_term

    def compact_term(self, r):
        term = 1 - r / 0.2
        term = tf.math.maximum(term, 0) ** 2
        return term

import pymc as pm
from pymc.variational.approximations import MeanField
from aesara.tensor.random.op import RandomVariable
import aesara.tensor as at
from typing import List, Tuple


def _logdet_fisher(q):
    if isinstance(q, MeanField):
        return at.sum(at.log(q.std))
    else:
        raise TypeError("logdet_fisher only implemented for MeanField so far")


class ThetaRV(RandomVariable):
    # See docs here: https://www.pymc.io/projects/docs/en/latest/contributing/implementing_distribution.html
    name: str = "theta_mu"
    ndim_supp: int = 1  # theta 'output' is a vector, so ndim=1
    ndims_params: List[int] = [1, 1]  # theta 'inputs' are mu, sigma, each of which is a vector
    _print_name: Tuple[str, str] = ("θ", r"\theta")
    dtype: str = "floatX"

    @classmethod
    def rng_fn(cls, rng, *args, **kwargs):
        # See https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Flat.html
        raise NotImplementedError("Cannot sample from the prior over theta")


theta = ThetaRV()


class MixingDistribution(pm.Continuous):
    rv_op = theta

    @classmethod
    def dist(cls, wrapped_model: pm.Model, lam=2.0, **kwargs):
        _q = MeanField(model=wrapped_model)
        params = [_q.params, lam]
        return super().dist(params, **kwargs)

    def moment(rv, size, *args, **kwargs):
        # TODO - return some good starting value for theta
        #    See https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Flat.html
        pass

    def logp(self, *args, **kwargs):
        # TODO - evaluate log psi(θ), where θ is somehow passed in here
        #    See https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Flat.html
        # This is the key definition of log psi(theta)
        return 1 / 2 * _logdet_fisher(self._q) - self.lam * self._kl.apply(f=None)


def isvi(model=None, **kwargs):
    model = pm.modelcontext(model)
    with pm.Model() as psi_wrapper:
        MixingDistribution("theta", wrapped_model=model, **kwargs)
    return pm.sample(model=psi_wrapper, **kwargs)

import pymc as pm
import aesara.tensor as at
from pymc.variational.operators import KL


class MixingDistribution:
    """This class handles the construction of the various inputs to pm.DensityDist. The MixingDistribution.logp method
    is the log density *over the variational parameters*
    """

    def __init__(self, p: pm.Model, lam: float = 2.0, nmc: int = 50):
        self.q = pm.MeanField(model=p)
        self.kl = KL(self.q)
        self.lam = lam
        self.nmc = nmc

    @property
    def dist_params(self):
        # q.params is a shared_tensor, which means it has get_value and set_value methods but params cannot otherwise
        # be used directly as distribution parameters
        return tuple(p.get_value() for p in self.q.params)

    @property
    def ndim_supp(self):
        return sum([p.ndim for p in self.q.params])

    @property
    def ndims_params(self):
        return [p.ndim for p in self.q.params]

    def logp(self, *theta):
        # Copy 'theta' (values given from the sampler) into the aesara tensors defining q's mean and std
        for param, value in zip(self.q.params, theta):
            # !!! This is semantically what we want but gives an error. I want to take the given values of 'theta' and
            #   copy it into q.mu and q.rho (note that q.params = [q.mu, q.rho])
            param.set_value(value)

        # we are omitting the d*log(2) term because we can drop constant offsets without affecting behavior
        log_det_fisher = -2 * at.sum(at.log(at.diag(at.slinalg.cholesky(self.q.cov))))
        # Equation (10) from Lange et al (2022)
        log_mixing_density = 1/2 * log_det_fisher - self.lam * self.kl.apply(f=None)

        # Wrap the log mixing density in a function that sets the number of samples for q's internal monte carlo approximation
        # !!! I may be returning the wrong type here? DensityDist expects logp to return a scalar, but I'm returning an
        #   Aesara Function (I think). Will this cause problems?
        # FIXME - d=True sets sampling to be determinsitic. Is this what we want?
        return self.q.set_size_and_deterministic(log_mixing_density, s=self.nmc, d=True)


def sample_stochastic_mixture(model=None, lam=2.0, nmc=50):
    """Entry point for sampling variational parmeters from a mixing distribution, AKA ψ(θ), from Lange et al (2022).

    Lange et al (2022). "Interpolating between sampling and variational inference with infinite stochastic
        mixtures". https://arxiv.org/abs/2110.09618

    Parameters
    ----------
    p : pm.Model
        The pymc Model we would like to do inference on.
    lam : float
        The hyper-parameter λ from Lange et al (2022). When λ=1, the mixture behaves like sampling. As λ→∞, the
        mixture behaves like variational inference.
    nmc : int
        The number of samples to use for the monte carlo approximation of the KL divergence.

    """
    # TODO someday we would like to be able to adapt lam and nmc on the fly - maybe we'll need to implement a custom
    #  step function for this?
    model = pm.modelcontext(model)
    mixing_dist = MixingDistribution(model, lam, nmc)

    # Create a new pymc model whose random variables are the parameters of q
    with pm.Model() as stochastic_mixture:
        # Create a custom distribution over the variational parameters whose logp is provided by MixingDistribution.logp
        # and implements the log density described in equation (10) of Lange et al (2022)
        pm.DensityDist("theta",
                       mixing_dist.dist_params,
                       ndim_supp=mixing_dist.ndim_supp,
                       ndims_params=mixing_dist.ndims_params,
                       logp=mixing_dist.logp)

        # !!! This gives an error having to do with not having a default 'moment' function
        return pm.sample(draws=1000, step=pm.HamiltonianMC(), model=stochastic_mixture)

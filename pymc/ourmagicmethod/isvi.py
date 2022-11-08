import pymc as pm
from pymc.variational.approximations import MeanField
from typing import Optional, Union, Sequence, List
from aesara.graph.basic import Variable


class PsiWrapper(pm.Model):
    def __init__(self, wrapped_model, lam=2.0, qmethod="meanfield", num_kl_samples=10):
        super(PsiWrapper, self).__init__()
        self.wrapped_model = wrapped_model
        self.lam = lam

        # q(x;theta) is the variational approximation. IDK where 'theta' is defined except that there is some q.mean and q.cov
        inf_kwargs = {}  # Placeholder (see inference.py)
        self.q = MeanField(model=wrapped_model, **inf_kwargs)

    def logp(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
        sum: bool = True,
    ) -> Union[Variable, List[Variable]]:
        """COMPUTE LOG(PSI)
        """
        theta = vars
        # TODO - make this...
        # STEP 1: set parameters of q equal to theta. q.mean.set(theta[:2]), q.cov.set(...)
        # STEP 2: get logdet(Fisher(q)) # TODO ourselves
        # STEP 3: get kl(q||p)  (look inside ADVI)
        return 1/2*logdet_fisher - self.lam * kl_qp

    def dlogp(
        self,
        vars: Optional[Union[Variable, Sequence[Variable]]] = None,
        jacobian: bool = True,
    ) -> Variable:
        """COMPUTE GRAD OF LOG(PSI) WRT THETA
        """
        theta = vars
        ... # TODO (if pm.sample uses this)

    def logp_dlogp_function(self, grad_vars=None, tempered=False, **kwargs):
        """COMPUTE BOTH
        """
        theta = vars
        ... # TODO (if pm.sample uses this)


def isvi(model=None):
    if model is None:
        model = pm.modelcontext(model)
    wrapped = PsiWrapper(model)
    pm.sample(model=wrapped)
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

#%%
with pm.Model() as model:
    x = pm.Normal("x", mu=0, sigma=4)
    y = pm.Normal("y", mu=x**2, sigma=1)

#%% Run ours

our_result = pm.sample_stochastic_mixture(model=model)

#%% Run and plot ADVI
advi_result = pm.fit(model=model, method="advi")

mu, cov = advi_result.mean.eval(), advi_result.cov.eval()

def plot_gauss(mu, cov, *args, **kwargs):
    L = np.linalg.cholesky(cov)
    t = np.linspace(0, 2*np.pi)
    x, y = np.cos(t), np.sin(t)
    combined = mu[:, None] + L @ np.stack([x, y], axis=0)
    plt.plot(*combined, *args, **kwargs)


plot_gauss(mu, cov, '-r')

#%% Run and plot HMC
hmc_result = pm.sample(1000, step=pm.HamiltonianMC(), model=model)

plt.scatter(hmc_result.posterior["x"], hmc_result.posterior["y"])

#%%

plt.show()
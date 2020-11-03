#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

#%%%% imports
# %pylab inline
# %load_ext autoreload
# %autoreload 2
#%reload_ext autoreload

import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *
from matplotlib.pyplot import rcParams
rcParams.update({"font.size":15})

#%%%% cd working directory
import os
os.chdir("/Users/cham/projects/sb2")
print(os.getcwd())

#%%%% code
"""
https://phoenix.astro.physik.uni-goettingen.de/
"""

from astropy.io import fits
flux_hi = fits.getdata("/Users/cham/Documents/slides/stellar_parameters/demo_ccf/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
wave_hi = fits.getdata("/Users/cham/Documents/slides/stellar_parameters/demo_ccf/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
figure()
plot(wave_hi, flux_hi)
xlim(3000, 25000)

#%%
xx = np.arange(10)
figure()
for i in range(3):
    yy = np.ones_like(xx) * i
    plot(xx-i, yy,"s", ms=10, mec="k")
    
#%%
import joblib
r = joblib.load("/Users/cham/projects/speclib/ferre/R7500_regli.dump")
#%%
from laspec.normalization import normalize_spectrum_spline
def plot_star(ax, p, **kwargs):
    
    ax.plot(r.wave, flux_norm)
    return

#%%    
rcParams["font.size"]=20
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharex=False, sharey=True)
lw = 3
plot_teff=10000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
axs[0].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
axs[1].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)

plot_teff=8000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
axs[0].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
axs[1].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)

plot_teff=6000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
axs[0].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
axs[1].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)

plot_teff=4000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
axs[0].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
axs[1].plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)

plot_teff=5800
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
axs[0].plot(r.wave, flux_norm, label="obs", c="k", lw=lw)
axs[1].plot(r.wave, flux_norm, label="obs", c="k", lw=lw)

axs[0].grid(True)
axs[1].grid(True)
axs[1].legend()
axs[0].set_xlim(5160, 5190)
axs[1].set_xlim(6550, 6580)
axs[0].set_ylim(0, 1.5)
axs[0].set_xlabel("$\lambda$ [$\mathrm{\AA}]$")
axs[1].set_xlabel("$\lambda$ [$\mathrm{\AA}]$")
axs[0].set_ylabel("Normalized flux")
axs[0].set_title("Mg I")
axs[1].set_title("H$\\alpha$")
fig.tight_layout()

#%%
def f(x):
    x = np.array(x)
    return np.exp(-0.5*(x-5900)**2/1000**2)
figure(figsize=(10, 7))
x = [4000, 6000, 8000, 10000]
plot(x, f(x),"s",  c='darkcyan', ms=15, mec="k")
xx = np.linspace(4000, 10000, 100)
plot(xx, f(xx), 'k-', lw=3)
ylabel("Likelihood / Similarity")
xlabel("$T_\mathrm{eff}$ [K]")
grid(True)
fig.tight_layout()

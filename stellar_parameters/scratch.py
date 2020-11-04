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
rcParams["font.size"]=25
fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharex=False, sharey=True)
lw = 3
plot_teff=10000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
ax.plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
plot_teff=8000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
ax.plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
plot_teff=6000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
ax.plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
plot_teff=4000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
ax.plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)

ax.set_xlim(6550, 6578)

#%%
rcParams["font.size"]=25
fig, ax = plt.subplots(1, 1, figsize=(10, 7), sharex=False, sharey=True)
lw = 3
plot_teff=6000
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
ax.plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw)
plot_teff=5800
flux_norm, flux_cont = normalize_spectrum_spline(r.wave, r.predict_spectrum([plot_teff, 4.5, 0, 0]), niter=3)
ax.plot(r.wave, flux_norm, label="{:d} K".format(plot_teff), lw=lw,c="k")

ax.set_xlim(6550, 6578)
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

#%%
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, y)

zz = xx**2+yy**2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz,cmap=cm.plasma)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel("$\\theta_2$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("Negative Likelihood")

#%%
rcParams["font.size"]=25
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharex=False, sharey=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def g(x):
    return np.sin(x*np.pi)


x = np.linspace(-1, 1, 30)
y = g(x)
axs[0].plot(x, y, "k-", lw=3, label="Truth")
axs[0].plot(x, y, "s", ms=10, label="Grid",mec="k")

x = np.linspace(-1, 1, 100)
y = g(x)
axs[1].plot(x, y, "k-", lw=3, label="Truth")
xn = np.random.uniform(-1, 1, 100)
xn = np.sort(xn)
yn = np.random.normal(0,.2, 100)+g(xn)
axs[1].plot(xn, yn, 'o', color="darkcyan", label="Observation",ms=10,mec="k")


from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
svr = SVR(C=10, epsilon=0.05)
svr.fit(xn.reshape(-1,1), yn)
axs[1].plot(xn, svr.predict(xn.reshape(-1,1)), "-", color="m", lw=3,label="SVR")

nn = MLPRegressor(activation="logistic", hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=10000,solver="lbfgs")
nn.fit(xn.reshape(-1,1), yn)
axs[1].plot(xn, nn.predict(xn.reshape(-1,1)), "-", color="lightcoral", lw=3,label="NN")


for i in range(2):
    axs[i].legend(loc="upper left")
    axs[i].set_xlim(-1,1)
    axs[i].set_ylim(-1.3,1.3)
    axs[i].set_xlabel("a specific parameter $\\theta$")
    axs[i].set_ylabel("Normalized Flux")
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
axs[0].set_title("Case A. Synthetic spectra")
axs[1].set_title("Case B. Empirical spectra")
fig.tight_layout()

#%%


rcParams["font.size"]=25
fig, ax = plt.subplots(1, 1, figsize=(7, 7), sharex=False, sharey=True)
ax.plot(4.5,1,"s", mec="k", mfc="w", ms=50)

ax.text(4.5, 1, "$\\theta$".format(i), va="center", ha="center",fontsize=10)

svr_xx, svr_yy = np.arange(10), np.zeros(10) 
ax.plot(svr_xx, svr_yy, "s", mec="k", mfc="w", ms=25)
for i in range(10):
    if i==8:
        ax.text(svr_xx[i], svr_yy[i],"...", va="center", ha="center",fontsize=10) 
    if i==9:
        ax.text(svr_xx[i], svr_yy[i],"SVR$_N$", va="center", ha="center",fontsize=10) 
    else:
        ax.text(svr_xx[i], svr_yy[i],"SVR$_{}$".format(i), va="center", ha="center",fontsize=10)
    plot([4.5, svr_xx[i]], [1,svr_yy[i]],'k-',zorder=-5)
#%%
rcParams["font.size"]=25
fig, axs = plt.subplots(1,2, figsize=(12, 6),)

n=100
xx = np.random.randn(n)
axs[0].plot(xx+np.random.normal(0, 0.03, n),xx+np.random.normal(0, 0.03, n),'o',color="darkcyan",mec="k",ms=10,alpha=0.7)
axs[0].set_xticks([])
axs[0].set_xticklabels([])
axs[0].set_yticks([])
axs[0].set_yticklabels([])
axs[0].set_xlabel("$\\theta_1$ (e.g., [Mg/H])")
axs[0].set_ylabel("$\\theta_2$ (e.g., [Ca/H])")

xx = np.linspace(-5, 15, 500)
axs[1].plot(xx,1-1*(np.exp(-(xx-3)**2)+0.5*np.exp(-(xx-8)**2)),lw=3)
axs[1].plot(xx,2-0.5*(np.exp(-(xx-3)**2)+0.5*np.exp(-(xx-8)**2)),lw=3)
axs[1].vlines([3,8], -1, 3, linestyle="--")
axs[1].text(3, 1, "Mg")
axs[1].text(8, 1, "Ca")
axs[1].set_xlim(-5, 15)
axs[1].set_ylim(-1, 3)
axs[1].set_xticks([])
axs[1].set_xticklabels([])
axs[1].set_yticks([])
axs[1].set_yticklabels([])
axs[1].set_xlabel("$\\lambda$ [$\mathrm{\AA}$]")
axs[1].set_ylabel("Flux")
fig.tight_layout()
#%%
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel("$\\theta_1$")
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_ylabel("$\\theta_2$")


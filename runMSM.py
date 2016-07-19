#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import spacepy, matplotlib
from spacepy.toolbox import binHisto
import numpy as np
import datetime as dt
import substorm_model as msm
import matplotlib.pyplot as plt
import spacepy.plot as splot
splot.style('spacepy')

mu0 = 4e-7*np.pi

satname = 'WIND' #'ACE'
if satname=='ACE':
    acedata = spacepy.datamodel.fromHDF5('BAS_ACEdata.h5')
    acedata['time'] = spacepy.datamodel.dmarray([dt.datetime.strptime(z, '%Y-%m-%dT%H:%M:%S') for z in acedata['time']])
    vel = np.sqrt(acedata['vx']**2 + acedata['vy']**2 + acedata['vz']**2)*1e3
    b2 = acedata['bx']**2+acedata['by']**2+acedata['bz']**2
    btot = np.sqrt(b2)*1e-9
    theta = np.arctan2(acedata['by'],acedata['bz'])
    pow_in = np.sin(theta/2.)**4 * vel * btot**2
elif satname=='WIND':
    data = spacepy.datamodel.fromHDF5('Wind_NAL.h5')
    pow_in = data['input']

#delta = dt.timedelta(0,60)
#numericDelta = delta.days*86400+delta.seconds
#minlen = dt.timedelta(hours=100)
delta = 60
numericDelta = 60
minlen = 60*60*100

istart, iend = msm.findContiguousData(data['time'], delta, 
    minLength=minlen)
results = msm.msm(numericDelta, data['time'], pow_in, istart, 
    iend, tau0=2.69*3600, restartmode='random') #ACE is 2.75; Wind is 2.69

# plot histogram of inter-substorm intervals
def convert_tdelt(inputlist, units='hours'):
    if units=='hours':
        fac = 3600.0
    elif units=='minutes':
        fac = 60.0
    else:
        raise TypeError
    try:
        outputlist = [(el.days*86400 + el.seconds)/fac for el in inputlist]
    except AttributeError: #probably serial time, not datetimes
        try:
            outputlist = inputlist/fac
        except TypeError: #not an array
            outputlist = [el/fac for el in inputlist]
    return outputlist

isi = convert_tdelt(results['tau_valid'], units='minutes')
isi_hr = convert_tdelt(results['tau_valid'], units='hours')
tau_ax = np.arange(0,30*60,.2)

try:
    from sklearn.neighbors.kde import KernelDensity
    kern_type = 'epanechnikov'
    kern_lab = '{0}{1} KDE'.format(kern_type[0].upper(), kern_type[1:])
    kernel = KernelDensity(kernel=kern_type, bandwidth=60).fit(isi[:, np.newaxis])
    kde_plot = np.exp(kernel.score_samples(tau_ax[:, np.newaxis]))
except ImportError:
    from scipy import stats
    kern_lab = 'Gaussian KDE'
    kernel = stats.gaussian_kde(isi, bw_method='scott')
    kde_plot = kernel.evaluate(tau_ax)

fig, ax = splot.set_target(None)
ax.hist(isi_hr, bins=np.arange(0,25,0.5), histtype='step', normed=True, lw=1.5, label='Binned Data')
ax.plot(tau_ax/60., kde_plot*60., lw=1.5, label=kern_lab)
ax.set_xlim([0,25])
ax.set_ylabel('Probability')
ax.set_xlabel(r'Inter-substorm Interval, $\tau$ [hours]') #raw string req'd (else \t in \tau becomes [tab]au
ax.legend()
fig.suptitle('MSM$_{Python}$: ' + '{0} (1998-2002)'.format(satname))
plt.show()

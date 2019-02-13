#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import spacepy
from spacepy.toolbox import binHisto
import spacepy.plot as splot
import substorm_model as msm
splot.style('spacepy')

sdict = dict()
sdict['ACE'] = {'delta': dt.timedelta(seconds=60),
                'minlen': dt.timedelta(hours=100),
                'tau0': 2.75*3600,
                }
sdict['WIND'] = {'delta': 60,
                 'minlen': 60*60*100,
                 'tau0': 2.69*3600,
                 }

numericDelta = 60

satname = 'ACE'#'WIND' #'ACE'
if satname=='ACE':
    data = spacepy.datamodel.fromHDF5('BAS_ACEdata.h5')
    data['time'] = spacepy.datamodel.dmarray([dt.datetime.strptime(z.decode('UTF-8'), '%Y-%m-%dT%H:%M:%S') for z in data['time']])
    vel = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)*1e3
    b2 = data['bx']**2 + data['by']**2 + data['bz']**2
    btot = np.sqrt(b2)*1e-9
    theta = np.arctan2(data['by'], data['bz'])
    pow_in = np.sin(theta/2.)**4 * vel * btot**2
elif satname=='WIND':
    data = spacepy.datamodel.fromHDF5('Wind_NAL.h5')
    pow_in = data['input']

istart, iend = msm.findContiguousData(data['time'], sdict[satname]['delta'], 
    minLength=sdict[satname]['minlen'])
results = msm.msm(numericDelta, data['time'], pow_in, istart, 
    iend, tau0=sdict[satname]['tau0'], restartmode='random') #ACE is 2.75; Wind is 2.69

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
    return np.asarray(outputlist)

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
ax.hist(isi_hr, bins=np.arange(0,25,0.5), histtype='step', density=True, lw=1.5, label='Binned Data')
ax.plot(tau_ax/60., kde_plot*60., lw=1.5, label=kern_lab)
ax.set_xlim([0,25])
ax.set_ylabel('Probability')
ax.set_xlabel(r'Inter-substorm Interval, $\tau$ [hours]') #raw string req'd (else \t in \tau becomes [tab]au
ax.legend()
fig.suptitle('MSM$_{Python}$: ' + '{0} (1998-2002)'.format(satname))
plt.show()

#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import spacepy, matplotlib
from spacepy.toolbox import binHisto
import numpy as np
import datetime as dt
import substorm_model as msm
import matplotlib.pyplot as plt

mu0 = 4e-7*np.pi

acedata = spacepy.datamodel.fromHDF5('BAS_ACEdata.h5')
acedata['time'] = spacepy.datamodel.dmarray([dt.datetime.strptime(z, '%Y-%m-%dT%H:%M:%S') for z in acedata['time']])
#acedata['time'] = spacepy.time.Ticktock(acedata['time'], 'ISO').UTC
vel = np.sqrt(acedata['vx']**2 + acedata['vy']**2 + acedata['vz']**2)*1e3
b2 = acedata['bx']**2+acedata['by']**2+acedata['bz']**2
btot = np.sqrt(b2)*1e-9
theta = np.arctan2(acedata['by'],acedata['bz'])
pow_in = np.sin(theta/2.)**4 * vel * btot**2

delta = dt.timedelta(0,60)

istart, iend = msm.findContiguousData(acedata['time'], delta, 
    minLength=dt.timedelta(hours=100))
numericDelta = delta.days*86400+delta.seconds
results = msm.msm(numericDelta, acedata['time'], pow_in, istart, 
    iend, tau0=2.75*3600)

# plot histogram of inter-substorm intervals
def tdelt2hrs(inputlist):
    outputlist = [(el.days*86400 + el.seconds)/3600 for el in inputlist]
    return outputlist
isi = tdelt2hrs(results['tau_valid'])
binw, nbins = binHisto(isi)

plt.hist(isi, bins=nbins, histtype='step', normed=True)
plt.ylabel('Probability')
plt.xlabel(r'Inter-substorm Interval, $\tau$ [hours]') #raw string req'd (else \t in \tau becomes [tab]au
plt.title('MSM$_{Python}$: ACE (1998-2002)')
plt.show()

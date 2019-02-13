#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
try:
    import unittest_pretty as utp
except:
    pass
import unittest
import spacepy, matplotlib
from spacepy.toolbox import binHisto
import numpy as np
import datetime as dt
import substorm_model as msm

def tdelt2hrs(inputlist):
    outputlist = [(el.days*86400 + el.seconds)/3600 for el in inputlist]
    return outputlist

class msmPropertiesTests(unittest.TestCase):

    def test_regress_FM04(self):
        '''Match F&M2004 (intersubstorm interval mean, stddev; # onsets'''
        mu0 = 4e-7*np.pi
        acedata = spacepy.datamodel.fromHDF5('BAS_ACEdata.h5')
        acedata['time'] = spacepy.datamodel.dmarray([dt.datetime.strptime(z.decode('UTF-8'), '%Y-%m-%dT%H:%M:%S') for z in acedata['time']])
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
        isi = tdelt2hrs(results['tau_valid'])
        self.assertEqual(2398, len(isi))
        msmMean = np.mean(isi)
        self.assertAlmostEqual(5.2005212677231079, msmMean)
        msmStd = np.std(isi)
        self.assertAlmostEqual(4.6785038008585831, msmStd)
        
    def test_recurrence_constant(self):
        '''Test that mean recurrence tends to given recurrence under constant driver
        
        Warning: this test is very slow
        '''
        recur = 2.7
        seq_len = 10000000
        out = msm.msm(60, range(seq_len), [4000]*seq_len, [0], [seq_len-1], tau0 = recur*3600)
        self.assertAlmostEqual(recur, np.mean(out['tau'])/60, places=4)

        
if __name__ == "__main__":
    try:
        utp.main()
    except:
        unittest.main()

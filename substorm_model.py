#!/usr/bin/python
# -*- coding: utf-8 -*-
'''Minimal substorm model port to Python'''
from __future__ import division
import numbers
from re import I
import numpy as np
import datetime as dt

def msm(delta, time, pow_in, istart, iend, tau0=2.7*3600, restartmode='mean',
        extendedout=False, seed=None):
    """Minimal substorm model (Freeman and Morley, GRL, 2004)
    
    Inputs:
    =======
    time - input series of times. Can be datetime objects or serial time
    delta - resolution of x (currently must be serial)
    pow_in - input series of solar wind power
    istart, iend - indices of start/end of contiguous blocks of input
    tau0 - recurrence constant (in seconds)
    restartmode - mode of determining energy level at which to start integrating
                  options are:
                  'mean' (mean power of whole input dataset * tau)
                  'random' (random energy content)
                  'onset' (force onset at start/restart of model run)
    extendedout - if True, add energy content as function of time to output
    
    Returns:
    ========
    out_dict = {'p_onset': p_onset, 't': t, 'tau': tau, 'f_dump': f_dump, 'f_thr': f_thr, 
        'n_valid': len(valid), 'n': len(t), 't_valid': t_valid, 'tau_valid': tau_valid, 
        'f_dump_valid': f_dump[valid], 'valid': valid, 'tau0': tau0}
    
    Author:
    =======
    Original -- Mervyn Freeman, British Antarctic Survey
    Port to Python by Steve Morley, Los Alamos National Lab.
    smorley@lanl.gov
    """
    np.random.seed(seed) if seed is not None else np.random.seed(1066)
    threshold = tau0*np.mean(pow_in) # Threshold for 1st s/storm after data gap
    rsmodes = {'mean': threshold,
               'random': np.random.random()*threshold,
               'onset': 0.0}
    if restartmode not in rsmodes: raise ValueError('Invalid mode for restarting energy content selected.')

    if extendedout: time_used = []
    f_thr = []
    f_dump = []
    #f_thr = value of s.w. input at onset #f_dump = energy dumped at onset
    t = [] #time of substorm onsets
    p_onset = [] #index of substorm onsets
    tau = [] #interval of substorm onsets
    valid_mask, csum = [], [] # cumulative sum of solar wind input
    n_onset = 0

    if isinstance(delta, dt.timedelta):
        numericDelta = delta.total_seconds()
    elif isinstance(delta, numbers.Number):
        numericDelta = delta
    else: #what is this?
        raise ValueError('Invalid input delta - datatype not supported')
    
    # Find substorms in each interval of contiguous solar wind data
    for j, st_ind in enumerate(istart):
        # Find index limits of interval
        en_ind = iend[j]
        
        # Initialise start of substorm sequence after data gap
        csum.append(-1*rsmodes[restartmode])
        first_after_gap = True
        
        # Find substorms in interval
        for i in range(st_ind+1, en_ind+1):
            # Increment cumulative sum of energy input
            csum.append(csum[-1] + (pow_in[i] + pow_in[i-1])/2*numericDelta)
            if extendedout:
                time_used.append(time[i])
            # Check whether substorm threshold is reached
            if (csum[-1] >= 0): # Substorm occurs
                # Record time of onset
                t.append(time[i])
                p_onset.append(i)
                # Calculate energy dump and record
                dump = (pow_in[i] + pow_in[i-1])/2*tau0
                f_dump.append(dump)
                f_thr.append(csum[-1]) # Record cumulative sum at onset
                valid_mask.append(first_after_gap)
                # If not first onset of interval then record 
                #time since last substorm onset
                if not first_after_gap:
                    dum = t[n_onset] - t[n_onset-1]
                    tau.append(dum)
                else:
                    tau.append(False)# Waittime not def for 1st onset after gap
                    first_after_gap = False
                csum[-1] = csum[-1] - dump # Subtract energy dump from cum sum
                n_onset += 1 # Increment substorm onset counter

    tau_valid = np.ma.array(tau, mask=valid_mask).compressed()
    p_valid = np.ma.array(p_onset, mask=valid_mask).compressed()
    t_valid = np.ma.array(t, mask=valid_mask).compressed()

    out_dict = {'p_onset': p_onset, 't': t, 'tau': tau, 'f_dump': f_dump,
                'f_thr': f_thr, 'n_valid': len(tau_valid), 'n': len(t),
                't_valid': t_valid, 'tau_valid': tau_valid}
    if extendedout:
        out_dict['energy'] = csum
        out_dict['time'] = time_used

    return out_dict


def findContiguousData(x, delta, minLength=None):
    """Find all intervals of contiguous data in x exceeding min_length

    Contiguous data are defined as neighbouring points separated by delta
    i.e., x[i+1] - x[i] = delta

    If min_length is undefined then min_length = Delta
    
    Inputs:
    =======
    x, input series of times. Can be datetime objects or serial time
    delta, expected resolution of x.
    minLength [defaults to delta], minimum length of contiguous interval required.
    
    Returns:
    ========
    istart, iend - indices of starts and ends of contiguous blocks
    
    Author:
    =======
    Original -- Mervyn Freeman, British Antarctic Survey
    Port to Python by Steve Morley, Los Alamos National Lab.
    smorley@lanl.gov/morley_steve@hotmail.com
    """

    if not minLength:
        minLength = delta

    if type(x)==list:
        x = np.array(x)
    #now ensure type consistency of array contents for datetime input
    if isinstance(x[0], dt.datetime):
        try:
            assert type(delta) == type(minLength)
            assert isinstance(delta, dt.timedelta)
        except:
            return 'findContiguousData: inconsistent data types for time objects'
    else:
        #assume serial time input. If delta/minLength are timedelta, convert assuming serial time in seconds
        if isinstance(delta, dt.timedelta): delta = delta.total_seconds()
        if isinstance(minLength, dt.timedelta): minLength = minLength.total_seconds()
        

    #Calculate distance between neighbouring data points in array x
    dx = x[1:]-x[0:-1]

    #Find positions i where X neighbours are non-contiguous
    #i.e., X(i+1) - X(i) > Delta
    #Store in array igaps
    igaps, = (dx > delta).nonzero()
    igaps1, nigaps = igaps+1, len(igaps)

    #Now find intervals of contiguous data exceeding min_length
    #Contiguous data interval starts at the end of a non-contiguous interval
    #and ends at the start of the next non-contiguous interval.

    #Start of series X is start of first potentially contiguous interval
    istart_c = [0]
    #Find starts of other potentially contiguous intervals
    #from ends of non-contiguous intervals
    end_nc = x[igaps+1].tolist()
    start_c = end_nc
    istart_c = igaps1.tolist()
    start_c.insert(0, x[0])
    istart_c.insert(0, 0)
    #Find ends of potentially contiguous intervals
    #from starts of non-contiguous intervals
    end_c = x[igaps].tolist() #start_nc
    iend_c = igaps.tolist()
    #Add end of series X as end of last potentially contiguous interval
    end_c.append(x[-1])
    iend_c.append(len(x)-1)

    #Find lengths of all potentially contiguous intervals
    length = [cEnd - start_c[i] for i, cEnd in enumerate(end_c)]
    #Find those whose length exceeds min_length
    ilong, = (np.array(length) > minLength).nonzero()

    #Return start and end indices of these intervals
    istart = [istart_c[ind] for ind in ilong]
    iend = [iend_c[ind] for ind in ilong]

    return istart, iend


def amplify_limit(imf_magn, ampfac=7, limit=60):
    """amplification-limitation procedure
    
    Amplification factor defaults to 7 following Reiff et al.
    Limit is set to 60nT.
    See Morley, Freeman & Tanskanen (2007) for details.
    """
    imf = np.atleast_1d(imf_magn)
    cap = np.ones_like(imf)*limit/ampfac
    test = np.c_[imf, cap]
    b_amplim = np.min(test, axis=-1)
    return b_amplim


def calc_epsilon(data, amplim=False, amplim_params={}):
    """Calculate Akasofu's epsilon from input SW/IMF data

    Assumes dict-like with vx, vy, vz, bx, by, bz
    v must be km/s
    b must be nT
    """
    # magnitude of v in m/s
    vel = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)*1e3
    # btotal in T
    b2 = data['bx']**2 + data['by']**2 + data['bz']**2
    btot = np.sqrt(b2)
    # clock angle
    theta = np.arctan2(data['by'], data['bz'])
    # check for amp-lim
    if amplim:
        if not amplim_params:
            amplim_params = dict(limit=60, ampfac=7)
        btot = amplify_limit(btot, **amplim_params)
    epsilon = np.sin(theta/2.)**4 * vel * (1e-9*btot)**2
    return epsilon*1e6


def transpolar_voltage(power_in):
    """Calculate transpolar voltage
    """
    term1 = 20*(np.pi*1e6*power_in)**(1/3)
    term2 = 1.4*np.pi*1e6*power_in
    cpcp = term1+term2
    return cpcp


def synth_DP2(power_in):
    """Synthetic AU index, driven by solar wind input only"""
    a_tv = 2.07  # see section 3.3 of Morley, Freeman & Tanskanen 2007
    dp2 = -a_tv*transpolar_voltage(power_in)
    return dp2


def synth_DP1(power_in):
    pass


def synthAL(power_in, energy_out):
    """Synthetic AL index, driven by solar wind and MSM energy output"""
    return dp2(power_in) + dp1(power_in, energy_out)


def synthAU(power_in):
    """Synthetic AU index. Invert sign on directly-driven (DP2) component of synthetic AL"""
    return -1*dp2(power_in)


def syntheticAE(power_in, energy_out):
    """Synthetic Auroral Electrojet (AE) indices described in Morley et al., 2007"""
    return synthAU(power_in) - synthAL(power_in, energy_out)
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os
from patchclamp_analysis.ephys_utilities import protocol_baseline_and_stim


def input_resistance_analyzer(abf, dVm_limits = [-30, 10],to_plot=False,figopt={'type':'jpg','dpi':300}):
    '''Calulates the series of delta Vs and delta Is and fits with a line to find the resistance.'''

    results={}
    if to_plot:
        fig,ax = plt.subplots(1,2,figsize=(4,2.5))

    stim_currents = []
    ss_voltage = []
    is_base, is_stim = protocol_baseline_and_stim(abf)
    for s in abf.sweepList:
        abf.setSweep(s)
        delta_v, _, _ = sweep_VIR(abf.sweepY, abf.sampleRate, is_stim = is_stim)
        delta_I, _, _    = sweep_VIR(abf.sweepC, abf.sampleRate, is_stim = is_stim) # repurpose but for command current
        stim_currents.append( delta_I)
        ss_voltage.append(delta_v)
        if to_plot:
            ax[1].plot(abf.sweepX,abf.sweepY,'k')

    stim_currents = np.array(stim_currents)
    ss_voltage = np.array(ss_voltage)
    vm_in_range = [(dV>=dVm_limits[0] and dV<=dVm_limits[1]) for dV in ss_voltage]

    dI_to_fit = stim_currents[vm_in_range]
    dV_to_fit = ss_voltage[vm_in_range]

    inputR_fit = {}
    slope, intercept , r_value, p_value, std_err = sci.stats.linregress(dI_to_fit, dV_to_fit)
    Rsqr = r_value**2

    dV_hat = dI_to_fit*slope+intercept

    fit_err = (dV_hat - dV_to_fit)**2
    std = np.std(fit_err)
    z_err = (fit_err-np.mean(fit_err))/std
    outlier = z_err > 3

    if np.any(outlier):
        dI_to_fit = dI_to_fit[np.logical_not(outlier)]
        dV_to_fit = dV_to_fit[np.logical_not(outlier)]
        slope, intercept , r_value, p_value, std_err = sci.stats.linregress(dI_to_fit, dV_to_fit)
        Rsqr = r_value**2
        dV_hat = dI_to_fit*slope+intercept

    if to_plot:
        ax[0].scatter(stim_currents, ss_voltage)
        ax[0].scatter(dI_to_fit, dV_to_fit,color='red')
        ax[0].plot( dI_to_fit, dV_hat )
        ax[0].set_xlabel('Current (pA)')
        ax[0].set_ylabel('Delta V (mV)')
        ax[0].set_title('Input_R')
        # plt.show()
        try:    os.makedirs('Saved_Figs/Input_Resitance/')
        except:     None
        fig.savefig( 'Saved_Figs/Input_Resitance/Input_Resitance_' + abf.abfID +'.'+figopt['type'],dpi=figopt['dpi'])


    results['Input_Resistance_MO'] = slope*1E9*1E-6   # to ohms to megaohms
    results['Rin_Rsqr'] = Rsqr
    results['sag_slope'], results['sag_Rsqr'] = measure_sag(abf,ss_window_ms=10)
    return results

def sweep_VIR(trace,rate,is_stim = None, window_t=0.100):
    '''Takes a trace and calulates the steady state delta V from a stimulus in Current Clamp'''
    if any(is_stim == None):
        is_stim = [True for i in trace]
    base_v = trace[:np.where(is_stim==True)[0][0]]
    cutoff = 5
    nyq = rate/2
    normal_cutoff = cutoff / nyq
    b, a = sci.signal.butter(3, normal_cutoff, btype='low')
    filtered_step_v = sci.signal.filtfilt(b, a, trace[is_stim])
    window_wid = int(window_t*rate)
    med_base_v = np.median(base_v[-window_wid:-1])
    med_stim_v = np.median(filtered_step_v[-window_wid:-1])
    delta_v = med_stim_v - med_base_v
    return delta_v, med_base_v, med_stim_v

def measure_sag(abf,ss_window_ms=50,Ih_val = -75):
    fig,ax = plt.subplots(1,2,figsize=(3,2))
    peaks=[]
    steady_states=[]
    is_base, is_stim = protocol_baseline_and_stim(abf)
    sigma=50
    for swp in abf.sweepList:
        abf.setSweep(swp,0)
        v_resp = abf.sweepY[is_stim]
        v_smooth = sci.ndimage.gaussian_filter1d(v_resp, sigma=sigma)
        swp_min = np.min(v_resp)
        ss_ind = int(abf.sampleRate*ss_window_ms/1000)
        swp_ss = np.median(v_resp[-ss_ind:])
        peaks.append(swp_min)
        steady_states.append(swp_ss)
        ax[0].plot(abf.sweepX[is_stim],v_smooth,'k')
    ax[0].set_title(f'smoothed s={sigma}')


    peaks = np.array(peaks)
    steady_states = np.array(steady_states)
    sag = steady_states-peaks
    ax[1].scatter(peaks,sag)

    ih_range = peaks < Ih_val
    x = peaks[ih_range]
    y = sag[ih_range]

    # Fit a line: y = m*x + b
    if len(x)<3:
        return np.nan, np.nan
    coeffs = np.polyfit(x, y, deg=1)
    ax[1].scatter(x,y,color='r')
    fit_line = np.poly1d(coeffs)
    ax[1].plot(x, fit_line(x), color='red', label="Fit for peaks < -90 mV")
    ax[1].set_title('Sag_Slope')
    sag_slope = coeffs[0]
    fig.savefig( 'Saved_Figs/Input_Resitance/Ih_Sag_' + abf.abfID +'.'+'jpg')

    # Predicted values from the fit
    y_pred = fit_line(x)
    # Residual sum of squares
    ss_res = np.sum((y - y_pred) ** 2)
    # Total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # R-squared
    r_squared = 1 - (ss_res / ss_tot)


    return sag_slope, r_squared

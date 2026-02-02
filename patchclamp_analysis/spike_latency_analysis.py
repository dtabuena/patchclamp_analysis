import numpy as np
import matplotlib.pyplot as plt
import os

from patchclamp_analysis.ephys_utilities import find_spike_in_trace


def latencey_analyzer(abf,spike_args,to_plot=False,figopt={'type':'jpg','dpi':300}):
    '''Analyze abf for spike latency'''
    results = {}
    latencey_list = []
    v_hold_list = []
    ap_list = []
    rheo_list = []
    for s in abf.sweepList:
        abf.setSweep(s)
        latencey, v_hold, ramp_ap_thresh, ramp_rheobase = analyze_ramp_sweep(abf,spike_args,to_plot=to_plot,figopt=figopt)
        latencey_list.append(latencey)
        v_hold_list.append(v_hold)
        ap_list.append(ramp_ap_thresh)
        rheo_list.append(ramp_rheobase)

    # Nearest to -70mv
    vhold_err = (abs(np.array(v_hold_list) + 70))
    best = np.where(vhold_err==np.min(vhold_err))[0][0]
    results['Spike_latency_(ms)'] = latencey_list[best]
    results['Ramp_AP_thresh'] = ap_list[best]
    results['Ramp_Vh'] = v_hold_list[best]
    results['Ramp_Rheobase'] = rheo_list[best]

    return results



def analyze_ramp_sweep(abf,spike_args,to_plot=False,figopt={'type':'jpg','dpi':300}):
    'Receives sweep data and finds the first AP and returns it.'
    'Also retuns Vhold for quality control.'

    sweepX = abf.sweepX
    sweepY = abf.sweepY
    sweepC = abf.sweepC
    rate = abf.sampleRate

    is_base = sweepC==sweepC[0]
    is_stim = np.logical_not(sweepC==sweepC[0])
    ramp_start_ind = np.min(np.where(is_base==False))
    v_hold = np.mean( sweepY[0:ramp_start_ind])
    # print(sweepX,sweepY)


    dVds, over_thresh, inds, mean_spike_rate = find_spike_in_trace(sweepY, rate,spike_args,is_stim=is_stim)

    if len(inds)==0:
        print('no spikes found')
        return np.nan,v_hold,np.nan,np.nan
    latencey = sweepX[np.min(inds)-ramp_start_ind]*1000
    ramp_ap_thresh = sweepY[np.min(inds)]
    ramp_rheobase = sweepC[np.min(inds)]


    if to_plot:
        # plt.scatter(sweepX,dVds,color='k')
        fig, axs, =plt.subplots(1,2,figsize=[8,1],width_ratios=[1, 8])
        for a in axs:
            a.plot(sweepX,sweepY,color='k')
            a.scatter(sweepX[inds],sweepY[inds],color='r' )
        zoom_x_relativ = np.array([ 0.75, 1.5])
        zoom_x = zoom_x_relativ*(latencey/1000+sweepX[ramp_start_ind])
        axs[0].set_xlim(zoom_x)
        axs[1].set_xlim([.05,1.25])
        try:    os.makedirs('Saved_Figs/Spike_latency/')
        except:     None
        plt.savefig( 'Saved_Figs/Spike_latency/Spike_latency'+'_' + abf.abfID +'.'+figopt['type'],dpi=figopt['dpi'])
        plt.show()


    return latencey, v_hold, ramp_ap_thresh,ramp_rheobase

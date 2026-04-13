import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os

from patchclamp_analysis.ephys_utilities import (
    protocol_baseline_and_stim,
    find_spike_in_trace,
    spikes_per_stim,
)


def rheobase_analyzer_V2(abf,
                        spike_args =  {'spike_thresh':10, 'high_dv_thresh': 30,'low_dv_thresh': -10,'window_ms': 2},
                        to_plot=False,
                        verbose=False,
                        single_spike=True,figopt={'type':'jpg','dpi':300}):

    ''' File Analyzer for Rheobase etc  '''

    results = {} # default return

    # Rheobase Measure:
    if len(abf.sweepList)<2:
        results = {'message': 'Not enough Sweeps'}
        return results
    else:
        is_base, is_stim = protocol_baseline_and_stim(abf)
        if not np.any(is_stim):
            results = {'message': 'No stim detected'}
            return results
        spike_results = spikes_per_stim(abf, spike_args)
        stim_currents = spike_results['stim_currents']
        spike_counts = spike_results['spike_counts']
        spike_rates = spike_results['spike_rates']
        v_before_AP = spike_results['v_before_spike1']
        v_before_stim = spike_results['v_before_stim']

        single_spikes = spike_counts==1
        zero_spikes = spike_counts==0
        if single_spike:
            none_to_one = np.full(single_spikes.shape, False)
            none_to_one[1:] = np.logical_and(single_spikes[1:], zero_spikes[:-1])
            first_spike_stim = np.where(none_to_one)[0]
        else:
            some_spikes = spike_counts>0
            none_to_some = np.full(single_spikes.shape, False)
            none_to_some[1:] = np.logical_and(some_spikes[1:], zero_spikes[:-1])
            first_spike_stim = np.where(none_to_some)[0]


    if first_spike_stim.size == 0:
        return results
    else:
        if first_spike_stim.size >1:
            first_spike_stim = first_spike_stim[0]
        else:
            first_spike_stim = first_spike_stim[0]
        results['Rheobase'] = stim_currents[first_spike_stim]
        results['Vhold_spike'] = v_before_stim[first_spike_stim]
        # results['AP_thresh'] = v_before_AP[first_spike_stim]

    if isinstance(first_spike_stim, (int, np.integer)):
        abf.setSweep(first_spike_stim)
        ap_params = single_ap_stats(abf,spike_args,window_ms=[-3, 9.5],to_plot=to_plot,verbose=verbose)
        results.update(ap_params)


    if to_plot:
        rheo_fig, ax = plt.subplots(1,1,figsize=(1.75,1.5))

        try:    os.makedirs('Saved_Figs/Rheobase/')
        except:     None
        for s in abf.sweepList:
            abf.setSweep(s)
            ax.plot(abf.sweepX,abf.sweepY,label = str(stim_currents[s]) + ' pA')
        ax.legend(loc='upper left', bbox_to_anchor=(1.1,1)) #,
        plt.show()
        plt.tight_layout()
        rheo_fig.savefig( 'Saved_Figs/Rheobase/Rheobase' + '_' + abf.abfID +'.'+figopt['type'],dpi=figopt['dpi'])


    return results


def single_ap_stats(abf,spike_args,window_ms=[-3, 6.5],rise_fraction=0.90,to_plot=True,verbose=False,up_sample = True):

    x_trace = abf.sweepX
    y_trace = abf.sweepY
    sample_rate = abf.sampleRate

    if up_sample:
        factor = 4
        x_new = np.linspace(x_trace[0],x_trace[-1], num=len(x_trace)*factor )
        interp_func = sci.interpolate.interp1d(x_trace, y_trace, kind='quadratic')
        y_trace = interp_func(x_new)
        x_trace = x_new
        sample_rate = sample_rate*factor

    is_stim = np.full(y_trace.shape,True)
    dVds, over_thresh, inds, mean_spike_rate = find_spike_in_trace(y_trace,sample_rate,spike_args,is_stim = is_stim)

    # get AP
    window_ind = np.arange(window_ms[0]/1000*sample_rate,window_ms[1]/1000*sample_rate)
    ap_start_ind = inds[0]
    ap_indicies = np.array(window_ind+ap_start_ind,dtype='int')
    spike_trace_x = x_trace[ap_indicies]
    spike_trace_y = y_trace[ap_indicies]
    spike_trace_dvds = np.diff(spike_trace_y,prepend=spike_trace_y[0])*sample_rate/1000

    ## Stats on AP
    'AP threshhold'
    ap_thresh_ind = int(abs(window_ind[0]))
    ap_thresh= spike_trace_y[ap_thresh_ind]

    'AP Max'
    v_max = np.max(spike_trace_y)
    v_max_ind = np.argmax(spike_trace_y)
    ap_amplitutude = v_max-ap_thresh
    v_min = np.min(spike_trace_y)

    'APD 50'
    v_half = np.mean([v_max,ap_thresh])
    ap_above_half = spike_trace_y>=v_half
    bool_dif = np.diff(ap_above_half,prepend=0)
    half_start = np.where(bool_dif == 1)[0][0]
    half_stop = np.where(bool_dif == -1)[0][0]
    ap50_width_ms = (spike_trace_x[half_stop] - spike_trace_x[half_start])*1000

    'FastAfterHype'

    fahp_wind = np.arange(v_max_ind,len(spike_trace_y)-v_max_ind,dtype='int')
    fast_after_hyperpol = np.min(spike_trace_y[fahp_wind])
    fast_after_hyperpol_ind = np.argmin(spike_trace_y[fahp_wind])+v_max_ind
    fast_after_hyperpol = fast_after_hyperpol - ap_thresh

    'Rise and Fall time'
    fractional_peak = ap_thresh+rise_fraction*(v_max-ap_thresh)
    fractional_base = ap_thresh+(1-rise_fraction)*(v_max-ap_thresh)
    rising_bool = np.array(spike_trace_y>=fractional_base) * np.array(spike_trace_y<=fractional_peak) * np.array(spike_trace_dvds>0)
    falling_bool = np.array(spike_trace_y>=fractional_base) * np.array(spike_trace_y<=fractional_peak) * np.array(spike_trace_dvds<0)

    rising_bool_diff=np.diff(rising_bool,prepend=0)
    first_stop = np.where(rising_bool_diff==-1)[0][0]
    rising_bool[first_stop:]=False

    falling_bool_diff=np.diff(falling_bool,prepend=0)
    first_stop = np.where(falling_bool_diff==-1)[0][0]
    falling_bool[first_stop:]=False

    rise_time_ms = len(spike_trace_x[rising_bool])/sample_rate*1000
    fall_time_ms = len(spike_trace_x[falling_bool])/sample_rate*1000

    'dvdt stats'
    dv_max = np.max(spike_trace_dvds)
    dv_min = np.min(spike_trace_dvds)

    if to_plot:
        fig, axs = plt.subplots(1,2,figsize=(2,1))
        for ax in axs:
            ax.plot(spike_trace_x,spike_trace_y,'k.-',zorder=-1)

            ax.scatter(spike_trace_x[ap_thresh_ind],spike_trace_y[ap_thresh_ind],color='red')
            ax.axhline(spike_trace_y[ap_thresh_ind],color='red',linewidth=.1)

            ax.scatter(spike_trace_x[fast_after_hyperpol_ind],spike_trace_y[fast_after_hyperpol_ind],color='blue')
            ax.axhline(spike_trace_y[fast_after_hyperpol_ind],color='blue',linewidth=.1)

            ax.plot([spike_trace_x[half_start],spike_trace_x[half_stop]],[v_half]*2,'orange')

            ax.plot(spike_trace_x[rising_bool],spike_trace_y[rising_bool],color='magenta' )
            ax.plot(spike_trace_x[falling_bool],spike_trace_y[falling_bool],color='cyan' )

            ax.set_ylim(np.min([v_min*1.1]),top=np.max([v_max,40]))

        buf = 1.2
        axs[1].set_ylim([(ap_thresh+fast_after_hyperpol*buf),(ap_thresh-fast_after_hyperpol*buf/2)])
        axs[1].set_yticks([ap_thresh,ap_thresh+fast_after_hyperpol])
        plt.tight_layout()
        
        os.makedirs('Saved_Figs/AP_Params/', exist_ok=True)
        fig.savefig( 'Saved_Figs/AP_Params/AP_Params_' + abf.abfID +'.png',dpi=300)


    ap_params = {'ap_amplitutude':ap_amplitutude,
                'fast_after_hyperpol':fast_after_hyperpol,
                'AP_thresh':ap_thresh,
                'v_half':v_half,
                'ap50_width_ms':ap50_width_ms,
                'rise_time_ms':rise_time_ms,
                'fall_time_ms':fall_time_ms,
                'dv_max':dv_max,
                'dv_min':dv_min,}
    return ap_params

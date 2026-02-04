
import numpy as np
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from patchclamp_analysis.ephys_utilities import (
    protocol_baseline_and_stim,
    spikes_per_stim,
    find_spike_in_trace,
    movmean,
)


def gain_analyzer_v2(abf,spike_args =  {'spike_thresh':10, 'high_dv_thresh': 25,'low_dv_thresh': -5,'window_ms': 2}, to_plot = 0,
                  max_fit_steps=8,rel_slope_cut=.7,Vh_hilo = [-60,-80],figopt={'type':'jpg','dpi':300},factor=2):
    '''Analyze Single ABF of increasing current injections for firing rate gain'''
    '''to_plot scales from 0:2, no plot, plot just the final fitting, plot every sweep for spike detection'''

    results= {} # init results dict
    if len(abf.sweepList)<5: return results # not enough sweeps to analyze

    is_base, is_stim = protocol_baseline_and_stim(abf) # find base lines and stims

    spike_results= spikes_per_stim(abf,spike_args, mode='count', to_plot=to_plot)
    stim_currents = spike_results['stim_currents']
    spike_counts = spike_results['spike_counts']
    spike_rates = spike_results['spike_rates']
    V_before_stim = spike_results['V_before_stim']
    fire_dur = spike_results['fire_dur']
    isi_rates = spike_results['isi_rates']
    spike_times = spike_results['spike_times']


    Vh_ok = [i for i in range(len(V_before_stim)) if V_before_stim[i]>np.min(Vh_hilo)]
    Vh_ok = [i for i in Vh_ok if V_before_stim[i]<np.max(Vh_hilo)]

    stim_currents = np.array([stim_currents[i] for i in Vh_ok])
    spike_counts = np.array([spike_counts[i] for i in Vh_ok])
    V_before_stim = np.array([V_before_stim[i] for i in Vh_ok])
    spike_rates = np.array([spike_rates[i] for i in Vh_ok])
    isi_rates = np.array([isi_rates[i] for i in Vh_ok])


    if sum(spike_counts)==0: return results   #if no spikes return none
    plot_name = abf.abfID
    if_fit = fit_firing_gain(stim_currents, spike_counts, spike_rates,
                             abf, spike_times, isi_rates, to_plot=to_plot>0,
                             plot_name=plot_name, figopt=figopt,
                             max_fit_steps=max_fit_steps, rel_slope_cut=rel_slope_cut)

    if sum(spike_counts)>0:
        max_fire_sweep = np.where(spike_counts==np.max(spike_counts))[0][0]
        ADR_sAHP_ind = np.where(stim_currents>0)[0][0]
        ADR_sAHP_ind = int(ADR_sAHP_ind*factor)
        ADR_sAHP_ind = np.min([ADR_sAHP_ind,max_fire_sweep])

        try: results['sAHP']=calc_slow_afterhyp(abf,ADR_sAHP_ind)
        except: results['sAHP']=np.nan
        try: results['ADR'],_=calc_adapt_ratio(abf,ADR_sAHP_ind,spike_args,to_plot=False)
        except: results['ADR']=np.nan


    sweep = np.argmax(spike_results['spike_counts'])
    phase_fig = ap_phase(abf,sweep,spike_results['spike_times'][sweep])
    phase_fig.savefig( 'Saved_Figs/Firing_Gain/Phase'+'_' + plot_name +'.'+figopt['type'])


    results['Gain_(HzpA)']=if_fit['slope']
    results['Gain_R2']=if_fit['R2']
    results['Spike_Counts']=dict(zip(stim_currents, spike_counts))
    # results['Firing_Duration_%']=fire_dur
    results['Gain_Vh']=V_before_stim
    results['V_stim']= calc_vm_stim(abf,is_stim,spike_counts,isi_rates,to_plot=False)

    adapt_res = adaption_analysis_v2(abf, spike_results, to_plot=to_plot>0,plot_name=plot_name,figopt=figopt)
    results['max_adapt'] = adapt_res['max_adapt']
    results['adapt_thresh_90'] = adapt_res['adapt_thresh_90']
    results['isi_ratios'] = adapt_res['isi_ratios']
    results['max_freq_isi_trace'] = adapt_res['max_freq_isi_trace']

    results['inact_current_pA'] = if_fit['inact_current']

    return results





def adaption_analysis_v2(abf, spike_results, to_plot=False, plot_name='recording', figopt={'type':'jpg','dpi':300}):
    # Extract spike data from results
    spike_times = spike_results['spike_times']
    mean_inst_rates = spike_results['isi_rates']
    spike_rates = spike_results['spike_rates']
    stim_currents = spike_results['stim_currents']

    # Skip if insufficient spiking
    max_spikes = np.max(spike_rates)
    if max_spikes < 2:
        return np.nan, np.nan

    # Plot spike raster with instantaneous firing rates
    if to_plot:
        fig, (ax_spike_count, ax_isi, ax_max_freq_isi) = plt.subplots(3,1,figsize=(2,3))
    colors = plt.cm.viridis(np.linspace(0, 1, len(spike_times)))
    colors =  colors[::-1]
    for si in range(len(spike_times)):
        s = spike_times[si]
        label = str(len(s)) + ' at ' + str(mean_inst_rates[si]) + ' hz'
        last_x = s[-1:]
        last_y = np.arange(len(s))[-1:] + 1
        if to_plot:
            ax_spike_count.plot(s, np.arange(len(s)) + 1, color=colors[si])
            ax_spike_count.scatter(last_x, last_y, color=colors[si], label=label)
            # if len(last_x) > 0:
            #     ax_spike_count.text(last_x[0], last_y[0], str(stim_currents[si]) + 'pA', ha='left', va='bottom')


    isi_ratios = np.full(len(spike_times),np.nan)
    for i,st in enumerate(spike_times):
        if len(st)>5:
            isi = np.diff(st)*1000
            rel_isi = isi/isi[0]
            spike_no = np.arange(len(isi))+1
            ax_isi.plot(spike_no, isi, color=colors[i], marker='o', label=str(int(stim_currents[i]))+'pA')
            isi_ratios[i] = rel_isi[-1]
        if len(st) == max_spikes:
            isi = np.diff(st)*1000
            rel_isi = isi/isi[0]
            spike_no = np.arange(len(isi))+1
            ax_max_freq_isi.plot(spike_no, rel_isi, color=colors[i], marker='o', label=str(int(stim_currents[i]))+'pA')
            max_freq_isi_trace = {s:r for s,r in zip(spike_no, rel_isi)}
            ax_max_freq_isi.set_xlabel('Spike Number (#)')
            ax_max_freq_isi.set_ylabel('Adapt Ratio')
            ax_max_freq_isi.axhline(1,color = 'k',linestyle=':')
            ax_max_freq_isi.set_ylim(bottom=0)


    if to_plot:
        ax_spike_count.set_xlabel('Spike Time (s)')
        ax_spike_count.set_ylabel('Spike Number (#)')
        ax_spike_count.set_xlim(-.4, 1)
        handles, labels = ax_spike_count.get_legend_handles_labels()
        ax_spike_count.legend(handles[-4:], labels[-4:], loc='upper left', bbox_to_anchor=(0, 1))

        ax_isi.set_xlabel('Spike Number (#)')
        ax_isi.set_ylabel('Inter-Spike Interval (ms)')
        # handles, labels = ax_isi.get_legend_handles_labels()
        # ax_isi.legend(handles[-4:], labels[-4:], loc='upper left', bbox_to_anchor=(0, 1))
        plt.tight_layout()

    # Calculate adaptation: 1 - (spike_rate / instantaneous_rate)
    sweep_adaption = [1 - (sr/mir if mir != 0 else 0) for sr, mir in zip(spike_rates, mean_inst_rates)]
    sweep_adaption = [np.nan if sa < 0 else sa for sa in sweep_adaption]

    # Find threshold where adaptation < 10%
    non_adapting = np.array(sweep_adaption) < 0.1
    if np.sum(non_adapting) == 0:
        adapt_thresh_90 = np.nan
    else:
        adapt_thresh_90 = np.max(stim_currents[non_adapting])

    # Calculate max adaptation for sweeps with sufficient instantaneous rate
    sweep_adaption = [sweep_adaption[si] for si in range(len(spike_times)) if mean_inst_rates[si]*2 > max_spikes]
    max_adapt = np.nanmax(sweep_adaption)

    if to_plot:
        fig.savefig('Saved_Figs/Firing_Gain/Adaption' + '_' + plot_name + '.' + figopt['type'], dpi=figopt['dpi'])

    isi_ratios = {s:r for s,r in zip(stim_currents,isi_ratios)}

    adapt_res = {'max_adapt':max_adapt,
                 'adapt_thresh_90':adapt_thresh_90,
                 'isi_ratios':isi_ratios,
                 'max_freq_isi_trace':max_freq_isi_trace,}

    return adapt_res



def calc_vm_stim(abf,is_stim,spike_counts,isi_rates,to_plot=False):
    stim_traces=[]
    stim_cur = []
    for s in abf.sweepList:
        abf.setSweep(s,0)
        stim_traces.append(abf.sweepY[is_stim])
        stim_cur.append( np.median(abf.sweepC[is_stim]))
    vm_list = [np.median(st) for st in stim_traces]
    if to_plot:
        fig, ax = plt.subplots(1,2,figsize=(2,1.25))
        ax[0].plot(stim_cur,vm_list,'ko-')
        ax[1].set_ylabel('vm')
        ax[1].set_xlabel('current')
        ax[1].plot(vm_list,spike_counts,'ko-')
        ax[1].set_ylabel('spikes')
        ax[1].set_xlabel('vm')
        ax[1].set_xlim([-80,35])
        max_fire = np.max(spike_counts)
        v_at_max = vm_list[np.where(spike_counts==max_fire)[0][0]]
        ax[1].axline( (v_at_max,0) ,(v_at_max,max_fire))
    return vm_list

def calc_inactivation(isi_rates, spike_counts, stim_currents, inact_thresh=0.9):
    # Calculate ratio of spike count to ISI-based rate estimate
    isi_ratio = np.divide(spike_counts, isi_rates, out=np.zeros_like(spike_counts, dtype=float), where=isi_rates!=0)

    # Find first sweep after max spike count where ratio drops below threshold
    max_ind = np.argmax(spike_counts)
    inactivating = isi_ratio <= inact_thresh
    after_max = np.cumsum(np.ones_like(isi_ratio)) >= max_ind
    where_true = np.where(np.logical_and(inactivating, after_max))[0]

    # Return current and pulse number where inactivation occurs
    if len(where_true) > 0:
        inact_pulse_num = where_true[0]
        inact_current = stim_currents[inact_pulse_num]
    else:
        inact_pulse_num = np.nan
        inact_current = stim_currents[-1] + 0.1

    return inact_current, inact_pulse_num


def ap_phase(abf, sweep, spike_times, up_sample=True, window_ms=[-1, 8]):
    _, is_stim = protocol_baseline_and_stim(abf)
    abf.setSweep(sweep)
    trace = abf.sweepY[is_stim]
    time = abf.sweepX[is_stim]
    time = time - time[0]
    sample_rate = abf.sampleRate

    if up_sample:
        factor = 4
        x_new = np.linspace(time[0], time[-1], num=len(time) * factor)
        interp_func = sci.interpolate.interp1d(time, trace, kind='quadratic')
        trace = interp_func(x_new)
        time = x_new
        sample_rate *= factor

    window_pts = np.arange(window_ms[0] / 1000 * sample_rate, window_ms[1] / 1000 * sample_rate).astype(int)
    half_window = max(abs(window_pts[0]), abs(window_pts[-1]))

    spike_inds = (spike_times * sample_rate).astype(int)

    # Only keep spikes that are fully inside the trace
    valid = (spike_inds > half_window) & (spike_inds < len(trace) - half_window)
    spike_inds = spike_inds[valid]
    if len(spike_inds) == 0:
        raise ValueError("No spikes within safe bounds for phase plot.")

    spike_ind_mat = np.expand_dims(spike_inds, -1) + window_pts
    spike_mat = trace[spike_ind_mat]
    dv_mat = np.diff(spike_mat, axis=1) * sample_rate / 1000  # dV/dt in V/s
    v_mat = spike_mat[:, :-1]

    # Create color-mapped plot
    num_colors = len(spike_inds)
    viridis_colors = mpl.cm.viridis(np.linspace(0, 1, num_colors))

    phase_fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.set_prop_cycle(mpl.cycler('color', viridis_colors))
    ax.plot(v_mat.T, dv_mat.T)
    ax.grid()
    ax.set_xlim(-60, 60)
    ax.set_ylim(-150, 325)
    ax.set_ylabel('V/s')
    ax.set_xlabel('mV')

    return phase_fig



def calc_adapt_ratio(abf,ADR_sAHP_ind,spike_args,to_plot=False):
    '''Calculate the adaption Ratio using ISI ratios of a specified sweep'''

    is_base, is_stim = protocol_baseline_and_stim(abf)
    abf.setSweep(ADR_sAHP_ind)

    dVds, over_thresh, inds, mean_spike_rate = find_spike_in_trace(abf.sweepY,abf.sampleRate,spike_args,is_stim=is_stim,mode='count')

    isi_series = np.diff(np.array(inds)/abf.sampleRate)
    ADR = isi_series[0]/isi_series[-1]

    if to_plot:
        fig, ax = plt.subplots(1,2)
        ax[0].plot(abf.sweepX[is_stim],abf.sweepY[is_stim],'k')
        ax[0].scatter(abf.sweepX[inds],abf.sweepY[inds],color='r')
        ax[1].plot(isi_series*1000,'-o',color='k')

    return ADR, isi_series


def calc_slow_afterhyp(abf,ADR_sAHP_ind):
    '''Calculate the slow after hyperpolarization of a specified sweep'''

    is_base, is_stim = protocol_baseline_and_stim(abf)
    abf.setSweep(ADR_sAHP_ind)

    stim_start_ind = np.min(np.where(is_stim))
    stim_stop_ind = np.max(np.where(is_stim))

    pre_stim_inds = np.where(is_base[0:stim_start_ind])
    post_stim_inds = np.where(is_base[stim_stop_ind:])

    base_Vm = abf.sweepY[pre_stim_inds]
    after_Vm = abf.sweepY[post_stim_inds]
    slow_afterhyp = np.mean(base_Vm) - np.min(after_Vm)
    return slow_afterhyp






def fit_firing_gain(stim_currents, spike_counts, spike_rates, abf,spike_times,isi_rates,to_plot=False,plot_name='',figopt={'type':'jpg','dpi':300},max_fit_steps=8,rel_slope_cut=.7):
    '''Gathers the firing rate of each stimuli and fits the linear portion of the curve to return the Gain in Hz/pA (the slope)'''

    is_pos_slope = np.diff(spike_counts,prepend=0)>0
    is_pos_slope = movmean(np.diff(spike_counts,prepend=0),4)>0
    peak_ind = np.where(spike_counts==np.max(spike_counts))[0]
    if len(peak_ind)>1:
        peak_ind = np.min(peak_ind)

    spike_slope = np.diff(spike_counts,prepend=np.nan)
    max_spike_slope = np.percentile(spike_slope[spike_slope>0],80)
    rel_spike_slope = spike_slope/max_spike_slope
    good_jerk = rel_spike_slope>.7
    first = spike_slope==spike_counts
    good_jerk[first]=True

    before_peak = np.arange(len(spike_counts))<=peak_ind
    is_nonzero = np.array(spike_counts)>0
    use_for_fit = np.logical_and.reduce((is_pos_slope,is_nonzero,before_peak,good_jerk))
    use_for_fit = np.logical_and.reduce((use_for_fit,np.cumsum(use_for_fit)<=max_fit_steps))

    if np.sum(use_for_fit)==1:
        last_zero = np.where(use_for_fit)[0][0]-1
        use_for_fit[last_zero]=1
        use_for_fit[peak_ind]=1

    if_fit = {}
    if_fit['stim_currents'] = stim_currents
    if_fit['spike_rates'] = spike_rates
    if 0 == np.sum(spike_rates):
        # print('no spikes detected')
        if_fit['slope'] = np.nan
        if_fit['rel_slope'] = np.nan
        if_fit['intercept'] = np.nan
        if_fit['rel_intercept'] = np.nan
        if_fit['R2'] = 0
        if_fit['inact_current'] = np.nan
        return if_fit

    if np.sum(spike_rates>0)<3:
        if_fit['slope'] = np.nan
        if_fit['rel_slope'] = np.nan
        if_fit['intercept'] = np.nan
        if_fit['rel_intercept'] = np.nan
        if_fit['R2'] = 0
        if_fit['inact_current'] = np.nan
        return if_fit


    if_fit['slope'], if_fit['intercept'] , r_value, p_value, std_err = sci.stats.linregress(stim_currents[use_for_fit], spike_rates[use_for_fit])
    if_fit['R2'] = r_value**2

    if_fit['inact_current'], inact_pulse_num = calc_inactivation(isi_rates,spike_counts,stim_currents,inact_thresh=0.9)

    big_marker = plt.rcParams['lines.markersize']*2
    if to_plot:
        my_fig, ax = plt.subplots(1,2, figsize=[3,1.5],gridspec_kw={'width_ratios': [2, 1]})
        my_fig.suptitle(plot_name)
        ax[1].scatter( if_fit['stim_currents'] ,if_fit['spike_rates'], color='k' )
        ax[1].plot( if_fit['stim_currents'], if_fit['slope']* if_fit['stim_currents']+if_fit['intercept'])
        ax[1].scatter(if_fit['stim_currents'][peak_ind],if_fit['spike_rates'][peak_ind],s=big_marker, color='c',marker="X")
        ax[1].scatter( if_fit['stim_currents'][use_for_fit] ,if_fit['spike_rates'][use_for_fit], color='m' )
        ax[1].set_xlabel('current')
        ax[1].set_ylabel('Spike Rate (Hz)')
        (min,max) = ax[1].get_ylim()
        ax[1].text(0, max/2, 'R**2='+str(round(if_fit['R2'],2)),fontsize='large')


        ax[1].scatter(if_fit['stim_currents'], isi_rates,color='orange')

        if not np.isnan(inact_pulse_num):
            ax[1].scatter(if_fit['stim_currents'][inact_pulse_num], if_fit['spike_rates'][inact_pulse_num],marker="+",color='r')

        n = len(abf.sweepList)
        colors = plt.cm.viridis(np.linspace(0,1,n))
        for s in abf.sweepList:
            abf.setSweep(s)
            ax[0].plot(abf.sweepX,abf.sweepY,color=colors[s])

        os.makedirs('Saved_Figs/Firing_Gain/',exist_ok=True)
        plt.tight_layout()
        plt.show()
        my_fig.savefig( 'Saved_Figs/Firing_Gain/Firing_Gain'+'_' + plot_name + figopt['type'],dpi=figopt['dpi'])
    return if_fit


def check_inactivation( time, trace, is_stim, sample_rate, dVds, inds, mean_spike_rate, to_plot=0 ):
    time_ms = time*1000
    sum_isi = None
    rel_firing_duration = None
    if len(inds)>0:
        stim_time = time_ms[np.where(is_stim)[0][0]]
        firing_duration = time[inds[-1]]
        rel_firing_duration = firing_duration /(np.max(time[is_stim]*1000)-stim_time)
    return rel_firing_duration














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

def gain_analyzer_v2(abf, spike_args={'spike_thresh':10, 'high_dv_thresh': 25, 'low_dv_thresh': -5, 'window_ms': 2}, to_plot=False,
                     max_fit_steps=8, rel_slope_cut=.7, Vh_hilo=[-60,-80], figopt={'type':'jpg','dpi':300}, factor=2):
    '''Analyze Single ABF of increasing current injections for firing rate gain'''

    results = {}
    if len(abf.sweepList) < 5: return results

    is_base, is_stim = protocol_baseline_and_stim(abf)

    spike_results = spikes_per_stim(abf, spike_args, mode='count')
    stim_currents = spike_results['stim_currents']
    spike_counts  = spike_results['spike_counts']
    spike_rates   = spike_results['spike_rates']
    v_before_stim = spike_results['v_before_stim']
    isi_rates     = spike_results['isi_rates']
    spike_times   = spike_results['spike_times']

    Vh_ok = [i for i in range(len(v_before_stim)) if v_before_stim[i] > np.min(Vh_hilo)]
    Vh_ok = [i for i in Vh_ok if v_before_stim[i] < np.max(Vh_hilo)]

    stim_currents = np.array([stim_currents[i] for i in Vh_ok])
    spike_counts  = np.array([spike_counts[i] for i in Vh_ok])
    v_before_stim = np.array([v_before_stim[i] for i in Vh_ok])
    spike_rates   = np.array([spike_rates[i] for i in Vh_ok])
    isi_rates     = np.array([isi_rates[i] for i in Vh_ok])

    if sum(spike_counts) == 0: return results

    gain_rheo_sweep = np.where(spike_counts > 0)[0][0]
    results['gain_rheo'] = stim_currents[gain_rheo_sweep]

    plot_name = abf.abfID
    if_fit = fit_firing_gain(stim_currents, spike_counts, spike_rates,
                             abf, spike_times, isi_rates, to_plot=to_plot,
                             plot_name=plot_name, figopt=figopt,
                             max_fit_steps=max_fit_steps, rel_slope_cut=rel_slope_cut)

    results['sAHP'] = calc_slow_afterhyp(abf, to_plot=to_plot, plot_name=plot_name)

    sweep = np.argmax(spike_results['spike_counts'])
    phase_fig = ap_phase(abf, sweep, spike_results['spike_times'][sweep])
    phase_fig.savefig('Saved_Figs/Firing_Gain/Phase_' + plot_name + '.' + figopt['type'])

    results['Gain_(HzpA)']  = if_fit['slope']
    results['Gain_R2']      = if_fit['R2']
    results['Spike_Counts'] = dict(zip(stim_currents, spike_counts))
    results['Gain_Vh']      = v_before_stim
    results['V_stim']       = calc_vm_stim(abf, is_stim, spike_counts, isi_rates, to_plot=False)

    adapt_res = adaption_analysis_v3(spike_results, gain_rheo_sweep, factor=factor,
                                     to_plot=to_plot, plot_name=plot_name, figopt=figopt)
    results.update(adapt_res)

    return results

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

    phase_fig, ax = plt.subplots(figsize=(1.25, 1.25))
    ax.set_prop_cycle(mpl.cycler('color', viridis_colors))
    ax.plot(v_mat.T, dv_mat.T)
    ax.grid()
    ax.set_xlim(-60, 60)
    ax.set_ylim(-150, 325)
    ax.set_ylabel('V/s')
    ax.set_xlabel('mV')

    return phase_fig




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



def calc_slow_afterhyp(abf,to_plot=False,plot_name=""):
    '''Calculate the slow after hyperpolarization of a specified sweep'''

    is_base, is_stim = protocol_baseline_and_stim(abf)

    stim_start_ind = np.min(np.where(is_stim))
    stim_stop_ind = np.max(np.where(is_stim))

    pre_stim_mask = is_base & (np.arange(len(is_base)) < stim_start_ind)
    post_stim_mask = is_base & (np.arange(len(is_base)) > stim_stop_ind)

    if to_plot:
        fig, ax = plt.subplots(1, 2, figsize=(3, 1))
    cmap = plt.colormaps['Greys'] # Needed for the zip iterator even if not plotting
    colors = cmap(np.linspace(0.3, 1.0, len(abf.sweepList))) # Needed for the zip iterator even if not plotting

    slow_list = []
    stim_level = []
    for s, color in zip(abf.sweepList, colors):
        abf.setSweep(s)
        stim_level.append(abf.sweepC[stim_start_ind])
        y_trace = abf.sweepY

        base_Vm = np.mean(y_trace[pre_stim_mask])
        sAHP_val = np.min(y_trace[post_stim_mask]) - base_Vm
        slow_list.append(sAHP_val)

        post_stim_global_inds = np.where(post_stim_mask)[0]
        sAHP_local_ind = np.argmin(y_trace[post_stim_mask])
        sAHP_global_ind = post_stim_global_inds[sAHP_local_ind]

        y_plot = y_trace - y_trace[0]
        if to_plot:
            ax[0].plot(abf.sweepX, y_plot, color=color)
            ax[0].scatter(abf.sweepX[sAHP_global_ind], y_plot[sAHP_global_ind], c='m', s=3, zorder=99)

    slow_afterhyp = np.min(slow_list)
    if to_plot:
        ax[0].axhline(0, color='red', linestyle=':')
        ax[0].set_ylim([np.min([slow_afterhyp * 1.1, -5]), +8])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Rel Vm (mv)')
        

        ax[1].scatter(stim_level,slow_list,c=colors,s=3)
        ax[1].set_ylim(np.min([-5,slow_afterhyp*1.1,2]))
        ax[1].axhline(0, color='red', linestyle=':')
        ax[1].set_ylabel('sAHP (mv)')
        ax[1].set_xlabel('Stim (pA)')
        plt.tight_layout()
        plt.draw()
        fig.savefig( 'Saved_Figs/Firing_Gain/Slow_After_Hyperpolarization'+'_' + plot_name+'.jpg')
    return slow_afterhyp


def adaption_analysis_v3(spike_results, gain_rheo_sweep, factor=2, inact_thresh=0.9, 
                          outlier_isi_factor=3, ADR_min_spikes=10,
                          to_plot=False, plot_name='recording', figopt={'type':'jpg','dpi':300}):

    spike_times   = spike_results['spike_times']
    isi_rates     = spike_results['isi_rates']
    spike_rates   = spike_results['spike_rates']
    spike_counts  = spike_results['spike_counts']
    stim_currents = spike_results['stim_currents']

    stim_currents = np.array(stim_currents)
    spike_counts  = np.array(spike_counts)
    spike_rates   = np.array(spike_rates)
    isi_rates     = np.array(isi_rates)

    max_spikes     = np.max(spike_rates)
    max_fire_sweep = np.argmax(spike_counts)

    # --- select ADR sweep: closest to 2x gain_rheo current, enough spikes ---
    valid_sweeps_adr = np.where(np.array([len(st) for st in spike_times]) >= ADR_min_spikes)[0]
    target_current   = stim_currents[gain_rheo_sweep] * factor
    ADR_sweep        = valid_sweeps_adr[np.argmin(np.abs(stim_currents[valid_sweeps_adr] - target_current))]
    ADR_sweep        = int(ADR_sweep) if len(valid_sweeps_adr) > 0 else None

    # --- ADR: first/last ISI for chosen sweep, outliers excluded ---
    st_adr = spike_times[ADR_sweep]
    if len(st_adr) >= 2:
        isi_adr       = np.diff(np.array(st_adr)) * 1000
        isi_adr_thresh = np.median(isi_adr) * outlier_isi_factor
        isi_adr_clean  = isi_adr[isi_adr <= isi_adr_thresh]
        ADR = isi_adr_clean[0] / isi_adr_clean[-1] if len(isi_adr_clean) >= 2 else np.nan
    else:
        ADR = np.nan

    # --- per-sweep ISI ratios, max-freq ISI trajectory, and outlier masks ---
    isi_ratios         = np.full(len(spike_times), np.nan)
    max_freq_isi_trace = {}
    max_freq_isi_trace_raw = {}
    max_spike_sweep    = np.argmax(spike_counts)
    sweep_isis         = []
    sweep_masks        = []

    for i, st in enumerate(spike_times):
        if len(st) > 5:
            isi          = np.diff(np.array(st)) * 1000
            isi_thresh   = np.median(isi) * outlier_isi_factor
            outlier_mask = isi > isi_thresh
            isi_clean    = isi[~outlier_mask]
            if len(isi_clean) >= 2:
                isi_ratios[i] = isi_clean[-1] / isi_clean[0]
            if i == max_spike_sweep:
                rel_isi       = isi / isi[0]
                rel_isi_clean = np.where(outlier_mask, np.nan, rel_isi)
                spike_no      = np.arange(len(isi)) + 1
                max_freq_isi_trace     = {int(s): r for s, r in zip(spike_no, rel_isi_clean)}
                max_freq_isi_trace_raw = {int(s): r for s, r in zip(spike_no, rel_isi)}
            sweep_isis.append(isi)
            sweep_masks.append(outlier_mask)
        else:
            sweep_isis.append(np.array([]))
            sweep_masks.append(np.array([], dtype=bool))

    isi_ratios = {float(s): r for s, r in zip(stim_currents, isi_ratios)}

    # --- max_adapt ---
    sweep_adaption = np.array([1 - (sr / mir if mir != 0 else 0) for sr, mir in zip(spike_rates, isi_rates)])
    sweep_adaption[sweep_adaption < 0] = np.nan
    valid_sweeps = isi_rates * 2 > max_spikes
    max_adapt = np.nanmax(sweep_adaption[valid_sweeps]) if np.any(valid_sweeps) else np.nan

    # --- inact_current ---
    isi_ratio_arr = np.divide(spike_counts, isi_rates, out=np.zeros_like(spike_counts, dtype=float), where=isi_rates != 0)
    max_ind       = np.argmax(spike_counts)
    after_max     = np.arange(len(spike_counts)) >= max_ind
    inactivating  = isi_ratio_arr <= inact_thresh
    where_inact   = np.where(np.logical_and(inactivating, after_max))[0]
    if len(where_inact) > 0:
        inact_pulse_num = where_inact[0]
        inact_current   = stim_currents[inact_pulse_num]
    else:
        inact_pulse_num = np.nan
        inact_current   = stim_currents[-1] + 0.1

    # --- plotting ---
    if to_plot:
        colors = plt.cm.viridis(np.linspace(0, 1, len(spike_times)))[::-1]
        fig, (ax_spike_count, ax_isi, ax_max_freq_isi) = plt.subplots(3, 1, figsize=(2, 3))

        # subplot 1: spike raster
        for si, st in enumerate(spike_times):
            if len(st) == 0:
                continue
            ax_spike_count.plot(st, np.arange(len(st)) + 1, color=colors[si])
            ax_spike_count.scatter(st[-1], len(st), color=colors[si], zorder=5)

        ax_spike_count.set_xlabel('Spike Time (s)')
        ax_spike_count.set_ylabel('Spike Number (#)')
        ax_spike_count.set_xlim(-0.4, 1)

        # subplot 2: ISI traces, outliers as hollow circles
        for i, (isi, outlier_mask) in enumerate(zip(sweep_isis, sweep_masks)):
            if len(isi) == 0:
                continue
            spike_no = np.arange(len(isi)) + 1
            ax_isi.plot(spike_no, isi, color=colors[i], marker='o', zorder=1)
            if np.any(outlier_mask):
                ax_isi.scatter(spike_no[outlier_mask], isi[outlier_mask],
                               s=15, marker='o', color='white', zorder=2, edgecolor=colors[i])

        ax_isi.set_xlabel('Spike Number (#)')
        ax_isi.set_ylabel('Inter-Spike Interval (ms)')

        # subplot 3: adapt ratio for max firing sweep
        if max_freq_isi_trace:
            sn       = np.array(list(max_freq_isi_trace.keys()))
            ri_clean = np.array(list(max_freq_isi_trace.values()))
            ri_raw   = np.array(list(max_freq_isi_trace_raw.values()))
            outlier_mask_plot = np.isnan(ri_clean)
            ax_max_freq_isi.plot(sn, ri_raw, c=colors[i], marker='o', zorder=1)
            if np.any(outlier_mask_plot):
                ax_max_freq_isi.scatter(sn[outlier_mask_plot], ri_raw[outlier_mask_plot],
                                        s=10, marker='o', color='white', zorder=2, edgecolor=colors[i],)
            ax_max_freq_isi.axhline(1, color='k', linestyle=':')
            ax_max_freq_isi.set_ylim(bottom=0)
            ax_max_freq_isi.set_xlabel('Spike Number (#)')
            ax_max_freq_isi.set_ylabel('Adapt Ratio')

        plt.tight_layout()
        os.makedirs('Saved_Figs/Firing_Gain/', exist_ok=True)
        fig.savefig('Saved_Figs/Firing_Gain/Adaption_' + plot_name + '.' + figopt['type'], dpi=figopt['dpi'])

    return {
        'ADR':                  ADR,
        'ADR_sweep':            ADR_sweep,
        'ADR_current':          stim_currents[ADR_sweep] if ADR_sweep is not None else np.nan,
        'max_adapt':            max_adapt,
        'isi_ratios':           isi_ratios,
        'max_freq_isi_trace':   max_freq_isi_trace,
        'inact_current':        inact_current,
        'inact_pulse_num':      inact_pulse_num,
    }

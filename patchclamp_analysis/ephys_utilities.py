import numpy as np
import matplotlib.pyplot as plt



def find_spike_in_trace(trace,rate,spike_args,refract=0.005,is_stim = None ,mode='count',sanity_check=True,to_plot=0):
    '''
    Takes in a voltage trace from current clamp mode and uses derivative (dVds) to find action potentials.
    Returns the dVds trace, boolean array indicating if dVds>threshold, inicies where dV crossed threshold,
    and the mean firing rate given # spikes in trace of given length. Optional ways to count are:
    isi (1/interspike interval) or count (spike count per second). Default is count.


    Spike args is a dict containing thesholds for spike dtection including dvds rising threshold and falling thresholds.
    spike_thresh: the minimum dVdT for determining AP occurnance. Also used as the 'start' of AP
    high_dv_thresh: An AP must also cross this higher threshold to be considered. eg: after crossing 20mv/ms the AP must continue to rise to >40mV/ms,
    this is used to filter out incomplete APs that may be undergoing depolarization/inactivation block.
    low_dv_thresh & window_ms: strong stimuli may triger the spike_thresh purely based on charging the membrane.
    Thus to be considered a true AP, there must also be a falling phase to the wave from that occurs shortly after, eg.  min(dVdS)<-20mV/ms within 2ms of waveform start
    '''


    high_dv_thresh = spike_args['high_dv_thresh']
    low_dv_thresh = spike_args['low_dv_thresh']
    spike_thresh = spike_args['spike_thresh']
    window_ms = spike_args['window_ms']

    if any(is_stim == None):
        is_stim = [True for i in trace]
    dVds = np.diff(trace, prepend=trace[0])*rate/1000
    over_thresh = dVds>spike_thresh
    over_thresh[np.logical_not(is_stim)] = False
    refract_window = int(np.round((refract*rate)))
    inds = [t for t in np.arange(refract_window,len(over_thresh)) if all([over_thresh[t], all(over_thresh[t-refract_window:t]==False)])]
    if sanity_check:
        old_inds = inds
        inds = []
        for i in old_inds:
            samp_window = window_ms/1000 * rate
            ind_range = np.arange(i-samp_window,i+samp_window).astype(int)
            ind_range = ind_range[ind_range<len(dVds)]
            nearby_dVds = dVds[ind_range]
            if False: print(i,'max', np.max(nearby_dVds))
            if False: print(i,'min', np.min(nearby_dVds))
            if np.max(nearby_dVds)>high_dv_thresh and np.min(nearby_dVds) < low_dv_thresh:
                inds.append(i)
                if False: print(inds)
    if to_plot>2:
        fig1, axs1 = plt.subplots(1,figsize = [9,2])
        axs1.plot(np.arange(len(dVds))/rate,dVds,zorder=1)
        axs1.scatter((np.arange(len(dVds))/rate)[inds],dVds[inds],color='red',zorder=2)
        plt.show()
    if len(inds)<1:
        mean_spike_rate = 0
    else:
        if mode=='isi':
            mean_spike_rate = np.mean(rate/np.diff(inds))
        elif mode=='count':
            mean_spike_rate = len(inds)/(np.sum(is_stim)/rate)
        else:
            print('invalid mode. using default (count)')
    return dVds, over_thresh, inds, mean_spike_rate

def protocol_baseline_and_stim(abf):
    'Return two boolean arrays, distiguishing holding I/V and electrical stimuli'
    # use command signal variance to determine stimulus periods
    commands = []
    for s in abf.sweepList:
        abf.setSweep(sweepNumber=s)
        commands.append(abf.sweepC)
    commands = np.stack(commands)

    std = np.std(commands, axis=0)
    is_base = std==0
    is_stim = np.logical_not(is_base)
    return is_base, is_stim

def command_match(abf,error_thresh = .05):
    abf.setSweep(0,channel=0)
    desired_command = abf.sweepC
    desired_command -= desired_command[0]
    mean_d = np.mean(desired_command)
    var_d = np.mean(desired_command)

    abf.setSweep(0,channel=1)
    observed_command = abf.sweepY
    observed_command -= observed_command[0]
    mean_o = np.mean(observed_command)
    var_o = np.mean(observed_command)

    if abs((var_d-var_o)/var_o) > error_thresh:
        print('Not Correct Command')
        fig,ax = plt.subplots(2,1)
        ax[0].plot(observed_command)
        ax[1].plot(desired_command)
        fig.suptitle(abf.abfFilePath)
        return False

    if abs((mean_d-mean_o)/mean_o) > error_thresh:
        print('Not Correct Command')
        fig,ax = plt.subplots(2,1)
        ax[0].plot(observed_command)
        ax[1].plot(desired_command)
        fig.suptitle(abf.abfFilePath)
        return False

    return True

def mean_inst_firing_rate(spike_times):
    mean_inst_rates = [1/np.mean(np.diff(s)) if len(s) > 1 else np.nan for s in spike_times]
    mean_inst_rates = [np.round(r,1) for r in mean_inst_rates]
    mean_inst_rates = [0 if np.isnan(r) else r for r in mean_inst_rates ]
    return mean_inst_rates

def initial_inst_firing_rate(sweepX,inds_list,num_spikes=2,to_plot=False):
    num_spikes = np.max([num_spikes,2])
    rate_list = []
    for inds in inds_list:
        times = [sweepX[i] for i in inds]
        if len(times>=num_spikes):
            isi=np.mean(np.diff(times[:num_spikes+1]))
            rate = 1/isi
        else: rate = 0

        rate_list.append(rate)
    rate_list = np.array(rate_list)
    return rate_list

def movmean(x, n=3):
    """Simple moving-average function to smooth the signal."""
    return np.convolve(x, np.ones(n)/n, mode='same')

def rms_noise(x):
    return np.sqrt(    np.sum((x-x.mean())**2)/len(x)    )  

def mono_exp(time, peak, tau, ss):
    'A single exponential decay function'
    return (peak * np.exp(-time / (tau)) + ss)

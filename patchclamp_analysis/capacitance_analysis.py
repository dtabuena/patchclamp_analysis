import numpy as np
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os

from patchclamp_analysis.ephys_utilities import movmean, mono_exp

def membrane_analyzer(abf, to_plot=False, verbose=False,report_params=None):

    '''Analyze ABF of square VC pulses to determine membrane properties'''

    results= {} # init results dict

    mem_params_df = fit_Icapacitave_mean_current(abf,to_plot=to_plot,verbose=verbose)
    pclamp_mem_params_df = pclamp_mem_test(abf,to_plot = to_plot, verbose = verbose)

    if pclamp_mem_params_df.empty:
        if verbose:
            print(f"[WARNING] pclamp_mem_test() returned an empty DataFrame for {abf.abfID}")
        return results

    mem_params_df = mem_params_df.join(pclamp_mem_params_df,how='outer')

    if report_params is None:
        report_params = ['Ra', 'Rm', 'Cm', 'tau',	'Cmq',	'Cmf',	'Cmqf', 'Cm_pc']

    for pulse_dur in mem_params_df.index:
        for parameter in mem_params_df.columns:
            if parameter in report_params:
                results[parameter+'_'+str(pulse_dur)] = mem_params_df.at[pulse_dur,parameter]
    return results


def fit_Icapacitave_mean_current(abf, to_plot=False, verbose=False):
    'Takes in an abf file and finds all pulses. Pulses with matching duration are averaged together.'
    'For each pulse duration the mean pulse is fit using the methods described at https://swharden.com/blog/2020-10-11-model-neuron-ltspice/ '
    'For each pulse length returns, Ra, Rm, and three Cm measures (Cmf, Cmq, Cmqf).'
    'Respectively these are capacitance determined by: fitting tau and computing,'
    'calculating the area under the capcitave transient, and calculating the area'
    'under the fit line.'

    command = abf.sweepC
    # trace,time,command,rate,

    base_v = command[0]
    step_v = np.median( command[np.logical_not(command==base_v)])
    is_base = command==base_v
    is_step = command==step_v

    delta_V = abs(step_v-base_v)

    step_start = np.logical_and(is_base[:-1], is_step[1:])
    step_stop = np.logical_and(is_step[:-1], is_base[1:])

    starts = np.where(step_start)[0]
    stops = np.where(step_stop)[0]

    assert len(starts)==len(stops), 'unable to match pulse starts and stops'
    assert any(( starts-stops)<0), 'unable to match pulse starts and stops'
    assert len(starts)>0, 'no pulse found'
    # parse_pulses

    params = []
    p_len_list = []
    Icap_list = []
    step_time_list = []
    # for s in abf.sweepList:
    for s in abf.sweepList:
        abf.setSweep(s)
        trace = abf.sweepY
        sweep_time = abf.sweepX
        if (base_v>step_v):
            trace = -trace
        for p in np.arange(len(starts)):
            pulse_start = starts[p]
            pulse_stop = stops[p]
            pulse_len = stops[p] - starts[p]
            p_len_list.append(pulse_len)
            pulse_index = np.arange(int(pulse_start-pulse_len*0.05),pulse_stop)

            step_times = sweep_time[pulse_index]
            step_times = step_times-sweep_time[starts[p]]
            step_time_list.append(step_times)

            Icap_transient = trace[pulse_index]
            Icap_list.append(Icap_transient)

    p_len_list = np.array(p_len_list)/abf.sampleRate*1000
    pulse_set = np.array(sorted(set(p_len_list)))
    mem_params_df = pd.DataFrame(None,index=pulse_set,columns=['>90%','Ib','Iss','Ip','Ra','Rm','tau','Cmq','Cmf','Cmqf'])

    if to_plot:
        fig, axs = plt.subplots(1,len(pulse_set),figsize=[4, 1.5])
        # fig.suptitle(abf.abfFilePath)
        if verbose: print(abf.abfFilePath)
        if str(type(axs)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
            axs = [axs]


    for p in pulse_set:
        # pulse_dur =p/abf.sampleRate*1000
        matching_traces = [Icap_list[n] for n in np.arange(len(p_len_list)) if p_len_list[n]==p ]
        matching_traces = np.stack(matching_traces)

        # mean_trace = np.mean(matching_traces,axis=0)
        mean_trace = np.median(matching_traces,axis=0)
        mean_time = np.mean(np.stack([step_time_list[n] for n in np.arange(len(p_len_list)) if p_len_list[n]==p ]),axis=0)

        sweep_var = abs((matching_traces-mean_trace)/mean_trace)
        outlier_percent = round(np.mean(sweep_var>1.645)*100,3)
        # base_ind = np.arange(len(mean_time))
        base_t = np.mean(mean_time[mean_time<0])
        base_I = np.mean(mean_trace[mean_time<0])

        steady_state_t = np.mean(mean_time[mean_time>mean_time[-1]*0.95])
        steady_state_I = np.mean(mean_trace[mean_time>mean_time[-1]*0.95])


        peak_I = np.max(mean_trace)
        peak_t = mean_time[mean_trace==peak_I]
        if peak_t.shape[0]>1: peak_t = min(peak_t)
        Icap_curve = (mean_trace[mean_time>=peak_t])
        Icap_curve_t = mean_time[mean_time>=peak_t]


        rel_dif_Icap = movmean(np.diff(Icap_curve,append=Icap_curve[-1]),10)/peak_I
        excess_plat_t = Icap_curve_t[rel_dif_Icap>=0]
        if len(excess_plat_t)>0:
            excess_plat_start = np.min(excess_plat_t)*10
            if excess_plat_start >0.005:
                Icap_curve = Icap_curve[Icap_curve_t<excess_plat_start]
                Icap_curve_t = Icap_curve_t[Icap_curve_t<excess_plat_start]
                steady_state_t = np.mean(Icap_curve_t[Icap_curve_t>Icap_curve_t[-1]*0.95])
                steady_state_I = np.mean(Icap_curve[Icap_curve_t>Icap_curve_t[-1]*0.95])



        delta_I_steady = steady_state_I - base_I
        delta_I_peak = peak_I - steady_state_I
        Ra = (delta_V*1e-3)/(delta_I_peak*1e-12) *1e-6 #(O/MO)
        Rm = ((delta_V*1e-3) - Ra*1e6 * delta_I_steady*1e-12) / (delta_I_steady*1e-12) *1e-6 #(O/MO)
        Q = np.sum(Icap_curve-steady_state_I) * (1/abf.sampleRate)
        Cmq = Q / delta_V*1000


        try:
            bounds=([peak_I*0.1,.0001,0], [peak_I*1.5,500, steady_state_I*3])
            p0 = (peak_I, 0.02 , steady_state_I) # start with values near those we expect
            fit_params, cv = sci.optimize.curve_fit(mono_exp, Icap_curve_t[int(0.0005*abf.sampleRate):], Icap_curve[int(0.0005*abf.sampleRate):], p0, bounds=bounds) #
            peak_hat, tau_hat, ss_hat = fit_params
            Icap_hat =  mono_exp(Icap_curve_t, peak_hat, tau_hat, ss_hat)
            perr = np.sqrt(np.diag(cv))
            Cmf = tau_hat / (1/(1/(Ra*1e6) + 1/(Rm*1e6)))
            Cmf = Cmf*1e12

        except:
            Cmf = np.nan
            Icap_hat = np.empty_like(Icap_curve_t)
            Icap_hat[:] =np.nan
            ss_hat = np.nan
            tau_hat = np.nan

        Cmqf = np.sum(Icap_hat-ss_hat) * (1/abf.sampleRate) / delta_V*1000

        param_list = [outlier_percent,base_I,steady_state_I,peak_I,Ra,Rm,tau_hat,Cmq,Cmf,Cmqf]
        for ci in range(len(mem_params_df.columns)):
            col = mem_params_df.columns[ci]
            mem_params_df.at[p,col] = param_list[ci]


        if to_plot:
            i = int(np.where(p==pulse_set)[0][0])
            mean_time_0 = -mean_time[0]
            axs[i].plot(mean_time_0+mean_time,matching_traces.T,color = (0.8,0.8,0.8))
            axs[i].plot(mean_time_0+mean_time,mean_trace,color='k')
            axs[i].plot(mean_time_0+Icap_curve_t[[0,-1]],base_I*np.array([1,1]),color='r',linestyle = 'dotted')
            axs[i].scatter(mean_time_0+peak_t,peak_I,color='r',zorder=5)
            axs[i].plot(mean_time_0+Icap_curve_t[[0,-1]],steady_state_I*np.array([1,1]),color='r',linestyle = 'dotted')
            axs[i].plot(mean_time_0+Icap_curve_t[int(0.001*abf.sampleRate):],Icap_curve[int(0.001*abf.sampleRate):],color='m')
            axs[i].plot(mean_time_0+Icap_curve_t, Icap_hat,'c',linestyle = 'dashed')
            axs[i].set_xlim([0,mean_time_0+Icap_curve_t[-1]*1.2]) #(mean_time_0+peak_t)*0.7
            axs[i].text( mean_time_0+peak_t,peak_I ,'     Cmq='+str(np.round(mem_params_df.at[p,'Cmq'],1)) + 'pF')
            # axs[i].set_title(str(p)+'ms')

    if verbose: display(mem_params_df)
    if to_plot:
        plt.show()
        plt.tight_layout()
        fig.subplots_adjust(top=0.8)
        try:    os.makedirs('Saved_Figs/Membrane_Fit/')
        except:     None
        fig.savefig( 'Saved_Figs/Membrane_Fit/Membrane_Fit'+'_' + abf.abfID +'.png')
    return mem_params_df



def pclamp_mem_test(abf,to_plot = False, verbose =False,dpi=300):
    # load file if name given instead of true abf
    command = abf.sweepC*1e-3
    trace = abf.sweepY*1e-12
    sweep_time = abf.sweepX

    # make all pos
    if np.mean(command)<0:
        command = -command
        trace = -trace



    # Find step and recovery
    base_v = command[0]
    # plt.plot(sweep_time,command)
    step_v = np.median( command[np.logical_not(command==base_v)])
    dvdt = np.diff(command,prepend=command[0])
    # plt.plot(sweep_time,dvdt)
    up_step = np.where(dvdt==np.max(dvdt))[0]
    # print('up_step',up_step)
    down_step = np.where(dvdt==np.min(dvdt))[0]
    # print('down_step',down_step)
    updn_ticks = down_step - up_step
    # print('updn_ticks',updn_ticks)

    pulse_dur_set = np.sort(list(set(updn_ticks)))
    # print('pulse_dur_set',pulse_dur_set)


    mem_params_df = pd.DataFrame(None,index=pulse_dur_set/abf.sampleRate*1000,columns=['Tau_pc','Rm_pc','Ra_pc','Cm_pc'])

    if to_plot:
        fig, axs_pc = plt.subplots(1,len(pulse_dur_set),figsize=[4, 1.5])

    'For Each Pulse Duration Length'
    for p in pulse_dur_set:
        pi = np.where(p == pulse_dur_set)[0][0]
        # print('pi',pi)
        'Average up the pulses'
        matching_starts = [up_step[i] for i in range(len(up_step)) if updn_ticks[i]==p ]
        tick_range = np.arange(p*2)
        pulse_indicies_mat = np.add.outer(matching_starts,tick_range)
        pulse_trace_set = trace[pulse_indicies_mat]
        mean_pulse_trace = np.mean(pulse_trace_set,axis = 0)
        mean_pulse_trace = np.median(pulse_trace_set,axis = 0)
        mean_pulse_command = np.mean(command[pulse_indicies_mat],axis = 0)

            # plt.show()
        'Get pclamp fitting variables'
        'Get Is and Vs'
        V1 = np.max(mean_pulse_command)
        V2 = np.min(mean_pulse_command)
        delta_V = V1-V2
        I1_index = range(int(p*0.8),p)
        I1 = np.mean(mean_pulse_trace[I1_index])

        I2_index = I1_index + p
        I2 = np.mean(mean_pulse_trace[I2_index])
        delta_I = I1-I2

        if to_plot:

            axs_pc[pi].plot(pulse_trace_set.transpose()*1E12,color='grey')
            axs_pc[pi].plot(mean_pulse_trace*1E12,color='k')

            axs_pc[pi].plot(I1_index,I1*np.ones_like(I1_index)*1E12,color='magenta') # ,linewidth=3
            axs_pc[pi].plot(I2_index,I2*np.ones_like(I1_index)*1E12,color='magenta') # ,linewidth=3


        'Fitting Tau'
        def linearized_exp_decay(time,tau,beta):
            'Linear form of ln(y) for exponential decay'
            return -(1/tau)*(time+ beta)

        peak_window = np.arange(0,int(.004*abf.sampleRate))
        peak_I = np.max(mean_pulse_trace[peak_window])
        ind_of_peak = np.where(mean_pulse_trace==peak_I)[0][0]


        single_pulse_trace = mean_pulse_trace[np.arange(ind_of_peak,p)]
        single_pulse_time = np.arange(ind_of_peak,p)/abf.sampleRate
        fraction_to_fit = [0.05, 0.90]

        'LinearFraction'
        I_max = np.max(single_pulse_trace)
        I_min = np.min(single_pulse_trace)
        delta = I_max - I_min
        lower = I_min + delta*fraction_to_fit[0]
        upper = I_min + delta*fraction_to_fit[1]
        trimmed_fit_range = np.logical_and(single_pulse_trace>lower, single_pulse_trace<upper)
        trace_to_fit = single_pulse_trace[trimmed_fit_range]
        time_to_fit = single_pulse_time[trimmed_fit_range]
        if to_plot:
            axs_pc[pi].plot(time_to_fit*abf.sampleRate,trace_to_fit*1E12,color='green') # ,linewidth=3

        'linear fit of ln_trace'
        '(with baseline shift to avoid log(x<0)'
        trace_to_fit = trace_to_fit
        shift = abs(np.min(trace_to_fit))
        ln_trace = np.log(trace_to_fit+shift*2)


        if len(time_to_fit)>1:
            [tau_hat, beta_hat], cv = sci.optimize.curve_fit(linearized_exp_decay, time_to_fit, ln_trace) #
        else:
            tau_hat = np.nan
            beta_hat = np.nan

        I_hat = linearized_exp_decay(time_to_fit,tau_hat, beta_hat)
        if to_plot:
            axs_pc[pi].plot(time_to_fit*abf.sampleRate,(np.exp(I_hat)-shift*2)*1E12,color='turquoise') # ,linewidth=3

        'Calculate Pclamp Values'
        delta_I = I1-I2
        Q2 = delta_I * tau_hat # This doesnt make sense to me
        I_ss = np.mean([I1,I2])

        Q1_ind = np.where(single_pulse_trace>I1)[0]
        Q1 = np.sum(single_pulse_trace[Q1_ind] - I1) / abf.sampleRate


        'Calculate Pclamp Values'
        Qt = Q1 + Q2
        Cm = Qt / delta_V
        Rt = delta_V/delta_I

        'Plot Area'
        patch_points = np.ones([len(Q1_ind)*2,2])
        if to_plot:
            patch_points[:,0] = np.concatenate((Q1_ind,np.flip(Q1_ind)))
            patch_points[:,1] = np.concatenate((single_pulse_trace[Q1_ind],I1*np.ones_like(Q1_ind)))*1E12
            poly = mpl.patches.Polygon(patch_points, color='orange')
            axs_pc[pi].add_patch(poly)
            axs_pc[pi].set_title(f'Cm_pc = {Cm*1e12:.1f}pF')


        'Iterateively Solve Ra using Newton-Raphson Method'
        Ra_guess = 20*1e6
        delta_guess = 1e10
        tol = 1
        while delta_guess>tol:
            f_of_guess = Ra_guess**2 - Ra_guess*Rt + Rt*(tau_hat/Cm)
            f_prime_of_guess = Ra_guess/2 - Rt
            Ra_guess_new = Ra_guess - (f_of_guess/f_prime_of_guess)
            delta_guess = Ra_guess_new - Ra_guess
            Ra_guess = Ra_guess_new
        Ra = Ra_guess
        Rm = Rt - Ra


        if verbose:
            print('tau_hat',tau_hat*1000,'ms')
            print('Cm',Cm*1e12,'pF')
            print('Rt',Rt*1e-6,'MO')
            print('Ra',Ra*1e-6,'MO')
            print('Rm',Rm*1e-6,'MO')


        'Return a dataframe of parameters'
        p_ms = int(p/abf.sampleRate*1000)
        mem_params_df.at[p_ms,'Tau_pc'] = tau_hat
        mem_params_df.at[p_ms,'Rm_pc'] = Rm*1e-6
        mem_params_df.at[p_ms,'Ra_pc'] = Ra*1e-6
        mem_params_df.at[p_ms,'Cm_pc'] = Cm*1e12

        if to_plot:
            try:    os.makedirs('Saved_Figs/Membrane_Fit_PC/')
            except:     None
            fig.savefig( 'Saved_Figs/Membrane_Fit_PC/Membrane_Fit_PC'+'_' + abf.abfID +'.png',dpi=dpi)
            # plt.show()
    return mem_params_df

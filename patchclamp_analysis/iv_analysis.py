import numpy as np
import matplotlib.pyplot as plt
import os

from patchclamp_analysis.ephys_utilities import protocol_baseline_and_stim



def IV_analyzer_v4(abf, Na_window=[.2, 3], K_window=[40,50], to_plot=True, leak_threshold=-200,
                   use_PN=True, PN_voltages=[-60,-80], use_baseline_subtraction=True, to_plot_PN=False,
                   figopt={'type':'jpg'}):

    """
    1) Collects valid sweeps (filters blown seals and high leak).
    2) Optionally performs P/N subtraction using n_pulses most negative sweeps.
    3) Optionally performs baseline subtraction (after P/N if enabled).
    4) Measures Na-peak in Na_window.
    5) Measures K-mean in K_window.
    6) Plots three subplots (Na-window, K full trace, IV).

    Parameters:
        leak_threshold : float, default=-200
            Minimum acceptable baseline current in pA.
        use_PN : bool, default=True
            Whether to perform P/N subtraction for leak correction.
        n_pulses : int, default=3
            Number of most negative pulses to use for P/N subtraction template.
        use_baseline_subtraction : bool, default=True
            Whether to baseline-subtract sweeps (subtracts first point) after P/N correction.
        to_plot_PN : bool, default=False
            Whether to show P/N diagnostic plots.

    Returns:
        results = {
            f"Na_{Na_window}": { voltage: peak, ... },
            f"K_{K_window}":   { voltage: mean, ... }
        }
    """

    is_base, is_stim = protocol_baseline_and_stim(abf)
    t0_relative = abf.sweepX[np.where(is_stim)[0][0]]*1000
    na_start, na_stop = [x + t0_relative for x in Na_window]
    k_start, k_stop   = [x + t0_relative for x in K_window]

    # Convert ms to seconds for the analysis windows
    na_start_s = na_start / 1000
    na_stop_s = na_stop / 1000
    k_start_s = k_start / 1000
    k_stop_s = k_stop / 1000

    # Calculate durations
    na_delta_s = na_stop_s - na_start_s
    k_delta_s = k_stop_s - k_start_s

    # Collect valid sweeps
    valid_sweeps = collect_valid_sweeps(abf, is_base, leak_threshold)
    if len(valid_sweeps)==0:
        results = {'message':'No valid Sweeps'}
        return results
    # Optionally perform P/N subtraction
    if use_PN:
        sweeps, PN_info = PN_subtraction(valid_sweeps, PN_voltages, is_base, to_plot_PN)
    else:
        sweeps = valid_sweeps

    # Optionally perform baseline subtraction (after P/N)
    if use_baseline_subtraction:
        for sweep in sweeps:
            sweep['current'] = sweep['current'] - sweep['current'][0]

    # Prepare data containers
    na_peaks = []
    na_voltages = []
    k_means = []
    k_voltages = []

    # ---- Create figure with 3 subplots: (Na-window, K full, IV) ----
    if to_plot:
        fig, (ax_na, ax_k, ax_iv) = plt.subplots(1, 3, figsize=(4,1.5))
    else:
        ax_na = ax_k = ax_iv = None

    # ----------------- NA MEASUREMENT & PLOT (SUBPLOT 1) ------------------ #
    margin_fraction = 0.2
    na_plot_start_s = na_start_s - margin_fraction * na_delta_s
    na_plot_stop_s  = na_stop_s  + margin_fraction * na_delta_s

    for sweep_data in sweeps:
        time = sweep_data['time']
        current = sweep_data['current']
        command_v = sweep_data['voltage']

        # Identify the analysis region for Na
        na_mask = (time >= na_start_s) & (time <= na_stop_s)
        if not np.any(na_mask):
            continue

        I_analysis = current[na_mask]
        t_analysis = time[na_mask]

        # Baseline is median within that window (for peak detection)
        baseline = np.median(I_analysis)
        delta_i = I_analysis - baseline
        peak_idx = np.argmax(np.abs(delta_i))
        # peak = baseline + largest dev
        peak_value = baseline + delta_i[peak_idx]
        peak_time = t_analysis[peak_idx]

        na_peaks.append(peak_value)
        na_voltages.append(command_v)

        # -------------- PLOTTING --------------
        if ax_na is not None:
            na_plot_mask = (time >= na_plot_start_s) & (time <= na_plot_stop_s)
            ax_na.plot(time[na_plot_mask]*1000, current[na_plot_mask], 'k')
            ax_na.scatter(peak_time*1000, peak_value, color='m',  zorder=5)

    # Set up the NA subplot
    if ax_na is not None:
        ax_na.set_xlim([na_plot_start_s*1000, na_plot_stop_s*1000])
        ax_na.set_xlabel("Time (ms)")
        ax_na.set_ylabel("I (pA)")
        ax_na.set_title(f"Na Window\n({Na_window[0]}-{Na_window[1]} ms)")

    # ----------------- K MEASUREMENT & PLOT (SUBPLOT 2) ------------------ #
    for sweep_data in sweeps:
        time = sweep_data['time']
        current = sweep_data['current']
        command_v = sweep_data['voltage']

        # measurement mask for K
        k_mask = (time >= k_start_s) & (time <= k_stop_s)
        if not np.any(k_mask):
            continue

        I_analysis = current[k_mask]
        I_mean = np.median(I_analysis)

        k_means.append(I_mean)
        k_voltages.append(command_v)

        # -------------- PLOTTING --------------
        if ax_k is not None:
            ax_k.plot(time*1000, current, 'k')
            k_mid_t = (k_start_s + k_stop_s)/2
            ax_k.scatter(k_mid_t*1000, I_mean, color='c',  zorder=5)

    # Set up the K subplot
    if ax_k is not None:
        ax_k.set_xlim([K_window[0]-k_delta_s*500, K_window[1]+k_delta_s*500])
        ax_k.set_xlabel("Time (ms)")
        ax_k.set_ylabel("I (pA)")
        ax_k.set_title(f"K Window\n({K_window[0]}-{K_window[1]} ms)")

    # --------------------- BUILD RESULTS DICT --------------------- #
    results = {}
    # Na
    key_na = f"IV_Na_{Na_window[0]:g}_{Na_window[1]:g}"
    results[key_na] = {}
    for v, peak in zip(na_voltages, na_peaks):
        results[key_na][v] = peak

    # K
    key_k = f"IV_K_{K_window[0]:g}_{K_window[1]:g}"
    results[key_k] = {}
    for v, mean_i in zip(k_voltages, k_means):
        results[key_k][v] = mean_i

    # --------------------- I–V PLOT (SUBPLOT 3) --------------------- #
    if to_plot and ax_iv is not None:
        ax_iv.set_title("I–V")
        ax_iv.set_xlabel("Voltage (mV)")
        ax_iv.set_ylabel("I (pA)")

        # Sort by voltage for a cleaner line
        unique_na_voltages = sorted(set(na_voltages))
        unique_k_voltages  = sorted(set(k_voltages))

        # Extract in sorted order
        sorted_na_peaks = [results[key_na][v] for v in unique_na_voltages]
        sorted_k_means  = [results[key_k][v]   for v in unique_k_voltages]

        # Plot them
        ax_iv.plot(unique_na_voltages, sorted_na_peaks, '-o', color='m')
        ax_iv.plot(unique_k_voltages,  sorted_k_means,  '-o', color='c')

        # Add lines at x=0 and y=0
        ax_iv.axhline(0, color='k', linewidth=.25)
        ax_iv.axvline(0, color='k', linewidth=.25)

        # Add lines at x=0 and y=0
        if np.min(sorted_na_peaks) > -300:
            ax_iv.set_ylim(ymin=-300)
        if np.max(sorted_k_means) < 300:
            ax_iv.set_ylim(ymax=300)


    # ----------- Adjust layout, Save, and Return ----------- #
    if to_plot:
        plt.tight_layout()
        os.makedirs('Saved_Figs/IV_Curves/', exist_ok=True)
        save_name = f"Saved_Figs/IV_Curves/IV_Curves_{abf.abfID}.{figopt['type']}"
        plt.savefig(save_name)
        plt.show()

    return results


def collect_valid_sweeps(abf, is_base, leak_threshold):
    """
    Collects all valid sweeps, filtering out blown seals and high leak.

    Parameters:
        abf: pyabf object
        is_base: baseline mask from protocol_baseline_and_stim
        leak_threshold: float, minimum acceptable baseline current in pA

    Returns:
        valid_sweeps: list of dicts with keys: 'sweep', 'time', 'current',
                      'voltage', 'v_hold'
    """
    valid_sweeps = []

    for s in abf.sweepList:
        abf.setSweep(s, channel=0)

        # Skip blown seals
        if abs(abf.sweepY[-1]) > 1000:
            continue

        # Check baseline leak current
        baseline_current = np.median(abf.sweepY[is_base])
        if baseline_current < leak_threshold:
            continue

        # Get command voltage
        abf.setSweep(s, channel=1)
        v_hold = np.median(abf.sweepY)
        abf.setSweep(s, channel=0)
        cmd_trace = abf.sweepC + v_hold
        command_v = np.median(cmd_trace[~is_base])
        command_v = round(command_v / 10) * 10

        valid_sweeps.append({
            'sweep': s,
            'time': abf.sweepX.copy(),
            'current': abf.sweepY.copy(),
            'voltage': command_v,
            'v_hold': v_hold
        })

    return valid_sweeps



def PN_subtraction(valid_sweeps, PN_voltages, is_base, to_plot):
    """
    Performs P/N subtraction using explicitly specified voltages as leak template.
    Normalizes each template sweep by its step amplitude before averaging.

    Parameters:
        valid_sweeps: list of sweep dicts from collect_valid_sweeps
        PN_voltages: list of voltages (mV) to use for template, e.g. [-80, -60]
        is_base: baseline mask from protocol_baseline_and_stim
        to_plot: bool, whether to create diagnostic plots

    Returns:
        corrected_sweeps: list of sweep dicts with 'current' replaced by corrected values
        PN_info: dict with 'voltages', 'template', and 'holding_potential'
    """

    v_hold = valid_sweeps[0]['v_hold']

    # Find sweeps matching requested voltages
    PN_sweeps = [s for s in valid_sweeps if s['voltage'] in PN_voltages]

    if len(PN_sweeps) == 0:
        PN_info = {'info': 'PN sweeps not detected'}
        return valid_sweeps, PN_info


    # Find separate baseline regions
    base_indices = np.where(is_base)[0]
    breaks = np.where(np.diff(base_indices) > 1)[0]

    if len(breaks) > 0:
        first_region = base_indices[:breaks[0]+1]
        last_region = base_indices[breaks[-1]+1:]
    else:
        first_region = base_indices
        last_region = base_indices

    n_first = max(1, int(len(first_region) * 0.1))
    n_last = max(1, int(len(last_region) * 0.1))
    first_base_indices = first_region[:n_first]
    last_base_indices = last_region[-n_last:]

    # Normalize each template sweep by its step amplitude, then average
    normalized_templates = []
    for s in PN_sweeps:
        current = s['current'].copy()

        # Baseline subtract
        baseline = (np.mean(current[first_base_indices]) +
                    np.mean(current[last_base_indices])) / 2
        current = current - baseline

        # Normalize by step amplitude (voltage - holding)
        step_amplitude = s['voltage'] - v_hold
        current = current / step_amplitude

        normalized_templates.append(current)

    # Template is now in units of pA/mV
    leak_template = np.median(normalized_templates, axis=0)

    # Apply P/N subtraction to all sweeps
    corrected_sweeps = []

    for sweep_info in valid_sweeps:
        V_test = sweep_info['voltage']
        current = sweep_info['current']

        # Scale by step amplitude directly (template already normalized)
        step_amplitude = V_test - v_hold
        corrected_current = current - (leak_template * step_amplitude)

        corrected_sweep = sweep_info.copy()
        corrected_sweep['current'] = corrected_current
        corrected_sweeps.append(corrected_sweep)

    PN_info = {
        'voltages': PN_voltages,
        'template': leak_template,
        'holding_potential': v_hold,
    }

    if to_plot:
        fig, ax = plt.subplots(3, 1, figsize=(3, 3))

        time = valid_sweeps[0]['time'] * 1000

        for sweep_info in valid_sweeps:
            ax[0].plot(time, sweep_info['current'], 'k')
        ax[0].set_ylabel("I (pA)")
        ax[0].set_title("Raw Traces")

        # Show the raw template traces in gray
        for s in PN_sweeps:
            ax[1].plot(time, s['current'], color='gray')
        # Show averaged template scaled to +10mV in red
        correction_10mV = leak_template * 10
        ax[1].plot(time, correction_10mV, 'r')
        ax[1].axvspan(time[first_base_indices[0]], time[first_base_indices[-1]],
                      color='lightgray', zorder=0)
        ax[1].axvspan(time[last_base_indices[0]], time[last_base_indices[-1]],
                      color='lightgray', zorder=0)
        ax[1].set_ylabel("I (pA)")
        ax[1].set_title(f"Correction for +10mV (template: {PN_voltages})")

        for corrected_sweep in corrected_sweeps:
            ax[2].plot(time, corrected_sweep['current'], 'k')
        ax[2].set_ylabel("I (pA)")
        ax[2].set_xlabel("Time (ms)")
        ax[2].set_title("Corrected Sweeps")

        plt.tight_layout()
        plt.show()

    return corrected_sweeps, PN_info

def get_sub_files(rootdir):
    'Recursively search subfolders and return a list of all files'
    file_list =[]
    for rootdir, dirs, files in os.walk(rootdir):
            file_list.extend([os.path.join(rootdir,f) for f in files])
    return file_list


def qc_sweep(sweepX,sweepC,sweepY,command_offset,is_IC,is_VC,sampleRate,
             max_leak=100, 
             max_high_freq_noise = 10,
             max_low_freq_noise = 10,
             Vhold_range = 5, to_plot=False):
    'Recieves a sweep and and calculates leak, noise and holding potential'
    'returns a dict of calculated values and a dict of boolean indicating'
    'pass/fail, (True/False)'


    stim_buffer_time = 250 #ms
    filtered_command = movmean((sweepC==sweepC[0])*1, stim_buffer_time/1000*sampleRate)
    ss_no_stim_bool = filtered_command==1
    ss_no_stim_idx = np.arange(len(ss_no_stim_bool))[ss_no_stim_bool]
    no_stim_sig = sweepY[ss_no_stim_bool]
    no_stim_t = sweepX[ss_no_stim_idx]
    baseline = np.mean( no_stim_sig )

    QC_checks = {}
    QC_values = {}
    if is_IC:
        QC_values['V_hold'] = baseline
        QC_values['I_leak'] = command_offset
    if is_VC:
        QC_values['V_hold'] = command_offset
        QC_values['I_leak'] = baseline
    
    QC_checks['V_hold'] = abs(QC_values['V_hold'] - -70)< Vhold_range
    QC_checks['I_leak'] = QC_values['I_leak']len(LF_noise_idx): LF_noise_idx = np.random.choice(LF_noise_idx, size=int(0.1*sampleRate))
    LF_noise_signal = sweepY[ LF_noise_idx ] 
    LF_noise = rms_noise(LF_noise_signal)
    QC_checks['LF_noise'] = LF_noise

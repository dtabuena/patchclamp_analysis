def init_func_arg_dicts():
    func_dict = {}
    arg_dict = {}

    # Spike arg defentitions
    spike_args_gain={'spike_thresh':10, 'high_dv_thresh': 20,'low_dv_thresh': -5,'window_ms': 3}
    spike_args_rheo ={'spike_thresh':15, 'high_dv_thresh': 30,'low_dv_thresh': -15,'window_ms': 2}


    func_dict['VC - 3min GapFree']= rmp_analyzer
    arg_dict['VC - 3min GapFree'] = [True] # [to_plot?]
    func_dict['I0 - 3min GapFree']= rmp_analyzer
    arg_dict['I0 - 3min GapFree'] = [True] # [to_plot?]

    func_dict['IC - Rheobase']= rheobase_analyzer
    arg_dict['IC - Rheobase'] = [spike_args_rheo, True, False, False]  # [spike_args, to_plot, verbose, force_singlespike]

    func_dict['IC - Gain - D10pA']= gain_analyzer
    arg_dict['IC - Gain - D10pA']= [spike_args_gain, 1, 4, .7,[-60,-80]]  # [spike_args, to_plot [0:2], max_fit_steps, rel_slope_cut, Vh_hilo]

    func_dict['IC - Gain - D20pA']= gain_analyzer
    arg_dict['IC - Gain - D20pA']= [spike_args_gain, 1, 4, .7,[-60,-80]]  # [spike_args, to_plot [0:2], max_fit_steps, rel_slope_cut, Vh_hilo]

    func_dict['IC - Gain - D25pA']= gain_analyzer
    arg_dict['IC - Gain - D25pA']= [spike_args_gain, 1, 4, .7,[-60,-80]]  # [spike_args, to_plot [0:2], max_fit_steps, rel_slope_cut, Vh_hilo]

    func_dict['IC - Gain - D50pA']= func_dict['IC - Gain - D20pA']
    arg_dict['IC - Gain - D50pA']= arg_dict['IC - Gain - D20pA']

    func_dict['VC - MemTest-10ms-160ms']= membrane_analyzer
    arg_dict['VC - MemTest-10ms-160ms']= [True, False, ['Ra', 'Rm', 'Cm', 'tau',	'Cmq',	'Cmf',	'Cmqf', 'Cm_pc']]  # [to_plot, verbose]

    func_dict['IC - Latentcy 800pA-1s']= latencey_analyzer
    arg_dict['IC - Latentcy 800pA-1s']= [spike_args_gain, True]  # [spike_args, to_plot]

    func_dict['IC - R input']= input_resistance_analyzer
    arg_dict['IC - R input']= [[-30, 10] ,True]  # [dVm_limits, to_plot]

    func_dict['VC - Multi IV - 150ms'] = IV_analyzer_v2
    arg_dict['VC - Multi IV - 150ms']= [{'IV_Early':(4.5, 30),'IV_Steady_State':(70,80)} ,[False, True]]  # [measure_windows, to_plot]


    return func_dict, arg_dict

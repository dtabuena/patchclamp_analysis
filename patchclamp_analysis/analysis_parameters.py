from patchclamp_analysis.rmp_analyzer import rmp_analyzer
from patchclamp_analysis.rheobase_analyzer import rheobase_analyzer_V2
from patchclamp_analysis.gain_analyzer import gain_analyzer_v2
from patchclamp_analysis.membrane_analyzer import membrane_analyzer
from patchclamp_analysis.latencey_analyzer import latencey_analyzer
from patchclamp_analysis.input_resistance_analyzer import input_resistance_analyzer
from patchclamp_analysis.IV_analyzer import IV_analyzer_v4



def init_func_arg_dicts_h5():

    # Spike argument definitions
    spike_args_gain = {'spike_thresh': 10, 'high_dv_thresh': 20, 'low_dv_thresh': -5, 'window_ms': 3}
    spike_args_rheo = {'spike_thresh': 15, 'high_dv_thresh': 30, 'low_dv_thresh': -15, 'window_ms': 2}
    spike_args_late = {'spike_thresh': 10, 'high_dv_thresh': 30, 'low_dv_thresh': -5, 'window_ms': 3}

    analyzer_configs = {
        'VC - 3min GapFree': {
            'func': rmp_analyzer,
            'to_plot': True
        },

        'I0 - 3min GapFree': {
            'func': rmp_analyzer,
            'to_plot': True
        },

        'IC - Rheobase': {
            'func': rheobase_analyzer_V2,
            'spike_args': spike_args_rheo,
            'to_plot': True,
            'verbose': False,
            'single_spike': False
        },

        'IC - Gain - D10pA': {
            'func': gain_analyzer_v2,
            'spike_args': spike_args_gain,
            'to_plot': 1,
            'max_fit_steps': 4,
            'rel_slope_cut': 0.7,
            'Vh_hilo': [-60, -80]
        },

        'IC - Gain - D20pA': {
            'func': gain_analyzer_v2,
            'spike_args': spike_args_gain,
            'to_plot': 1,
            'max_fit_steps': 4,
            'rel_slope_cut': 0.7,
            'Vh_hilo': [-60, -80]
        },

        'IC - Gain - D25pA': {
            'func': gain_analyzer_v2,
            'spike_args': spike_args_gain,
            'to_plot': 1,
            'max_fit_steps': 4,
            'rel_slope_cut': 0.7,
            'Vh_hilo': [-60, -80]
        },

        'IC - Gain - D50pA': {
            'func': gain_analyzer_v2,
            'spike_args': spike_args_gain,
            'to_plot': 1,
            'max_fit_steps': 4,
            'rel_slope_cut': 0.7,
            'Vh_hilo': [-60, -80]
        },

        'VC - MemTest-10ms-160ms': {
            'func': membrane_analyzer,
            'to_plot': True,
            'verbose': False
        },

        'IC - Latentcy 800pA-1s': {
            'func': latencey_analyzer,
            'spike_args': spike_args_late,
            'to_plot': True
        },

        'IC - R input': {
            'func': input_resistance_analyzer,
            'dVm_limits': [-30, 10],
            'to_plot': True
        },

        'VC - Multi IV - 150ms': {
            'func': IV_analyzer_v4,
            'Na_window': [0.2, 10],
            'K_window': [130, 140],
            'to_plot': True,
            'leak_threshold': -400,
            'use_PN': True,
            'PN_voltages': [-60, -80, -90],
            'use_baseline_subtraction': True,
            'to_plot_PN': True
        },

        'VC - Multi IV - 500ms': {
            'func': IV_analyzer_v4,
            'Na_window': [0.2, 10],
            'K_window': [130, 140],
            'to_plot': True,
            'leak_threshold': -400,
            'use_PN': True,
            'PN_voltages': [-60, -80, -90],
            'use_baseline_subtraction': True,
            'to_plot_PN': True
        },

    }
    return analyzer_configs

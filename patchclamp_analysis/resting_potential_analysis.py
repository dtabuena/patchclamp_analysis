import numpy as np
import matplotlib.pyplot as plt
import os

def rmp_analyzer(abf,to_plot=False,figopt={'type':'jpg'}):
    results = {'Rmp_mV': np.nan}
    if 'mV' not in abf.adcUnits[0]:
        print('mV not in abf.adcUnits[0]')
        return synaptic_analysis(abf,to_plot)
    abf.setSweep(0,0)
    command = abf.sweepC[:abf.sampleRate*3]
    vm = abf.sweepY[:abf.sampleRate*3]
    abf.setSweep(0,1)
    command_ch2 = abf.sweepY[:abf.sampleRate*3]
    if to_plot:
        fig, ax = plt.subplots(1,2,figsize=(3,1.5))
        ax[0].hist(command,20)
        ax[0].hist(command_ch2,20)
        ax[0].set_xlabel('Command Current (pA)')
        ax[1].hist(vm,20)
        ax[1].set_xlabel('Membrane Pot (mv)')
        os.makedirs('Saved_Figs/Resting_VM/', exist_ok=True)
        fig.savefig( 'Saved_Figs/Resting_VM/Resting_VM_' + abf.abfID +'.'+figopt['type'])
        plt.tight_layout()
        plt.show()
    mean_rmp = np.mean(vm)
    mean_command = np.mean(command_ch2)
    if abs(mean_command)<15:
        results = {'Rmp_mV': mean_rmp,'command':mean_command}
    return results

def synaptic_analysis(abf,to_plot=False):
    # Not yet implemented
    results = {'Rmp_mV':np.nan,'synaptic': np.nan}
    return results
    

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:25:21 2024

@author: pmeshkiz
"""

"""
NOTE: The "display" option was deliberately considered only for single wave.
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt

def CWT(InputSignal, wavelet, scales, Freq_Sampling, display=False):
    
    if InputSignal.ndim == 1:
        Output, frequencies = pywt.cwt(InputSignal, scales, wavelet, sampling_period=1/Freq_Sampling)
        Output = np.abs(Output) # Extract the amplitude with respect to each frequency
        Output = np.flipud(Output) # changing the order of each column for better representation
        if display == True:
            frequencies = frequencies / 1e6 # increasing the scale of frequencies
            plt.figure(figsize=[25,15])
            plt.imshow(Output,extent=[0,len(InputSignal)*1e6/Freq_Sampling,frequencies.min(),frequencies.max()],
                       aspect='auto')
            plt.colorbar(label='Magnitudes')
            plt.ylabel('Frequencies (MHz)', fontsize=18)
            plt.xlabel('Time (microseconds)', fontsize=18)
            plt.title('The spectogram of the sample', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
            
    elif InputSignal.ndim > 1:
        Output = np.zeros([InputSignal.shape[0], InputSignal.shape[1],
                           len(scales), InputSignal.shape[2]])
        for i in range(InputSignal.shape[0]):
            for j in range(InputSignal.shape[1]):

                COEFFs, frequencies = pywt.cwt(InputSignal[i,j,:], scales, wavelet, sampling_period=1/Freq_Sampling)
                COEFFs = np.abs(COEFFs)
                COEFFs = np.flipud(COEFFs)
                # COEFFs = COEFFs.astype(np.float16)
                Output[i,j,:,:] = COEFFs
                
    return Output.astype(np.float32)
                    
            
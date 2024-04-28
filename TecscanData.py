# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:13:20 2024

@author: pmeshkiz
"""

import numpy as np
import struct


"""
Assigning working directory
"""
DataDirectory = r'C:\Users\pmeshkiz\OneDrive - Arizona State University (1)\ISNDE\Projects\AM parts(Two cubes)\Data\10' 
InfoFile = r'\Interface.dat.txt'
DataFile = r'\Interface.dat'

"""
Extracting inspection information
"""

def Extract_Inspection_Data(WorkDirectory, InfoFile):
    ScanPoints, IndexPoints, SPWaveform = 0, 0, 0
    with open(WorkDirectory + InfoFile,'r') as file:
        ScanPoints, IndexPoints, SPWaveform = 0, 0, 0
        Linesinfo = file.readlines()
        for rows in Linesinfo:
            infosinline = rows.split(':')
            if 'scan points' in infosinline[0].replace('\x00', '').lower():
                ScanPoints = int(infosinline[1].replace('\x00', '').strip())
            elif 'index points' in infosinline[0].replace('\x00', '').lower():
                IndexPoints = int(infosinline[1].replace('\x00', '').strip())
            elif 'per waveform' in infosinline[0].replace('\x00', '').lower():
                SPWaveform = int(infosinline[1].replace('\x00', '').strip())
            elif 'sampling frequency' in infosinline[0].replace('\x00', '').lower():
                FreqValue = infosinline[1].replace('\x00', '').split(' ')
                Sampling_Freq = float(FreqValue[1].strip())* 10**6
                
    return ScanPoints, IndexPoints, SPWaveform, Sampling_Freq

# ScanP, IndexP, SamPW = Extract_Inspection_Data(DataDirectory,InfoFile)


"""
Reconstructing Binary files
"""
def BinaryData_Reconstruction(WorkDirectory, DataFile, InfoFile):    
    with open(WorkDirectory + DataFile, 'rb') as Dfile:
        # Extracting Sacn points as ScanP, Index points as IndexP, and Sample per waveform as SamPW.
        ScanP, IndexP, SamPW, Sampling_Freq = Extract_Inspection_Data(WorkDirectory, InfoFile)
        
        D = np.zeros((ScanP, IndexP, SamPW), dtype=np.int16) # Initial Matrix
        B = Dfile.read() # read the binary file
        A = np.array(struct.unpack('<' + str(int(len(B) / 2)) + 'h', B), dtype=np.int16)
        for b in range(IndexP):
            for a in range(ScanP):
                start_index = SamPW * (a + ScanP * b)
                end_index = start_index + SamPW
                D[a, b, :] = A[start_index:end_index]           
    return D
    
# UTData = BinaryData_Reconstruction(DataDirectory, DataFile, InfoFile)

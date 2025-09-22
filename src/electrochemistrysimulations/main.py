'''
===================================================================================================
Copyright (C) 2023 Steven Linfield

This file is part of the oscilloscope-reader package. This package is free software: you can 
redistribute it and/or modify it under the terms of the GNU General Public License as published by 
the Free Software Foundation, either version 3 of the License, or (at your option) any later 
version. This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU General Public License for more details. You should have received a copy of the GNU General 
Public License along with oscilloscope-reader. If not, see https://www.gnu.org/licenses/
===================================================================================================

Package title:      oscilloscope-reader
Repository:         https://github.com/MoonshinetheP/oscilloscope-reader
Date of creation:   22/10/2022
Main author:        Steven Linfield (MoonshinetheP)
Collaborators:      None
Acknowledgements:   None

Filename:           reader.py

===================================================================================================

Description: 
This is the main file of the oscilloscope-reader package. All other files can be operated from 
here, although if it is necessary, many of the files can be operated individually from main.
 
===================================================================================================

How to use this file:
    
    1. Scroll down to the third point and choose the waveform you want to use, commenting out the 
       other waveform
    2. 
       d) If the window is set larger than the number of data points per interval, it can cause 
          distortion, whilst lower window values will leave some transient features
       e) In current sampling, the sampling window cannot be in the first or last 1% of an interval
    7. Scroll down to the sixth point and choose whether you want to display and/or save plots of 
       the analysed data
    8. Run the python file

The potential waveform data will be saved in a .txt file in the /data folder of the current working
directory, whilst the analysed oscilloscope data will be saved in a .txt file in the /analysis 
folder of the current working directory. If saved, the plotted data will be saved in a .png file in
the /plots folder of the current working drectory.

===================================================================================================
'''


import os
import time
from errno import EEXIST

import waveforms as wf
import simulations as sim
import plot as plt


'''1. MAKE THE /DATA, /ANALYSIS, & /PLOTS FOLDERS''' 
cwd = os.getcwd()

try:
    os.makedirs(cwd + '/data')
except OSError as exc:
    if exc.errno == EEXIST and os.path.isdir(cwd + '/data'):
        pass
    else: 
        raise


try:
    os.makedirs(cwd + '/plots')
except OSError as exc:
    if exc.errno == EEXIST and os.path.isdir(cwd + '/plots'):
        pass
    else: 
        raise

'''2. DEFINE THE START TIME'''
start = time.time()  

'''3. DESCRIBE THE WAVEFORM THAT WAS USED IN THE EXPERIMENT OR IS TO BE USED IN THE SIMULATION'''
shape = wf.CV(Eini = 0, Eupp = 0.5, Elow = 0, dE = 0.001, sr = 0.1, ns = 1)


data = sim.Diffusive(input = shape, E0 = 0.25, k0 = 0.005, a = 0.5, cR = 0.005, cO = 0.000000, DR = 5E-06, DO = 5E-06, Cd = 0.000020, Ru = 250, Nernstian = False, BV = True, MH = False, electrical = False, shot = False, thermal = False, r = 0.15, expansion = 1.05)


'''6. VISUALISE THE ANALYSIS'''
plt.Plotter(shape, data)

'''7. SAVE THE DATA'''
with open(f'{cwd}/data/{time.strftime("%Y-%m-%d %H-%M-%S")} {shape.subtype} waveform.txt', 'w') as file:
    for ix, iy, iz in shape.output():
        file.write(str(ix) + ',' + str(iy) + ',' + str(iz) + '\n')

        
'''8. DEFINE THE END TIME'''
end = time.time()
print(f'The oscilloscope file took {end-start} seconds to analyse')
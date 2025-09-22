'''
===================================================================================================
Copyright (C) 2023 Steven Linfield

This file is part of the electrochemistry-simulations package. This package is free software: you 
can redistribute it and/or modify it under the terms of the GNU General Public License as published 
by the Free Software Foundation, either version 3 of the License, or (at your option) any later 
version. This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU General Public License for more details. You should have received a copy of the GNU General 
Public License along with electrochemistry-simulations. If not, see https://www.gnu.org/licenses/
===================================================================================================

Package title:      electrochemistry-simulations
Repository:         https://github.com/MoonshinetheP/electrochemistry-simulations
Date of creation:   09/03/2023
Main author:        Steven Linfield (MoonshinetheP)
Collaborators:      None
Acknowledgements:   Oliver Rodriguez (oliverrdz), Guy Denuault

Filename:           simulations.py

===================================================================================================

Description:

This is the main simulation module of the electrochemistry-simulations package and can be used to 
generate simulations of diffusion-based electrochemical reactions. The module uses the output of 
the waveforms module to prepare a .txt file containing the time, potential, and current. 

===================================================================================================

How to use this file:
    
    1.  Scroll down to the section titled 'RUNNING THE SIMULATION FROM MAIN' (near the bottom)
    2.  Describe the waveform using the appropriate class from the waveforms (wf) module
    3.  Define the parameters of the simulation in the 'Diffusive' class
    4.  Run the file in Python
    
The programme will attempt to make a new folder in the current working directory to keep all the 
data. The name of this folder is /data by default, but this can be edited. If the /data folder 
already exists, then this programme will save data into this folder.

===================================================================================================

Note:

The code within this file is split into several sections. Each section starts with a title and a 
brief description of what the code within it does. Comments are placed throughout the code, with 
one hashtag (#) indicating a description of what the code below does and two hashtags (##) 
indicating the formal definition of a variable.

===================================================================================================
'''




'''MODULES'''
import sys
import os
import time
import numpy as np
import waveforms as wf
import plot as plt
import capacitance as cap

from mechanism import E, C, Reactions
from matrices import Matrices
from solver import Solver

from errno import EEXIST
from PyQt5.QtWidgets import QApplication




'''SIMULATION CLASS'''
class Diffusive:
    """Simulation of a diffusion-based electrochemical reaction through solving Fick's 2nd law of diffusion with an initial value problem solver from scipy"""

    def __init__(self, input, mechanism, Nernstian, BV, MH, r, expansion):

        
        # The actual waveform object passed into the simulation class
        self.input = input
        # Index (i.e. the list of data points) derived from the waveform object                                      
        self.index = self.input.index                          
        # Time for simulations derived from the waveform object
        self.t = self.input.t                                   
        # Potential for simulations derived from the waveform object
        self.E = self.input.E
        # Time for plotting simulation data                                   
        self.tPLOT = self.input.tPLOT
        # Potential for plotting simulation data                           
        self.EPLOT = self.input.EPLOT                           

        self.mechanism = mechanism
        ## Faraday constant (in C/mol)
        self.F = 96485                                          
        ## Ideal gas constant (in J/Kmol)
        self.R = 8.314                                          
        ## Standard temperature (in K)
        self.Temp = 298                                         
        


        ## Nersntian kinetics (True or False)
        self.Nernstian = Nernstian            
        ## Butler-Volmer kinetics (True or False)                  
        self.BV = BV
        ## Marcus-Hush kinetics (True or False)                                            
        self.MH = MH

        # Variable to hold the number of kinetic models which have been selected
        self.kinetics = 0
        # Loops through the kinetic models and adds 1 to model counter if the model is True                              
        for ix in [self.Nernstian, self.BV, self.MH]:
            if ix == True:
                self.kinetics += 1

        # If no kinetic model is chosen, an error message is show and the simulation is cancelled
        if self.kinetics == 0:
            print('\n' + 'No kinetic model was chosen' + '\n')
            sys.exit()

        # If more than one kinetic model is chosen, an error message is shown and the simulation is cancelled
        if self.kinetics >= 2:
            print('\n' + 'More than one kinetic model was chosen' + '\n')
            sys.exit()


        # Electrode radius (in cm)
        self.r = r 
        # Expansion factor of the spatial grid                                             
        self.expansion = expansion                              


        self.Dmax = self.mechanism.Dmax
        self.d = self.Dmax/self.Dmax
        ## Dimensionless time
        self.T = (self.Dmax * self.t) / (self.r ** 2)
        ## Dimensionless time step
        self.dT = np.diff(self.T)
        self.sT = np.array(())
        self.sT = np.append(self.sT, np.amin(self.dT))
        
  
   
        
        # Maximum dimensionless time
        self.Tmax = self.T[-1]
        
        ## Maximum dimensionless distance 
        self.Xmax = 6 * np.sqrt(self.d * self.Tmax)

        # Dimensionless potential
        for ix in self.mechanism.markers:
            if len(ix['Oxidised']) != 0:
                self.E0 = ix['Oxidised'][1]
                self.theta = (self.F / (self.R * self.Temp)) * (self.E - self.E0)
                ix['Oxidised'][1] = self.theta
                
                self.k0 = ix['Oxidised'][2]
                self.K0 = (self.k0 * self.r) / self.Dmax 
                ix['Oxidised'][2] = self.K0

            if len(ix['Reduced']) != 0:
                self.E0 = ix['Reduced'][1]
                self.theta = (self.F / (self.R * self.Temp)) * (self.E - self.E0)
                ix['Reduced'][1] = self.theta

                self.k0 = ix['Reduced'][2]
                self.K0 = (self.k0 * self.r) / self.Dmax 
                ix['Reduced'][2] = self.K0

            if len(ix['Consumed']) != 0:
                self.k1 = ix['Consumed'][1]
                self.K1 = (self.k1 * (self.r**2)) / self.Dmax 
                ix['Consumed'][1] = self.K1

            if len(ix['Produced']) != 0:
                self.k1 = ix['Produced'][1]
                self.K1 = (self.k1 * (self.r**2)) / self.Dmax 
                ix['Produced'][1] = self.K1
        # Dimensionless standard rate constant
                                  




        # For the sake of stability, dT divided by dX squared must be less than 0.5. Since dT is dictated by the waveform, dX is calculated as below to achieve this stability criteria
        self.dX = np.sqrt(2.05 * self.dT[0])
        # The first distance in the array is 0 (i.e. the electrode surface)
        self.x = np.array([0])
        # Whilst the final element of the array is below the maximum distance, another point is added
        while self.x[-1] < self.Xmax:
            self.x = np.append(self.x, self.x[-1] + self.dX)
            self.dX *= self.expansion


        ## Number of points in the spatial grid
        self.n = int(self.x.size)
        ## Number of points in the dimensionless potential waveform                               
        self.m = int(self.E.size)                           

        # A matrix of bulk concentrations with dimensions of n x m are prepared for both the reduced and oxidised species
        matrices = Matrices(geometry = '1D', spatial = (self.x,), coordinates = (self.n,self.m), mechanism = self.mechanism)
        #for now spatial and coordinates are user entered, need a geometry making class to automate
        
        # Flux calculated during sweep, step, and hybrid type waveforms
        self.flux = np.array([])



        for k in range(1,self.m):
            self.thisflux = np.array([])
            if k == 300:
                pass
            for ix in self.mechanism.markers:
                solved = Solver(k, (self.dT,self.sT), ix, self.mechanism.markers, '1 Dimensional', (self.x,), self.kinetics)
                pass
                if len(ix['Oxidised']) != 0:
                    self.thisflux = np.append(self.thisflux, (self.F * np.pi * self.r * self.mechanism.cmax * self.mechanism.Dmax * (ix['Concentration array'][1, k] - ix['Concentration array'][0, k]) / (self.x[1] - self.x[0])))
                if len(ix['Reduced']) != 0:
                    self.thisflux = np.append(self.thisflux, (-self.F * np.pi * self.r * self.mechanism.cmax * self.mechanism.Dmax * (ix['Concentration array'][1, k] - ix['Concentration array'][0, k]) / (self.x[1] - self.x[0])))#need to do for dO and DR
            self.flux = np.append(self.flux, np.sum(self.thisflux)) 
        
        self.i = self.flux





'''RUNNING THE SIMULATION FROM MAIN'''

if __name__ == '__main__':
        

    '''2. DEFINE THE START TIME'''
    start = time.time()

    shape = wf.CV(Eini = 0, Eupp = 0.6, Elow = 0, dE = 0.001, sr = 0.01, ns = 1)
    #shape = wf.DPV(Eini = 0, Efin = 0.5, dEs = 0.005, dEp = 0.02, pt = 0.05, rt = 0.15, st = 0.001, detailed = False, sampled = False, alpha = 0.5)
    E1 = E((['G'], ['H']), ([1],[1]), ([0],[1]), ([0.000005],[0]), ([5E-6],[5E-6]), E0 = 0.3, k0 = 0.0005, a = 0.5)
    #E2 = E((['I'], ['J']), ([1],[1]), ([1],[0]), ([0],[0]), ([5E-6],[5E-6]), E0 = 0.2, k0 = 0.1, a = 0.5)
    #C1 = C((['H'], ['I']), ([1],[1]), ([1],[1]), ([0],[0]), ([5E-6],[5E-6]), k1 = 0.000001)

    '''4. RUN THE SIMULATION'''
    instance = Diffusive(input = shape, mechanism = Reactions(E1), Nernstian = False, BV = True, MH = False, r = 0.1, expansion = 1.05)

    app = QApplication(sys.argv)
    window = plt.MainWindow(instance)
    window.show()
    sys.exit(app.exec_())

    filepath = os.getcwd() + '/data/sonata.txt'
    with open(filepath, 'w') as file:
        for ix, iy in zip(instance.E, instance.i):
            file.write(str(ix) + ',' + str(iy) + '\n')
            
    '''5. DEFINE THE END TIME'''
    end = time.time()
    print(end-start)
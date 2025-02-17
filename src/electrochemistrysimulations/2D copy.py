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
import capacitance as cap
import noise as noise
import matplotlib.pyplot as plt

from errno import EEXIST
from scipy.sparse import diags as diagonals
from scipy.integrate import solve_ivp as solver


'''SIMULATION CLASS'''
class Diffusive:
    """Simulation of a diffusion-based electrochemical reaction through solving Fick's 2nd law of diffusion with an initial value problem solver from scipy"""

    def __init__(self, input, E0, k0, a, cR, cO, DR, DO, Cd, Ru, Nernstian, BV, MH, electrical, shot, thermal, r, expansion):

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


        # Variables associated with sweep type waveforms are imported
        if self.input.type == 'sweep':
            ## Initial potential for a sweep waveform
            self.Eini = self.input.Eini
            ## Upper vertex potential for a sweep waveform (ignored in LSV if dE is negative)                        
            self.Eupp = self.input.Eupp                         
            ## Lower vertex potential for a sweep waveform (ignored in LSV if dE is positive) 
            self.Elow = self.input.Elow                         
            ## Step potential for a sweep waveform (equates to number of data points)
            self.dE = self.input.dE
            ## Scan rate for a sweep waveform
            self.sr = self.input.sr                             
            ## Number of scans for a sweep waveform
            self.ns = self.input.ns                             
            
            ## Whether the waveform should be detailed or not (set to False for a sweep waveform)
            self.detailed = self.input.detailed                               


        

        '''
        -------------------------------------------------------------------------------------------
        MECHANISM VARIABLES
        -------------------------------------------------------------------------------------------

        Variables associated with the electrochemical reaction defined by the user in the instance
        of the Diffusive class are imported. 
        
        -------------------------------------------------------------------------------------------    
        '''
        
        ## Standard redox potential (in V)
        self.E0 = E0                                            
        ## Standard rate constant (in cm s^-1)
        self.k0 = k0                                            
        ## Transfer coefficient (no units)
        self.a = a                                              
        ## Concentration of the reduced species (in mol/cm^3)
        self.cR = cR                                            
        ## Concentration of the oxidised species (in mol/cm^3)
        self.cO = cO                                            
        ## Diffusion coefficient of the reduced species (in cm^2 s^-1)
        self.DR = DR                                            
        ## Diffusion coefficient of the reduced species (in cm^2 s^-1)
        self.DO = DO                                            

        ## Faraday constant (in C/mol)
        self.F = 96485                                          
        ## Ideal gas constant (in J/Kmol)
        self.R = 8.314                                          
        ## Standard temperature (in K)
        self.Temp = 298                                         
        


        '''
        -------------------------------------------------------------------------------------------
        SPATIAL GRID VARIABLES
        -------------------------------------------------------------------------------------------
        
        In this section, the variables associated with the spatial coordinates are imported.    
        
        -------------------------------------------------------------------------------------------
        '''
        
        # Electrode radius (in cm)
        self.r = r 
        # Expansion factor of the spatial grid                                             
        self.expansion = expansion                              


        
        '''
        -------------------------------------------------------------------------------------------
        DIMENSIONLESS VARIABLES
        -------------------------------------------------------------------------------------------

        The variables imported above can be redefined as dimensionless variables.  

        -------------------------------------------------------------------------------------------  
        '''
        
        # The concentration of each species is contained in an array
        concentrations = np.array([self.cR, self.cO])
        ## Maximum concentration in the concentration array
        self.cmax = np.amax(concentrations)
        ## Dimensionless concentration of the reduced species
        self.CR = self.cR / self.cmax
        ## Dimensionless concentration of the oxidised species
        self.CO = self.cO / self.cmax

        # The diffusion coefficient of each species is contained in an array
        diffusions = np.array([self.DR, self.DO])
        ## Maximum diffusion coefficient in the diffusion coefficient array
        self.Dmax = np.amax(diffusions)
        ## Dimensionless diffusion coefficient of the reduced species
        self.dR = self.DR / self.Dmax
        ## Dimensionless diffusion coefficient of the oxidised species
        self.dO = self.DO / self.Dmax
        ## Dimensionless maximum diffusion coefficient (i.e. 1)
        self.d = self.Dmax / self.Dmax

        ## Dimensionless time
        self.T = (self.Dmax * self.t) / (self.r ** 2)
        ## Dimensionless time step
        self.dT = np.diff(self.T)

       
        # 
        if self.detailed == False:
            if self.input.type == 'sweep' or self.input.type == 'hybrid':
                self.sT = np.array([])
                self.sT = np.append(self.sT, np.amin(self.dT))

        # Maximum dimensionless time
        self.Tmax = self.T[-1]
        
        ## Maximum dimensionless distance 
        self.Zmax = 6 * np.sqrt(self.d * self.Tmax)

        # Dimensionless potential
        self.theta = (self.F / (self.R * self.Temp)) * (self.E - self.E0)

        # Dimensionless standard rate constant
        self.K0 = (self.k0 * self.r) / self.Dmax
        
        
        ## Double layer capacitance (in F)
        self.Cd = Cd                                            
        ## Uncompensated resistance (in Ohms)
        self.Ru = Ru                                                     
        

        self.Re = self.r/self.r

        # Maximum dimensionless distance
        self.Zmax = 6 * np.sqrt(self.T[-1])
        self.Rmax = self.Re + 6 * np.sqrt(self.T[-1])


        '''SPATIAL GRID'''
                # For the sake of stability, dT divided by dX squared must be less than 0.5. Since dT is dictated by the waveform, dX is calculated as below to achieve this stability criteria
        self.h = np.sqrt(2.05 * self.dT[0])
                # The first distance in the array is 0 (i.e. the electrode surface)
        self.R = np.array([0], dtype = np.float64)
                # Whilst the final element of the array is below the maximum distance, another point is added


        self.rh = 0.1
        while self.R[-1] < self.Re / 2:
                self.R = np.append(self.R, self.R[-1] + self.rh)
                self.rh *= self.expansion
        while self.R[-1] < self.Re:
                self.R = np.append(self.R, self.R[-1] + self.rh)
                self.rh = self.rh/self.expansion
        self.Rep = self.R.size
        while self.R[-1] < self.Rmax:
                self.R = np.append(self.R, self.R[-1] + self.rh)
                self.rh *= self.expansion

        self.h = np.sqrt(2.05 * self.dT[0])
        self.Z = np.array([0], dtype = np.float64)

        while self.Z[-1] < self.Zmax:
            self.Z = np.append(self.Z, self.Z[-1] + self.h)
            self.h *= self.expansion


        self.l = int(self.R.size)        ## Number of points in the spatial grid
        self.n = int(self.Z.size)        ## Number of points in the spatial grid
        self.m = int(self.theta.size)        ## Number of points in the dimensionless potential waveform


        self.C_R = np.ones((self.l, self.n, self.m)) * self.CR 
        # A matrix of bulk concentrations with dimensions of n x m are prepared for both the reduced and oxidised species 
        self.C_O = np.ones((self.l, self.n, self.m)) * self.CO

        # An array of ones are generated for the coefficients of the expanded Fick's 2nd law
        self.alpha_RZ = np.ones(self.n - 1)
        self.beta_RZ = np.ones(self.n)
        self.gamma_RZ = np.ones(self.n - 1)
        
        self.alpha_RR = np.ones(self.l - 1)
        self.beta_RR = np.ones(self.l)
        self.gamma_RR = np.ones(self.l - 1)
        
        self.alpha_OZ = np.ones(self.n - 1)
        self.beta_OZ = np.ones(self.n)
        self.gamma_OZ = np.ones(self.n - 1)

        self.alpha_OR = np.ones(self.l - 1)
        self.beta_OR = np.ones(self.l)
        self.gamma_OR = np.ones(self.l - 1)
          
        for ix in range(1, self.n - 1):
            try: 
                self.Zplus = self.Z[ix + 1] - self.Z[ix]
            except: pass
            self.Zminus = self.Z[ix] - self.Z[ix - 1]
            self.denominator = 1 / (self.Zminus + self.Zplus) # why not dT[0] work?
            
            self.alpha_RZ[ix - 1] *= 2 * self.dR * (1 / self.Zminus) * self.denominator
            self.beta_RZ[ix] *= (-2 + (2 * self.dR * self.denominator) * ((-1/self.Zplus) + (-1/self.Zminus)))
            try:
                self.gamma_RZ[ix] *= 2 * self.dR * (1 / self.Zplus) * self.denominator
            except: pass

            self.alpha_OZ[ix - 1] *= 2 * self.dO * (1 / self.Zminus) * self.denominator
            self.beta_OZ[ix] *= (-2 + (2 * self.dO * self.denominator) * ((-1/self.Zplus) + (-1/self.Zminus)))
            try:
                self.gamma_OZ[ix] *= 2 * self.dO * (1 / self.Zplus) * self.denominator
            except: pass
        
        for ix in range(1, self.l - 1):
            try: 
                self.Rplus = self.R[ix + 1] - self.R[ix]
            except: pass
            self.Rminus = self.R[ix] - self.R[ix - 1]
            self.denominator = 1 / (self.Rminus + self.Rplus) # why not dT[0] work?
            
            self.alpha_RR[ix - 1] *= 1 * self.dR * self.denominator * ((2/self.Rminus) - (1/self.R[ix]))
            self.beta_RR[ix] *= (-2 + (2 * self.dR * self.denominator *((-1/self.Rplus) + (-1/self.Rminus))))
            try:
                self.gamma_RR[ix] *= 1 * self.dR * self.denominator * ((2/self.Rplus) + (1/self.R[ix]))
            except: pass

            self.alpha_OR[ix - 1] *= 1 * self.dO * self.denominator * ((2/self.Rminus) - (1/self.R[ix]))
            self.beta_OR[ix] *= (-2 + (2 * self.dO * self.denominator *((-1/self.Rplus) + (-1/self.Rminus))))
            try:
                self.gamma_OR[ix] *= 1 * self.dO * self.denominator * ((2/self.Rplus) + (1/self.R[ix]))
            except: pass
        #probably can make diagonal not square and pop in a  initial term - need to think
            
        
        RZ = diagonals([self.alpha_RZ, self.beta_RZ, self.gamma_RZ], [-1,0,1]).toarray()
        RZ[0,:] = np.zeros(self.n)
        RZ[0,0] = 1

        RR = diagonals([self.alpha_RR, self.beta_RR, self.gamma_RR], [-1,0,1]).toarray()


        OZ = diagonals([self.alpha_OZ, self.beta_OZ, self.gamma_OZ], [-1,0,1]).toarray()
        OZ[0,:] = np.zeros(self.n)
        OZ[0,0] = 1

        OR = diagonals([self.alpha_OR, self.beta_OR, self.gamma_OR], [-1,0,1]).toarray()


        def reducedZ(t,y):
            return np.dot(RZ,y)
        
        def reducedR(t,y):
            return np.dot(RR,y)
        
        def oxidisedZ(t,y):
            return np.dot(OZ,y)

        def oxidisedR(t,y):
            return np.dot(OR,y)
        
        # Flux calculated during sweep, step, and hybrid type waveforms
        self.flux = np.array([])


        #will be something like: for each R column, work out the boundary condition. Then, work out Z axis diffusion (using existing method) for each column at 1/2 timestep, then work out R axis diffusion (will need to rearrange the shape of the array and make a new matrix. Not sure what the coefficients will look like) for each row at full timestep.
        # maybe I could flatten the matrix into an array and make the dot solve for a more complex matrix    
        for k in range(1,self.m):
            for column in range(0, self.Rep):
                self.C_R[column,0, k] = (self.C_R[column,1, k - 1] + self.Z[1] * self.K0 * np.exp(-self.a * self.theta[k - 1]) * (self.C_O[column,1, k - 1] + (self.dR/self.dO) * self.C_R[column,1, k - 1]))/(self.Z[1] * self.K0 * (np.exp((1 - self.a) * self.theta[k - 1]) + (self.dR/self.dO) * np.exp((-self.a) * self.theta[k - 1])) + 1)
                
                self.C_O[column,0, k] = (self.C_O[column,1, k - 1] + self.Z[1] * self.K0 * np.exp((1-self.a) * self.theta[k - 1]) * (self.C_O[column,1, k - 1] + (self.dR/self.dO) * self.C_R[column,1, k - 1]))/(self.Z[1] * self.K0 * (np.exp((1 - self.a) * self.theta[k - 1]) + (self.dR/self.dO) * np.exp((-self.a) * self.theta[k - 1])) + 1)
            
            for column in range(0, self.l):    
                oxidation = solver(reducedZ, [0, self.dT[k - 1]], self.C_R[column,:,k - 1], t_eval=self.sT, method='RK45')
                self.C_R[column,1:-1, k] = oxidation.y[1:-1, -1]

                reduction = solver(oxidisedZ, [0, self.dT[k - 1]], self.C_O[column,:,k - 1], t_eval=self.sT, method='RK45')
                self.C_O[column,1:-1, k] = reduction.y[1:-1, -1]
    
            for row in range(1, self.n - 1):
                oxidation = solver(reducedR, [0, self.dT[k - 1]], self.C_R[:,row,k], t_eval=self.sT, method='RK45')
                self.C_R[:, row, k] = oxidation.y[:, -1]
 
                reduction = solver(oxidisedR, [0, self.dT[k - 1]], self.C_O[:,row,k], t_eval=self.sT, method='RK45')
                self.C_O[:,row, k] = reduction.y[:,-1]
            
            self.flux = np.append(self.flux, np.sum((self.F * 2* np.pi * self.r * self.cmax * self.Dmax) * ((self.C_R[:self.Rep,1, k] - self.C_R[:self.Rep,0, k]) / (self.Z[1] - self.Z[0])) - (self.F * 2 * np.pi * self.r * self.cmax * self.Dmax) * ((self.C_O[:self.Rep,1, k] - self.C_O[:self.Rep,0, k]) / (self.Z[1] - self.Z[0]))))
 
        self.i = self.flux


'''RUNNING THE SIMULATION FROM MAIN'''

if __name__ == '__main__':
    
    '''1. MAKE A /DATA FOLDER'''
    cwd = os.getcwd()    
    try:
        os.makedirs(cwd + '/data')
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(cwd + '/data'):
            pass
        else: 
            raise
    

    '''2. DEFINE THE START TIME'''
    start = time.time()


    '''3. DESCRIBE THE WAVEFORM'''
    '''Sweeps'''
    #shape = wf.LSV(Eini = 0, Eupp = 0.5, Elow = 0, dE = 0.001, sr = 0.1, ns = 1)
    shape = wf.CV(Eini = 0, Eupp = 0.5, Elow = 0, dE = 0.001, sr = 1.0, ns = 1)
    
    '''STEPS'''
    #shape = wf.CA(dE = [0.5], dt = [1], st = 0.001)
    
    '''PULSES'''
    #shape = wf.DPV(Eini = 0, Efin = 0.5, dEs = 0.005, dEp = 0.02, pt = 0.05, rt = 0.15, st = 0.001, detailed = True, sampled = False, alpha = 0.5)
    #shape = wf.SWV(Eini = 0, Efin = 0.5, dEs = 0.005, dEp = 0.02, pt = 0.1, rt = 0.1, st = 0.001, detailed = True, sampled = True, alpha = 0.25)
    #shape = wf.NPV(Eini = 0, Efin = 0.5, dEs = 0.005, dEp = 0.02, pt = 0.05, rt = 0.15, st = 0.001, detailed = False, sampled = True, alpha = 0.25)
    
    '''HYBRID'''
    #shape = wf.CSV(Eini = -0.4, Eupp = 0.05, Elow = -0.4, dE = 0.002, sr = 0.5, ns = 1, st = 0.0001, detailed = True, sampled = True, alpha = 0.1)
    #shape = wf.AC(Eini = 0, Eupp = 0.5, Elow = 0, dE = 0.001, sr = 0.1, ns = 1, st = 0.001, detailed = True, sampled = True, alpha = 0.25)
    

    '''4. RUN THE SIMULATION'''
    instance = Diffusive(input = shape, E0 = 0.25, k0 = 0.1, a = 0.5, cR = 0.000005, cO = 0.000000, DR = 5E-06, DO = 5E-06, Cd = 0.000020, Ru = 250, Nernstian = False, BV = True, MH = False, electrical = False, shot = False, thermal = False, r = 0.00125, expansion = 1.05)
    #instance = Adsorbed(input = shape, E0 = 0.25, k0 = 1, a = 0.5, SC = 10E-10, Nernstian = True, BV = False, r = 0.15)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
    fig.tight_layout(pad = 5)

    number = instance.R.size * instance.Z.size
    xdata = np.empty((number))
    ydata = np.empty((number))
    for i in range(1,instance.R.size):
       xdata[int((i-1)*instance.Z.size) : int((i)* instance.Z.size)] = instance.R[i-1]
       ydata[int((i-1)*instance.Z.size):int((i)*instance.Z.size)] = instance.Z

    left = ax1.scatter(xdata, ydata, color = 'blue', s = 0.1, label = None, visible = True)       # plots the potential waveform from waveforms.py on the left-hand subplot

    ax1.axvline(instance.Re, color = 'red')
    ax1.set_xlim(np.amin(instance.R) - (0.1 * (np.amax(instance.R) - np.amin(instance.R))), np.amax(instance.R) + (0.1 * (np.amax(instance.R) - np.amin(instance.R))))        # sets the x-axis of the right-hand subplot to +/- 10% of the oscilloscope data's potential range
    ax1.set_ylim(np.nanmin(instance.Z) - (0.1 * (np.nanmax(instance.Z) - np.nanmin(instance.Z))), np.nanmax(instance.Z) + (0.1 * (np.nanmax(instance.Z) - np.nanmin(instance.Z))))        # sets the y-axis of the right-hand subplot to +/- 10% of the oscilloscope data's current range
    ax1.set_title('2D spatial grid', pad = 15, fontsize = 20)       # defines the title and settings of the right-hand subplot
    ax1.set_xlabel('R', labelpad = 5, fontsize = 15)        # defines the x-axis label and settings of the right-hand subplot
    ax1.set_ylabel('Z', labelpad = 5, fontsize = 15)        # defines the y-axis label and settings of the right-hand subplot
        

    '''DATA PLOT'''
    right, = ax2.plot(instance.E[1:], instance.i, linewidth = 1, linestyle = '-', color = 'red', marker = None, label = None, visible = True)     # plots the oscilloscope data from operations.py on the right-hand subplot
       
    ax2.set_xlim(np.amin(instance.E) - (0.1 * (np.amax(instance.E) - np.amin(instance.E))), np.amax(instance.E) + (0.1 * (np.amax(instance.E) - np.amin(instance.E))))        # sets the x-axis of the right-hand subplot to +/- 10% of the oscilloscope data's potential range
    ax2.set_ylim(np.nanmin(instance.i) - (0.1 * (np.nanmax(instance.i) - np.nanmin(instance.i))), np.nanmax(instance.i) + (0.1 * (np.nanmax(instance.i) - np.nanmin(instance.i))))        # sets the y-axis of the right-hand subplot to +/- 10% of the oscilloscope data's current range
    ax2.set_title('i vs. E', pad = 15, fontsize = 20)       # defines the title and settings of the right-hand subplot
    ax2.set_xlabel('E / V', labelpad = 5, fontsize = 15)        # defines the x-axis label and settings of the right-hand subplot 
    ax2.set_ylabel('i / A', labelpad = 5, fontsize = 15)        # defines the y-axis label and settings of the right-hand subplot
        
    plt.show()
    plt.close()

    '''5. DEFINE THE END TIME'''
    end = time.time()
    print(end-start)
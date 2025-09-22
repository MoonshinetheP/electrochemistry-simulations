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
import plot as plt

from errno import EEXIST
from scipy.sparse import diags as diagonals
from scipy.integrate import solve_ivp as solver


'''SIMULATION CLASS'''
class Diffusive:
    """Simulation of a diffusion-based electrochemical reaction through solving Fick's 2nd law of diffusion with an initial value problem solver from scipy"""

    def __init__(self, input, E0, k0, a, cR, cO, DR, DO, Cd, Ru, Nernstian, BV, MH, electrical, shot, thermal, r, expansion):
        '''
        -------------------------------------------------------------------------------------------
        WAVEFORM VARIABLES
        -------------------------------------------------------------------------------------------
        
        Several variables first need to be imported from the waveforms module through the input 
        argument before they can be used in the simulation. Specific waveform types are defined 
        here in order to differentiate between variables that are available in some waveforms, but 
        not others. For sweeps, the detailed variable is set to False by default. For steps, the 
        detailed variable is set to True by default. For all other waveforms, this variable is set 
        according to the value entered in the waveform generation.

        -------------------------------------------------------------------------------------------
        '''
        
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


        # Variables associated with step type waveforms are imported
        if self.input.type == 'step':
            ## Applied potential for a step waveform
            self.dE = self.input.dE                             
            ## Duration for which the potential is applied in a step waveform
            self.dt = self.input.dt                             
            
            ## Whether the waveform should be detailed or not (set to True for a step waveform)
            self.detailed = self.input.detailed                                


        # Variables associated with pulse type waveforms are imported
        if self.input.type == 'pulse':
            ## Initial potential for a pulse waveform
            self.Eini = self.input.Eini                         
            ## Final potential for a pulse waveform
            self.Efin = self.input.Efin                         
            ## Step potential for a pulse waveform
            self.dEs = self.input.dEs                           
            ## Pulse amplitude (not peak to peak) for a pulse waveform
            self.dEp = self.input.dEp                           
            ## Duration of the pulse in a pulse waveform
            self.pt = self.input.pt                             
            ## Duration of the rest period in a pulse waveform
            self.rt = self.input.rt                             
            ## Sampling time for a detailed pulse waveform (ignored if detailed == False)
            self.st = self.input.st                             

            ## Whether the pulse waveform should be detailed or not
            self.detailed = self.input.detailed                 


        # Variables associated with hybrid type waveforms are imported
        if self.input.type == 'hybrid':
            ## Initial potential for a hybrid waveform
            self.Eini = self.input.Eini                         
            ## Upper vertex potential for a hybrid waveform
            self.Eupp = self.input.Eupp                         
            ## Lower vertex potential for a hybrid waveform
            self.Elow = self.input.Elow                         
            ## Step potential for a hybrid waveform
            self.dE = self.input.dE                             
            ## Scan rate for a hybrid waveform
            self.sr = self.input.sr                             
            ## Number of scans for a hybrid waveform
            self.ns = self.input.ns                             
            ## Sampling time for a detailed hybrid waveform (ignored if detailed == False)
            self.st = self.input.st                             

            ## Whether the hybrid waveform should be detailed or not
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
        KINETIC MODELS
        -------------------------------------------------------------------------------------------

        The kinetic models which the user selected in the instance of the Diffusive class are 
        counted and checked. Error messages are given if the user has selected no valid kinetic models or more than one 
        valid kinetic model. 

        ------------------------------------------------------------------------------------------- 
        '''

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



        '''
        -------------------------------------------------------------------------------------------
        NOISE MODELS
        -------------------------------------------------------------------------------------------
        
        In this section, the noise models which the user has selected are imported.

        -------------------------------------------------------------------------------------------
        '''
        
        ## Electrical noise (True or False)
        self.electrical = electrical
        ## Shot noise (True or False)
        self.shot = shot
        ## Thermal noise (True or False)
        self.thermal = thermal



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
        if self.input.type == 'pulse':
            #
            self.pT = np.array([])
            #
            self.rT = np.array([])
            #
            for ix in range (0, self.dT.size):
                if ix % 2 == 0:
                    self.pT = np.append(self.pT, self.dT[ix])
                else:
                    self.rT = np.append(self.rT, self.dT[ix])
        
        # 
        if self.detailed == False:
            if self.input.type == 'sweep' or self.input.type == 'hybrid':
                self.sT = np.array([])
                self.sT = np.append(self.sT, np.amin(self.dT))
            
            if self.input.type == 'pulse':
                self.psT = np.empty((self.pT.size, 1))
                for ix in range(0, self.pT.size):
                    self.psT[ix,:] = np.amin(self.pT)
                
                self.rsT = np.empty((self.rT.size, 1))
                for iy in range(0, self.rT.size):
                    self.rsT[iy,:] = np.amin(self.rT)
                
        else:
            if self.input.type == 'pulse':
                self.psT = np.empty((self.pT.size, self.input.pp))
                for ix in range(0, self.pT.size):
                    self.psT[ix,:] = np.arange(0, np.amin(self.pT), (self.Dmax * self.st) / (self.r ** 2))
                
                self.rsT = np.empty((self.rT.size, self.input.rp))
                for iy in range(0, self.rT.size):
                    self.rsT[iy,:] = np.arange(0, np.amin(self.rT), (self.Dmax * self.st) / (self.r ** 2))
            
            #
            if self.input.type == 'hybrid':
                #
                self.sT = np.array([])
                #
                self.sT = np.append(self.sT, np.arange(0, np.amin(self.dT), (self.Dmax * self.st) / (self.r ** 2)))
                #
                if self.sT[-1] > self.dT[1]:
                    #
                    self.sT = self.sT[:-1]
        
        # Maximum dimensionless time
        self.Tmax = self.T[-1]
        
        ## Maximum dimensionless distance 
        self.Xmax = 6 * np.sqrt(self.d * self.Tmax)

        # Dimensionless potential
        self.theta = (self.F / (self.R * self.Temp)) * (self.E - self.E0)

        # Dimensionless standard rate constant
        self.K0 = (self.k0 * self.r) / self.Dmax
        
        
        
        '''
        -------------------------------------------------------------------------------------------
        CELL CHARACTERISTICS
        -------------------------------------------------------------------------------------------
        
        In this section, variables representing the characteristics of the cell are imported. The
        double layer capacitance and the uncompensated resistance are later used to calculate the
        capacative contribution to an analogue response.

        -------------------------------------------------------------------------------------------
        '''      
        
        ## Double layer capacitance (in F)
        self.Cd = Cd                                            
        ## Uncompensated resistance (in Ohms)
        self.Ru = Ru                                            



        '''
        -------------------------------------------------------------------------------------------
        EXPANDING SPATIAL GRID
        -------------------------------------------------------------------------------------------

        The spatial grid is an array of distances between which the change in concentration is 
        calculated. The finer the grid, the more continuous the calculation, but at the expense of
        computation time. To save on computation time, an expansion factor can be introduced which
        increases the distance between subsequent points, allowing for a finer grid at the 
        electrode surface and a coarse grid in the bulk solution. If the expansion factor is set to
        1, then there is no expansion.
         
        -------------------------------------------------------------------------------------------    
        '''                
        
        # For the sake of stability, dT divided by dX squared must be less than 0.5. Since dT is dictated by the waveform, dX is calculated as below to achieve this stability criteria
        self.dX = np.sqrt(2.05 * self.dT[0])
        # The first distance in the array is 0 (i.e. the electrode surface)
        self.x = np.array([0])
        # Whilst the final element of the array is below the maximum distance, another point is added
        while self.x[-1] < self.Xmax:
            self.x = np.append(self.x, self.x[-1] + self.dX)
            self.dX *= self.expansion
        


        '''
        -------------------------------------------------------------------------------------------
        DIFFUSION MATRICES
        -------------------------------------------------------------------------------------------
            
        Fick's 2nd law is discretised

        -------------------------------------------------------------------------------------------
        '''

        ## Number of points in the spatial grid
        self.n = int(self.x.size)
        ## Number of points in the dimensionless potential waveform                               
        self.m = int(self.theta.size)                           

        # A matrix of bulk concentrations with dimensions of n x m are prepared for both the reduced and oxidised species
        self.C_R = np.ones((self.n, self.m)) * self.CR          
        self.C_O = np.ones((self.n, self.m)) * self.CO

        # An array of ones are generated for the coefficients of the expanded Fick's 2nd law
        self.alpha_R = np.ones(self.n - 1)
        self.beta_R = np.ones(self.n)
        self.gamma_R = np.ones(self.n - 1)
        
        self.alpha_O = np.ones(self.n - 1)
        self.beta_O = np.ones(self.n)
        self.gamma_O = np.ones(self.n - 1)
          
        for ix in range(1, self.n - 1):
            self.xplus = self.x[ix + 1] - self.x[ix]
            self.xminus = self.x[ix] - self.x[ix - 1]
            self.denominator = self.xminus * (self.xplus ** 2) + self.xplus * (self.xminus **2)
            
            self.alpha_R[ix - 1] *= 2 * self.dR * self.xplus / self.denominator
            self.beta_R[ix] *= -2 * self.dR * (self.xminus + self.xplus) / self.denominator
            self.gamma_R[ix] *= 2 * self.dR * self.xminus / self.denominator
            
            self.alpha_O[ix - 1] *= 2 * self.dO * self.xplus / self.denominator
            self.beta_O[ix] *= -2 * self.dO * (self.xminus + self.xplus) / self.denominator
            self.gamma_O[ix] *= 2 * self.dO * self.xminus / self.denominator
        
            

        R = diagonals([self.alpha_R, self.beta_R, self.gamma_R], [-1,0,1]).toarray()
        R[0,:] = np.zeros(self.n)
        R[0,0] = self.CR    
        
        O = diagonals([self.alpha_O, self.beta_O, self.gamma_O], [-1,0,1]).toarray()
        O[0,:] = np.zeros(self.n)
        O[0,0] = self.CO   

        def reduced(t,y):
            return np.dot(R,y)
        
        def oxidised(t,y):
            return np.dot(O,y)



        '''
        -------------------------------------------------------------------------------------------
        CONTAINERS
        -------------------------------------------------------------------------------------------
            
        Before running the main simulation loop, several containers are made to contain the results
        of the simulation

        -------------------------------------------------------------------------------------------
        '''
        
        # Flux calculated during sweep, step, and hybrid type waveforms
        self.flux = np.array([])
        # Flux calculated during the pulse period of a pulse type waveform
        self.pulseflux = np.array([])
        # Flux calculated during the rest period of a pulse type waveform
        self.restflux = np.array([])



        '''
        -------------------------------------------------------------------------------------------
        MAIN LOOP
        -------------------------------------------------------------------------------------------
            
        Here is the main part of the simulation, where Fick's 2nd law is solved. For each point in 
        the dimensionless potential waveform, the boundary condition of the selected kinetic model 
        is used to calculate the concentration of each species at the electrode surface. Then, an
        initial values problem solver is used to update each point in the concentration matrix 
        using the diffusion matrix. Finally, the flux between the first two grid points is used to
        calculate the flux of species towards the electrode, which is recalculated into current.

        -------------------------------------------------------------------------------------------
        '''
        
        for k in range(1,self.m):
            '''Boundary conditions'''
            if self.Nernstian == True:
                '''Nernstian'''
                self.C_R[0, k] = (self.C_R[1, k - 1] + (self.dR/self.dO) * self.C_O[1, k - 1])/(1 + (self.dR/self.dO) * np.exp(self.theta[k-1]))

                self.C_O[0, k] = (self.C_R[1, k - 1] + (self.dR/self.dO) * self.C_O[1, k - 1])/((self.dR/self.dO) + np.exp(-self.theta[k-1]))
            
            if self.BV == True:
                '''Butler-Volmer'''
                self.C_R[0, k] = (-self.C_R[1, k - 1] + (self.x[1] - self.x[0]) * self.K0 * np.exp(-self.a * self.theta[k - 1]) * (self.C_O[1, k - 1] + (self.dR/self.dO) * self.C_R[1, k - 1]))/((self.x[1] - self.x[0]) * self.K0 * (np.exp((1 - self.a) * self.theta[k - 1]) + (self.dR/self.dO) * np.exp((-self.a) * self.theta[k - 1])) - 1)

                self.C_O[0, k] = (-self.C_O[1, k - 1] + (self.x[1] - self.x[0]) * self.K0 * np.exp((1 - self.a) * self.theta[k - 1]) * (self.C_R[1, k - 1] + (self.dO/self.dR) * self.C_O[1, k - 1]))/((self.x[1] - self.x[0]) * self.K0 * (np.exp(-self.a * self.theta[k - 1]) + (self.dO/self.dR) * np.exp((1 - self.a) * self.theta[k - 1])) - 1)
           
            if self.input.type == 'sweep' or self.input.type == 'hybrid':
                oxidation = solver(reduced, [0, self.dT[k - 1]], self.C_R[:,k - 1], t_eval=self.sT, method='RK45')
                self.C_R[1:-1, k] = oxidation.y[1:-1, -1]
                if self.detailed == True:
                    self.C_Rdet = oxidation.y[1:-1, :]
                
                reduction = solver(oxidised, [0, self.dT[k - 1]], self.C_O[:,k - 1], t_eval=self.sT, method='RK45')
                self.C_O[1:-1, k] = reduction.y[1:-1, -1]
                if self.detailed == True:
                    self.C_Odet = reduction.y[1:-1, :]

            if self.input.type == 'pulse':
                if k % 2 != 0:
                    oxidation = solver(reduced, [0, self.dT[k - 1]], self.C_R[:,k - 1], t_eval=self.psT[(k-1)//2], method='RK45')
                if k % 2 == 0:
                    oxidation = solver(reduced, [0, self.dT[k - 1]], self.C_R[:,k - 1], t_eval=self.rsT[(k-2)//2], method='RK45')
            
                self.C_R[1:-1, k] = oxidation.y[1:-1, -1]

                if self.detailed == True:
                    self.C_Rdet = oxidation.y[1:-1, :]
           
                if k % 2 != 0:
                    reduction = solver(oxidised, [0, self.dT[k - 1]], self.C_O[:,k - 1], t_eval=self.psT[(k-1)//2], method='RK45')
                if k % 2 == 0:
                    reduction = solver(oxidised, [0, self.dT[k - 1]], self.C_O[:,k - 1], t_eval=self.rsT[(k-2)//2], method='RK45')
                
                self.C_O[1:-1, k] = reduction.y[1:-1, -1]

                if self.detailed == True:
                    self.C_Odet = reduction.y[1:-1, :]
            
            if self.input.type == 'sweep':
                self.flux = np.append(self.flux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_R[1, k] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_O[1, k] - self.C_O[0, k]) / (self.x[1] - self.x[0])))

            if self.input.type == 'pulse':
                if self.detailed == False:
                    if k % 2 != 0:
                        self.pulseflux = np.append(self.pulseflux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_R[1, k] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_O[1, k] - self.C_O[0, k]) / (self.x[1] - self.x[0])))
                    if k % 2 == 0:
                        self.restflux = np.append(self.restflux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_R[1, k] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_O[1, k] - self.C_O[0, k]) / (self.x[1] - self.x[0])))
                if self.detailed == True:
                    if k % 2 != 0:
                        for ix in range(0, self.input.pp):
                            self.flux = np.append(self.flux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_Rdet[0, ix] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_Odet[0, ix] - self.C_O[0, k]) / (self.x[1] - self.x[0])))
                    if k % 2 == 0:
                        for ix in range(0, self.input.rp):
                            self.flux = np.append(self.flux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_Rdet[0, ix] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_Odet[0, ix] - self.C_O[0, k]) / (self.x[1] - self.x[0])))

            if self.input.type == 'hybrid':
                if self.detailed == False:
                    self.flux = np.append(self.flux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_R[1, k] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_O[1, k] - self.C_O[0, k]) / (self.x[1] - self.x[0])))
                if self.detailed == True:
                    if self.input.sampled == False:
                        for ix in range(0, self.input.sp):
                            self.flux = np.append(self.flux, (self.F * np.pi * self.r * self.cR * self.DR) * ((self.C_Rdet[0, ix] - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((self.C_Odet[0, ix] - self.C_O[0, k]) / (self.x[1] - self.x[0])))
                    if self.input.sampled == True:
                        self.flux = np.append(self.flux, (self.F * np.pi * self.r * self.cR * self.DR) * ((np.average(self.C_Rdet[0, self.input.sp - round(self.input.sp * (1 - self.input.alpha)):]) - self.C_R[0, k]) / (self.x[1] - self.x[0])) - (self.F * np.pi * self.r * self.cO * self.DO) * ((np.average(self.C_Odet[0, self.input.sp - round(self.input.sp * (1 - self.input.alpha)):]) - self.C_O[0, k]) / (self.x[1] - self.x[0])))
       
        

        '''
        -------------------------------------------------------------------------------------------
        POST SIMULATION CALCULATIONS
        -------------------------------------------------------------------------------------------
            
        After the main loop, any other operations that need to be performed are done so here. For
        example, the capacitance is calculated using the same input as the simulation, but to
        simplify the simulation space, the calculation is done in a separate module. Also the pulse
        signal is calculated by subtracting the rest flux from the pulse flux.

        -------------------------------------------------------------------------------------------
        '''
        #self.fluxcapacitance = cap.Capacitance(input = self.input, Cd = self.Cd, Ru = self.Ru)
        #if self.input.type == 'pulse':
        #    if self.detailed == False:
        #        self.flux = self.pulseflux - self.restflux
            
        #    else:
        #        pass
        


        '''
        -------------------------------------------------------------------------------------------
        OUTPUTS
        -------------------------------------------------------------------------------------------



        -------------------------------------------------------------------------------------------    
        '''
        self.i = self.flux
        self.output = zip(self.tPLOT, self.EPLOT, self.flux)
        self.concentrations = zip(self.C_O, self.C_R)

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
    shape = wf.LSV(Eini = 0, Eupp = 0.5, Elow = 0, dE = 0.001, sr = 0.1, ns = 1)
    #shape = wf.CV(Eini = 0, Eupp = 0.5, Elow = 0, dE = 0.001, sr = 0.1, ns = 1)
    
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
    instance = Diffusive(input = shape, E0 = 0.25, k0 = 0.1, a = 0.5, cR = 0.005, cO = 0.000000, DR = 5E-06, DO = 5E-06, Cd = 0.000020, Ru = 250, Nernstian = False, BV = True, MH = False, electrical = False, shot = False, thermal = False, r = 0.15, expansion = 1.05)
    #instance = Adsorbed(input = shape, E0 = 0.25, k0 = 1, a = 0.5, SC = 10E-10, Nernstian = False, BV = True, r = 0.15)
    
    plt.Plotter(shape, instance, display = True, animate = True, save = False)
    
    '''5. DEFINE THE END TIME'''
    end = time.time()
    print(end-start)

    '''6. SAVE THE DATA'''
    filepath = cwd + '/data/' + f'{shape.subtype}' + '.txt'
    with open(filepath, 'w') as file:
        for ix, iy, iz in instance.results():
            file.write(str(ix) + ',' + str(iy) + ',' + str(iz) + '\n')
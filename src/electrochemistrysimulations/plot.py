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
Acknowledgements:   Oliver Rodriguez (oliverrdz)
    
Filename:           plot.py

===================================================================================================
How to use this file:
    

===================================================================================================
'''

import time
import collections as col
import numpy as np
import functools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QLineEdit, QRadioButton

#SOme things I need to make happen:
# Start and end buttons need to alter position slider value
# Play and reverse buttons need to alter position slider value


class MainWindow(QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data = data
        
        self.setWindowTitle('Electrochemical simulations')
        #self.setWindowIcon()
        self.setGeometry(150, 150, 2400, 1200)
        self.setStyleSheet("background-color: white;")

        self.plot = Plotter(data)

        self.WaveformPlot = FigureCanvas(self.plot.fig1)
        self.DataPlot = FigureCanvas(self.plot.fig2)
        self.ConcentrationProfile = FigureCanvas(self.plot.fig3)
        self.DiffusionMap = FigureCanvas(self.plot.fig4)

        self.direction = 1
        self.StartButton = QPushButton(text = '|<')
        self.StartButton.clicked.connect(lambda: (self.setPosition(0, self.selected), self.setSlider(0)))

        self.RewindButton = QPushButton(text = '<')
        self.RewindButton.clicked.connect(lambda: (self.timer.start(1), self.setDirection(-1)))

        self.PauseButton = QPushButton(text = '||')
        self.PauseButton.clicked.connect(lambda: self.timer.stop())

        self.PlayButton = QPushButton(text = '>')
        self.PlayButton.clicked.connect(lambda: (self.timer.start(1), self.setDirection(1)))

        self.EndButton = QPushButton(text = '>|')
        self.EndButton.clicked.connect(lambda: (self.setPosition(self.data.m - 1, self.selected), self.setSlider(self.data.m - 1)))

        self.LoopButton = QCheckBox(text = 'Loop?')
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animatePlots)

        self.SpeedSlider = QSlider()
        self.SpeedSlider.setOrientation(QtCore.Qt.Horizontal)
        self.SpeedSlider.setMinimum(0)
        self.SpeedSlider.setMaximum(8)
        self.SpeedSlider.setValue(4)

        self.SpeedTextBox = QLineEdit()
        self.SpeedTextBox.setText(f'{self.SpeedSlider.value() * 0.25}')
        
        self.PositionSlider = QSlider()
        self.PositionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.PositionSlider.setMinimum(0)
        self.PositionSlider.setMaximum(data.m - 1)
        self.PositionSlider.setValue(data.m - 1)
        self.PositionSlider.sliderMoved.connect(lambda: self.setPosition(self.PositionSlider.value(), self.selected))

        self.breakPositions = []
        self.PositionTextBox = QLineEdit(text = f'{self.PositionSlider.value()}')
        self.PositionTextBox.textChanged.connect(lambda: (self.setPosition(int(self.PositionTextBox.text()), self.selected), self.setSlider(int(self.PositionTextBox.text()))))
        self.PositionAddButton = QPushButton(text = '+')
        self.PositionAddButton.clicked.connect(lambda: self.breakPositions.append(self.PositionTextBox.value()))
        self.PositionResetButton = QPushButton(text = 'Reset')
        self.PositionAddButton.clicked.connect(lambda: self.breakPositions.clear())
        
        self.SelectionBoxes = col.namedtuple('Selection', self.plot.markerlist)._make({} for _ in self.plot.markerlist)
        for n in range(0, len(self.SelectionBoxes)):
                self.SelectionBoxes[n].update({'Number': n})
                self.SelectionBoxes[n].update({'Button': QRadioButton(f'{self.plot.markerlist[n]}')})
                self.SelectionBoxes[n]['Button'].toggled.connect(lambda: self.btnstate())
        
        
        self.selected = 0
        self.SelectionBoxes[0]['Button'].setChecked(True)

        RadioLayout = QHBoxLayout() 
        
        for ix in range(0, len(self.SelectionBoxes)):
            RadioLayout.addWidget(self.SelectionBoxes[ix]['Button'])

        self.RadioButtons = QtWidgets.QWidget()
        self.RadioButtons.setLayout(RadioLayout)



        layout = QGridLayout()

        layout.addWidget(self.WaveformPlot, 0, 0, 6, 6)
        layout.addWidget(self.DataPlot, 0, 6, 6, 6)
        layout.addWidget(self.ConcentrationProfile, 6, 0, 6, 12)
        layout.addWidget(self.DiffusionMap, 1, 12, 9, 9)
        
        layout.addWidget(self.RadioButtons, 0, 12, 1, 9)

        layout.addWidget(self.StartButton, 10, 12, 1, 1)
        layout.addWidget(self.RewindButton, 10, 13, 1, 1)
        layout.addWidget(self.PauseButton,10, 14, 1, 1)
        layout.addWidget(self.PlayButton, 10, 15, 1, 1)
        layout.addWidget(self.EndButton, 10, 16, 1, 1)
        layout.addWidget(self.LoopButton, 10, 17, 1, 1)
        layout.addWidget(self.SpeedSlider, 10, 18, 1, 2)
        layout.addWidget(self.SpeedTextBox, 10, 20, 1, 1)

        layout.addWidget(self.PositionSlider, 11, 12, 1, 6)
        layout.addWidget(self.PositionTextBox, 11, 18, 1, 1)
        layout.addWidget(self.PositionAddButton, 11, 19, 1, 1)
        layout.addWidget(self.PositionResetButton, 11, 20, 1, 1)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        
        self.show()

 
    '''FUNCTIONS'''
    def setDirection(self, direction):
        self.direction = direction

    def animatePlots(self):
        current = self.PositionSlider.value()
        current += self.direction
        self.setPosition(current, self.selected)
        self.setSlider(current)
        '''if self.LoopButton.isChecked() == False:
            if current != self.data.m - 1 and current != 0:
                current += self.direction
                self.setPosition(current, self.selected)
                self.setSlider(current)
            else:
                self.timer.stop()
        else:
            if self.direction == -1 and current == 0:
                current == self.data.m - 1
                self.setPosition(current, self.selected)
                self.setSlider(current)
            elif self.direction == 1 and current == self.data.m - 1:
                current == 0
                self.setPosition(current, self.selected)
                self.setSlider(current)
            else:
                current += self.direction
                self.setPosition(current, self.selected)
                self.setSlider(current)'''
        
    def setPosition(self, value, species):
        '''UPDATING WAVEFORM PLOT'''
        self.plot.line1.set_xdata(self.data.t[:value])
        self.plot.line1.set_ydata(self.data.E[:value])
        
        self.plot.fig1.canvas.draw()


        '''UPDATING DATA PLOT'''
        self.plot.line2.set_xdata(self.data.E[:value])
        self.plot.line2.set_ydata(self.data.i[:value])
        
        self.plot.fig2.canvas.draw()  


        '''UPDATING CONCENTRATION PROFILES'''
        for n in self.plot.cprofile:
            n['Line'].set_ydata(self.data.mechanism.markers[n['Axes']]['Concentration array'][:,int(value)])
        
        self.plot.fig3.canvas.draw()
        

        '''UPDATING DIFFUSION MAP'''
        self.plot.ax4.clear()
        self.array = np.empty((self.data.n, 100))
        for ix in range(0,100):
            self.array[:, ix] = self.data.mechanism.markers[species]['Concentration array'][:,value]
        self.plot.ax4.contourf(np.linspace(0, self.data.r, 100), self.data.x, self.array, levels = np.linspace(-0.1, 1.1, 120), cmap='plasma')
        self.plot.fig4.canvas.draw()
    
    def setSlider(self, value):
        self.PositionSlider.setValue(value)

    def btnstate(self):
        for ix in range(0, len(self.SelectionBoxes)):
            if self.SelectionBoxes[ix]['Button'].isChecked():
                self.selected = self.SelectionBoxes[ix]['Number']
                self.setPosition(self.PositionSlider.value(), self.selected)


class Plotter:
    '''Plots potential waveforms and analysed oscilloscope data generated from other files in the package\n
    
    Requires:\n
'''

    def __init__(self, data):

        '''PARAMETER INITIALISATION'''
        self.data = data        #
        
        self.waveformX = self.data.t
        self.waveformY = self.data.E
        self.dataX = self.data.E
        self.dataY = self.data.i


        cprofilelist = []
        self.markerlist = []
        colours = mcolors.TABLEAU_COLORS
        colourkeys = list(colours.keys())
        for n in range(1, len(self.data.mechanism.markers) + 1):
            cprofilelist.append(f'bottom{n}')
            self.markerlist.append(self.data.mechanism.markers[n - 1]['Species'])
        
        self.cprofile = col.namedtuple('Lines', cprofilelist)._make({} for _ in cprofilelist)
        for n in range(0, len(self.cprofile)):
            self.cprofile[n].update({'Axes': n})
            self.cprofile[n].update({'Line': 0})


        '''WAVEFORM PLOT'''
        self.fig1, self.ax1 = plt.subplots()
        self.line1, = self.ax1.plot(self.data.t, self.data.E, linewidth = 1, linestyle = '-', color = 'blue', marker = None, label = None, visible = True)       # plots the potential waveform from waveforms.py on the left-hand subplot
       
        self.ax1.set_xlim(np.amin(self.data.t) - (0.1 * (np.amax(self.data.t) - np.amin(self.data.t))), np.amax(self.data.t) + (0.1 * (np.amax(self.data.t) - np.amin(self.data.t))))      # sets the x-axis limits of the left-hand subplot to +/- 10% of the waveform's time range
        self.ax1.set_ylim(np.amin(self.data.E) - (0.1 * (np.amax(self.data.E) - np.amin(self.data.E))), np.amax(self.data.E) + (0.1 * (np.amax(self.data.E) - np.amin(self.data.E))))      # sets the y-axis limits of the left-hand subplot to +/- 10% of the waveform's potential range
        self.ax1.set_title('E vs. t', pad = 15, fontsize = 20)       # defines the title and settings of the left-hand subplot
        self.ax1.set_xlabel('t / s', labelpad = 5, fontsize = 15)        # defines the x-axis label and settings of the left-hand subplot
        self.ax1.set_ylabel('E / V', labelpad = 5, fontsize = 15)        # defines the y-axis labe and settings of the left-hand subplot
        

        '''DATA PLOT'''
        self.fig2, self.ax2 = plt.subplots()
        self.line2, = self.ax2.plot(self.data.E[1:], self.data.i, linewidth = 1, linestyle = '-', color = 'red', marker = None, label = None, visible = True)     # plots the oscilloscope data from operations.py on the right-hand subplot
       
        self.ax2.set_xlim(np.amin(self.data.E) - (0.1 * (np.amax(self.data.E) - np.amin(self.data.E))), np.amax(self.data.E) + (0.1 * (np.amax(self.data.E) - np.amin(self.data.E))))        # sets the x-axis of the right-hand subplot to +/- 10% of the oscilloscope data's potential range
        self.ax2.set_ylim(np.nanmin(self.data.i) - (0.1 * (np.nanmax(self.data.i) - np.nanmin(self.data.i))), np.nanmax(self.data.i) + (0.1 * (np.nanmax(self.data.i) - np.nanmin(self.data.i))))        # sets the y-axis of the right-hand subplot to +/- 10% of the oscilloscope data's current range
        self.ax2.set_title('i vs. E', pad = 15, fontsize = 20)       # defines the title and settings of the right-hand subplot
        self.ax2.set_xlabel('E / V', labelpad = 5, fontsize = 15)        # defines the x-axis label and settings of the right-hand subplot 
        self.ax2.set_ylabel('i / A', labelpad = 5, fontsize = 15)        # defines the y-axis label and settings of the right-hand subplot
        

        '''CONCENTRATION PROFILES'''
        self.fig3, self.ax3 = plt.subplots()
        for n in self.cprofile:
            n['Line'], = self.ax3.plot(self.data.x*0.1, self.data.mechanism.markers[n['Axes']]['Concentration array'][:,1], linewidth = 1, linestyle = '-', color = colours[colourkeys[n['Axes']]], marker = None, label = self.markerlist[n['Axes']], visible = True)
            
        self.ax3.set_xlim(0, self.data.x[-1]*0.1)
        self.ax3.set_ylim(-0.1 , 1.1)
        self.ax3.set_title('Concentration profiles', pad = 15, fontsize = 20)       # defines the title and settings of the left-hand subplot
        self.ax3.set_xlabel('x / cm', labelpad = 5, fontsize = 15)        # defines the x-axis label and settings of the left-hand subplot
        self.ax3.set_ylabel('C / Cmax', labelpad = 5, fontsize = 15)        # defines the y-axis labe and settings of the left-hand subplot
        self.ax3.legend()


        '''DIFFUSION MAP'''
        self.array = np.empty((self.data.n, 100))
        for ix in range(0,100):
            self.array[:, ix] = self.data.mechanism.markers[0]['Concentration array'][:,-1]
        
        self.fig4, self.ax4 = plt.subplots()
        self.ax4.contourf(np.linspace(0, self.data.r, 100), self.data.x, self.array, levels = np.linspace(-0.1, 1.1, 120), cmap='plasma')
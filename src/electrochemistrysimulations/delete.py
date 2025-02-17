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

Package title:      electrochemistry-simulations
Repository:         https://github.com/MoonshinetheP/electrochemistry-simulations
Date of creation:   09/03/2023
Main author:        Steven Linfield (MoonshinetheP)
Collaborators:      None
Acknowledgements:   Oliver Rodriguez (oliverrdz), Guy Denuault

Filename:           delete.py

===================================================================================================

Description:

This file contains additional code which can be used to empty the /data folder.
 
===================================================================================================

How to use this file:
    
    1. Run the python file

===================================================================================================
'''


import os


class Eraser:

    '''Deletes all items from the selected folders'''

    def __init__(self):
        
        '''PARAMETER DEFINITIONS'''
        cwd = os.getcwd()       # finds the current working directory

        '''FILE DELETION'''
        for ix in os.listdir(cwd + '/data'):      # loops through all files in the data directory
            try:
                os.remove(cwd + '/data' + '/' + ix)       # and removes each one
            except:
                raise       # raises an error just in case something gets in the way
        
        for iy in os.listdir(cwd + '/plots'):      # loops through all files in the data directory
            try:
                os.remove(cwd + '/plots' + '/' + iy)       # and removes each one
            except:
                raise       # raises an error just in case something gets in the way


"""
===================================================================================================
DELETING DATA FROM MAIN
===================================================================================================
"""

if __name__ == '__main__':

    Eraser()
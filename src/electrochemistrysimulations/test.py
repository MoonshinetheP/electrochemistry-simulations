import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def function(t,y):
    return y*2

def EulerCromer(x, y0):
    
    y = np.array([])
    y = np.append(y, y0)

    for ix in range(1, len(x)):
        y = np.append(y, y[-1] + (x[ix]-x[ix - 1]) * function(x[ix - 1], y[-1]))

    return y

def RungeKutta(x, y0):
    '''Runge-Kutta solution to differential equations'''
    
    y = np.array([])
    y = np.append(y, y0)

    for ix in range(1, len(x)):
        h = x[ix] - x[ix - 1]
        k1 = function(x[ix - 1], y[ix - 1])
        k2 = function(x[ix - 1] + h/2, y[ix - 1] + h*k1/2)
        k3 = function(x[ix - 1] + h/2, y[ix - 1] + h*k2/2)
        k4 = function(x[ix - 1] + h, y[ix - 1] + h*k3)
              
        y = np.append(y, y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4))

    return y

def SolveIVP(x,y0):
    solver = solve_ivp(function, t_span = [x[0], x[-1]], y0 = [y0], method='RK45', t_eval= x)
    return solver.y[0]

if __name__ == '__main__':

    x = np.linspace(2, -2, 50, endpoint = True)
    y0 = 4

    fig = plt.figure(figsize = (20, 10))
    fig.tight_layout(pad = 5)
    
    plt.plot(x, (x**3), color = 'black', linestyle = '-', label = None, visible = True)
    plt.plot(np.linspace(2, -2, 200, endpoint = True), EulerCromer(np.linspace(2, -2, 200, endpoint = True), y0), color = 'blue', linestyle = '-.', label = None, visible = True)       # plots the potential waveform from waveforms.py on the left-hand subplot
    plt.plot(x, RungeKutta(x, y0), color = 'red', linestyle = ':', label = None, visible = True) 
    plt.plot(x, SolveIVP(x, y0), color = 'green', linestyle = '--', label = None, visible = True)


    plt.show()
    plt.close()

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:26:22 2024

@author: johnl
"""
import numpy as np
import main as main
#Newton Raphson Method of Solution 

def derivative(xbar, func, dt, params, node_ids):
    delta = dt
    derivatives = []
    for i, x in enumerate(xbar):
        #get the finite difference for a single function 
        pertXbar = xbar.copy()
        pertXbar[i] = x + delta
        #Evaluate f(x+dx)
        pertUp = func.int_nrg(params, pertXbar, node_ids)
        pertXbar[i] = x - delta
        #Evaluate f(x-dx)
        pertDown = func.int_nrg(params, pertXbar, node_ids)
        #To approximate the first derivative, we use the following equation:
        #  f'(x) = (f(x+dx) - f(x-dx) ) / (2dx)
        firOrdDiff = (pertUp - pertDown) / (2*delta)
        derivatives.append(firOrdDiff)
    return derivatives 

def iterate(inputs, funcs, params, node_ids, epsilon=1e-2, dt = 0.000000001, loops = None, core = None):
    diffScore = 100e100
    it = 0
    epsilon = epsilon
    hist = [inputs]
    stop_cri = 50e10
    converge = False
    while diffScore >= epsilon and it < 50e10:
        jacobian = np.zeros((len(inputs), len(funcs)))
        F = np.zeros((len(funcs), 1))
        for i, func in enumerate(funcs):
            dervs = derivative(inputs, func, dt, params, node_ids)
            jacobian[i][:] = np.array(dervs)
            F[i][0] = func.int_nrg(params, inputs, node_ids) 
        invertJacob = np.linalg.inv(jacobian)
        diff = np.matmul(invertJacob, -F)
        diffScore = np.linalg.norm(diff)/np.linalg.norm(inputs)
        inputsold = inputs
        it += 1
        inputs = [inp + diff[j][0] for j, inp in enumerate(inputs)] 
        #main.update_state(params, loops, core)
        diffScore = np.linalg.norm(np.array(inputs)-np.array(inputsold))
        hist.append(inputs)
    if it < stop_cri:
        converge = True
        
    return inputs, it, hist, converge  
   
def pctderivative(xbar, func, dt):
    delta = dt
    derivatives = []
    for i, x in enumerate(xbar):
        #get the finite difference for a single function 
        pertXbar = xbar.copy()
        pertXbar[i] = x + delta
        #Evaluate f(x+dx)
        pertUp = func.calc(pertXbar)
        pertXbar[i] = x - delta
        #Evaluate f(x-dx)
        pertDown = func.calc(pertXbar)
        #To approximate the first derivative, we use the following equation:
        #  f'(x) = (f(x+dx) - f(x-dx) ) / (2dx)
        firOrdDiff = (pertUp - pertDown) / (2*delta)
        derivatives.append(firOrdDiff)
    return derivatives 

def pctiterate(inputs, funcs, epsilon=1e-2, dt = 0.000000001, loops = None, core = None):
    diffScore = 100e100
    it = 0
    epsilon = epsilon
    hist = []
    stop_cri = 50e10
    converge = False
    while diffScore >= epsilon and it < 50e10:
        jacobian = np.zeros((len(inputs), len(funcs)))
        F = np.zeros((len(funcs), 1))
        for i, func in enumerate(funcs):
            dervs = pctderivative(inputs, func, dt)
            jacobian[i][:] = np.array(dervs)
            F[i][0] = func.calc(inputs) 
        invertJacob = np.linalg.inv(jacobian)
        diff = np.matmul(invertJacob, -F)
        diffScore = np.linalg.norm(diff)/np.linalg.norm(inputs)
        inputsold = inputs
        it += 1
        inputs = [inp + diff[j][0] for j, inp in enumerate(inputs)] 
        #main.update_state(params, loops, core)
        diffScore = np.linalg.norm(np.array(inputs)-np.array(inputsold))
        hist.append(inputs)
    if it < stop_cri:
        converge = True
        
    return inputs, it, hist, converge  
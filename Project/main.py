# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:47:06 2024

@author: johnl
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
import copy
from pyXSteam.XSteam import XSteam
from pyfluids import Fluid, FluidsList, Input
#User Defined classes/functions

import parameters as st
gc = 32.174*3600**2

def find_newTC(T, data):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    return data[1] - steamTable.u_pt(data[0], T)

def get_rho(node, P):
    """
    Return the density rho_k for a node

    Parameters
    ----------
    node : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    Returns
    -------
    rho : TYPE
        DESCRIPTION.

    """
    
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    T = node.T 
    rho = steamTable.rho_pt(P, T)
    return rho
def f(node):
    reynolds = node.Re
    inv_roughness = 1 / (0.00005/node.D)
    param_a = 1 / (1 + (reynolds / 2712)**8.4)
    param_b = 1 / (1 + (reynolds / (150 * inv_roughness))**1.8)
    exponent_a = 2 * (param_a - 1) * param_b
    exponent_b = 2 * (param_a - 1) * (1 - param_b)
    friction = (64 / reynolds)**param_a * (0.75 * np.log10(reynolds / 5.37))**exponent_a * (0.88 * np.log10(6.82 *inv_roughness))**exponent_b
    return friction
def update_node(params, node, bal_u = -123456):
    #Update the T, v, Re, Pr u for node
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    if bal_u == -123456:
        node.set_nrg(steamTable.u_pt(params.rv.p, params.rv.T_out), params.rv.p)
        nrg =  steamTable.u_pt(params.rv.p, params.rv.T_out) * 2325.99999994858 #btu/lbm to j/kg
    else:
        node.set_nrg(bal_u, params.rv.p)
        nrg = bal_u *2325.99999994858 #btu/lbm to j/kg
    p = params.rv.p * 6894.75729 #psig to Pa
    TC = Fluid(FluidsList.Water).with_state(Input.pressure(p), Input.internal_energy(nrg)).temperature
    #TC = scipy.optimize.fsolve(find_newTC, params.rv.T_out, args=[params.rv.p, nrg])[0]

    node.T = TC#TC*9/5+32
    node.v = steamTable.my_pt(params.rv.p, node.T) # lb/ft-h
    node.Re = (node.m*node.D)/(node.A*node.v)
    node.Pr = node.v*steamTable.Cp_pt(params.rv.p, node.T)/steamTable.tc_pt(params.rv.p, node.T)
    node.f = f(node)  

def update_state(params, loops, core):
    #Updates the mass flux and the T, velocity, reynolds number, and u values
    for loop in loops:
        for node in loop.loop:
            node.m = loop.m_flux
            update_node(params, node, node.nrg)
    for node in core.core:
        node.m = core.m_flux
        update_node(params, node, node.nrg)        

#Get the k factors for your loop nodes
def calc_loop_ks(params, nodes, flow = 1):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    for node in nodes:
        next_node = node.next
        d1 = node.D 
        d2 = next_node.D
        #First assume straight pipe
        if d1 > d2:
            node.k = 1/2*(1-(d2/d1)**2)
        else:
            node.k = (1-(d1/d2)**2)
        #We are going into the dc
        if next_node.n == 16:
            node.k += 1 
        #We are going around an elbow 
        elif next_node.n in [10, 11]:
            node.k += 0.3
        node.m = params.rv.mass_flux*(52.74)/4    
        update_node(params, node)


#Get your k factors for your core nodes 
def calc_core_ks(params, nodes):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    for node in nodes:
        #Fix the k of the inlet and outlet nodes
        if node.n in [5, 16]:
            node.k = 4.25
        else:
            node.k = params.rv.grid_loss*params.rv.spacers/4
        node.m = params.rv.mass_flux*52.74 
        update_node(params, node)
               
#Generate initial conditions
def fill_params(params):       
    rv = st.reactor_vessel()
    sg = st.steam_generator()
    params.rv = rv
    params.sg = sg

#Create your very own PWR!
def make_pwr(num_loops):
    #Initiate the state from problem params
    all_params = st.params()  
    fill_params(all_params)
    #Inititate core nodes based on geometry 
    core_nodes = [16, 1, 2, 3, 4, 5]
    core = st.core(core_nodes, all_params)
    #Initiate nodes based on geometry of problem
    loop_list = []
    loops_nodes = [6,7,8,9,10,11,12,13,14,15]
    for n in range(0,num_loops):
        loop = st.loop(loops_nodes, all_params)
        loop.loop[-1].next = core.core[0]
        loop.loop[0].prev = core.core[-1]
        calc_loop_ks(all_params, loop.loop)
        loop_list.append(loop)
        
    core.core[0].prev = [loop_list[0].loop[-1], loop_list[1].loop[-1]]
    core.core[-1].next = [loop_list[0].loop[0], loop_list[1].loop[0]]
    calc_core_ks(all_params, core.core)
    return core, loop_list, all_params
 
#directly solve for new mass flux iterative values
def step_massflux(params, loops, core):
    a = []
    b = []
    n = params.n_loops
    for loop in loops:
        a1, b1 = loop.momentum(params)
        a.append(a1)
        b.append(b1)
    ac, bc = core.momentum(params)
    m1_dt = (b[0]*a[1]+(n-1)*b[0]*ac-(n-1)*b[1]*ac+a[1]*b[1]) / (a[0]*a[1]+(n-1)*a[0]*ac+a[1]*ac)
    dP_dt = b[0]-a[0]*m1_dt
    m2_dt = (b[1]-dP_dt)/a[1]
    mc_dt = m1_dt + (n-1)*m2_dt
    loops[0].m_flux = m1_dt
    loops[1].m_flux = m2_dt
    core.m_flux = mc_dt
    core.dp = dP_dt
    update_state(params, loops, core)
    return params, loops, core  
    
def balance_nrg(params, loops, core):
    inputs = []
    funcs = []
    node_ids = []
    loop = 1 
    node0 = loops[0].loop[0]
    inputs.append(node0.nrg)
    funcs.append(node0)
    node_ids.append(node0.node_id)
    node0 = node0.next
    while node0.n != 16 or loop < 2:
        inputs.append(node0.nrg)
        funcs.append(node0)
        node_ids.append(node0.node_id)
        node0 = node0.next
        if type(node0) == list:
            node0 = node0[loop]
            loop += 1
       
    inputs, it, hist, converge = iterate(inputs, funcs, params, node_ids, 1e-5, params.dt, loops, core)  
    #print(inputs, it, converge)
    loop = 1 
    node0 = loops[0].loop[0]
    update_node(params, node0, inputs[0])
    node0 = node0.next
    i = 1
    while node0.n != 16 or loop < 2:
        update_node(params, node0, inputs[i])
        node0 = node0.next
        i += 1
        if type(node0) == list:
            node0 = node0[loop]
            loop += 1
    return core, loops, params, node_ids

def get_dP_pump(core, loops, params):
    p = params.rv.p 
    
    node_p = [p]
    dps = []
    for node in core.core:
        dP = node_p[-1] - (node.k+node.f*node.l/node.D)*0.5*(node.m/(node.get_rho(p)*node.A*gc))**2 - node.get_rho(p)*node.dH#/144
        dps.append(-(node.k+node.f*node.l/node.D)*0.5*(node.m/(node.get_rho(p)*node.A*gc))**2 - node.get_rho(p)*node.dH/144)
        node_p.append(dP)
    for node in loops[0].loop:
        dP = node_p[-1] - (node.k+node.f*node.l/node.D)*0.5*(node.m/(node.get_rho(p)*node.A*gc))**2 - node.get_rho(p)*node.dH#/144
        dps.append(- (node.k+node.f*node.l/node.D)*0.5*(node.m/(node.get_rho(p)*node.A*gc))**2 - node.get_rho(p)*node.dH/144)
        node_p.append(dP)       
        
    #dP_loop = loops[0].get_dP(all_params)
    #dP_core = core.get_dP(all_params)
    pump_dP = node_p[0] - node_p[-1]
    print(pump_dP)
    loops[0].get_pump_curve(params, pump_dP)
    loops[1].get_pump_curve(params, pump_dP)      
    
def grapher(core, loops, params):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    show = []
    node_ids = []
    loop = 1 
    node0 = loops[0].loop[0]
    #cur_p = params.rv.p
    #UPDATE THIS LINE TO SHOW THINGS
    show.append(node0.T)
    #cur_p -= node0.calc_dP(params.rv.p)[0]
    node_ids.append(node0.node_id)
    node0 = node0.next
    while node0.n != 16 or loop < 2:
        show.append(node0.T)
       # cur_p -= node0.calc_dP(params.rv.p)[0]
        node_ids.append(node0.node_id)
        node0 = node0.next
        if type(node0) == list:
            node0 = node0[loop]
            loop += 1 
    #plt.hlines(steamTable.h_pt(params.rv.p, params.rv.T_out), 0, 26)
    plt.scatter(node_ids, show)
    #print(show)
    #plt.ylim(min(show)-1, max(show)+1)
    plt.show()

def get_loop_list(loops):
    show = []
    node_ids = []
    loop = 1 
    node0 = loops[0].loop[0]
    #UPDATE THIS LINE TO SHOW THINGS
    show.append(node0)
    node_ids.append(node0.node_id)
    node0 = node0.next
    while node0.n != 16 or loop < 2:
        show.append(node0)
        node_ids.append(node0.node_id)
        node0 = node0.next
        if type(node0) == list:
            node0 = node0[loop]
            loop += 1 
    return show

def find_eta(core, loops, params):
    init_core = copy.copy(core)
    init_loops = copy.copy(loops)
    run_secs = 1
    all_params.dt = 0.1
    its = int(round(run_secs/all_params.dt, 0))
    mflux = [loops[0].loop[0].nrg]
    diff = 1e10
    it = 0
    while np.abs(diff) > 1e-3 and it < 1e4:
        for i in range(0, its):
            params, loops, core = step_massflux(all_params, loops, core)
            core, loops, params, node_ids = balance_nrg(params, loops, core)
            mflux.append(loops[0].loop[0].nrg)
            it += 1 
        #Get to steady state difference
        diff = mflux[-1] - mflux[-2]
        if diff < 0:
            all_params.sgf += 0.001
        else: 
            all_params.sgf -= 0.001  
        loops = init_loops
        core = init_core
    return all_params.sgf
    
if __name__ == '__main__':
    from newtonRaphson import iterate 
    #assemble the reactor for the problem 
    core, loops, all_params = make_pwr(2)
    for loop in loops:
        loop.rcp_p_r = 4225 #guess and check method
    core, loops, all_params, node_ids = balance_nrg(all_params, loops, core)
    loop_list = get_loop_list(loops)
    #get_dP_pump(core, loops, all_params)
    all_params, loops, core = step_massflux(all_params, loops, core)
    #grapher(core, loops, all_params)
    #sgf = find_eta(core, loops, all_params)
    #print(sgf)
    all_params.sgf = 0.001#0.0022308#sgf
    run_secs = 1
    its = int(round(run_secs/all_params.dt, 0))
    mflux = [loops[0].loop[0].m]
    diff = 1e10
    it = 0
    #while np.abs(diff) > 1e-3 and it < 1e4:
    for i in range(0, its):
        all_params, loops, core = step_massflux(all_params, loops, core)
        core, loops, all_params, node_ids = balance_nrg(all_params, loops, core)
        mflux.append(loops[0].loop[0].m)
        diff = mflux[-1] - mflux[-2]
        it += 1 

        #print(mflux[-1])
        
            
    print(loops[0].m_flux, loops[0].m_flux_0)
    print(core.m_flux, all_params.rv.mass_flux*52.74)
    print(loops[1].m_flux)
    plt.plot(mflux)
    plt.show()
    #grapher(core, loops, all_params)








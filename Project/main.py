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
import pickle as pkl
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
        node.set_nrg(round(steamTable.u_pt(params.p, params.rv.T_out-5),4), params.p)
        nrg =  steamTable.u_pt(params.p, params.rv.T_out) * 2325.99999994858 #btu/lbm to j/kg
    else:
        node.set_nrg(round(bal_u,4), params.p)
        nrg = bal_u *2325.99999994858 #btu/lbm to j/kg
    p = params.p * 6894.75729 #psig to Pa
    TC = Fluid(FluidsList.Water).with_state(Input.pressure(p), Input.internal_energy(nrg)).temperature
    #TC = scipy.optimize.fsolve(find_newTC, params.rv.T_out, args=[params.rv.p, nrg])[0]
#params.rv.T_out#
    node.T = TC*9/5+32#params.rv.T_out#TC*9/5+32
    node.v = steamTable.my_pt(params.p, node.T) # lb/ft-h
    node.Re = (node.m*node.D)/(node.A*node.v)
    node.Pr = node.v*steamTable.Cp_pt(params.p, node.T)/steamTable.tc_pt(params.p, node.T)
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
    params.p = rv.p
    params.mw = rv.MW
    params.sg_tubes = sg.sg_tubes
    

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
    
    diff = 10e10
#    new_inputs = [i for i in funcs]
    org_nrg = [i.nrg for i in funcs]
    
    while diff > 0.001:
        diffs = []
        data  = []
        start_nrg = [i.nrg for i in funcs]
        #prev_nrg = [i.prev.nrg for i in funcs]
        #print('---')round()
        for i, node in enumerate(funcs):
            prev_node = node.prev 
            q = 0
            if node.name == 'sg':
                q = node.q_dot_sg(params)*20
            elif node.name == 'core':
                q = node.q_dot_core(params)
            grow = org_nrg[i]*node.l*node.A*node.get_rho(params.p)/(params.dt/(60*60))#/100
            grow_nou = node.l*node.A*node.get_rho(params.p)/(params.dt/(60*60))#/100
            if node.name == 'up':
                new_u = (q+prev_node.m*prev_node.nrg+grow)/(node.m+grow_nou)
            elif node.name == 'lp':
                new_u = (q+node.prev[0].m*node.prev[0].nrg+node.prev[1].m*3*node.prev[1].nrg+grow)/(node.m+grow_nou)
                new_u = (prev_node[0].nrg + 3*prev_node[1].nrg)/4
            elif prev_node.name == 'up':
                new_u = prev_node.nrg#(q+node.prev.m*node.prev.nrg/4+grow)/(node.m+grow_nou)
            elif node.name == 'cl' or node.name == 'dc':
                new_u = prev_node.nrg
            else:
                new_u = (q+prev_node.m*prev_node.nrg+grow)/(node.m+grow_nou)
            node.nrg = round(new_u,4)
            update_node(params, node, round(new_u,4))
            diffs.append(new_u)
            if len(node.prev) > 1:
                data.append((node.prev[0].node_id, node.prev[1].node_id, node.node_id))
            else:
                data.append((node.prev.node_id,node.node_id))
        diff = np.linalg.norm(np.array(start_nrg)-np.array(diffs))
        #print('---')
    #print('break')
    inputs = [i.nrg for i in funcs]
    #inputs, it, hist, converge = iterate(inputs, funcs, params, node_ids, 1e-5, params.dt, loops, core)  
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
    p = params.p 
    
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

def calc_clad_temp(node, params):
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    q = node.q_dot_core(params)
    wall = params.rv.fuel_rods*np.pi*params.rv.r_d*node.l
    C = 0.042*(params.rv.rod_pitch/params.rv.r_d)-0.024
    Nu = C * node.Re**0.8*node.Pr**(1/3)
    hc = steamTable.tc_pt(params.p, node.T)*Nu/node.D
    wall_T = q/(hc*wall) + node.T
    return wall_T

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
    all_params.sgf = 0.0100#1
    all_params.rcp_l1 = 5570
    all_params.rcp_l2 = 5570
    all_params.sg_tubes = all_params.sg.sg_tubes
    for loop in loops:
        loop.rcp_p_r = 5570#4253 #guess and check method
    core, loops, all_params, node_ids = balance_nrg(all_params, loops, core)
    loop_list = get_loop_list(loops)
    all_params.ptrip = False
    loops[0].ptrip = False
    #loops[0].rcp_p_r = 1e-5

    loops[1].ptrip = False
    all_params.beta = 3.8
    
    with open('steady_state.pkl', 'rb') as fid:
        state_data = pkl.load(fid)   
        all_params, loops, core = state_data
    
    time_sec = [0]
    st_time = 0
    all_nodes_list = []
    loop_id = 1
    node_first = loops[0].loop[0]
    all_nodes_list.append(node_first)
    node0 = node_first.next
    while node0.n != 16 or loop_id < 2:
        all_nodes_list.append(node0)
        node0 = node0.next
        if type(node0) == list:
            node0 = node0[loop_id]
            loop_id += 1
    all_params.sgf = 0.0097#100#0.0042
    run_secs = 13
    its = int(round(run_secs/all_params.dt, 0))
    mflux = [loops[0].loop[0].m]
    mflux2 = [loops[1].loop[0].m]
    nrg = [core.core[-1].T]
    diff = 1e10
    it = 0
    #while np.abs(diff) > 1e-3 and it < 1e4:
 #   for i in range(st_time, its):
 #       all_params, loops, core = step_massflux(all_params, loops, core)
 #       core, loops, all_params, node_ids = balance_nrg(all_params, loops, core)
 #       mflux.append(loops[0].loop[0].m)
 #       mflux2.append(loops[1].loop[0].m)
 #       nrg.append(loops[0].loop[0].T)
 #       diff = mflux[-1] - mflux[-2]
 #       it += 1 
 #       time_sec.append(all_params.dt*it)
 #       all_params.t = all_params.dt*it
 #       if np.mod(it, 1000) == 0:
 #           plt.plot([j.n for j in all_nodes_list], [j.nrg for j in all_nodes_list], label=f'{it}')
 #           print(mflux[-1])
 #   plt.plot([i.n for i in all_nodes_list], [i.nrg for i in all_nodes_list], label=f'{it}')
#    state_data = (all_params, loops, core)
#    with open('steady_state.pkl', 'wb') as fid:
#        pkl.dump(state_data, fid)
    wall_Ts = []
    wall_Ts.append( [calc_clad_temp(i, all_params) for i in core.core[1:-1]])
    wall_Ts.append( [calc_clad_temp(i, all_params) for i in core.core[1:-1]])
    Tcs = [(loops[0].loop[-1].T, loops[1].loop[-1].T)]
    Tcs.append((loops[0].loop[-1].T, loops[1].loop[-1].T))
    #plt.figure()
    mflux.append(loops[0].loop[0].m)
    mflux2 .append(loops[1].loop[0].m)
    nrg .append(core.core[-1].T)
    time_sec .append(all_params.t)
    pump_trip = all_params.t
    print(core.m_flux, all_params.rv.mass_flux*52.74)
    print(pump_trip)
    it = int(all_params.t/all_params.dt)
    st_time += it
    #initialize pump trip
    all_params.ptrip = True
    loops[0].ptrip = True
    #loops[0].rcp_p_r = 1e-5

    loops[1].ptrip = False
    all_params.beta = 3.8 #0.16 for peak core temp

    #plug sg holes:
    perc_plug = 1#.845#0.9
    all_params.sg_tubes = all_params.sg.sg_tubes*perc_plug
    loop_id = 1
    node0 = loops[0].loop[0]
    while node0.n != 16 or loop_id < 2:
        print(node0.node_id, node0.name)
        if node0.name == 'sg' and node0.node_id < 15:
            #node0.LD += node0.LD*1/perc_plug
            #node0.k += node0.k*1/perc_plug
            node0.A = node0.A * perc_plug 
            print('new K', node0.k, node0.LD)
        node0 = node0.next
        if type(node0) == list:
            node0 = node0[loop_id]
            loop_id += 1 
    

    run_secs = 5
    its = int(round(run_secs/all_params.dt, 0))
    max_iter = its+st_time
    print('Starting Run: ', end='')

    for i in range(0+st_time, its+st_time):
        step_massflux(all_params, loops, core)
        balance_nrg(all_params, loops, core)
        mflux.append(loops[0].loop[0].m)
        mflux2.append(loops[1].loop[0].m)
        nrg.append(core.core[-1].T)
        Tcs.append((loops[0].loop[-1].T, loops[1].loop[-1].T))
        it += 1    
        all_params.ptime += all_params.dt
        all_params.t = all_params.dt*it  
        time_sec.append(it*all_params.dt)
        wall_Ts.append( [calc_clad_temp(i, all_params) for i in core.core[1:-1]])
        if time_sec[-1] == float(15) and all_params.ptrip:
            all_params.trip = True
        if all_params.trip == True:
            all_params.time_since_scram += all_params.dt


    print('|')
    print('final l1 mass flow', loops[0].m_flux)
    print('final l2 mass flow',loops[1].m_flux)
    print('%reduction', loops[0].m_flux/loops[0].m_flux_0)
    print('Num Clogged tubes', all_params.sg.sg_tubes-all_params.sg_tubes)
    print(loops[0].loop[0].m, core.core[-1].T) 
    #grapher(core, loops, all_params)
    #    plt.legend()
    #plt.legend()
    min_x = pump_trip-1 
    max_x = all_params.t
    steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    Tsat = steamTable.tsat_p(all_params.p)
    print(f' Core inlet temp: {core.core[0].T} Core exit temp: {core.core[-1].T}')
    print(nrg[-1], core.core[-1].T)       
    print(loops[0].m_flux, loops[0].m_flux_0)
    print(core.m_flux, all_params.rv.mass_flux*52.74)
    print(loops[1].m_flux)
    T1 = []
    T2 = []
    T3 = []
    T4 = []
    for it in wall_Ts:
        T1.append(it[0])
        T2.append(it[1])
        T3.append(it[2])
        T4.append(it[3])
    plt.figure()
    L1 = []
    L2 = []
    for tempy in Tcs:
        L1.append(tempy[0])
        L2.append(tempy[1])
    
    plt.figure()
    plt.plot(time_sec, L1, label = 'Loop 1 Tc [F]')
    plt.plot(time_sec, L2, label = 'Loop 2 Tc [F]')    
    plt.vlines(13, min(L1), max(L1),  label = 'Pump Trip', color = 'black', linestyle='--', alpha = 0.3)
    plt.vlines(15, min(L1), max(L1),  label = 'Scram', color = 'black', linestyle='--', alpha = 0.3)
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [F]')
    plt.xlim(min_x, max_x)
    plt.legend()
    plt.show()
    plt.figure()
    plt.title(f'One Pump Locked Rotor, Beta: {all_params.beta}')
    plt.plot(time_sec, T1, label = 'Node 1 Clad T [F]')
    plt.plot(time_sec, T2, label = 'Node 2 Clad T [F]')
    plt.plot(time_sec, T3, label = 'Node 3 Clad T [F]')
    plt.plot(time_sec, T4, label = 'Node 4 Clad T [F]')
    plt.vlines(13, min(T1), Tsat, label = 'Clogged Tubes', color = 'black', linestyle='--', alpha = 0.3)
    #plt.vlines(15, min(T1), Tsat, label = 'Scram', color = 'black', linestyle='--', alpha = 0.3)
    plt.hlines(Tsat, time_sec[0], time_sec[-1], label = f'Saturation Temp = {round(Tsat,1)} [F]', linestyle='--', color='r')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [F]')
    plt.xlim(min_x, max_x)
    plt.legend()
    plt.figure()
    plt.plot(time_sec, mflux, label = 'Loop 1 Mass Flow [lbm/hr]')
    plt.plot(time_sec, mflux2, label = 'Loop 2 Mass Flow [lbm/hr]')
    plt.vlines(13, min(mflux), mflux2[-1],  label = 'Clogged Tubes', color = 'black', linestyle='--', alpha = 0.3)
    #plt.vlines(15, min(mflux), mflux2[-1], label = 'Scram', color = 'black', linestyle='--', alpha = 0.3)
    plt.hlines(loops[0].m_flux_0*0.9, time_sec[0], time_sec[-1], label = '90% Rated Mass Flow [lbm/hr]', linestyle='--', color='r')
    plt.xlabel('Time [s]')
    plt.ylabel('Mass Flow [lbm/hr]')
    plt.xlim(min_x, max_x)
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(time_sec, nrg, label = 'Core Exit Temp [F]')
    plt.hlines(Tsat, time_sec[0], time_sec[-1], label = f'Saturation Temp = {round(Tsat,1)} [F]', linestyle='--', color='r')
    plt.vlines(13, min(nrg), Tsat, label = 'Clogged Tubes', color = 'black', linestyle='--', alpha = 0.3)
    #plt.vlines(15, min(nrg), Tsat, label = 'Scram', color = 'black', linestyle='--', alpha = 0.3)
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [F]')
    plt.xlim(min_x, max_x)
    plt.legend()
    plt.show()








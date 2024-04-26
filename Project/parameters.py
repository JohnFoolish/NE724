# -*- coding: utf-8 -*-
"""
Main Script for the project
"""
from pyXSteam.XSteam import XSteam
import numpy as np
gc = 32.174*3600**2 #to get to lbm/ft-hr^2 
g = 32.174*3600**2
class params():
    def __init__(self):
        self.node_id = 0 
        self.n_loops = 4
        self.dt = 1e-2
        self.t = 0
        self.rcp_l1 = None
        self.rcp_l2 = None
        self.p = None
        self.trip = False
        self.ptrip = False
        self.ptime = 0
        self.time_since_scram = 0
        self.rv = None
        self.up = None
        self.lp = None
        self.sg = None
        self.sgf = None
        self.mw = None
        self.beta = None
        self.adjust = 1
        self.sg_tubes = None
        self.loops = []
        
        self.all_nodes = {1: ("core",3.0, 0.4635, 52.74, 4988.86, 3.0, -1),
                 2: ("core", 3.0, 0.4635, 52.74, 4988.86, 3.0, -1),
                 3: ("core", 3.0, 0.4635, 52.74, 4988.86, 3.0, -1),
                 4: ("core", 3.0, 0.4635, 52.74, 4988.86, 3.0, -1),
                 5: ("up", 1.5, 158, 136.2, 0, 1.5, 4.25),
                 6: ("hl", 20, 29, 4.59, 0, 0, 1),
                 7: ("sg", 9.54, 0.6075, 13.35, 1054.9, 9.54, 1),
                 8: ("sg", 9.54, 0.6075, 13.35, 1054.9, 9.54, 1),
                 9: ("sg", 9.54, 0.6075, 13.35, 1054.9, 9.54, 1),
                 10: ("sg", 9.54, 0.6075, 13.35, 1054.9, 0, 1),
                 11: ("sg", 9.54, 0.6075, 13.35, 1054.9, -9.54, 1),
                 12: ("sg", 9.54, 0.6075, 13.35, 1054.9, -9.54, 1),
                 13: ("sg", 9.54, 0.6075, 13.35, 1054.9, -9.54, 1),
                 14: ("cl", 40, 27.5, 4.12, 0, 0, 1),
                 15: ("dc", 18.4, 15, 6.77, 0, -13.5, 1),
                 16: ("lp", 7.2, 173, 163.2, 0, 0, 4.25)}
        self.PDF = {1: 0.586,
               2: 1.414,
               3: 1.414,
               4: 0.586}
        self.core_nodes = len(self.PDF)
        

class fuel_rod():
    def __init__(self):
        self.r_d = 0.374/12 
        self.c_t = 0.0225/12 
        self.gap_cond = 10000 
        self.k_c = 9.6
        self.S=0.506/12        
        self.r_f = self.r_d/2 - self.c_t 
    
class reactor_vessel():
    def __init__(self):
        self.MW = 3411
        self.core_height = 144/12 #ft
        self.rod_locs = 55777
        self.fuel_rods = 50927
        self.r_d = 0.374/12 #ft
        self.p_d = 0.3225/12 #ft
        self.c_t = 0.0225/12 #ft
        self.gap_cond = 1000 #BTU/hr-ft^2-F
        self.k_c = 9.6 #BTU/hr-ft-F
        self.rod_pitch = 0.496/12 #ft
        self.spacers = 8
        self.grid_loss = 0.5
        self.mass_flux = 2.48e6 #lbm/hr-ft^2
        self.T_in = 552 #F
        self.T_out = 616 #F
        self.p = 2250 #psia
        self.core_in = 4.25 
        self.core_exit = 4.25        
class upper_plenum():
    def __init__(self):
        #Upper Plenum
        self.pn_l = 1.5 #ft
        self.pn_ed = 158/12 #ft
        self.pn_vol = 1373.7 #ft^3
class hot_leg():
    def __init__(self):
        #HL
        self.hl_n = 4 
        self.hl_l = 20 #ft
        self.hl_d = 29/12 #ft
        self.hl_LD = 20
        self.hl_loss_in = 0.5
        self.hl_loss_exit = 1
class steam_generator():
    def __init__(self):
        #SG 
        self.sg_n = 4
        self.sg_p = 1000 #psia
        self.sg_tubes = 6633 
        self.sg_t_id = 0.6075/12 #ft
        self.sg_t_l = 66.8 #ft
        self.sg_LD = 55
        self.sg_loss_in = 0.5
        self.sg_loss_exit = 1
        self.sg_node_l = None
        self.sg_node_num = None
class cold_leg():
    def __init__(self):
        #CL
        self.cl_n = 4
        self.cl_l = 40 #ft
        self.cl_d = 27.4/12 #ft
        self.cl_LD = 18
        self.cl_loss_in = 0.5
        self.cl_loss_exit = 4.6
#class downcomer():
#    def __init__(self):
        #Downcomer
        self.dc_ID = 173/12 #ft
        self.dc_OD = 158/12 #ft
        self.dc_l = 18.4 #ft
class lower_plenum():
    def __init__(self):
        #Lower PLenum
        self.lp_l = 7.2 #ft
        self.lp_d = 173/12 #ft
        self.ld_v = 784.4 #ft^3

class node_base():
    def __init__(self, i, node_list, node_id):
        steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
        name, l, D, A, P, dH, k2 = node_list
        self.node_id = node_id
        self.n = i #node
        self.l = l #ft
        self.D = D/12 #ft
        self.A = A #ft^2
        self.P = P #ft
        self.dH = dH #ft
        self.name = name
        self.sgf = 0.0002308
        self.LD = None
        self.p = None
        self.m = None
        self.Re = None
        self.next = None
        self.prev = None
        self.nrg = None #internal energy
        self.T = None #F
        self.k = None
        self.f = None #Darby Friction
        self.rho = None
        self.m = None
        self.dP = None
        self.Pr = None
        self.og_nrg = None
    def __len__(self):
        return 1
    def set_nrg(self, new_nrg, P):
        steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
        self.nrg = new_nrg
    def int_nrg(self, params, inputs, node_ids):
        dt = params.dt/(60*60)
        P = params.p
        rho = self.get_rho(P)
        V = self.l*self.A
        u = self.nrg
        if self.name == 'sg':
            q = self.q_dot_sg(params)
        elif self.name == 'core':
            q = self.q_dot_core(params)
        else:
            q = 0
        counter, = np.where(np.array(node_ids) == self.node_id)[0]
        if len(self.next) > 1:
            count1, = np.where(np.array(node_ids) == self.next[0].node_id)[0]
            count2, = np.where(np.array(node_ids) == self.next[1].node_id)[0]
            count_m, = np.where(np.array(node_ids) == self.prev.node_id)[0]
            u_dt_p = inputs[count1] 
            u_dt_p2 = inputs[count2]*3
            m_dt = self.m
            u_dt = inputs[counter]
            u_dt_m = inputs[count_m]
            val = V*rho*(u_dt-u)/dt + m_dt/2*(1-np.abs(m_dt)/m_dt)*u_dt_p + m_dt/2*(1-np.abs(m_dt)/m_dt)*u_dt_p2+np.abs(m_dt)*u_dt-m_dt/2*(1+np.abs(m_dt)/m_dt)*u_dt_m-q 
        elif len(self.prev) > 1:
            count1, = np.where(np.array(node_ids) == self.prev[0].node_id)[0]
            count2, = np.where(np.array(node_ids) == self.prev[1].node_id)[0]
            count_p, = np.where(np.array(node_ids) == self.next.node_id)[0]
            u_dt_m = inputs[count1] 
            u_dt_m2 = inputs[count2]*3
            m_dt = self.m
            u_dt = inputs[counter]
            u_dt_p = inputs[count_p]
            val = V*rho*(u_dt-u)/dt + m_dt/2*(1-np.abs(m_dt)/m_dt)*u_dt_p+np.abs(m_dt)*u_dt-m_dt/2*(1+np.abs(m_dt)/m_dt)*u_dt_m-m_dt/2*(1+np.abs(m_dt)/m_dt)*u_dt_m2-q 
        else:
            counter_p, = np.where(np.array(node_ids) == self.next.node_id)[0]
            counter_m, = np.where(np.array(node_ids) == self.prev.node_id)[0]
            m_dt = self.m    
            u_dt = inputs[counter]
            u_dt_p = inputs[counter_p]
            u_dt_m = inputs[counter_m]
            #if self.next.n == 15:
            #    u_dt_p = inputs[0]
            #    u_dt_m = inputs[counter-1]            
            
        #if counter == len(node_ids)-1:
        #    u_dt_p = inputs[0]
        #    u_dt_m = inputs[counter-1]
        #elif counter == 0:
        #    u_dt_m = inputs[-1]
        #    u_dt_p = inputs[counter+1]
        
        #else:
        #    u_dt_m = inputs[counter-1]
        #    u_dt_p = inputs[counter+1]
        val = V*rho*(u_dt-u)/dt + m_dt/2*(1-np.abs(m_dt)/m_dt)*u_dt_p+np.abs(m_dt)*u_dt-m_dt/2*(1+np.abs(m_dt)/m_dt)*u_dt_m-q 
        return val#/10000000
    
    def get_rho(self, P):
        #return fluid density of node based on pressure 
        steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
        T = self.T
        rho = steamTable.rho_pt(P, T)
        return rho
    def q_dot_sg(self, params):
        #use for SG nodes to get the heat flux out
        steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
        P = params.sg.sg_p
        Tsat = steamTable.tsat_p(P)
        #Calculate heat exchanger area
        k_water = steamTable.tc_pt(params.p, self.T)
        h = k_water*(0.023*self.Re**0.8*self.Pr**3)/self.D#+1000
        sg_tubes = params.sg.sg_tubes
        if self.node_id < 15:
            sg_tubes = params.sg_tubes
        
        AX = (sg_tubes*params.sg.sg_t_id*np.pi*self.l)/(1/h + params.sgf)        
        qdot = AX*(Tsat - self.T)
        return qdot#*50
    def q_dot_core(self, params):
        #use for core nodes to get the heat flow in 
        MW = params.rv.MW*3412142.450123 #BTU /hr
        pdf = params.PDF[self.n]
        n = params.core_nodes
        if params.trip:
            ts = params.time_since_scram
            t = params.t
            return MW/n*pdf*0.0622*(ts**(-0.2)-(t)**(-0.2))
        return MW/n*pdf
    def clad_temp(self, params):
        #use for core nodes to get their clad temp 
        steamTable = XSteam(XSteam.UNIT_SYSTEM_FLS)
    def calc_dP(self, P):
        rho = self.get_rho(P)
        node_length = self.l/self.A 
        node_darby = self.f*self.l/self.D * (self.m/self.A*rho)**2 * 1/(2*gc*rho)
        node_geom = self.k/(2*rho*gc)*(self.m/(self.A*rho))**2
        node_pres = rho*self.dH*g/gc
        #p_drop =  (self.k+self.f*self.LD)*1/2*(self.m/(self.A*rho))**2 - rho*self.dH#node_darby + node_geom + node_pres
        p_drop = (self.k+self.f*self.l/self.D)*0.5*(self.m/(self.get_rho(P)*self.A*gc))**2 - self.get_rho(P)-self.dH
        return p_drop, [node_length, node_darby, node_geom, node_pres]
        
class loop():
    #Create an object that is the combination of nodes for assembling a loop
    def  __init__(self, loops_nodes, params):
        self.loop = []
        self.m_flux = params.rv.mass_flux*52.74/params.n_loops
        self.ptrip = False
        self.m_flux_0 = params.rv.mass_flux*52.74/params.n_loops #Should be nominal steady state mass flux
        for node in loops_nodes:
            node_id = params.node_id
            params.node_id += 1
            self.loop.append(node_base(node, params.all_nodes[node], node_id))
            if node == 6:
                self.loop[-1].LD = 20/7
                
            elif node == 15:
                self.loop[-1].LD = 18/7
            else:
                self.loop[-1].LD = 55/7
        self.rcp_p_r = None
        #self.get_pump_curve(params, self.rcp_p_r)  #needs a dp calculation here    
        for i, node in enumerate(self.loop):
            if i == 0:
                node.next = self.loop[i+1]    
            elif i == len(self.loop)-1:
                node.prev = self.loop[i-1]
            else:
                node.next = self.loop[i+1]
                node.prev = self.loop[i-1]
                
    def get_pump_curve(self, params, RCP_dp_r, trip=0, t = 0, b = 0):
        G = self.m_flux
        G_r = self.m_flux_0
        km = 0.01
        kp = 1e6        
        if not self.ptrip:
            RCP_dp = (1.094+0.089*G/G_r-0.183*(G/G_r)**2)*RCP_dp_r

        else:
            t = params.ptime
            b = params.beta
            if RCP_dp_r == 0:
                km = 1e9
                kp = 1e9
            elif RCP_dp_r == 1e-5:
                km = 0.00000001
                kp = 0.00000001
                RCP_dp_r = 0
            RCP_dp = (1.094+0.089*G/G_r-0.183*(G/G_r)**2)*RCP_dp_r*(1/(1+t/b))
        self.rcp_p = RCP_dp
        self.pumpk = 1/2*(km+kp) + 1/2*(np.abs(G)/G)*((km-kp))

    def momentum(self, param):
        dt = param.dt
        P = param.rv.p
        node_length = 0
        node_darby = 0
        node_geom = 0
        node_pres = 0
        self.get_pump_curve(param, self.rcp_p_r)
        for node in self.loop:   
            rho = node.get_rho(P)
            node_length += node.l/node.A *1/gc*1/dt
            node_darby += node.f*node.l/node.D * (1/node.A)**2 * 1/(2*gc*rho)
            node_geom += (node.k+node.LD*node.f)/(2*rho*gc)*(1/node.A)**2
            node_pres += rho*node.dH*g/gc*1/144
        node_geom += self.pumpk/(2*rho*gc)*(1/node.A)**2
        a = 1 * (node_length) + 2*(node_darby+node_geom)*self.m_flux
        b = 1 * node_length * self.m_flux + (node_darby+node_geom)*self.m_flux**2-node_pres + self.rcp_p
        return a, b    
    
    def get_dP(self, params):
        P = params.p
        node_darby = 0
        node_geom = 0
        node_pres = 0
        for node in self.loop:            
            rho = node.get_rho(P)
            node_darby += node.f*node.LD * (1/node.A)**2 * 1/(2*gc*rho)
            node_geom += node.k/(2*rho*gc)*(1/node.A)**2
            node_pres += rho*node.dH*g/gc*1/144 
        loop_dP = -(node_darby + node_geom)*self.m_flux*np.abs(self.m_flux) - node_pres
        return loop_dP

class core():
    def  __init__(self, core_nodes, params):
        self.core = []
        self.m_flux = params.rv.mass_flux*52.74
        self.dp = None
        for node in core_nodes:
            node_id = params.node_id
            params.node_id += 1
            self.core.append(node_base(node, params.all_nodes[node], node_id))   
            self.core[-1].m = params.rv.mass_flux*52.74
        for i, node in enumerate(self.core):
            if i == 0:
                node.next = self.core[i+1]    
            elif i == len(self.core)-1:
                node.prev = self.core[i-1]
            else:
                node.next = self.core[i+1]
                node.prev = self.core[i-1]
    def momentum(self, param):
        dt = param.dt#*(60*60)
        P = param.rv.p
    
        node_length = 0
        node_darby = 0
        node_geom = 0
        node_pres = 0
        for node in self.core:
            rho = node.get_rho(P)
            node_length += node.l/node.A*1/gc*1/dt
            node_darby += node.f*node.l/node.D * (1/node.A)**2 * 1/(2*gc*rho)
            node_geom += node.k/(2*rho*gc)*(1/node.A)**2
            node_pres += rho*node.dH*g/gc*1/144  

        a = 1 * (node_length) + 2*(node_darby+node_geom)*np.abs(self.m_flux)
        b = 1 * node_length *self.m_flux + (node_darby+node_geom)*self.m_flux**2-node_pres 
        return a, b
        
    def get_dP(self, params):
        P = params.p
        node_darby = 0
        node_geom = 0
        node_pres = 0
        for node in self.core:            
            rho = node.get_rho(P)
            node_darby += node.f*node.l/node.D * (1/node.A)**2 * 1/(2*gc*rho)
            node_geom += node.k/(2*rho*gc)*(1/node.A)**2
            node_pres += rho*node.dH*g/gc*1/144
        core_dP = -(node_darby + node_geom)*self.m_flux*np.abs(self.m_flux) - node_pres
        return core_dP
#    def get_dP(self, param):
#        nodes{i}.temp = given.reactor.inletTemperature; % F
#        nodes{i}.density = @() XSteam('rho_pT',given.reactor.pressure*0.0689475729,(nodes{i}.temp-32)*5/9)*62.42796/1000; % lbm/ft3
#        nodes{i}.internalEnergy = @() XSteam('u_pT',given.reactor.pressure*0.0689475729,(nodes{i}.temp-32)*5/9)*1/2.326; % BTU/lbm
#        nodes{i}.roughness = given.pipeRoughness; % ft
#        nodes{i}.viscosity = @() XSteam('my_pT', given.reactor.pressure*0.0689475729, (nodes{i}.temp-32)*5/9 )*2419.08833; % lb/ft-h
#        nodes{i}.reynoldsNum = @() (nodes{i}.mdot*nodes{i}.De)/(nodes{i}.Ax*nodes{i}.viscosity());



        
            



#if __name__ == '__main__':
#    all_params = params()  
#    loops_nodes = [6,7,8,9,10,11,12,13,14,15]
#    loopA = loop(loops_nodes, all_params)
#    loopB = loop(loops_nodes, all_params)
#    loopC = loop(loops_nodes, all_params)
#    loopD = loop(loops_nodes, all_params)
    
#    core_nodes = [16, 1, 2, 3, 4, 5]
#    core = core(core_nodes, all_params)


































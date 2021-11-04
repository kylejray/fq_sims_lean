from re import L
from sympy import *
from scipy.optimize import fsolve
import numpy as np

px, pxdc, beta, dbeta, g = symbols('phi_x phi_xdc beta beta_delta gamma', real=True)
p, pdc = var('phi phi_dc', real=True)


quadratic_p =  .5 * (p-px)**2
quadratic_pdc = .5* g  * (pdc-pxdc)**2
beta_term = beta *cos(.5*pdc)*cos(p)
dbeta_term = -dbeta * sin(.5*pdc) * sin(p)


U =quadratic_p + quadratic_pdc + beta_term + dbeta_term
U_assym = expand(U.subs({beta:0, g:0})-.5*p**2)

def find_px(db, p_dc, mode='min_of_max'):
    if db==0:
        return 0
    if mode=='min_of_max':
        ans_px = nsolve(U_assym.subs(p, acos(px/(dbeta*sin(.5*pdc)))).subs({pdc:p_dc, dbeta:db}), px, .1)
        if np.isclose(float(im(ans_px)),0):
            return float(re(ans_px))
        else:
            return -db*sin(p_dc/2)
    if mode=='min_of_mid':
        return float(-db*sin(p_dc/2))

required_attributes = {'C':4E-9, 'R':371, 'L':1E-9, 'alpha':2.07E-15/(2*np.pi)}

class DeviceParams():
    def __init__(self, value_dict={}):
        for dictionary in [required_attributes, value_dict]:
            for key, value in dictionary.items():
                setattr(self, key, value)

        if not hasattr(self, 'ell'):
                setattr(self, 'ell', self.L/24)
        if not hasattr(self, 'I_plus'):
                setattr(self, 'I_plus', 6.2*self.alpha/self.L)
        if not hasattr(self, 'I_minus'):
                setattr(self, 'I_minus', .2*self.alpha/self.L)
        
        self.refresh()
        
    def refresh(self):
        self.gamma = self.L/(2*self.ell)
        self.beta = self.I_plus*self.L/self.alpha
        self.dbeta = self.I_minus*self.L/self.alpha
        self.U_0 = self.alpha**2/self.L
    
    def change_vals(self, value_dict):
        for key, value in value_dict.items():
            setattr(self, key, value)
        self.refresh()

def fidelity(jumps):
    out = {}
    names = ['+ to -', '- to +']
    tot_fails = 0
    total =0
    for i, key in enumerate(jumps):
        length = len(jumps[key])
        succ, tot = sum(jumps[key]==2), sum(jumps[key]!=0)
        out[names[i]] = succ/tot
        tot_fails += tot-succ
        total += tot
    if total != length:
        print ('missed counting {} trajectories in fidelity'.format(length-total))
    out['overall'] = 1-tot_fails/total

    return out

    



             
        
        



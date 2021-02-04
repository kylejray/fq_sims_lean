from protocol_designer.potentials import Potential
from protocol_designer.protocol import Protocol, Compound_Protocol
import numpy as np

#first we defined the potential

def flux_qubit_pot(p,pdc, params):
    px, pxdc, gamma, beta, dbeta= params
    U = .5 * (p-px)**2 + .5*gamma*(pdc-pxdc)**2 + beta*np.cos(.5*pdc)*np.cos(p) + dbeta*np.sin(.5*pdc)*np.sin(p)
    return U

def flux_qubit_force(p,pdc, params):
    px, pxdc, gamma, beta, dbeta= params
    
    dp = (p-px) - beta*np.cos(.5*pdc)*np.sin(p) + dbeta*np.sin(.5*pdc)*np.cos(p)
    
    dpdc = gamma*(pdc-pxdc) - .5*beta*np.sin(.5*pdc)*np.cos(p) + .5*dbeta*np.cos(.5*pdc)*np.sin(p)
    
    return (-dp, -dpdc)
#realistic:
default_real = (-.1018, -2.3, 12, 6.2, .1)
#symmetric approximation:
default_symm = (0, -2.3, 12, 6.2, 0)
#domain
dom = ((-3,-3),(3,-1))

fq_pot = Potential(flux_qubit_pot, flux_qubit_force, 5, 2, default_params=default_symm, relevant_domain=dom)

#once the potential is done, we make some simple one-time-step protocols, taht will serves as buulding blocks

#the protocol below will start with the default parameters and then go to the 'flip parameters' at t=1
prm = np.zeros((5,2))
prm[:, 0] = fq_pot.default_params
prm[:, 1] = fq_pot.default_params
prm[1, 1] = -2*3.1416

t=(0,1)

flip_on = Protocol(t, prm)
flip_on.interpolation = 'step'

#the protocol below will start with the 'flip parameters' and go to the defaults at t=1
flip_off = flip_on.copy()
flip_off.reverse()

#same as above but shifted to run from t=1 to t=2 instead
flip_off_shift = flip_off.copy()
flip_off_shift.time_shift(1)


#combinig protocol steps into a Compound_Protocol allows for a full computational cycle
flip_prot = Compound_Protocol([flip_on,flip_off_shift])


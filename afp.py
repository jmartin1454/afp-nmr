#!/usr/bin/python

# Simulates AFP-NMR sequence by integrating the Bloch equations
# by Jeff

from math import *
import numpy as np
from scipy.integrate import ode
import scipy.constants as c
import matplotlib.pyplot as plt

# global parameters of the calculation
# constants of nature
pi=c.pi             # pi
hbar=c.hbar         # hbar
k=c.k               # Boltzmann's constant (SI)
go2pi=c.physical_constants['proton gyromag. ratio over 2 pi'][0]
                    # gamma/(2*pi) in Hz/uT
gamma=go2pi*2*pi    # gamma (radians/s)/uT
mu_p=c.physical_constants['proton mag. mom.'][0]
                    # proton magnetic moment (J/T)

# Boltzmann polarization
T=295               # Kelvin
c=mu_p/(k*T)/1e6    # (1/uT) Boltzmann polarization is Bz times this.

# relaxation times
T1=0.050           # seconds
T2=0.049           # seconds

# afp parameters
freq=25.8e3         # b1 frequency (Hz)
omega=freq*2*pi     # b1 frequency (rad/s)
centralb0=omega/gamma
                    # central b0 field (uT)
b1=0.01*centralb0   # b1 amplitude (uT)
deltab=b1*20.       # range to sweep (uT)
deltat=1.           # sweep time (seconds)
b00=centralb0-deltab/2
                    # starting b0
db0dt=deltab/deltat # b0 ramp rate (uT/s)

s0=[0,0,-c*b00]      # starting polarization vector
t0=0.               # starting time
t1=2               # ending time (seconds)
dt=.00001           # time step (seconds)

# lock-in parameters
tau=.01              # lock-in time constant (seconds)

# sparsification factor
sparse=100000
i=0

def b(t):
    bx= b1*cos(omega*t)
    by=-b1*sin(omega*t)
    n=int(t/deltat) # nth up (even) or down (odd)
    if(n%2==0):     # up sweep
        bz=b00+db0dt*(t-n*deltat)
    else:           # down sweep
        bz=b00+deltab-db0dt*(t-n*deltat)
    return [bx,by,bz]  # uT

def dsdt(t,s):
    # The Bloch equations in the non-rotating frame
    # f = dS/dt = gamma S x B + relaxation
    sx,sy,sz=s
    bx,by,bz=b(t)
    dsxdt=gamma*(sy*bz-sz*by)-sx/T2
    dsydt=gamma*(sz*bx-sx*bz)-sy/T2
    dszdt=gamma*(sx*by-sy*bx)-(sz-c*bz)/T1
    return [dsxdt,dsydt,dszdt]

r=ode(dsdt).set_integrator('dop853',atol=s0[2]*1e-8)
# Seems necessary to crank down atol for small initial polarizations?
r.set_initial_value(s0,t0)

s_demod_x=np.array([0,0,0]) # components of polarization demodulated with cos
s_demod_y=np.array([0,0,0]) # components of polarization demodulated with sin
sol=[]
sol_demod_x=[]
sol_demod_y=[]
while r.successful() and r.t < t1:
    r.integrate(r.t+dt) # do an integration step
    t=r.t               # time after the step
    s=r.y               # polarization vector after the step
    sx,sy,sz=s          # polarization vector after the step
    s=np.array(s)
    s_demod_x=s_demod_x+(-s_demod_x+s*cos(omega*t))*dt/tau
    s_demod_y=s_demod_y+(-s_demod_y+s*sin(omega*t))*dt/tau
    i+=1
    if(i%sparse):
        sol.append([t,sx,sy,sz,b(t)[2]])
        sol_demod_x.append([s_demod_x[0],s_demod_x[1],s_demod_x[2]])
        sol_demod_y.append([s_demod_y[0],s_demod_y[1],s_demod_y[2]])
        i=0

sol=np.array(sol)
sol_demod_x=np.array(sol_demod_x)
sol_demod_y=np.array(sol_demod_y)

plt.plot(sol[:,0],sol[:,1])
plt.plot(sol[:,0],sol[:,2])
plt.plot(sol[:,0],sol[:,3])
plt.plot(sol[:,0],sol_demod_x[:,0])
plt.xlabel('Time (s)')
plt.ylabel('Spin components and demodulation')
#plt.plot(sol[:,0],sol_demod_x[:,1])
#plt.plot(sol[:,0],sol_demod_x[:,2])
#plt.plot(sol[:,0],sol_demod_y[:,0])
#plt.plot(sol[:,0],sol_demod_y[:,1])
#plt.plot(sol[:,0],sol_demod_y[:,2])
plt.show()

plt.plot(sol[:,0],sol[:,4])
plt.xlabel('Time (s)')
plt.ylabel('B ($\mu$T)')
plt.show()

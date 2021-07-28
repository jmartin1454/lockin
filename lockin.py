#!/usr/bin/python3

from math import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fsamp=200000 # Hz
dt=1/fsamp # s
RC=1./10000. # s

A=1 # V
f=7000 # Hz
tau=0.02 # s

capital_T=0.04 # s, total time of simulation
n_repeat=4 # number of times to repeat

t=np.arange(0,capital_T*n_repeat,dt)
t_repeat=np.tile(np.arange(0,capital_T,dt),n_repeat)
Vin=A*np.cos(2*pi*f*t_repeat)*np.exp(-t_repeat/tau)+t_repeat/10/tau+np.random.normal(size=np.shape(t))*A*.1
plt.plot(t,Vin)

phi_ref_deg=90.
phi_ref_rad=(phi_ref_deg/360.0)*(2.0*np.pi)
f_ref=f+100

cos_ref=np.cos(2*pi*f_ref*t+phi_ref_rad)
sin_ref=np.sin(2*pi*f_ref*t+phi_ref_rad)


t_lockin=.001 # s

sos=signal.butter(4,1/t_lockin,'lp',fs=fsamp,output='sos')
filtered=signal.sosfilt(sos,Vin)
plt.plot(t,filtered)

x=signal.sosfilt(sos,cos_ref*Vin)
y=signal.sosfilt(sos,sin_ref*Vin)
plt.plot(t,x)
plt.plot(t,y)

plt.show()

plt.plot(t,np.unwrap(np.arctan2(y,x)))
plt.plot(t,np.sqrt(y**2+x**2))
plt.show()

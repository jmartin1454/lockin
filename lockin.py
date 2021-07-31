#!/usr/bin/python3

from math import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-f", "--file", dest="infile",
                  default="sample_short.txt", help="read data from file",
                  metavar="FILE")

(options, args) = parser.parse_args()


class input_waveform:
    def __init__(self,t,V):
        self.V=V
        self.t=t
        self.dt=t[1]-t[0]
        self.fs=1/self.dt

    @classmethod
    def fromfakedata(cls):
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
        return cls(t,Vin)

    @classmethod
    def fromfile(cls,fn):
        V,t=np.genfromtxt(fn,delimiter=',',unpack=True)
        print('File %s read successfully'%fn)
        return cls(t,V)

#data=input_waveform.fromfakedata()
fn=options.infile
data=input_waveform.fromfile(fn)
t=data.t
Vin=data.V
fs=data.fs
dt=t[1]-t[0]

plt.plot(t,Vin,label='input')


class lockin_amplifier:
    def __init__(self,t):
        self.dt=t[1]-t[0]
        self.fs=1/self.dt
        self.phi_ref_deg=0.
        self.phi_ref_rad=(self.phi_ref_deg/360.0)*(2.0*np.pi)
        self.f_ref=6940

        self.cos_ref=np.cos(2*pi*self.f_ref*t+self.phi_ref_rad)
        self.sin_ref=np.sin(2*pi*self.f_ref*t+self.phi_ref_rad)


        self.tau_lockin=.001 # s

        self.sos=signal.butter(4,1/self.tau_lockin,'lp',fs=self.fs,output='sos')
        #filtered=signal.sosfilt(sos,Vin)
        #plt.plot(t,filtered)

    def amplify(self,Vin):
        self.x=signal.sosfilt(self.sos,self.cos_ref*Vin)
        self.y=signal.sosfilt(self.sos,self.sin_ref*Vin)
        self.r=np.sqrt(self.y**2+self.x**2)
        self.theta=np.unwrap(np.arctan2(self.y,self.x))
        return self.x,self.y,self.r,self.theta

lockin=lockin_amplifier(t)
x,y,r,theta=lockin.amplify(Vin)

plt.plot(t,x,label='X')
plt.plot(t,y,label='Y')
plt.xlabel('t (s)')
plt.ylabel('V (V)')
plt.legend()
plt.show()

plt.plot(t,theta/2/pi,label=r'$\theta/2\pi$')
plt.plot(t,r/np.amax(r),label='R/max(R)')
#freq=np.diff(theta)/2/pi/dt # seems to add noise
#freq=np.append(freq,freq[-1]) # differentiation removes one entry
#plt.plot(t,freq,label='f-%f (Hz)'%lockin.f_ref)
plt.xlabel('t (s)')
plt.legend()
plt.show()

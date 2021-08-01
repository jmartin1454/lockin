#!/usr/bin/python3

from math import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import stats
from optparse import OptionParser


parser = OptionParser()

parser.add_option("-f", "--file", dest="infile",
                  default="sample_short.txt.gz", help="read data from file",
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

# http://www.ap.smu.ca/~agolob/phys2300/blog/climate-change/
# (Hope it's right... you can get it from Taylor or Bevington, too.)
def OLSfit(x, y, dy=None):
    """Find the best fitting parameters of a linear fit to the data through the
    method of ordinary least squares estimation. (i.e. find m and b for
    y = m*x + b)

    Args:
        x: Numpy array of independent variable data
        y: Numpy array of dependent variable data. Must have same size as x.
        dy: Numpy array of dependent variable standard deviations. Must be same
            size as y.

    Returns: A list with four floating point values. [m, dm, b, db]
    """
    if dy is None:
        #if no error bars, weight every point the same
        dy = np.ones(x.size)
    denom = np.sum(1 / dy**2) * np.sum((x / dy)**2) - (np.sum(x / dy**2))**2
    m = (np.sum(1 / dy**2) * np.sum(x * y / dy**2) -
         np.sum(x / dy**2) * np.sum(y / dy**2)) / denom
    b = (np.sum(x**2 / dy**2) * np.sum(y / dy**2) -
         np.sum(x / dy**2) * np.sum(x * y / dy**2)) / denom
    dm = np.sqrt(np.sum(1 / dy**2) / denom)
    db = np.sqrt(np.sum(x / dy**2) / denom)
    return([m, dm, b, db])

rnorm=r/np.amax(r)
plt.plot(t,theta/2/pi,label=r'$\theta/2\pi$')
plt.plot(t,rnorm,label='R/max(R)')
#freq=np.diff(theta)/2/pi/dt # seems to add noise
#freq=np.append(freq,freq[-1]) # differentiation removes one entry
#plt.plot(t,freq,label='f-%f (Hz)'%lockin.f_ref)

# The rest of this is just trying to define the ranges of data to fit,
# based on looking at rnorm and trying to figure it out.

class schmitt_trigger:
    def __init__(self,hi,lo):
        self.hi=hi
        self.lo=lo
    def return_trigger(self,data):
        trigger=np.empty_like(data)
        nowlevel=self.lo
        for i in range(len(data)):
            if(nowlevel==self.lo and data[i]>self.hi):
                nowlevel=self.hi
            if(nowlevel==self.hi and data[i]<self.lo):
                nowlevel=self.lo
            trigger[i]=nowlevel
        return trigger

schmitt=schmitt_trigger(.7,.3)
trigger=schmitt.return_trigger(rnorm)
plt.plot(t,trigger,label='trigger')

hilo=np.diff(trigger)
hilo=np.append(hilo,hilo[-1])
plt.plot(t,hilo,label='trigger transitions')
rising_edges=t[hilo>.3]
falling_edges=t[hilo<-.3]


maxtimes=np.empty_like(rising_edges)
mintimes=np.empty_like(maxtimes)
freq=np.empty_like(maxtimes)
for i in range(len(rising_edges)):
    maxtimes[i]=t[np.argmax(np.ma.masked_where((t<rising_edges[i])|(t>falling_edges[i]),rnorm))]
    if(i<len(rising_edges)-1):
        mintimes[i]=t[np.argmax(np.ma.masked_where((t<falling_edges[i])|(t>rising_edges[i+1]),-rnorm))]
    else:
        mintimes[i]=t[np.argmax(np.ma.masked_where((t<falling_edges[i]),-rnorm))]
    maxtimes[i]=maxtimes[i]+3*lockin.tau_lockin
    mintimes[i]=maxtimes[i]+.05
    mask=(t>maxtimes[i])&(t<mintimes[i])
    #res=stats.linregress(t[mask],theta[mask]/2/pi)
    #plt.plot(t[mask],res.intercept+res.slope*t[mask],'r')    
    #freq[i]=res.slope
    res=OLSfit(t[mask],theta[mask]/2/pi,1/rnorm[mask])
    plt.plot(t[mask],res[2]+res[0]*t[mask],'r')    
    freq[i]=res[0]
    
#print(maxtimes)
#print(mintimes)
print(freq)


plt.xlabel('t (s)')
plt.legend()
plt.show()

gamma=7000 # Hz/uT
true_freq=lockin.f_ref-freq
print(true_freq)
field=true_freq/gamma # uT
field=field*1e6 # pT
plt.plot(maxtimes,field,label='trigger')
plt.show()

plt.hist(field)
print(np.mean(field),np.std(field))
plt.show()
